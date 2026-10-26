"""fMRI expert: TabM-mini on the 136 standardised chart deviations.
  TabMmini  32 members, each with its own input scaling (random-sign start) and linear head, sharing an MLP d -> 128 -> 128
            (ReLU, dropout 0.1); 42,528 parameters for 136 inputs.
  train     AdamW (lr 1e-3, weight decay 1e-4), batch 64, gradient clipping 1.0; loss = mean over members of each member's
            binary cross-entropy; early stopping on the cross-entropy of the mean member probability (patience 20, at most
            300 epochs); the best weights are restored.
  output    log-odds of the mean member probability."""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import predict_mc

EPS = 1e-6


@torch.no_grad()
def mc_dropout(model, X, M, passes, seed):
    """MC-dropout predictions; the global torch random state is restored afterwards"""
    state = torch.get_rng_state()
    try:
        return predict_mc(model, X, M, passes, seed)
    finally:
        torch.set_rng_state(state)


class TabMmini(nn.Module):
    def __init__(self, d_in, k=32, w=128, p=0.1):
        super().__init__()
        self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        self.l1, self.l2, self.drop = nn.Linear(d_in, w), nn.Linear(w, w), nn.Dropout(p)
        self.hw = nn.Parameter(torch.empty(k, w).uniform_(-w ** -0.5, w ** -0.5))
        self.hb = nn.Parameter(torch.zeros(k))

    def forward(self, x):                                       # B x d -> B x k member logits
        h = x[:, None, :] * self.r[None]
        h = self.drop(F.relu(self.l1(h)))
        h = self.drop(F.relu(self.l2(h)))
        return (h * self.hw[None]).sum(-1) + self.hb

    def hidden(self, x):                                        # B x d -> B x w: the second hidden layer, mean over members
        h = x[:, None, :] * self.r[None]
        h = self.drop(F.relu(self.l1(h)))
        h = self.drop(F.relu(self.l2(h)))
        return h.mean(1)


def mean_prob(out):
    return torch.sigmoid(out).mean(-1) if out.dim() == 2 else torch.sigmoid(out)


class FMRIExpert(nn.Module):
    """TabMmini with the interface of the transformer branch: forward(x, m, return_embedding=False) -> log-odds of the mean member
    probability (or the 128-value embedding); m (missing flags) is accepted and not used"""
    def __init__(self, d_in, cfg):
        super().__init__()
        self.net = TabMmini(d_in, int(cfg["members"]), int(cfg["width"]), float(cfg["dropout"]))
        self.ln = nn.Identity()                                 # no parameters; predict() reads ln.normalized_shape for an empty input
        self.ln.normalized_shape = (int(cfg["width"]),)

    def forward(self, x, m=None, return_embedding=False):
        if return_embedding:
            return self.net.hidden(x)
        p = mean_prob(self.net(x)).clamp(EPS, 1 - EPS)
        return torch.log(p) - torch.log1p(-p)

    @torch.no_grad()
    def member_sd(self, x):
        """eval-mode SD (ddof 1) over the members' logits"""
        was = self.training
        self.eval()
        out = self.net(x).std(dim=1, unbiased=True)
        self.train(was)
        return out


def train_tabm(X, y, tr, es, cfg, seed):
    """train one TabM-mini and return it wrapped as FMRIExpert, with a training summary; X = standardised inputs (people x d),
    y in {0, 1}, tr / es = boolean masks of the training and early-stopping rows"""
    torch.set_num_threads(int(cfg["threads"]))
    torch.manual_seed(seed)
    model = TabMmini(X.shape[1], int(cfg["members"]), int(cfg["width"]), float(cfg["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    f32 = torch.float32
    Xt, yt = torch.as_tensor(X[tr], dtype=f32), torch.as_tensor(y[tr], dtype=f32)
    Xe, ye = torch.as_tensor(X[es], dtype=f32), torch.as_tensor(y[es], dtype=f32)
    g = torch.Generator().manual_seed(seed)
    best, state, best_ep, bad, ep, history = float("inf"), copy.deepcopy(model.state_dict()), 0, 0, 0, []
    batch, patience, max_epochs, clip = int(cfg["batch"]), int(cfg["patience"]), int(cfg["max_epochs"]), float(cfg["grad_clip"])
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(yt), generator=g)
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            out = model(Xt[idx])
            tgt = yt[idx][:, None].expand_as(out) if out.dim() == 2 else yt[idx]
            loss = F.binary_cross_entropy_with_logits(out, tgt)
            if not math.isfinite(loss.detach().item()):
                raise RuntimeError(f"non-finite training loss at epoch {ep} (seed {seed})")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(F.binary_cross_entropy(mean_prob(model(Xe)).clamp(EPS, 1 - EPS), ye))
        if not math.isfinite(v):
            raise RuntimeError(f"non-finite VAL-ES loss at epoch {ep} (seed {seed})")
        history.append(v)
        if v < best - 1e-6:
            best, state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_ep == 0:
        raise RuntimeError(f"no finite VAL-ES loss (seed {seed})")
    model.load_state_dict(state)
    model.eval()
    model.zero_grad(set_to_none=True)
    expert = FMRIExpert(X.shape[1], cfg)
    expert.net = model
    expert.eval()
    return expert, dict(epochs=ep, best_epoch=best_ep, val_es_bce=best, history=history)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def fbranch_logits(raw, has_fmri, train_mask):
    """logits for one seed; people without fMRI get the seed's mean logit over the training people with fMRI"""
    out = np.array(raw, dtype=np.float64, copy=True)
    fill = float(np.mean(raw[train_mask & has_fmri]))
    out[~has_fmri] = fill
    return out
