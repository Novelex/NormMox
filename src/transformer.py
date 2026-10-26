"""Transformer expert used by the deep fusion baselines.
Branch = feature tokenizer (token_j = z_j * W_j + B_j + miss_j * U, as in FT-Transformer) + a CLS token
       -> 2 pre-norm transformer layers (d 32, 4 heads, FFN 64, GELU, dropout 0.3 on both residual branches, attention dropout 0.1);
          the attention logits get Bias[h, type(i, j)]: one learned table per branch, shared by both layers (as in Graphormer)
       -> LayerNorm(CLS) -> Linear(32 -> 1) = the branch logit.
Relation types (token 0 = CLS):
  fMRI (136 network-pair tokens): number of shared networks 0 / 1 / 2 (2 only on the diagonal); CLS edges 'cls'.
  sMRI (185 tokens), in priority order: homolog > same Yeo-17 network > same hemisphere > volume-volume > thickness-volume > other;
  CLS edges 'cls'. A token with itself gets the ordinary type.
W, B and CLS start from N(0, 0.02^2), U and the relation bias at 0. Training: AdamW, mini-batches, gradient clipping, early stopping
on the binary cross-entropy of the early-stopping rows, best weights restored; fixed torch threads and seeds."""
import copy
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

FMRI_TYPES = ["shared0", "shared1", "shared2", "cls"]
STRUCT_TYPES = ["homolog", "same_network", "same_hemisphere", "vol_vol", "thick_vol", "other", "cls"]
SEED_BASE = 20261019


def torch_seed(rep, fold, seed, branch):
    """one fixed 32-bit seed per (repeat, fold, seed, branch); branch 0 = fMRI, 1 = structure, 2 = joint models"""
    return int(np.random.SeedSequence([SEED_BASE, int(rep), int(fold), int(seed), int(branch)]).generate_state(1)[0])


def network_map(yeo_map_path, cols):
    """thickness column -> Yeo-17 network from a region-to-network table; key column 'region' or 'destrieux', whichever matches more
    structure columns. Returns (map, key column, number of matching columns); callers must check that 148 columns match."""
    ym = pd.read_csv(yeo_map_path)
    key = max((c for c in ("region", "destrieux") if c in ym.columns), key=lambda c: int(ym[c].isin(cols).sum()))
    return dict(zip(ym[key], ym["network"].astype(str))), key, int(ym[key].isin(cols).sum())


def fmri_relations(cols):
    """(T + 1) x (T + 1) relation types for the network-pair tokens (names like 'ContA_DefaultB')"""
    nets = [frozenset(c.split("_")) for c in cols]
    T = len(cols)
    R = np.full((T + 1, T + 1), FMRI_TYPES.index("cls"), dtype=np.int64)
    for i in range(T):
        for j in range(T):
            R[i + 1, j + 1] = len(nets[i] & nets[j])
    return R


def hemisphere(c):
    if c.startswith("lh_") or c.startswith("Left-"):
        return "L"
    if c.startswith("rh_") or c.startswith("Right-"):
        return "R"
    return None


def homolog_key(c):
    for prefix in ("lh_", "rh_", "Left-", "Right-"):
        if c.startswith(prefix):
            return c[len(prefix):]
    return None


def struct_relations(cols, network_of):
    """(T + 1) x (T + 1) relation types for the structure tokens; network_of maps a thickness column to its Yeo-17 network"""
    T = len(cols)
    thick = [c.endswith("_thickness") for c in cols]
    hemi = [hemisphere(c) for c in cols]
    key = [homolog_key(c) for c in cols]
    R = np.full((T + 1, T + 1), STRUCT_TYPES.index("cls"), dtype=np.int64)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i != j and key[i] is not None and key[i] == key[j] and hemi[i] is not None and hemi[j] is not None and hemi[i] != hemi[j]:
                t = "homolog"
            elif a in network_of and b in network_of and network_of[a] == network_of[b]:
                t = "same_network"
            elif hemi[i] is not None and hemi[i] == hemi[j]:
                t = "same_hemisphere"
            elif not thick[i] and not thick[j]:
                t = "vol_vol"
            elif thick[i] != thick[j]:
                t = "thick_vol"
            else:
                t = "other"
            R[i + 1, j + 1] = STRUCT_TYPES.index(t)
    return R


class BiasedSelfAttention(nn.Module):
    def __init__(self, d, heads, attn_dropout):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(attn_dropout)

    def forward(self, x, rel_bias):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)     # each (B, heads, L, dh)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh) + rel_bias    # (B, heads, L, L) + (heads, L, L)
        att = self.drop(torch.softmax(scores, dim=-1))
        return self.out(torch.matmul(att, v).transpose(1, 2).reshape(B, L, d))


class Layer(nn.Module):
    def __init__(self, d, heads, ffn, dropout, attn_dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = BiasedSelfAttention(d, heads, attn_dropout)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rel_bias):
        x = x + self.drop(self.att(self.ln1(x), rel_bias))
        return x + self.drop(self.ff(self.ln2(x)))


class Branch(nn.Module):
    def __init__(self, relations, n_types, cfg):
        super().__init__()
        n_feat, d = relations.shape[0] - 1, cfg["d"]
        self.W = nn.Parameter(0.02 * torch.randn(n_feat, d))
        self.B = nn.Parameter(0.02 * torch.randn(n_feat, d))
        self.U = nn.Parameter(torch.zeros(d))
        self.cls = nn.Parameter(0.02 * torch.randn(1, 1, d))
        self.rel_bias = nn.Parameter(torch.zeros(cfg["heads"], n_types))               # Bias[h, type(i, j)], shared by the layers
        self.register_buffer("onehot", F.one_hot(torch.as_tensor(relations, dtype=torch.long), n_types).float(), persistent=False)
        self.layers = nn.ModuleList([Layer(d, cfg["heads"], cfg["ffn"], cfg["dropout"], cfg["attn_dropout"]) for _ in range(cfg["layers"])])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x, m, return_embedding=False):
        tok = x.unsqueeze(-1) * self.W + self.B + m.unsqueeze(-1) * self.U          # (batch, T, d)
        h = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)             # (batch, T + 1, d)
        rel_bias = torch.einsum("ht,ijt->hij", self.rel_bias, self.onehot)          # (heads, T + 1, T + 1)
        for layer in self.layers:
            h = layer(h, rel_bias)
        emb = self.ln(h[:, 0])                                                       # (batch, d): the CLS embedding
        return emb if return_embedding else self.head(emb).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _batches(X, M, batch):
    for s in range(0, len(X), batch):
        yield torch.as_tensor(X[s:s + batch], dtype=torch.float32), torch.as_tensor(M[s:s + batch], dtype=torch.float32)


@torch.no_grad()
def predict(model, X, M, batch=256, embedding=False):
    """logits (or CLS embeddings) in eval mode, no dropout"""
    if len(X) == 0:
        return np.zeros((0, model.ln.normalized_shape[0])) if embedding else np.zeros(0)
    model.eval()
    return torch.cat([model(x, m, return_embedding=embedding) for x, m in _batches(X, M, batch)]).numpy().astype(np.float64)


@torch.no_grad()
def predict_mc(model, X, M, passes, seed, batch=256):
    """MC-dropout logits: dropout kept on (the model has no batch-norm), `passes` passes from a fixed seed;
    returns the mean and the SD (ddof 1) per person"""
    if len(X) == 0:
        return np.zeros(0), np.zeros(0)
    model.train()
    torch.manual_seed(seed)
    runs = torch.stack([torch.cat([model(x, m) for x, m in _batches(X, M, batch)]) for _ in range(int(passes))]).numpy().astype(np.float64)
    model.eval()
    return runs.mean(axis=0), runs.std(axis=0, ddof=1)


def train_branch(X, M, y, tr, es, relations, n_types, mcfg, tcfg, seed):
    """train one branch on the training rows, early stopping on the early-stopping rows; returns the best model and a summary"""
    torch.set_num_threads(int(tcfg["threads"]))
    torch.manual_seed(seed)
    model = Branch(relations, n_types, mcfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    f32 = torch.float32
    Xt, Mt, yt = torch.as_tensor(X[tr], dtype=f32), torch.as_tensor(M[tr], dtype=f32), torch.as_tensor(y[tr], dtype=f32)
    Xe, Me, ye = torch.as_tensor(X[es], dtype=f32), torch.as_tensor(M[es], dtype=f32), torch.as_tensor(y[es], dtype=f32)
    order = torch.Generator().manual_seed((int(seed) ^ 0x9E3779B9) & 0xFFFFFFFF)       # batch order: its own stream, not the init stream
    best_bce, best_state, best_epoch, bad, history = float("inf"), copy.deepcopy(model.state_dict()), 0, 0, []
    epoch = 0
    for epoch in range(1, int(tcfg["max_epochs"]) + 1):
        model.train()
        perm = torch.randperm(len(yt), generator=order)
        for s in range(0, len(perm), int(tcfg["batch"])):
            idx = perm[s:s + int(tcfg["batch"])]
            loss = F.binary_cross_entropy_with_logits(model(Xt[idx], Mt[idx]), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(F.binary_cross_entropy_with_logits(model(Xe, Me), ye))
        history.append(val)
        if val < best_bce - 1e-6:
            best_bce, best_state, best_epoch, bad = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= int(tcfg["patience"]):
                break
    if best_epoch == 0 or not math.isfinite(best_bce):
        raise RuntimeError(f"training never reached a finite VAL-ES loss (seed {seed}, last values {history[-3:]})")
    model.load_state_dict(best_state)
    model.eval()
    return model, dict(epochs=epoch, best_epoch=best_epoch, val_es_bce=best_bce, history=history)
