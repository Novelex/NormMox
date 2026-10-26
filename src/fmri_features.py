"""Yeo-17 network connectivity (136 Fisher-z values) from every C-PAC func_preproc scan (ABIDE Preprocessed, no band-pass filter,
no global signal regression).
Modes:
  python src/fmri_features.py prepare   unzip the Yeo atlas and put the 17-network map (thin mask) on the 3 mm scan grid
  python src/fmri_features.py           all people, or this array task's share (SLURM_ARRAY_TASK_ID of SLURM_ARRAY_TASK_COUNT);
                                        workers = SLURM_CPUS_PER_TASK (default 4)
Per person (the scan is read once):
  brain mask = ABIDE's own func_mask (fallback: voxels whose signal is not flat, recorded); flat voxels inside the mask are left out
  no smoothing
  atlas: Yeo 2011 17 networks, MNI152 1 mm, thin mask (Yeo2011_17Networks_MNI152_FreeSurferConformed1mm), moved to the scan grid by
  nearest neighbour; 17 mean signals matched by label number 1..17 (a network with no voxel inside the mask gives NaN and a coverage of 0)
  drift: cosine high-pass 128 s (the discrete-cosine basis of fMRIPrep's cosine_XX regressors and nilearn's filter='cosine',
  removed by least squares) -> Pearson correlation -> Fisher z (arctanh).
No time point removed, no band-pass, no global signal, no extra detrending (C-PAC already removed linear + quadratic trends).
Saves data/derived/fmri_per_person/<FILE_ID>.npz: z[17,17] (label order, NaN diagonal), coverage[17], nvox[17], nvol, tr,
mask_source, grid_ok, flat_in_mask, mask_vox. Existing files are skipped (restartable).
Label k is named with the k-th network of the chart's list (data/raw/braincharts/docs/phenotypes_yeo17.txt) by src/fmri_merge.py.
"""
import os, sys, time, zipfile
import numpy as np
import nibabel as nib
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNC = f"{ROOT}/data/raw/func_preproc"
MASKD = f"{ROOT}/data/raw/func_mask"
ATLAS_ZIP = f"{ROOT}/data/raw/yeo/Yeo_JNeurophysiol11_MNI152.zip"
ATLAS_DIR = f"{ROOT}/data/raw/yeo"
PP = f"{ROOT}/data/derived/fmri_per_person"
REF_ID = os.environ.get("REF_ID", "OHSU_0050147")           # the scan whose grid the atlas is prepared on (every scan is checked against it)
ATLAS_FILE = "Yeo_JNeurophysiol11_MNI152/Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz"
HIGH_PASS = 1.0 / 128
NET = 17


def labels_on_grid(atlas_img, shape, affine):
    """Nearest neighbour: the atlas label at the MNI position of every voxel centre of the target grid (what nilearn does for label images)."""
    ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)]).astype(float)
    src = np.linalg.inv(atlas_img.affine) @ (affine @ vox)                 # target voxel -> MNI mm -> atlas voxel
    idx = np.floor(src[:3] + 0.5).astype(int)                                # nearest voxel, ties rounded up (scipy affine_transform order 0, as nilearn)
    data = np.asarray(atlas_img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    inside = np.all((idx >= 0) & (idx < np.array(data.shape)[:, None]), axis=0)
    out = np.zeros(ii.size, dtype=np.int16)
    out[inside] = data[idx[0, inside], idx[1, inside], idx[2, inside]]
    return out.reshape(shape)


def cosine_basis(n, tr, high_pass):
    """fMRIPrep / nilearn discrete-cosine drift basis, constant column included (as nilearn's); removing the mean does not change a correlation."""
    order = min(n - 1, int(np.floor(2 * n * high_pass * tr)))
    t = np.arange(n)
    X = np.zeros((n, order + 1))
    for k in range(1, order + 1):
        X[:, k - 1] = np.sqrt(2.0 / n) * np.cos((np.pi / n) * (t + 0.5) * k)
    X[:, -1] = 1.0
    return X


def remove_drift(sig, X):
    """Least-squares removal of the basis X from every column of sig (NaN columns stay NaN)."""
    out = np.array(sig, copy=True)
    ok = ~np.isnan(sig).any(axis=0)
    if ok.any():
        beta, *_ = np.linalg.lstsq(X, sig[:, ok], rcond=None)
        out[:, ok] = sig[:, ok] - X @ beta
    return out


def network_means(data, labels, mask):
    """Mean signal of each label 1..17 over its voxels inside the mask. Returns (T x 17 with NaN columns for empty networks, voxel counts)."""
    T = data.shape[3]
    sig = np.full((T, NET), np.nan)
    nvox = np.zeros(NET, dtype=int)
    for k in range(1, NET + 1):
        sel = (labels == k) & mask
        nvox[k - 1] = int(sel.sum())
        if nvox[k - 1] > 0:
            sig[:, k - 1] = data[sel].mean(axis=0)
    return sig, nvox


def fisher_z(sig):
    """Pearson correlation of the 17 signals -> Fisher z; NaN diagonal; NaN rows/columns for empty networks."""
    z = np.full((NET, NET), np.nan)
    ok = ~np.isnan(sig).any(axis=0)
    if ok.sum() >= 2:
        r = np.corrcoef(sig[:, ok].T)
        r = np.clip(r, -0.9999999, 0.9999999)
        zz = np.arctanh(r)
        np.fill_diagonal(zz, np.nan)
        idx = np.where(ok)[0]
        z[np.ix_(idx, idx)] = zz
    return z


_CACHE = {}


def atlas_for(shape, affine):
    """Label map on this grid: the prepared 3 mm map if the grid is the reference grid, else recomputed from the 1 mm atlas."""
    key = (tuple(shape), np.round(affine, 4).tobytes())
    if key not in _CACHE:
        ref = nib.load(f"{ATLAS_DIR}/yeo17_thin_3mm.nii.gz")
        if tuple(ref.shape[:3]) == tuple(shape) and np.allclose(ref.affine, affine, atol=1e-3):
            labels = np.asarray(ref.dataobj).astype(np.int16)
            grid_ok = True
        else:
            labels = labels_on_grid(nib.load(f"{ATLAS_DIR}/{ATLAS_FILE}"), shape, affine)
            grid_ok = False
        _CACHE[key] = (labels, np.array([(labels == k).sum() for k in range(1, NET + 1)]), grid_ok)
    return _CACHE[key]


def load_mask(fid, shape, affine, nonflat):
    """ABIDE's func_mask on the same grid, else the non-flat voxels. A flat voxel (constant in time) inside the mask is left out,
    so a network mean can never be constant (which would give an undefined correlation without a flag)."""
    p = f"{MASKD}/{fid}_func_mask.nii.gz"
    if os.path.exists(p):
        m = nib.load(p)
        if tuple(m.shape[:3]) == tuple(shape) and np.allclose(m.affine, affine, atol=1e-3):
            mask = np.asarray(m.dataobj) > 0
            return mask & nonflat, "func_mask", int((mask & ~nonflat).sum())
    return nonflat, "nonflat", 0


def person(fid, out_dir=PP):
    try:
        t0 = time.time()
        out_path = f"{out_dir}/{fid}.npz"
        if os.path.exists(out_path):
            return {"FILE_ID": fid, "skipped": True}
        img = nib.load(f"{FUNC}/{fid}_func_preproc.nii.gz")
        shape, affine = tuple(img.shape[:3]), img.affine
        zooms = img.header.get_zooms()
        tr, nvol = float(zooms[3]), int(img.shape[3])
        data = np.asarray(img.dataobj, dtype=np.float32)
        labels, atlas_count, grid_ok = atlas_for(shape, affine)
        nonflat = data.std(axis=3) > 0
        mask, mask_src, flat_in_mask = load_mask(fid, shape, affine, nonflat)
        sig, nvox = network_means(data, labels, mask)
        cov = nvox / np.maximum(atlas_count, 1)
        z = fisher_z(remove_drift(sig, cosine_basis(nvol, tr, HIGH_PASS)))
        os.makedirs(out_dir, exist_ok=True)
        tmp_path = f"{out_dir}/{fid}.tmp.npz"                       # written first, then renamed: a killed task cannot leave a half file
        np.savez_compressed(tmp_path, z=z, coverage=cov, nvox=nvox, nvol=nvol, tr=tr, mask_source=mask_src, grid_ok=grid_ok,
                            flat_in_mask=flat_in_mask, mask_vox=int(mask.sum()))
        os.replace(tmp_path, out_path)
        return {"FILE_ID": fid, "seconds": round(time.time() - t0, 1), "nvol": nvol, "tr": tr, "mask": mask_src, "grid_ok": grid_ok,
                "flat_in_mask": flat_in_mask, "min_cov": float(cov.min())}
    except Exception as e:
        return {"FILE_ID": fid, "error": f"{type(e).__name__}: {e}"}


def prepare():
    os.makedirs(ATLAS_DIR, exist_ok=True)
    with zipfile.ZipFile(ATLAS_ZIP) as zf:
        names = zf.namelist()
        if ATLAS_FILE not in names:
            print(f"FAIL  {ATLAS_FILE} not in the zip; zip holds: {names[:8]}"); sys.exit(1)
        zf.extract(ATLAS_FILE, ATLAS_DIR)
    ref = nib.load(f"{FUNC}/{REF_ID}_func_preproc.nii.gz")
    shape, affine = tuple(ref.shape[:3]), ref.affine
    print(f"reference grid {REF_ID}: {shape}, voxel {tuple(round(float(v), 2) for v in ref.header.get_zooms()[:3])} mm, {ref.shape[3]} volumes, TR {float(ref.header.get_zooms()[3])} s")
    img = nib.load(f"{ATLAS_DIR}/{ATLAS_FILE}")
    vals = np.unique(np.asarray(img.dataobj))
    if not set(vals.astype(int).tolist()) == set(range(NET + 1)):
        print(f"FAIL  atlas labels are {vals[:20]}, expected 0..17"); sys.exit(1)
    lab = labels_on_grid(img, shape, affine)
    nib.save(nib.Nifti1Image(lab.astype(np.int16), affine), f"{ATLAS_DIR}/yeo17_thin_3mm.nii.gz")
    counts = [int((lab == k).sum()) for k in range(1, NET + 1)]
    print(f"atlas: {img.shape} at {tuple(round(float(v), 2) for v in img.header.get_zooms()[:3])} mm -> on the 3 mm grid: voxels per network 1..17 = {counts} (total {sum(counts)})")
    if min(counts) < 20:
        print("FAIL  a network has fewer than 20 voxels on the 3 mm grid"); sys.exit(1)
    print("PREPARE: OK")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "prepare":
        prepare()
    else:
        ids = open(f"{ROOT}/data/raw/file_ids.txt").read().split()
        task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
        ntask = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
        mine = ids[task::ntask]
        os.makedirs(PP, exist_ok=True)
        print(f"task {task} of {ntask}: {len(mine)} people, {workers} workers")
        t0 = time.time()
        res = []
        with Pool(workers) as pool:
            for n, r in enumerate(pool.imap_unordered(person, mine, chunksize=1), 1):
                res.append(r)
                if n % 25 == 0 or n == len(mine):
                    print(f"  {n} of {len(mine)} people done after {time.time() - t0:.0f} s", flush=True)
        err = [r for r in res if "error" in r]
        done = [r for r in res if "seconds" in r]
        skipped = sum(1 for r in res if r.get("skipped"))
        for r in err[:20]:
            print("ERROR", r["FILE_ID"], r["error"])
        if done:
            print(f"done {len(done)} | skipped (already present) {skipped} | errors {len(err)} | median {np.median([r['seconds'] for r in done]):.1f} s per person | "
                  f"mask: func_mask {sum(r['mask'] == 'func_mask' for r in done)}, nonflat {sum(r['mask'] == 'nonflat' for r in done)} | grid as reference: {sum(r['grid_ok'] for r in done)} | "
                  f"flat voxels inside masks: {sum(r['flat_in_mask'] for r in done)} | lowest coverage {min(r['min_cov'] for r in done):.2f} | wall {time.time() - t0:.0f} s")
        if not done:
            print(f"nothing computed: {skipped} people already had their file, {len(err)} errors")
        print("TASK: OK" if not err else "TASK: FAIL")
        sys.exit(1 if err else 0)
