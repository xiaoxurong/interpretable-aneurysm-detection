"""Hand-crafted geometric descriptors for vessel segments (no deep learning).

These give the "is deep learning even needed?" baseline. Every feature has a
direct geometric reading, which makes the baseline interpretable in its own
right — a useful yardstick for a concept bottleneck model.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _eig_features(cov: np.ndarray, prefix: str) -> dict[str, float]:
    ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 1e-12, None))[::-1]
    s = ev.sum()
    return {
        f"{prefix}_l1": ev[0] / s,
        f"{prefix}_l2": ev[1] / s,
        f"{prefix}_l3": ev[2] / s,
        f"{prefix}_linearity": (ev[0] - ev[1]) / ev[0],
        f"{prefix}_planarity": (ev[1] - ev[2]) / ev[0],
        f"{prefix}_sphericity": ev[2] / ev[0],
    }


def segment_features(pts: np.ndarray, k: int = 16, n_sub: int = 512, n_bins_axis: int = 12, seed: int = 0) -> dict[str, float]:
    """Descriptor vector for one (N, 6) xyz+normal point cloud."""
    rng = np.random.default_rng(seed)
    xyz, nrm = pts[:, :3].astype(np.float64), pts[:, 3:6].astype(np.float64)
    c = xyz - xyz.mean(0)
    feats: dict[str, float] = {}

    # 1. Global shape: PCA of coordinates (elongated tube vs. blob)
    cov = np.cov(c.T)
    feats.update(_eig_features(cov, "xyz"))
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]
    proj = c @ evecs[:, ::-1]
    ext = proj.max(0) - proj.min(0)
    feats.update({"ext1": ext[0], "ext2": ext[1], "ext3": ext[2], "ext_ratio21": ext[1] / ext[0], "ext_ratio31": ext[2] / ext[0]})

    r = np.linalg.norm(c, axis=1)
    feats.update(
        {
            "r_mean": r.mean(),
            "r_std": r.std() / r.mean(),
            "r_max": r.max() / r.mean(),
            "r_skew": float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-12)),
        }
    )

    # 2. Tube-radius profile along the main axis — a sac shows up as a local bulge
    t = c @ axis
    radial = np.linalg.norm(c - np.outer(t, axis), axis=1)
    edges = np.quantile(t, np.linspace(0, 1, n_bins_axis + 1))
    prof = np.array([radial[(t >= lo) & (t <= hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    feats.update(
        {
            "rad_mean": radial.mean(),
            "rad_cv": radial.std() / radial.mean(),
            "rad_profile_max_over_median": prof.max() / (np.median(prof) + 1e-9),
            "rad_profile_cv": prof.std() / (prof.mean() + 1e-9),
            "rad_p95_over_median": np.percentile(radial, 95) / (np.median(radial) + 1e-9),
        }
    )

    # 3. Normal-direction dispersion (tubes: normals ⟂ axis; sacs: normals everywhere)
    feats.update(_eig_features(np.cov(nrm.T), "nrm"))
    cosax = np.abs(nrm @ axis)
    feats.update({"nrm_axis_cos_mean": cosax.mean(), "nrm_axis_cos_p90": np.percentile(cosax, 90)})

    # 4. Local surface variation (curvature proxy) from k-NN PCA on a subsample
    sub = rng.choice(len(xyz), size=min(n_sub, len(xyz)), replace=False)
    tree = cKDTree(xyz)
    _, nn = tree.query(xyz[sub], k=k)
    nb = xyz[nn] - xyz[nn].mean(1, keepdims=True)
    lc = np.einsum("nki,nkj->nij", nb, nb) / k
    le = np.sort(np.clip(np.linalg.eigvalsh(lc), 0, None), axis=1)
    sv = le[:, 0] / (le.sum(1) + 1e-12)
    feats.update({"curv_mean": sv.mean(), "curv_std": sv.std(), "curv_p90": np.percentile(sv, 90), "curv_max": sv.max()})
    nn_n = nrm[nn]
    nvar = 1 - np.linalg.norm(nn_n.mean(1), axis=1)  # 0 = flat patch
    feats.update({"nvar_mean": nvar.mean(), "nvar_p90": np.percentile(nvar, 90)})

    # 5. D2 shape distribution (Osada et al., 2002)
    i, j = rng.integers(0, len(xyz), size=(2, 4000))
    d2 = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    hist, _ = np.histogram(d2 / (d2.max() + 1e-9), bins=10, range=(0, 1), density=True)
    feats.update({f"d2_{b}": h for b, h in enumerate(hist)})
    return feats
