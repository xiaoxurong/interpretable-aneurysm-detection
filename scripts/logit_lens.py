"""
Logit Lens for OpenShape PointBERT backbone.

Tracks how the CLS token evolves through the 12 transformer sub-layers
(6 attention + 6 FFN) by projecting it into CLIP space via image_proj
and computing cosine similarity with concept text embeddings.

The analysis runs on the raw pretrained weights (what OpenShape learned
on ShapeNet) AND on features extracted through your fine-tuned IntrA model,
revealing where VLM alignment persists or degrades for OOD medical data.

Architecture inferred from checkpoint inspection:
    ppat.sa               PointNet SA, 9-dim input → 256-dim tokens
    ppat.lift             Conv1d(259, 512) + GELU + LayerNorm
    ppat.cls_token        (512,) learnable
    ppat.transformer.layers[0]   PreNorm + Attention  (even indices)
    ppat.transformer.layers[1]   PreNorm + FFN        (odd indices)
    ...                          (12 sub-layers total)
    image_proj            Linear(512, clip_dim)       separate in checkpoint

Usage:
    python logit_lens.py \
        --config       tti_config.yaml \
        --openShape_ckpt  path/to/model.pt \
        --fold         1 \
        --concept_names "irregular shape" "dome morphology" "vessel narrowing" \
                        "wall irregularity" "neck geometry" "surface curvature"
"""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from CCAM.dataset import IntrAClassificationDataset


# ─────────────────────────────────────────────────────────────────────────────
# 1.  PointBERT architecture (reconstructed from checkpoint keys)
# ─────────────────────────────────────────────────────────────────────────────

def index_points(points, idx):
    """Gather points by index.  points: (B, N, C),  idx: (B, S) or (B, S, K)."""
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_idx = (torch.arange(B, device=points.device)
                 .view(view_shape).repeat(repeat_shape))
    return points[batch_idx, idx, :]


def farthest_point_sample(xyz, npoint):
    """FPS. xyz: (B, N, 3) → (B, npoint) indices."""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.full((B, N), 1e10, device=device)
    farthest  = torch.randint(0, N, (B,), device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        c = xyz[torch.arange(B), farthest, :].unsqueeze(1)   # (B, 1, 3)
        dist = ((xyz - c) ** 2).sum(-1)                       # (B, N)
        distance = torch.min(distance, dist)
        farthest = distance.argmax(-1)
    return centroids


def query_ball(radius, nsample, xyz, new_xyz):
    """Ball query.  Returns (B, S, nsample) indices into xyz."""
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    idx = torch.zeros(B, S, nsample, dtype=torch.long, device=xyz.device)
    for b in range(B):
        for s in range(S):
            diff = xyz[b] - new_xyz[b, s]                     # (N, 3)
            dist = (diff ** 2).sum(-1)                         # (N,)
            ball = (dist <= radius ** 2).nonzero(as_tuple=False).squeeze(-1)
            if ball.numel() == 0:
                ball = dist.topk(nsample, largest=False)[1]
            if ball.numel() >= nsample:
                perm = torch.randperm(ball.numel(), device=xyz.device)[:nsample]
                ball = ball[perm]
            else:
                pad = ball[torch.randint(ball.numel(), (nsample - ball.numel(),),
                                         device=xyz.device)]
                ball = torch.cat([ball, pad])
            idx[b, s] = ball
    return idx


class PointNetSALayer(nn.Module):
    """
    Minimal PointNet++ Set Abstraction.
    Matches ppat.sa weights: mlp_convs [64,9,1,1], [128,64,1,1], [256,128,1,1]
    with corresponding mlp_bns (BatchNorm2d).

    Input:
        xyz      (B, 3, N)
        features (B, 6, N)   xyz + normals
    Output:
        new_xyz      (B, 3, npoint)
        new_features (B, 256, npoint)
    """

    def __init__(self, npoint=128, radius=0.2, nsample=32):
        super().__init__()
        self.npoint  = npoint
        self.radius  = radius
        self.nsample = nsample

        # in_channel = 9 = relative_xyz(3) + per_point_features(6)
        # MLP shape inferred from checkpoint: 9 → 64 → 64 → 256
        self.mlp_convs = nn.ModuleList([
            nn.Conv2d(9,   64,  1),
            nn.Conv2d(64,  64,  1),
            nn.Conv2d(64,  256, 1),
        ])
        self.mlp_bns = nn.ModuleList([
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(256),
        ])

    def forward(self, xyz, features):
        """
        xyz      : (B, 3, N)
        features : (B, 6, N)   xyz + normals
        """
        B, _, N = xyz.shape
        xyz_t = xyz.transpose(1, 2)                   # (B, N, 3)
        feat_t = features.transpose(1, 2)             # (B, N, 6)

        fps_idx   = farthest_point_sample(xyz_t, self.npoint)  # (B, npoint)
        new_xyz_t = index_points(xyz_t, fps_idx)                # (B, npoint, 3)

        # For speed, use k-NN instead of ball query
        diff = xyz_t.unsqueeze(2) - new_xyz_t.unsqueeze(1)     # (B, N, npoint, 3)
        dist = (diff ** 2).sum(-1)                               # (B, N, npoint)
        knn_idx = dist.topk(self.nsample, dim=1, largest=False)[1]  # (B, nsample, npoint)
        knn_idx = knn_idx.permute(0, 2, 1)                      # (B, npoint, nsample)

        grouped_xyz  = index_points(xyz_t,  knn_idx)            # (B, npoint, nsample, 3)
        grouped_feat = index_points(feat_t, knn_idx)            # (B, npoint, nsample, 6)
        relative_xyz = grouped_xyz - new_xyz_t.unsqueeze(2)     # (B, npoint, nsample, 3)

        # 9-dim input: relative_xyz(3) + per-point features(6)
        x = torch.cat([relative_xyz, grouped_feat], dim=-1)     # (B, npoint, nsample, 9)
        x = x.permute(0, 3, 1, 2)                               # (B, 9, npoint, nsample)

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            x = F.relu(bn(conv(x)))

        x = x.max(dim=-1)[0]                                    # (B, 256, npoint)
        return new_xyz_t.transpose(1, 2), x                     # (B,3,np), (B,256,np)


class Attention(nn.Module):
    """Multi-head attention with fused QKV (no bias on to_qkv)."""

    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.scale     = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.0),
        )

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.to_qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                             # each (B, L, H, head_dim)
        q = q.transpose(1, 2)                                   # (B, H, L, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.to_out(out)


class FeedForward(nn.Module):
    """2-layer FFN. net.0 and net.3 match checkpoint naming (GELU at 1, Dropout at 2)."""

    def __init__(self, d_model=512, ffn_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x):
        return self.net(x)


class PreNormResidual(nn.Module):
    """PreNorm + residual: x = x + fn(norm(x)). Matches {norm, fn} key structure."""

    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn   = fn

    def forward(self, x):
        return x + self.fn(self.norm(x))


class PointBERT(nn.Module):
    """
    OpenShape PointBERT (ppat) reconstructed from checkpoint.

    Forward returns list of CLS token states, one per complete block
    (after every attention+FFN pair), shape (B, 512) each.
    """

    D_MODEL = 512

    def __init__(self, npoint=128, radius=0.2, nsample=32,
                 num_heads=8, n_layers=12):
        super().__init__()
        D = self.D_MODEL

        self.sa        = PointNetSALayer(npoint, radius, nsample)
        # lift: Conv1d(259→512) + GELU + LayerNorm
        # 259 = SA output(256) + centroid xyz(3)
        self.lift      = nn.Sequential(
            nn.Conv1d(259, D, 1),
            nn.GELU(),
            nn.LayerNorm(D),
        )
        self.cls_token = nn.Parameter(torch.zeros(D))

        layers = []
        for _ in range(n_layers // 2):
            layers.append(PreNormResidual(D, Attention(D, num_heads)))
            layers.append(PreNormResidual(D, FeedForward(D, ffn_dim=1536)))
        self.transformer = nn.ModuleList(layers)

    @torch.no_grad()
    def forward_logit_lens(self, xyz, features):
        """
        Run the model and return CLS token at every complete block.

        Args:
            xyz      : (B, 3, N)
            features : (B, 6, N)   xyz + normals

        Returns:
            cls_states : list of (B, 512) tensors, length = n_layers // 2
                         cls_states[i] = CLS token after transformer block i
        """
        B = xyz.shape[0]

        # Tokenise
        new_xyz, tokens = self.sa(xyz, features)   # (B,3,npt), (B,256,npt)
        tokens_in = torch.cat([tokens, new_xyz], dim=1)   # (B, 259, npt)

        # Lift  — LayerNorm expects (B, L, C) so permute
        tokens_in = self.lift[:2](tokens_in)               # Conv1d + GELU → (B,512,npt)
        tokens_in = tokens_in.transpose(1, 2)              # (B, npt, 512)
        tokens_in = self.lift[2](tokens_in)                # LayerNorm

        # Prepend CLS
        cls = self.cls_token.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        x   = torch.cat([cls, tokens_in], dim=1)           # (B, npt+1, 512)

        # Transformer — extract CLS after each attn+FFN pair
        cls_states = []
        for i in range(0, len(self.transformer), 2):
            x = self.transformer[i](x)       # attention sub-layer
            x = self.transformer[i + 1](x)  # FFN sub-layer
            cls_states.append(x[:, 0, :].clone())  # (B, 512)

        return cls_states   # list of 6 tensors


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pointbert_from_ckpt(ckpt_path: str, device: torch.device) -> 'PointBERT':
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd   = ckpt['state_dict']

    # Strip 'module.ppat.' prefix
    ppat_sd = {}
    for k, v in sd.items():
        if k.startswith('module.ppat.'):
            ppat_sd[k[len('module.ppat.'):]] = v

    # Remap  transformer.layers.N.M.rest  →  transformer.{N*2+M}.rest
    #
    # The checkpoint uses a *nested* ModuleList:
    #   transformer.layers[block_idx][sub_idx]  (sub_idx 0=attn, 1=ffn)
    # Our model uses a *flat* ModuleList:
    #   transformer[flat_idx]  where flat_idx = block_idx*2 + sub_idx
    #
    # The old code incorrectly re-enumerated keys alphabetically (wrong new_i)
    # and kept the sub-index embedded in the new key → every key was wrong.
    final_sd = {}
    n_blocks_detected = 0
    for k, v in ppat_sd.items():
        if k.startswith('transformer.layers.'):
            parts = k.split('.')          # ['transformer','layers', N, M, ...]
            N         = int(parts[2])     # outer block index
            M         = int(parts[3])     # sub-layer index (0=attn, 1=ffn)
            flat_idx  = N * 2 + M
            rest      = '.'.join(parts[4:])
            final_sd[f'transformer.{flat_idx}.{rest}'] = v
            n_blocks_detected = max(n_blocks_detected, N + 1)
        else:
            final_sd[k] = v

    n_sublayers = n_blocks_detected * 2
    print(f"  Detected {n_blocks_detected} transformer blocks ({n_sublayers} sub-layers)")

    model = PointBERT(n_layers=n_sublayers).to(device)
    missing, unexpected = model.load_state_dict(final_sd, strict=False)

    if missing:
        print(f"  [warn] Missing keys ({len(missing)}): {missing[:4]} ...")
    if unexpected:
        print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:4]} ...")

    return model.eval()


def load_image_proj(ckpt_path: str, device: torch.device) -> nn.Linear:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw  = ckpt.get('image_proj')

    weight, bias = None, None

    if isinstance(raw, torch.Tensor):
        # Stored as a bare weight matrix
        weight = raw

    elif isinstance(raw, dict):
        # Stored as a state-dict sub-dict
        weight = raw['weight']
        bias   = raw.get('bias')

    else:
        # raw is None → try state_dict under 'module.image_proj.*'
        sd = ckpt.get('state_dict', {})
        for prefix in ('module.image_proj', 'image_proj'):
            if f'{prefix}.weight' in sd:
                weight = sd[f'{prefix}.weight']
                bias   = sd.get(f'{prefix}.bias')
                print(f"  image_proj found in state_dict['{prefix}']")
                break

    if weight is None:
        # image_proj is None → the backbone already outputs in CLIP space.
        # Alignment is baked into the PointBERT weights; use identity.
        print("  image_proj is None — using identity "
              "(PointBERT output assumed to be in CLIP space already)")
        return nn.Identity().to(device)

    proj = nn.Linear(weight.shape[1], weight.shape[0], bias=(bias is not None))
    proj.weight.data.copy_(weight)
    if bias is not None:
        proj.bias.data.copy_(bias)

    print(f"  image_proj: Linear({weight.shape[1]} → {weight.shape[0]})")
    return proj.to(device).eval()


def load_text_proj(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load text_proj from the OpenShape checkpoint.

    OpenShape keeps image features at d_model (512) and projects text
    embeddings from CLIP's native dim (e.g. 768 for ViT-L/14) into that
    same 512-dim space via text_proj.  We apply this to E_text so the
    cosine-similarity comparison is in a compatible space.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    raw  = ckpt.get('text_proj')

    weight = None
    if isinstance(raw, torch.Tensor):
        weight = raw
    elif isinstance(raw, dict):
        weight = raw['weight']

    if weight is None:
        # Return Python None so callers can detect "no projection available"
        print("  text_proj is None in checkpoint")
        return None

    # weight shape: [out_dim, in_dim]  e.g. [512, 768]
    proj = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    proj.weight.data.copy_(weight)
    print(f"  text_proj: Linear({weight.shape[1]} → {weight.shape[0]})")
    return proj.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_cls_states(model: PointBERT,
                       loader: DataLoader,
                       device: torch.device):
    """
    Returns:
        cls_per_block : list of (N, 512) arrays, length = 6 (one per block)
        labels        : (N,) array
    """
    model.eval()
    buffers = None   # initialised on first batch so size adapts to n_blocks
    all_y   = []

    for X, Y in tqdm(loader, leave=False):
        X = X.to(device)                 # (B, N_pts, 6)
        xyz      = X[:, :, :3].transpose(1, 2)   # (B, 3, N_pts)
        features = X.transpose(1, 2)             # (B, 6, N_pts)

        states = model.forward_logit_lens(xyz, features)   # list of n_blocks × (B, 512)
        if buffers is None:
            buffers = [[] for _ in range(len(states))]
        for i, s in enumerate(states):
            buffers[i].append(s.cpu().numpy())
        all_y.append(Y.numpy())

    cls_per_block = [np.concatenate(b, axis=0) for b in buffers]
    labels = np.concatenate(all_y, axis=0)
    return cls_per_block, labels


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Logit lens: concept alignment at each block
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def logit_lens(cls_per_block, labels, image_proj, E_text, device,
               text_proj=None):
    """
    Project each block's CLS token through image_proj → CLIP space,
    then compute cosine similarity with concept text embeddings.

    If text_proj is provided, E_text is projected first so that both
    sides live in the same embedding space (handles the 768 vs 512 mismatch
    when image_proj is identity but text embeddings are from a larger CLIP).

    Returns
    -------
    sims_all   : (n_blocks, K)  mean cosine similarity over all test samples
    sims_diff  : (n_blocks, K)  µ[aneurysm] − µ[healthy]
    """
    # Project text embeddings into the image feature space if needed
    if text_proj is not None:
        E_proj = text_proj(E_text)          # e.g. 768 → 512 via checkpoint matrix
    else:
        E_proj = E_text

    cls_dim  = cls_per_block[0].shape[1]   # e.g. 512
    text_dim = E_proj.shape[1]             # e.g. 512 or 768

    if text_dim != cls_dim:
        # Neither image_proj nor text_proj bridges the gap.
        # Fall back to PCA to bring text embeddings into the same dimensionality.
        # This is an approximation — ideally re-encode concepts with ViT-B/32.
        from sklearn.decomposition import PCA
        print(f"  [warn] text dim ({text_dim}) ≠ cls dim ({cls_dim}). "
              f"Applying PCA {text_dim}→{cls_dim} as fallback.")
        E_cpu = F.normalize(E_proj, dim=-1).cpu().numpy()
        pca   = PCA(n_components=cls_dim)
        E_np  = pca.fit_transform(E_cpu)
        E_np  = E_np / (np.linalg.norm(E_np, axis=1, keepdims=True) + 1e-8)
    else:
        E_np = F.normalize(E_proj, dim=-1).cpu().numpy()   # (K, feat_dim)
    K = E_np.shape[0]
    n_blocks = len(cls_per_block)

    sims_all  = np.zeros((n_blocks, K))
    sims_diff = np.zeros((n_blocks, K))

    for i, cls_np in enumerate(cls_per_block):
        cls_t   = torch.from_numpy(cls_np).float().to(device)
        clip    = image_proj(cls_t).cpu().numpy()         # (N, clip_dim)
        clip   /= np.linalg.norm(clip, axis=1, keepdims=True) + 1e-8
        sims    = clip @ E_np.T                           # (N, K)

        sims_all[i]  = sims.mean(axis=0)
        sims_diff[i] = (sims[labels == 1].mean(axis=0) -
                        sims[labels == 0].mean(axis=0))

    return sims_all, sims_diff


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Linear probing at each block
# ─────────────────────────────────────────────────────────────────────────────

def linear_probe(train_cls, train_labels, test_cls, test_labels):
    """
    Fit a balanced logistic regression probe at each block.
    Returns dict: block_idx → {'acc': float, 'auroc': float}
    """
    results = {}
    for i, (tr, te) in enumerate(zip(train_cls, test_cls)):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(tr)
        X_te   = scaler.transform(te)
        clf    = LogisticRegression(max_iter=1000, class_weight='balanced',
                                    C=1.0, random_state=42)
        clf.fit(X_tr, train_labels)
        preds  = clf.predict(X_te)
        probs  = clf.predict_proba(X_te)[:, 1]
        acc    = (preds == test_labels).mean()
        try:
            auroc = roc_auc_score(test_labels, probs)
        except ValueError:
            auroc = float('nan')
        results[i] = {'acc': acc, 'auroc': auroc}
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_logit_lens(sims_all, sims_diff, concept_names, probe_results,
                   output_dir, tag=""):
    n_blocks = sims_all.shape[0]
    K        = sims_all.shape[1]
    block_labels = [f"Block {i+1}" for i in range(n_blocks)]

    fig, axes = plt.subplots(1, 3, figsize=(max(14, K * 1.5), 5))
    fig.suptitle(f"Logit Lens — PointBERT (OpenShape){tag}",
                 fontsize=13, fontweight='bold')

    # ── Panel 1: mean cosine similarity ──────────────────────────────
    vmax = max(sims_all.max(), 0.01)
    im0  = axes[0].imshow(sims_all, cmap='YlOrRd', aspect='auto',
                          vmin=0, vmax=vmax)
    axes[0].set_xticks(range(K))
    axes[0].set_xticklabels(concept_names, rotation=35, ha='right', fontsize=9)
    axes[0].set_yticks(range(n_blocks))
    axes[0].set_yticklabels(block_labels)
    axes[0].set_title("Mean Cosine Sim\n(all samples)", fontsize=10)
    plt.colorbar(im0, ax=axes[0])
    for r in range(n_blocks):
        for c in range(K):
            axes[0].text(c, r, f"{sims_all[r,c]:.2f}", ha='center',
                         va='center', fontsize=7.5,
                         color='white' if sims_all[r,c] > 0.6*vmax else 'black')

    # ── Panel 2: aneurysm − healthy difference ────────────────────────
    vd   = max(abs(sims_diff).max(), 0.01)
    im1  = axes[1].imshow(sims_diff, cmap='RdBu_r', aspect='auto',
                          vmin=-vd, vmax=vd)
    axes[1].set_xticks(range(K))
    axes[1].set_xticklabels(concept_names, rotation=35, ha='right', fontsize=9)
    axes[1].set_yticks(range(n_blocks))
    axes[1].set_yticklabels(block_labels)
    axes[1].set_title("µ[Aneurysm] − µ[Healthy]\n(+ = more active for aneurysm)",
                      fontsize=10)
    plt.colorbar(im1, ax=axes[1])
    for r in range(n_blocks):
        for c in range(K):
            axes[1].text(c, r, f"{sims_diff[r,c]:+.2f}", ha='center',
                         va='center', fontsize=7.5)

    # ── Panel 3: linear probe AUROC ───────────────────────────────────
    aurocs = [probe_results[i]['auroc'] for i in range(n_blocks)]
    axes[2].barh(range(n_blocks), aurocs, color='steelblue', alpha=0.8)
    axes[2].axvline(0.5, color='red', linestyle='--', linewidth=1, label='chance')
    axes[2].set_yticks(range(n_blocks))
    axes[2].set_yticklabels(block_labels)
    axes[2].set_xlabel("Linear Probe AUROC")
    axes[2].set_title("Discriminability\n(linear probe)", fontsize=10)
    axes[2].set_xlim(0, 1)
    axes[2].legend(fontsize=8)
    axes[2].grid(axis='x', alpha=0.3)
    for i, v in enumerate(aurocs):
        axes[2].text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=8)

    plt.tight_layout()
    fname = os.path.join(output_dir,
                         f"logit_lens{tag.replace(' ', '_')}.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {fname}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',           required=True)
    p.add_argument('--openShape_ckpt',   required=True)
    p.add_argument('--fold',             type=int, default=1)
    p.add_argument('--concept_names',    nargs='+', default=None)
    p.add_argument('--output_dir',       default='logit_lens_results')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── concept embeddings ─────────────────────────────────────────────
    E_data = torch.load(config['data']['concept_embeddings'], map_location=device)
    E_text = E_data['embeddings'].float().to(device)
    K      = E_text.shape[0]
    concept_names = args.concept_names or [f'Concept-{i}' for i in range(K)]
    assert len(concept_names) == K
    print(f"Concepts: {concept_names}\n")

    # ── load PointBERT + image_proj + text_proj ───────────────────────
    print("Loading PointBERT from OpenShape checkpoint …")
    model      = load_pointbert_from_ckpt(args.openShape_ckpt, device)
    image_proj = load_image_proj(args.openShape_ckpt, device)
    text_proj  = load_text_proj(args.openShape_ckpt, device)

    # ── data ───────────────────────────────────────────────────────────
    fold      = args.fold
    fold_root = os.path.join(config['data']['fold_dir'], f'fold_{fold}')
    data_dir  = config['data']['data_dir']

    train_ds = IntrAClassificationDataset(
        os.path.join(fold_root, 'train_labels.csv'), data_dir, augment=False)
    test_ds  = IntrAClassificationDataset(
        os.path.join(fold_root, 'test_labels.csv'),  data_dir, augment=False)

    kw = dict(batch_size=64, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, **kw)

    # ── extract CLS states ─────────────────────────────────────────────
    print("\nExtracting CLS token states (train) …")
    train_cls, train_labels = extract_cls_states(model, train_loader, device)
    print("Extracting CLS token states (test) …")
    test_cls, test_labels = extract_cls_states(model, test_loader,  device)
    print(f"  Blocks: {len(test_cls)},  each shape: {test_cls[0].shape}")
    print(f"  Test class dist: {np.bincount(test_labels.astype(int))}")

    # ── logit lens ─────────────────────────────────────────────────────
    print("\nComputing concept alignment (logit lens) …")
    sims_all, sims_diff = logit_lens(test_cls, test_labels,
                                     image_proj, E_text, device,
                                     text_proj=text_proj)

    # ── linear probing ─────────────────────────────────────────────────
    print("\nFitting linear probes …")
    probe_results = linear_probe(train_cls, train_labels,
                                 test_cls,  test_labels)
    print(f"  {'Block':>8}  {'Acc':>8}  {'AUROC':>8}")
    for i, r in probe_results.items():
        print(f"  {i+1:>8}  {r['acc']*100:>7.2f}%  {r['auroc']:>8.4f}")

    # ── plot ───────────────────────────────────────────────────────────
    plot_logit_lens(sims_all, sims_diff, concept_names, probe_results,
                    args.output_dir)

    # ── print alignment shift across blocks ───────────────────────────
    print("\nConcept alignment at each block (aneurysm − healthy):")
    print(f"  {'':30s}  " + "  ".join(f"B{i+1:>4}" for i in range(len(test_cls))))
    for k, name in enumerate(concept_names):
        vals = "  ".join(f"{sims_diff[i,k]:+.3f}" for i in range(len(test_cls)))
        print(f"  {name:30s}  {vals}")

    print(f"\nDone. Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()