# Design notes & engineering decisions

This document records *why* the code looks the way it does — the evaluation
pitfalls specific to this dataset, and the bugs found when the original course
project was hardened into a reusable package.

## 1. Evaluation protocol

| Decision | Why |
|---|---|
| **Subject-grouped 5-fold CV** (103 subjects, 1,909 segments) | Each subject contributes 11–29 segments (median 18). Random segment-level splits put near-duplicate geometry from one patient in both train and test and inflate every metric. `tests/test_data_config.py` asserts zero subject overlap between train / val / test for every fold. |
| **Separate validation split per fold** | Used for early stopping *and* for choosing the decision threshold. The test fold is touched exactly once. |
| **Threshold chosen on validation (Youden's J)** | Prevalence is 11%. With a balanced sampler the model's probabilities are not calibrated to that prior, so a fixed 0.5 cut-off can predict almost everything as one class. ACC₀/ACC₁/F1 are reported at the validation-selected threshold; 0.5-threshold metrics are also saved (`*@0.5`). |
| **No fold is ever dropped** | The original `evaluate.py` excluded "collapsed" folds from the mean ± std. That makes the aggregate optimistic. Collapse is now *flagged* (`collapsed@0.5`) but every fold counts. |
| **Pooled out-of-fold AUROC with a subject-level (cluster) bootstrap CI** | Segments from one subject are correlated; resampling segments independently understates uncertainty. `pointcbm.metrics.cluster_bootstrap_ci` resamples subjects. |
| **AUPRC and ECE alongside AUROC** | AUROC is insensitive to class imbalance; AUPRC (baseline = prevalence 0.11) shows how useful positive predictions actually are. ECE records how far probabilities are from frequencies. |
| **Strong non-deep baseline on identical folds** | 44 hand-crafted geometric descriptors + gradient boosting. A deep model has to beat this to justify its complexity. |

## 2. Model design

* **Backbone contract.** Every encoder maps `(B, C_in, N) → (B, N, d_f)` with *no*
  global pooling, so the bottleneck can attend over individual points. New
  encoders are added with `@register("Name")` in `pointcbm/backbones/__init__.py`.
* **Concept Cross-Attention (CCAM).** `A = sigmoid(⟨W_f·LN(F), W_t·LN(E)⟩/√d)`.
  Each point gets an independent score per concept, so many points can fire for
  one concept and one point can fire for several concepts.
* **Top-k log-sum-exp pooling.** A smooth maximum over the top 10% of points. Plain `max`
  sends gradient to a single point. `mean` washes out a small sac on a long vessel.
  `β` is learnable and clamped to [1, 50].
* **Transparent head.** `logits = W·Ĉ + b`, plus an L1 penalty on `W`. In eval mode this is
  *exactly* linear in the concept scores (asserted in `tests/test_models.py`). That linearity is
  what makes test-time interventions and the concept-ablation analysis exact.
* **Black-box control.** `BlackBoxClassifier` uses the same backbone with
  `[max ‖ mean]` pooling and an MLP. The gap between the two heads is the "price of the bottleneck".
* **Random-concept control.** `data.concept_embeddings: random` swaps the CLIP bank for
  unit-norm Gaussian vectors of the same shape. This tests whether the language grounding
  matters, or whether the concepts just act as K learned prototypes.

## 3. Bugs fixed while refactoring

| Issue in the original code | Effect | Fix |
|---|---|---|
| Scripts imported `CCAM.model`, `CCAM.dataset`, `CCAM.loss`, but the package was `model/` | Nothing ran from a fresh clone | Installable `pointcbm` package (`pip install -e .`) |
| `print()` calls inside `ConceptBottleneck3D.forward` | Five log lines per batch, plus a slowdown | Removed |
| `train.py` built `OfficialPointNet()` without `d_f` / `in_channel` | Config values were silently ignored | Backbone registry passes the config through |
| Early stopping counted a non-improving epoch twice after warm-up | Patience of 15 effectively became ~8 | One counter, which only starts after warm-up |
| `ReduceLROnPlateau(verbose=True)` | Raises on current PyTorch | Removed; the plateau scheduler now tracks the monitored val metric |
| The balanced sampler loaded every point cloud from disk to read its label | ~1,200 extra file reads before each fold | Labels come from the CSV |
| Farthest-point sampling used a random start at eval time | PointNet++ test metrics changed between runs | Deterministic start in `eval()` mode |
| Same NumPy RNG state in every DataLoader worker | Workers repeated each other's augmentations | `worker_init_fn` reseeds each worker |
| `evaluate.py` excluded collapsed folds from the aggregate | Optimistic mean ± std | All folds reported; collapse is flagged |
| `__pycache__/`, `.DS_Store` committed; README paths (`utils/`, `OpenShape/`) didn't match the tree | Repo hygiene | `.gitignore`, and a README rewritten against the actual tree |

Parameter names were kept identical (`backbone.stn.*`, `ccam.proj_text`, `predictor`,
`beta`, …), so checkpoints trained with the original code still load with
`strict=True`. A unit test guards this.

## 4. Reproducibility

* `seed_everything(seed + fold)` for each fold; the sampler has its own seeded generator.
* Eval-time point subsampling uses a fixed per-sample RNG (`default_rng(idx)`).
* Every checkpoint stores its full config, concept names and the chosen threshold.
  `scripts/evaluate.py --checkpoint_dir <run>` needs nothing else.
* `oof_predictions.csv` stores every test probability and concept score.
  All tables and figures are regenerated from these files without retraining
  (`scripts/analyze_results.py`).

## 5. Data files

| Path | Contents |
|---|---|
| `data/processed_IntrA_6d/points/*.npy` | 1,909 segments × 2,048 points × (xyz, unit normal), derived from the IntrA meshes |
| `data/processed_IntrA_6d/fold_k/{train,val,test}_labels.csv` | Subject-grouped splits |
| `data/metadata/concepts.json` | The 6-concept natural-language bank |
| `data/metadata/concept_embeds.pt` | CLIP ViT-L/14 text embeddings (6 × 768); `_ViT-B-32` variant is 6 × 512 |
| `data/OpenShape/heatmaps/*.pt` | Zero-shot OpenShape point–concept similarity maps (2,048 × 21 concepts), used as optional pseudo-labels (`lambda_align`) and as the zero-shot baseline |
