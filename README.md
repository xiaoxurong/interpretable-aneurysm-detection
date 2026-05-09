# PC-CBM: 3D Point Cloud Concept Bottleneck Model for Intracranial Aneurysm Diagnosis

A concept bottleneck model for interpretable classification of intracranial
aneurysms from 3D point cloud data. PC-CBM routes all predictive reasoning
through a spatial cross-attention bottleneck grounded in natural language
geometric concepts, enabling transparent, clinician-auditable diagnosis.

## Results

5-fold cross-validation on the IntrA dataset (6D input: xyz + normals):

| Backbone    | Loss           | ACC₀            | ACC₁            | AUROC           | F1              |
|-------------|----------------|-----------------|-----------------|-----------------|-----------------|
| PointNet++  | Focal          | 0.530 ± 0.234   | 0.853 ± 0.163   | 0.827 ± 0.107   | 0.483 ± 0.143   |
| PointNet++  | Focal + SupCon | **0.877 ± 0.014** | 0.733 ± 0.133 | **0.876 ± 0.032** | **0.726 ± 0.029** |
| DGCNN       | Focal          | 0.559 ± 0.288   | 0.659 ± 0.334   | 0.699 ± 0.046   | 0.455 ± 0.099   |
| DGCNN       | Focal + SupCon | 0.278 ± 0.250   | **0.930 ± 0.095** | 0.786 ± 0.033 | 0.313 ± 0.181   |
| PointNet    | Focal          | 0.584 ± 0.215   | 0.672 ± 0.220   | 0.687 ± 0.083   | 0.482 ± 0.101   |
| PointNet    | Focal + SupCon | 0.280 ± 0.395   | 0.833 ± 0.236   | 0.642 ± 0.111   | 0.277 ± 0.238   |

## Installation

```bash
git clone https://github.com/xiaoxurong/POINTCBM-767.git
cd POINTCBM-767
pip install -r requirements.txt
```

## Data Setup

1. Download the IntrA dataset from https://github.com/intra3d2019/IntrA
   and place it under `data/IntrA/`.
2. Generate the 5-fold cross-validation splits and place CSV files under
   `data/splits/fold_{1..5}/`.

The expected directory structure is:

```
data/
├── points/
│   └── *.npy
├── processed_IntrA_6d/
│   ├── fold_1/
│   │   ├── train_labels.csv
│   │   └── test_labels.csv
│   └── ...
└── metadata/
    ├── concepts.json
    └── concept_embeds.pt
```

## Generating Concept Embeddings

Edit `data/metadata/concepts.json` to define your concept bank, then run:

```bash
python scripts/concept_embeddings.py \
    --input data/metadata/concepts.json \
    --output data/metadata/concept_embeds.pt \
    --model ViT-B/32
```

## Training

```bash
python scripts/train.py --config configs/train_intra.yaml
```

Key fields to set in `configs/train_intra.yaml`:
- `model.backbone`: `PointNetPlusPlus`, `DGCNN`, or `OfficialPointNet`
- `data.concept_embeddings`: path to your `.pt` embeddings file
- `data.csv_dir`: path to your fold splits
- `loss_weights.lambda_supcon`: set to `0.0` to disable SupCon

Checkpoints are saved to the directory specified by
`training.checkpoint_dir`, one per fold (`best_model_fold_{1..5}.pth`).

## Evaluation

```bash
python scripts/evaluate.py --config configs/evaluate_intra.yaml
```

Set `evaluation.checkpoint_dir` in the config to point to the folder
containing your saved fold checkpoints. Metrics are reported per fold
and aggregated as mean ± std across folds.

## Logit Lens Analysis (OpenShape)

```bash
python scripts/logit_lens.py --config configs/evaluate_intra.yaml
```

Requires the OpenShape checkpoint to be available under `OpenShape/`.
See `OpenShape/README.md` for download instructions.

## Project Structure
```
POINTCBM-767/
├── backbones/              # PointNet, PointNet++, DGCNN implementations
├── model/                  # dataset.py, loss.py, model.py (PC-CBM)
├── configs/                # YAML configuration files
├── data/                   # Dataset, splits, and concept metadata
├── OpenShape/              # Pretrained 3D VLM (OpenShape/PointBERT)
├── utils/                  # Evaluation metrics and helper functions
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── logit_lens.py
│   └── concept_embeddings.py
└── requirements.txt
```