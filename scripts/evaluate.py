import os
import argparse
import yaml
from backbones import DGCNN
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, classification_report

# Import your custom modules
from CCAM.model import ConceptBottleneck3D, VanillaPointNet, OfficialPointNet, PointNetPlusPlus
from CCAM.dataset import IntrAClassificationDataset
from utils.metrics import compute_segmentation_metrics, compute_concept_metrics
from backbones.DGCNN import DGCNN, EdgeConv

@torch.no_grad()
def evaluate_model(model, dataloader, E_text, device, task="classification", fold=None):
    model.eval()

    all_y_true = []
    all_y_pred_probs = []
    all_y_pred_classes = []

    fold_label = f"Fold {fold}" if fold else "Test Set"
    print(f"\nRunning inference on {fold_label} for {task.upper()}...")

    for X, Y_target in tqdm(dataloader, leave=False):
        X = X.to(device)
        Y_target = Y_target.to(device)

        Y_hat, C_hat, A = model(X, E_text)

        if task == "classification":
            probs = torch.softmax(Y_hat, dim=1)[:, 1]
            preds = torch.argmax(Y_hat, dim=1)
            all_y_true.extend(Y_target.cpu().view(-1).numpy())
            all_y_pred_probs.extend(probs.cpu().view(-1).numpy())
            all_y_pred_classes.extend(preds.cpu().view(-1).numpy())

        elif task == "segmentation":
            preds = torch.argmax(Y_hat, dim=2)
            all_y_true.extend(Y_target.cpu().view(-1).numpy())
            all_y_pred_classes.extend(preds.cpu().view(-1).numpy())

    # --- Metrics Calculation ---
    metrics = {}

    if task == "classification":
        auc = roc_auc_score(all_y_true, all_y_pred_probs)
        f1_macro = f1_score(all_y_true, all_y_pred_classes, average='macro')

        tn, fp, fn, tp = confusion_matrix(
            all_y_true, all_y_pred_classes, labels=[0, 1]
        ).ravel()

        class_0_acc = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        class_1_acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        overall_acc = (tp + tn) / (tn + fp + fn + tp)

        # ── collapse detection ────────────────────────────────────────────
        # A fold is collapsed if overall accuracy ≈ majority-class baseline
        # AND minority (aneurysm) recall is near zero.
        all_y_true_arr = np.array(all_y_true)
        majority_baseline = max(np.bincount(all_y_true_arr)) / len(all_y_true_arr)
        is_collapsed = (
            abs(overall_acc - majority_baseline) < 0.02
            and class_1_acc < 0.05
        )
        metrics['collapsed'] = is_collapsed

        if is_collapsed:
            print(f"  ⚠️  [{fold_label}] COLLAPSED — overall acc={overall_acc*100:.1f}% "
                  f"≈ majority baseline={majority_baseline*100:.1f}%, "
                  f"aneurysm recall={class_1_acc*100:.1f}%. "
                  f"This fold will be excluded from aggregate results.")

        metrics['overall_acc'] = overall_acc
        metrics['class_0_acc'] = class_0_acc
        metrics['class_1_acc'] = class_1_acc
        metrics['f1_macro']    = f1_macro
        metrics['auroc']       = auc

        print(f"🎯 CLASSIFICATION PERFORMANCE ({fold_label})")
        print(f"Overall Accuracy:  {overall_acc * 100:.2f}%")
        print(f"Class 0 (Healthy): {class_0_acc * 100:.2f}%")
        print(f"Class 1 (Aneurysm):{class_1_acc * 100:.2f}%")
        print(f"Macro F1 Score:    {f1_macro:.4f}")
        print(f"AUROC:             {auc:.4f}")

    elif task == "segmentation":
        task_metrics = compute_segmentation_metrics(all_y_true, all_y_pred_classes)
        print(f"🎯 SEGMENTATION PERFORMANCE ({fold_label})")
        print(f"Point Accuracy: {task_metrics['point_accuracy'] * 100:.2f}% "
              f"| mIoU: {task_metrics['miou']:.4f}")
        metrics['point_accuracy'] = task_metrics['point_accuracy']
        metrics['miou']           = task_metrics['miou']
        metrics['collapsed']      = False   # not applicable for segmentation

    return metrics


# ── Aggregate across folds, excluding collapsed ones ─────────────────────────

def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """
    Compute mean ± std across folds, excluding any collapsed folds.

    Parameters
    ----------
    fold_metrics : list of dicts returned by evaluate_model, one per fold.

    Returns
    -------
    dict with keys: mean_{metric}, std_{metric} for each metric,
    plus 'n_valid_folds' and 'collapsed_folds'.
    """
    valid   = [m for m in fold_metrics if not m.get('collapsed', False)]
    invalid = [i + 1 for i, m in enumerate(fold_metrics) if m.get('collapsed', False)]

    if not valid:
        raise RuntimeError("All folds collapsed — cannot compute aggregate metrics.")

    if invalid:
        print(f"\n[NOTE] Excluded collapsed fold(s): {invalid}")
        print(f"       Aggregating over {len(valid)} / {len(fold_metrics)} folds.")

    scalar_keys = ['overall_acc', 'class_0_acc', 'class_1_acc', 'f1_macro', 'auroc']
    summary = {'n_valid_folds': len(valid), 'collapsed_folds': invalid}

    for key in scalar_keys:
        values = np.array([m[key] for m in valid])
        summary[f'mean_{key}'] = values.mean()
        summary[f'std_{key}']  = values.std()

    # ── pretty print ─────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"🏆  FINAL {len(valid)}-FOLD CROSS-VALIDATION RESULTS")
    if invalid:
        print(f"    (collapsed folds excluded: {invalid})")
    print("=" * 52)
    for key in scalar_keys:
        label = {
            'overall_acc': 'Overall Accuracy ',
            'class_0_acc': 'Class 0 (Healthy)',
            'class_1_acc': 'Class 1 (Aneurysm)',
            'f1_macro':    'Macro F1 Score   ',
            'auroc':       'AUROC            ',
        }[key]
        print(f"{label} : "
              f"{summary[f'mean_{key}'] * 100:6.2f}% "
              f"± {summary[f'std_{key}'] * 100:5.2f}%")
    print("=" * 52)

    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate the 3D CBM (5-Fold)")
    parser.add_argument('--config', type=str, required=True, help="Path to evaluate_intra.yaml")
    args = parser.parse_args()

    # --- 1. Load Config & Device ---
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = config['evaluation']['checkpoint_dir']

    print(f"Evaluating 5-Fold Experiment: {checkpoint_dir}")
    print("=" * 50)

    # --- 2. Load Frozen Text Embeddings ---
    E_data = torch.load(config['data']['concept_embeddings'], map_location=device)
    E_text = E_data['embeddings'].to(device)
    num_concepts = E_text.shape[0]
    d_t = E_text.shape[1]

    # Collect each fold's full metrics dict (replaces the manual dict of lists)
    all_fold_metrics = []

    # --- 3. Loop over the 5 folds ---
    for fold in range(1, 6):
        csv_dir = config['data']['csv_dir']
        test_csv = os.path.join(csv_dir, f"fold_{fold}", "test_labels.csv")
        checkpoint_path = os.path.join(checkpoint_dir, f"best_model_fold_{fold}.pth")

        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint for Fold {fold} not found. Skipping...")
            continue

        test_dataset = IntrAClassificationDataset(
            csv_file=test_csv,
            data_dir=config['data']['data_dir']
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['evaluation']['batch_size'],
            shuffle=False
        )

        backbone_name = config['model']['backbone']
        if backbone_name == "VanillaPointNet":
            backbone = VanillaPointNet()
        elif backbone_name == "OfficialPointNet":
            backbone = OfficialPointNet()
        elif backbone_name == "PointNetPlusPlus":
            backbone = PointNetPlusPlus(
                out_channel=config['model']['d_f'],
                in_channel=config['model']['in_channel'],
            )
        elif backbone_name == "DGCNN":
            backbone = DGCNN(
                out_channel=config['model']['d_f'],
                in_channel=config['model']['in_channel'],
                k=config['model'].get('k', 20),
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        model = ConceptBottleneck3D(
            backbone=backbone,
            d_f=config['model']['d_f'],
            d_t=d_t,
            num_concepts=num_concepts,
            num_classes=config['model']['num_classes'],
            d_hidden=config['model']['d_hidden']
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            model.load_state_dict(checkpoint, strict=True)

        metrics = evaluate_model(
            model=model,
            dataloader=test_dataloader,
            E_text=E_text,
            device=device,
            task=config['experiment']['task'],
            fold=fold
        )
        all_fold_metrics.append(metrics)

    # --- 4. Aggregate, excluding collapsed folds ---
    if config['experiment']['task'] == "classification" and all_fold_metrics:
        aggregate_fold_metrics(all_fold_metrics)

if __name__ == "__main__":
    main()