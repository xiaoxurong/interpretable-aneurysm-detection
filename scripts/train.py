from CCAM import dataset
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import wandb
import yaml
from tqdm import tqdm
import os
import math

from CCAM.dataset import IntrAClassificationDataset
from CCAM.model import VanillaPointNet, ConceptBottleneck3D, OfficialPointNet, PointNetPlusPlus
from CCAM.loss import CCAMLoss
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Subset, WeightedRandomSampler
from backbones.DGCNN import DGCNN

def train_one_epoch(model, dataloader, criterion, optimizer, E_text, device, lambda_l1=1e-4):
    model.train()
    metrics = {'total': 0.0, 'task': 0.0, 'align': 0.0, 'sparse': 0.0, 'ortho': 0.0, 'supcon': 0.0}
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_idx, (X, Y_target) in enumerate(pbar):
        X = X.to(device)
        Y_target = Y_target.to(device)
        dummy_C_target = torch.zeros(X.size(0), model.ccam.proj_text.out_features).to(device)
        
        optimizer.zero_grad()
        Y_hat, C_hat, A = model(X, E_text)
        # check the distribution of C_hat values
        # print(f"Batch {batch_idx}: C_hat min={C_hat.min().item():.4f}, max={C_hat.max().item():.4f}, mean={C_hat.mean().item():.4f}")
        
        total_loss, l_task, l_align, l_sparse, l_ortho, l_supcon = criterion(
            Y_hat, Y_target, C_hat, dummy_C_target, A
        )
        
        l1_reg = model.predictor.weight.abs().sum()
        total_loss = total_loss + lambda_l1 * l1_reg

        total_loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(Y_hat, dim=1)
        probs = torch.softmax(Y_hat, dim=1)[:, 1] 
        
        all_targets.extend(Y_target.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        metrics['total'] += total_loss.item()
        metrics['task'] += l_task.item()
        metrics['align'] += l_align.item()
        metrics['sparse'] += l_sparse.item()
        metrics['ortho'] += l_ortho.item()
        metrics['supcon'] += l_supcon.item()
        
        pbar.set_postfix({
            'Total': f"{total_loss.item():.3f}",
            'Task': f"{l_task.item():.3f}"
        })
        
    for key in ['total', 'task', 'align', 'sparse', 'ortho', 'supcon']:
        metrics[key] /= len(dataloader)
        
    metrics['accuracy'] = (np.array(all_preds) == np.array(all_targets)).mean()
    metrics['precision'] = precision_score(all_targets, all_preds, zero_division=0)
    metrics['recall'] = recall_score(all_targets, all_preds, zero_division=0)
    metrics['f1'] = f1_score(all_targets, all_preds, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics['acc_class_0'] = tn / (tn + fp + 1e-8)
    metrics['acc_class_1'] = tp / (tp + fn + 1e-8)
    metrics['balanced_accuracy'] = (metrics['acc_class_0'] + metrics['acc_class_1']) / 2
    
    try:
        metrics['auroc'] = roc_auc_score(all_targets, all_probs)
    except ValueError:
        metrics['auroc'] = 0.5 
        
    return metrics

@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, E_text, device):
    model.eval()
    metrics = {'total': 0.0, 'task': 0.0, 'align': 0.0, 'sparse': 0.0, 'ortho': 0.0, 'supcon': 0.0}
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    for X, Y_target in dataloader:
        X = X.to(device)
        Y_target = Y_target.to(device)
        
        Y_hat, C_hat, A = model(X, E_text)
        dummy_C_target = torch.zeros_like(C_hat)
        total_loss, l_task, l_align, l_sparse, l_ortho, l_supcon = criterion(
            Y_hat.squeeze(), Y_target, C_hat, dummy_C_target, A
        )
        
        predictions = torch.argmax(Y_hat, dim=1)
        probs = torch.softmax(Y_hat, dim=1)[:, 1] 
        
        all_targets.extend(Y_target.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        metrics['total'] += total_loss.item()
        metrics['task'] += l_task.item()
        metrics['align'] += l_align.item()
        metrics['sparse'] += l_sparse.item()
        metrics['ortho'] += l_ortho.item()
        metrics['supcon'] += l_supcon.item()
    for key in ['total', 'task', 'align', 'sparse', 'ortho', 'supcon']:
        metrics[key] /= len(dataloader)
        
    metrics['accuracy'] = (np.array(all_preds) == np.array(all_targets)).mean()
    metrics['precision'] = precision_score(all_targets, all_preds, zero_division=0)
    metrics['recall'] = recall_score(all_targets, all_preds, zero_division=0)
    metrics['f1'] = f1_score(all_targets, all_preds, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics['acc_class_0'] = tn / (tn + fp + 1e-8)
    metrics['acc_class_1'] = tp / (tp + fn + 1e-8)
    metrics['balanced_accuracy'] = (metrics['acc_class_0'] + metrics['acc_class_1']) / 2
    try:
        metrics['auroc'] = roc_auc_score(all_targets, all_probs)
    except ValueError:
        metrics['auroc'] = 0.5
        
    return metrics

def train_single_fold(fold, config, E_text, num_concepts, d_t, device):
    """Handles everything required to train from scratch on one specific fold."""
    print(f"\n{'='*50}\n🚀 STARTING TRAINING FOR FOLD {fold}/5\n{'='*50}")

    # --- 1. W&B Initialization ---
    base_experiment_name = config['experiment']['name']
    
    wandb.init(
        project="IntrA-CBM-CV",
        group=base_experiment_name,
        name=f"{base_experiment_name}_Fold_{fold}",
        config=config,
        tags=[f"fold_{fold}"],
        reinit=True
    )

    # SWEEP HANDLING
    if wandb.run.sweep_id is not None:
        print(f"Detected W&B Sweep ID: {wandb.run.sweep_id}")
        if 'lambda_align' in wandb.config:
            config['loss_weights']['lambda_align'] = wandb.config.lambda_align
        if 'lambda_sparse' in wandb.config:
            config['loss_weights']['lambda_sparse'] = wandb.config.lambda_sparse
        if 'lambda_ortho' in wandb.config:
            config['loss_weights']['lambda_ortho'] = wandb.config.lambda_ortho

    # --- 2. Dynamic Data Paths ---
    output_root = config['data']['csv_dir']
    fold_dir = os.path.join(output_root, f"fold_{fold}")
    
    train_dataset = IntrAClassificationDataset(
        csv_file=os.path.join(fold_dir, "train_labels.csv"), 
        data_dir=config['data']['data_dir'], 
        augment=True
    )
    
    val_dataset = IntrAClassificationDataset(
        csv_file=os.path.join(fold_dir, "val_labels.csv"), 
        data_dir=config['data']['data_dir'], 
        augment=False
    )

    def make_balanced_sampler(dataset):
        labels = [dataset[i][1] for i in range(len(dataset))]
        labels = torch.tensor(labels)
        class_counts = torch.bincount(labels)
        weights = 1.0 / class_counts.float()
        sample_weights = weights[labels]
        return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)

    # --- 3. Initialize Fresh Model ---
    if config['model']['backbone'] == "VanillaPointNet":
        backbone = VanillaPointNet()
    elif config['model']['backbone'] == "OfficialPointNet":
        backbone = OfficialPointNet()
    elif config['model']['backbone'] == "PointNetPlusPlus":
        backbone = PointNetPlusPlus(
        out_channel=config['model']['d_f'],
        in_channel=config['model']['in_channel']  # 6 for xyz+normals
        )
    elif config['model']['backbone'] == "DGCNN":
        backbone = DGCNN(
        out_channel=config['model']['d_f'],
        in_channel=config['model']['in_channel'],
        k=config['model'].get('k', 20),   # add k: 20 to your config
        )
        
    model = ConceptBottleneck3D(
        backbone=backbone, 
        d_f=config['model']['d_f'], 
        d_t=d_t, 
        d_hidden=config['model']['d_hidden'],
        num_concepts=num_concepts, 
        num_classes=config['model']['num_classes'],
        task=config['experiment']['task']
    ).to(device)

    criterion = CCAMLoss(
    alpha=config['loss_weights']['alpha'],
    gamma=config['loss_weights']['gamma'],
    lambda_align=config['loss_weights']['lambda_align'],
    lambda_sparse=config['loss_weights']['lambda_sparse'],
    lambda_ortho=config['loss_weights']['lambda_ortho'],
    lambda_supcon=config['loss_weights']['lambda_supcon'], 
    temperature_supcon=config['loss_weights']['temperature'],
    task=config['experiment']['task']
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # --- 5. Directory Setup for Saving Checkpoints ---
    save_dir = os.path.join(config['experiment']['save_dir'], base_experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"best_model_fold_{fold}.pth")

    # --- 6. The Epoch Loop (With Early Stopping & Warmup) ---
    best_val_auroc = 0.0  # Switched to tracking AUROC
    epochs_without_improvement = 0
    
    # Retrieve patience and warmup from config (with defaults)
    patience = config['training'].get('patience', 15)
    warmup_epochs = config['training'].get('warmup_epochs', 30) 
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(1, num_epochs + 1):
        # Train & Validate
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, E_text, device)
        val_metrics = validate_one_epoch(model, val_loader, criterion, E_text, device)

        scheduler.step(val_metrics['task'])
        
        # Log to W&B
        wandb.log({
            "epoch": epoch,
            **{f"Train/{k}": v for k, v in train_metrics.items()},
            **{f"Val/{k}": v for k, v in val_metrics.items()},
        })
        
        # Console output
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"  Train | Loss: {train_metrics['total']:.4f} | Acc: {train_metrics['accuracy']:.3f} | F1: {train_metrics['f1']:.3f} | AUC: {train_metrics['auroc']:.3f}")
        print(f"  Val   | Loss: {val_metrics['total']:.4f} | Acc: {val_metrics['accuracy']:.3f} | F1: {val_metrics['f1']:.3f} | AUC: {val_metrics['auroc']:.3f}")

        # Extract Macro F1 Score
        current_val_metric = val_metrics['auroc'] 

        # Save Best Checkpoint (Based on Macro F1 Generalization)
        if current_val_metric > best_val_auroc:
            best_val_auroc = current_val_metric
            epochs_without_improvement = 0  # Reset patience counter
            
            torch.save({
                'epoch': epoch,
                'fold': fold,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auroc': best_val_auroc,
            }, checkpoint_path)
        
            
            print(f"  --> Saved new best model for Fold {fold}! Val AUROC: {best_val_auroc:.4f}")
            wandb.save(checkpoint_path, base_path=save_dir) # Instantly uploads to W&B
        else:
            epochs_without_improvement += 1
            
        if epoch <= warmup_epochs:
            epochs_without_improvement = 0 
            print(f"  --> [Warmup Phase] Evaluating concepts. Early stopping disabled.")
        else:
            if current_val_metric <= best_val_auroc:
                epochs_without_improvement += 1
            print(f"  --> No improvement for {epochs_without_improvement}/{patience} epochs.")
        
        # check beta.item every 10 epochs
        if epoch % 10 == 0:
            print(f"  Current beta value: {model.beta.item():.4f}")
            
        # Trigger Early Stopping
        if epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping triggered! Training halted for Fold {fold} at epoch {epoch}.")
            break

    # Close W&B Run for this fold
    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Train the 3D Concept Bottleneck Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading config: {config['experiment']['name']} | Device: {device}")

    # Load Frozen Text Embeddings ONCE before the loop starts
    E_data = torch.load(config['data']['concept_embeddings'], map_location=device)
    E_text = E_data['embeddings'].to(device)
    num_concepts = E_text.shape[0]
    d_t = E_text.shape[1]

    # Run the 5-Fold Cross Validation Sequentially
    for fold in range(1, 6):
        train_single_fold(fold, config, E_text, num_concepts, d_t, device)

    print(f"\n🎉 ALL 5 FOLDS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()