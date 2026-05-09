import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, jaccard_score, classification_report

def compute_classification_metrics(y_true, y_pred_probs, y_pred_classes):
    """Calculates standard classification metrics."""
    acc = accuracy_score(y_true, y_pred_classes)
    auc = roc_auc_score(y_true, y_pred_probs)
    report = classification_report(y_true, y_pred_classes, target_names=["Healthy", "Aneurysm"])
    
    return {"accuracy": acc, "auc": auc, "report": report}

def compute_segmentation_metrics(y_true, y_pred_classes):
    """Calculates 3D point-cloud segmentation metrics."""
    acc = accuracy_score(y_true, y_pred_classes)
    miou = jaccard_score(y_true, y_pred_classes, average='macro')
    
    return {"point_accuracy": acc, "miou": miou}

def compute_concept_metrics(c_true, c_pred):
    """Calculates how well the model predicts the VLM pseudo-labels."""
    c_true = np.array(c_true)
    c_pred = np.array(c_pred)
    mse = np.mean((c_true - c_pred) ** 2)
    
    return {"concept_mse": mse}