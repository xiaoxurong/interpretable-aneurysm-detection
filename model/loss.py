# from CCAM.model import C_hat
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Upgraded to Binary Cross Entropy for [0, 1] probability matching
        self.bce = nn.BCELoss()
        
    def forward(self, C_hat, C_target):
        mask = ((C_target > 0.8) | (C_target < 0.2)).float()
        return (F.binary_cross_entropy(C_hat, C_target, reduction='none') * mask).mean()

class SparsityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, A):
        # Shannon Entropy to force localized attention
        # Normalize A across points (dim=1) so it sums to 1.0 
        A_norm = A / (A.sum(dim=1, keepdim=True) + self.eps)
        entropy = -torch.sum(A_norm * torch.log(A_norm + self.eps), dim=1)
        return entropy.mean()

class OrthogonalityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, A):
        # Penalize overlapping attention between different concepts
        # A shape: (B, N, K)
        B, N, K = A.shape
        
        # 1. L2 Normalize the attention vectors along the spatial dimension (N)
        # This converts the dot product into pure cosine similarity
        A_norm = A / (A.norm(dim=1, keepdim=True) + self.eps)
        
        # 2. Vectorized Batch Matrix Multiplication (B, K, N) @ (B, N, K) -> (B, K, K)
        # sim_matrix[b, i, j] gives the cosine similarity between concept i and j for batch b
        sim_matrix = torch.bmm(A_norm.transpose(1, 2), A_norm)
        
        # 3. Mask out the diagonal (a concept's similarity with itself is always 1)
        mask = torch.ones((K, K), device=A.device) - torch.eye(K, device=A.device)
        mask = mask.unsqueeze(0) # Shape: (1, K, K) for broadcasting over batch
        
        off_diagonal_sim = sim_matrix.abs() * mask
        
        # 4. Average the overlapping similarities
        # Divide by B * K * (K - 1) to get the mean of all non-diagonal pairs
        loss_ortho = off_diagonal_sim.sum() / (B * K * (K - 1))
            
        return loss_ortho

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        """
        Args:
            alpha: weight for the minority (aneurysm) class.
                   Set > 0.5 to up-weight the rare class.
            gamma: focusing parameter. Higher = more focus on hard,
                   misclassified examples. 0 reduces to weighted CE.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, num_classes) — raw logits, NOT softmax
            targets: (B,) — integer class labels
        """
        # Standard CE per sample, shape: (B,)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Convert to probability of the true class
        pt = torch.exp(-ce_loss)
        
        # Alpha weighting: alpha for class 1 (aneurysm), 1-alpha for class 0
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1 - self.alpha, device=logits.device))
        
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class SupervisedContrastiveLoss(nn.Module):
    """
    SupCon loss applied to concept bottleneck vectors C_hat.

    Pulls same-class concept vectors together, pushes different-class apart.
    Reference: Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020).

    Args:
        temperature: Scales the dot products. Lower = sharper distribution.
                     Typical range: 0.07 – 0.3. Start with 0.1.
    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, C_hat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            C_hat  : (B, K) concept scores from the bottleneck
            labels : (B,)   ground truth class labels (0 or 1)
        Returns:
            Scalar loss.
        """
        B = C_hat.shape[0]

        # ── L2-normalize so dot product == cosine similarity ──────────────
        z = F.normalize(C_hat, dim=1)          # (B, K)

        # ── pairwise cosine similarity matrix ─────────────────────────────
        sim = torch.matmul(z, z.T) / self.temperature   # (B, B)

        # ── masks ─────────────────────────────────────────────────────────
        labels = labels.view(-1, 1)                          # (B, 1)
        same_class = (labels == labels.T).float()            # (B, B)
        # remove self-similarity from positives and denominator
        identity = torch.eye(B, device=C_hat.device)
        positives = same_class - identity                    # (B, B)
        negatives = (1 - same_class)                         # (B, B)
        denominator_mask = positives + negatives             # all pairs except self

        # ── numerical stability: subtract row max ─────────────────────────
        sim_max, _ = (sim * denominator_mask - 1e9 * identity).max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # ── log-sum-exp over all non-self pairs ───────────────────────────
        exp_sim = torch.exp(sim) * denominator_mask          # zero out self
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # ── average log-prob over positive pairs ──────────────────────────
        num_positives = positives.sum(dim=1)                 # (B,)

        # only compute loss for anchors that have at least one positive
        valid = num_positives > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=C_hat.device)

        log_prob = sim - log_denom                           # (B, B)
        mean_log_prob_pos = (log_prob * positives).sum(dim=1) / \
                            (num_positives + 1e-8)           # (B,)

        loss = -mean_log_prob_pos[valid].mean()
        return loss

class CCAMLoss(nn.Module):
    def __init__(self, class_weights=None, alpha=0.75, gamma=2.0, lambda_align=0, lambda_sparse=0.01, lambda_ortho=0.01, lambda_supcon=0.01, temperature_supcon=0.1, task="classification"):
        super().__init__()
        self.lambda_align = lambda_align
        self.lambda_sparse = lambda_sparse
        self.lambda_ortho = lambda_ortho
        self.lambda_supcon = lambda_supcon
        self.task = task
        # supervised contrastive loss for concept vectors
        self.supcon_criterion = SupervisedContrastiveLoss(temperature=temperature_supcon)

        self.task_criterion = FocalLoss(alpha=alpha, gamma=gamma)
        self.align_criterion = SemanticAlignmentLoss()
        self.sparse_criterion = SparsityLoss()
        self.ortho_criterion = OrthogonalityLoss()

    def forward(self, Y_hat, Y_target, C_hat, C_target, A):
        """
        Args:
            Y_hat: Task predictions from the predictor (B, Classes)
            Y_target: Ground truth class labels (B,)
            C_hat: Predicted concept scores [0, 1] from spatial max-pooling (B, K)
            C_target: Pseudo-labels [0, 1] from the 3D-VLM oracle (B, K)
            A: Dense spatial attention matrix (B, N, K)
        """
        # Calculate individual losses using the initialized classes
        loss_task = self.task_criterion(Y_hat, Y_target)
        if self.lambda_align > 0:
            loss_align = self.align_criterion(C_hat, C_target)
        else:
            loss_align = torch.tensor(0.0, device=C_hat.device)
        loss_sparse = self.sparse_criterion(A)
        loss_ortho = self.ortho_criterion(A)
        loss_supcon = self.supcon_criterion(C_hat, Y_target)

        # Combine everything together using the lambdas
        total_loss = loss_task + (self.lambda_align * loss_align) + \
                     (self.lambda_sparse * loss_sparse) + (self.lambda_ortho * loss_ortho) + \
                     (self.lambda_supcon * loss_supcon)
                     
        return total_loss, loss_task, loss_align, loss_sparse, loss_ortho, loss_supcon