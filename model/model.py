import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaPointNet(nn.Module):
    """
    A lightweight PointNet backbone for testing and ablation.
    Takes in [Batch, 3, Points] and outputs a global feature vector [Batch, 1024].
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x expected shape: [B, 3, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024) # Flatten to [B, 1024]
        return x

class TNet(nn.Module):
    """
    Spatial Transformer Network: Learns to align the point cloud or feature space.
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialize identity matrix for the transformation
        iden = torch.eye(self.k, requires_grad=True).repeat(B, 1, 1).to(x.device)
        x = x.view(-1, self.k, self.k) + iden
        return x

class OfficialPointNet(nn.Module):
    """
    PointNet with Input and Feature T-Nets, returning per-point features
    compatible with ConceptBottleneck3D / CCAM.

    Output interface matches PointNet++ and DGCNN:
        Input : (B, in_channel, N)
        Output: (B, N, out_channel)

    The original global max-pool is removed so CCAM can attend over all
    points. A Conv1d projection maps the 1024-dim per-point descriptors
    down to out_channel.

    Args:
        out_channel : per-point feature dimension (d_f). Default 128.
        in_channel  : 3 for xyz, 6 for xyz + normals. Default 6.
    """
    def __init__(self, out_channel=128, in_channel=6):
        super().__init__()
        self.stn  = TNet(k=in_channel)  # Input Transform — must match in_channel
        self.fstn = TNet(k=64)          # Feature Transform

        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        # Project per-point 1024-dim descriptors → out_channel
        self.proj = nn.Sequential(
            nn.Conv1d(1024, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        # 1. Input Alignment
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # 2. Feature Alignment
        trans_feat = self.fstn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        # 3. Per-point Feature Extraction  (B, 1024, N)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))

        # 4. Project to out_channel — no global pooling
        x = self.proj(x)                   # (B, out_channel, N)
        return x.transpose(1, 2)           # (B, N, out_channel)

class CCAM(nn.Module):
    """
    Concept Cross-Attention Module (g function)
    """
    def __init__(self, d_f, d_t, d_hidden):
        super().__init__()
        # Normalization to bridge the modality gap
        self.norm_f = nn.LayerNorm(d_f)
        self.norm_t = nn.LayerNorm(d_t)
        
        # Projections into shared latent space
        self.proj_features = nn.Linear(d_f, d_hidden)
        self.proj_text = nn.Linear(d_t, d_hidden)
        
        self.scale = d_hidden ** -0.5

    def forward(self, F_spatial, E_text):
        B = F_spatial.shape[0] # Get batch size
        
        # Apply LayerNorm THEN project
        feat_proj = self.proj_features(self.norm_f(F_spatial)) # Shape: (B, N, d_hidden)
        
        # Project text, add batch dimension, and expand to match B
        text_proj = self.proj_text(self.norm_t(E_text))        # Shape: (K, d_hidden)
        # text_proj = text_proj.unsqueeze(0).expand(B, -1, -1)   # Shape: (B, K, d_hidden)
        
        # Cross-attention dot product
        # (B, N, d_hidden) x (B, d_hidden, K) -> (B, N, K)
        attention_logits = torch.matmul(feat_proj, text_proj.T) * self.scale
        A = torch.sigmoid(attention_logits) 
        
        return A

class ConceptBottleneck3D(nn.Module):
    """
    The full CBM pipeline combining the backbone, CCAM, and transparent predictor.
    """
    def __init__(self, backbone, d_f, d_t, d_hidden, num_concepts, num_classes, task="classification", beta=10.0):
        super().__init__()
        self.task = task
        
        # 1. The spatial feature extractor (f)
        self.backbone = backbone

        self.feat_norm = nn.LayerNorm(d_f)
        
        # 2. The Concept Cross-Attention Module (g)
        self.ccam = CCAM(d_f, d_t, d_hidden)
        self.dropout = nn.Dropout(p=0.3)
        
        # 3. The Transparent Predictor (h)
        self.predictor = nn.Linear(num_concepts, num_classes)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)

    def forward(self, X, E_text):
        """
        Args:
            X: Input point clouds (B, N, 3)
            E_text: Frozen text embeddings (K, d_t)
        """
        # Step 1: Extract spatial features from the backbone
        # transpose X to [B, 3, N] for PointNet and back to [B, N, d_f] for CCAM
        X = X.transpose(1, 2) # [B, N, 3] -> [B, 3, N]
        F_spatial = self.backbone(X) # (B, N, d_f)
        print(f"Backbone output shape (F_spatial): {F_spatial.shape}")
        F_spatial = self.feat_norm(F_spatial) # Normalize features before attention
        
        # Step 2: Generate the dense spatial attention matrix
        A = self.ccam(F_spatial, E_text) # (B, N, K)
        print(f"Attention matrix shape (A): {A.shape}")
        # Step 3: Log-Sum-Exp Pooling
        # Formula: C_hat = (1/beta) * log( sum( exp( beta * A ) ) )
        # We subtract the max for numerical stability (Log-Sum-Exp trick)
        # weighted_A = self.beta * A
        # A_max, _ = torch.max(weighted_A, dim=1, keepdim=True)
        # Optional step 3: use top-k pooling instead of log-sum-exp for a sharper focus on the most relevant points
        k_value = int(A.shape[1] * 0.1)
        print(f"Using top-{k_value} pooling for sharper concept activation.")
        A_topk, _ = torch.topk(A, k=k_value, dim=1)
        print(f"Top-{k_value} attention shape: {A_topk.shape}")
        beta = torch.clamp(self.beta, min=1.0, max=50.0) # Optional: clamp beta to prevent extreme values
        weighted_A = beta * A_topk
        print(f"Weighted top-{k_value} attention shape: {weighted_A.shape}")
        A_max, _ = torch.max(weighted_A, dim=1, keepdim=True)
        
        # Step 3: Pool attention to get global concept scores (C_hat)
        C_hat = (A_max + torch.log(torch.sum(torch.exp(weighted_A - A_max), dim=1, keepdim=True))) / beta
        C_hat = C_hat.squeeze(1) # Final shape: (B, K)
        C_hat_dropped = self.dropout(C_hat)
        
        # Step 4: Final Task Prediction
        if self.task == "classification":
            # Predict based on global concept presence
            Y_hat = self.predictor(C_hat_dropped) # (B, num_classes)
        elif self.task == "segmentation":
            # Predict for every single point
            Y_hat = self.predictor(A)     # (B, N, num_classes)
        else:
            raise ValueError(f"Unknown task: {self.task}")
            
        return Y_hat, C_hat, A

def square_distance(src, dst):
    """Calculates squared distance between two tensor sets of points."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """Retrieves points from the point cloud using indices."""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """Iteratively samples the most distant points to get an even spread."""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """Finds points within a physical radius of a center point."""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetFeaturePropagation(nn.Module):
    """
    Upsamples features from a coarser level to a finer level using
    inverse-distance-weighted 3-NN interpolation, then concatenates
    with skip-connection features and runs an MLP.
    """
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Args:
            xyz1:    [B, 3, N]  target coordinates (finer resolution)
            xyz2:    [B, 3, S]  source coordinates (coarser resolution)
            points1: [B, D, N]  skip-connection features at xyz1, or None
            points2: [B, D', S] features at xyz2 to be propagated
        Returns:
            [B, mlp[-1], N]
        """
        # Convert to point-last layout for distance calculations
        xyz1 = xyz1.permute(0, 2, 1)       # [B, N, 3]
        xyz2 = xyz2.permute(0, 2, 1)       # [B, S, 3]
        points2 = points2.permute(0, 2, 1) # [B, S, D']

        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            # Degenerate case: only one source point, broadcast it.
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 3-nearest-neighbor inverse-distance-weighted interpolation.
            dists = square_distance(xyz1, xyz2)         # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3] # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm                   # [B, N, 3]

            interpolated_points = torch.sum(
                index_points(points2, idx) * weight.view(B, N, 3, 1),
                dim=2,
            )                                            # [B, N, D']

        # Skip connection: concat encoder features at this level
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)           # [B, N, D]
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)         # [B, D+D', N]

        # Pointwise MLP
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))

        return new_points                                # [B, mlp[-1], N]

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        # xyz: coordinates [B, C, N], points: features [B, D, N]
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = xyz, points
            grouped_xyz = xyz.view(xyz.shape[0], 1, xyz.shape[1], 3)
            if points is not None:
                grouped_points = points.view(points.shape[0], 1, points.shape[1], -1)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz
        else:
            new_xyz_idx = farthest_point_sample(xyz, self.npoint)
            new_xyz = index_points(xyz, new_xyz_idx)
            idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = index_points(xyz, idx)
            grouped_xyz -= new_xyz.view(new_xyz.shape[0], new_xyz.shape[1], 1, 3)
            if points is not None:
                grouped_points = index_points(points, idx)
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
            else:
                grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1) # [B, D, nsample, npoint]
        
        for i, conv in enumerate(self.mlp_convs):
            grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))
            
        new_points = torch.max(grouped_points, 2)[0] # Max pooling over local neighborhood
        
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points

# The segmentation version of PointNet++, serving as the U-Net backbone for CCAM
class PointNetPlusPlus(nn.Module):
    """
    Segmentation-style PointNet++ backbone.
    Input:  [B, C, N]  where C=6 (xyz + normals) or C=3 (xyz only)
    Output: [B, N, d_f]  (per-point features, ready for CCAM)
    """
    def __init__(self, out_channel=128, in_channel=6):  # default to 6D
        super().__init__()
        self.in_channel = in_channel

        # --- Encoder ---
        # in_channel for sa1 = 3 (xyz) + 3 (normals) = 6
        # sa2/sa3 are unchanged — they receive SA-extracted features, not raw input
        self.sa1 = PointNetSetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channel=in_channel, mlp=[64, 64, 128], group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channel=128 + 3, mlp=[128, 128, 256], group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=32, radius=0.8, nsample=64,
            in_channel=256 + 3, mlp=[256, 256, 512], group_all=False,
        )

        # --- Decoder --- unchanged
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128,       mlp=[128, 128, out_channel])

        self.out_channel = out_channel

    def forward(self, x):
        """
        x: [B, C, N]
          C=6 → split into xyz [B,3,N] and normals [B,3,N]
          C=3 → xyz only, no additional features
        returns: [B, N, out_channel]
        """
        l0_xyz = x[:, :3, :]                                        # always xyz
        l0_points = x[:, 3:, :] if self.in_channel > 3 else None   # normals or None

        # Encode
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  # [B,3,512],  [B,128,512]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # [B,3,128],  [B,256,128]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  # [B,3,32],   [B,512,32]

        # Decode
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  # [B,256,128]
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)  # [B,128,512]
        l0_points = self.fp1(l0_xyz, l1_xyz, None,      l1_points)  # [B,out_channel,N]

        return l0_points.transpose(1, 2).contiguous()  # [B, N, out_channel]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # --- FP sanity ---
    print("Testing PointNetFeaturePropagation...")
    fp = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256]).to(device)
    xyz1 = torch.randn(2, 3, 128).to(device)
    xyz2 = torch.randn(2, 3, 32).to(device)
    p1   = torch.randn(2, 256, 128).to(device)
    p2   = torch.randn(2, 512, 32).to(device)
    out = fp(xyz1, xyz2, p1, p2)
    assert out.shape == (2, 256, 128), f"FP shape wrong: {out.shape}"
    print(f"  FP output: {out.shape}  ✓")

    # --- Full pipeline ---
    print("\nTesting ConceptBottleneck3D end-to-end...")
    B, N, d_f, d_t, d_hidden, K, C = 4, 2048, 128, 512, 256, 6, 1

    dummy_points = torch.rand(B, N, 3).to(device)
    dummy_text_embeds = torch.rand(K, d_t).to(device)

    backbone = PointNetPlusPlus(out_channel=d_f).to(device)
    model = ConceptBottleneck3D(
        backbone=backbone,
        d_f=d_f, d_t=d_t, d_hidden=d_hidden,
        num_concepts=K, num_classes=C,
        task="classification",
    ).to(device)

    Y_hat, C_hat, A = model(dummy_points, dummy_text_embeds)

    # Expected shapes
    assert Y_hat.shape == (B, C), f"Y_hat shape wrong: {Y_hat.shape}"
    assert C_hat.shape == (B, K), f"C_hat shape wrong: {C_hat.shape}"
    assert A.shape == (B, N, K), f"A shape wrong: {A.shape}"
    assert (A >= 0).all() and (A <= 1).all(), "Sigmoid attention out of [0,1]"

    print(f"  Input points:   {dummy_points.shape}")
    print(f"  Input text:     {dummy_text_embeds.shape}")
    print("-" * 50)
    print(f"  Y_hat:          {Y_hat.shape}")
    print(f"  C_hat:          {C_hat.shape}   (range: [{C_hat.min():.3f}, {C_hat.max():.3f}])")
    print(f"  A:              {A.shape}   (range: [{A.min():.3f}, {A.max():.3f}])")
    print("  All shape checks passed ✓")