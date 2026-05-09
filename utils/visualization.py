import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def write_ply(points, colors, filename):
    """
    Writes 3D points and their RGB colors to a standard .ply file.
    No heavy 3D libraries (like Open3D) required!
    
    Args:
        points: numpy array of shape (N, 3)
        colors: numpy array of shape (N, 3), values 0-255
        filename: output path
    """
    N = points.shape[0]
    
    # Standard PLY header
    header = f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    # Combine points and colors
    data = np.hstack((points, colors))
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, data, fmt='%.6f %.6f %.6f %d %d %d')

def export_concept_attention(points, attention_scores, out_path, cmap_name='jet'):
    """
    Maps 1D attention scores to a 3D heatmap and saves it.
    
    Args:
        points: (N, 3) tensor or array
        attention_scores: (N,) tensor or array of raw attention values
        out_path: Where to save the .ply file
        cmap_name: Matplotlib colormap (jet, viridis, inferno, etc.)
    """
    # 1. Convert to numpy if they are PyTorch tensors
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(attention_scores):
        attention_scores = attention_scores.detach().cpu().numpy()
        
    # 2. Normalize attention scores to strictly [0.0, 1.0]
    # This ensures the highest attention point is always pure red
    a_min, a_max = attention_scores.min(), attention_scores.max()
    if a_max - a_min > 1e-6:
        normalized_attn = (attention_scores - a_min) / (a_max - a_min)
    else:
        normalized_attn = np.zeros_like(attention_scores)
        
    # 3. Apply Colormap
    cmap = plt.get_cmap(cmap_name)
    # cmap returns RGBA in [0, 1], we take RGB and scale to [0, 255]
    colors = (cmap(normalized_attn)[:, :3] * 255).astype(np.uint8)
    
    # 4. Save the file
    write_ply(points, colors, out_path)
    print(f"Saved attention heatmap to {out_path}")

def visualize_all_concepts(X_batch, A_batch, concept_names, save_dir="visualizations", shape_idx=0):
    """
    Wrapper to export a .ply file for EVERY concept for a specific shape in a batch.
    
    Args:
        X_batch: (B, N, 3) tensor of points
        A_batch: (B, N, K) tensor of attention matrices
        concept_names: List of strings (length K)
        shape_idx: Which shape in the batch to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract the single shape and its attention matrix
    points = X_batch[shape_idx] # (N, 3)
    attention = A_batch[shape_idx] # (N, K)
    
    for k, concept in enumerate(concept_names):
        # Clean the concept string for file naming
        safe_name = concept.replace(" ", "_").replace(",", "")
        out_name = f"shape_{shape_idx:03d}_concept_{safe_name}.ply"
        out_path = os.path.join(save_dir, out_name)
        
        # Pass the 1D attention slice (N,) for this specific concept
        export_concept_attention(points, attention[:, k], out_path)

# --- Quick Test Block ---
if __name__ == "__main__":
    # Simulate a single point cloud (a sphere)
    print("Generating a test 3D sphere...")
    u = np.random.uniform(0, 2 * np.pi, 2048)
    v = np.random.uniform(0, np.pi, 2048)
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)
    mock_points = np.vstack((x, y, z)).T
    
    # Simulate attention: Let's make the model "look" at the top of the sphere (high Z values)
    mock_attention = mock_points[:, 2] 
    
    # Save it!
    os.makedirs("test_viz", exist_ok=True)
    export_concept_attention(mock_points, mock_attention, "test_viz/test_heatmap.ply")
    print("Test complete! You can open test_viz/test_heatmap.ply in MeshLab to see a sphere with a red 'cap'.")