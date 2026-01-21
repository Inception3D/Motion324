"""
Visualization utilities for inference
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

try:
    import bpy
    from mathutils import Vector, Matrix, Euler
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False

def visualize_input_data(input_data, save_path="input_data_vis.png"):
    """
    Visualize input data including colors, normals (as colors), and ref_pcd.
    
    Args:
        input_data: Dictionary containing ref_shape_pcd, ref_shape_rgbs, ref_shape_normals, 
                   ref_pcd, ref_rgb, ref_normal
        save_path: Path to save the visualization image
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    points1 = input_data['ref_shape_pcd'][0].detach().cpu().numpy()
    rgbs1 = input_data['ref_shape_rgbs'][0].detach().cpu().numpy()
    rgbs1_clip = np.clip(rgbs1, 0, 1)
    ax1.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=rgbs1_clip, s=1)
    ax1.set_title("ref_shape_pcd RGB")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim([-0.5, 0.5])
    ax1.set_ylim([-0.5, 0.5])
    ax1.set_zlim([-0.5, 0.5])

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    normals1 = input_data['ref_shape_normals'][0].detach().cpu().numpy()
    normals_vis1 = (normals1 + 1) * 0.5
    ax2.scatter(points1[:, 0], points1[:, 1], points1[:, 2], c=normals_vis1, s=1)
    ax2.set_title("ref_shape_pcd Normals (as color)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim([-0.5, 0.5])
    ax2.set_ylim([-0.5, 0.5])
    ax2.set_zlim([-0.5, 0.5])

    ax3 = fig.add_subplot(gs[1, 0], projection='3d')
    points2 = input_data['ref_pcd'][0].detach().cpu().numpy()
    rgbs2 = input_data['ref_rgb'][0].detach().cpu().numpy()
    rgbs2_clip = np.clip(rgbs2, 0, 1)
    ax3.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c=rgbs2_clip, s=3)
    ax3.set_title("ref_pcd RGB")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim([-0.5, 0.5])
    ax3.set_ylim([-0.5, 0.5])
    ax3.set_zlim([-0.5, 0.5])

    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    normals2 = input_data['ref_normal'][0].detach().cpu().numpy()
    normals_vis2 = (normals2 + 1) * 0.5
    ax4.scatter(points2[:, 0], points2[:, 1], points2[:, 2], c=normals_vis2, s=3)
    ax4.set_title("ref_pcd Normals (as color)")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_xlim([-0.5, 0.5])
    ax4.set_ylim([-0.5, 0.5])
    ax4.set_zlim([-0.5, 0.5])

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Input data visualization saved to: {save_path}")

def visualize_pointcloud_prediction(result, sample, save_dir="./visualization_results", idx = 0, fps=8):
    """
    Visualize point cloud prediction results.
    
    Args:
        result: Model prediction results containing input_data, rgb, pcd_moved, loss_metrics.
        sample: Input data containing obj_name, rgb_video, point_clouds, uv_gt, color_gt.
        save_dir: Directory to save visualization results.
        fps: Animation frame rate.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    gt_point_clouds = sample['point_clouds'][idx].float().cpu().numpy()  # Shape: (T, N_points, D_features)
    pred_point_clouds = result['pcd_moved'][idx].float().cpu().numpy()   # Shape: (T, N_points, 3)
    rgb_frames = sample['rgb_video'][idx].float().cpu().numpy()          # Shape: (T, H, W, C)
    obj_name = sample['obj_name'][idx]
    
    T, N_points, _ = gt_point_clouds.shape
    
    if gt_point_clouds.shape[-1] > 3:
        gt_coords = gt_point_clouds[..., :3]
    else:
        gt_coords = gt_point_clouds
    
    pred_coords = pred_point_clouds
    
    
    all_coords = np.concatenate([gt_coords.reshape(-1, 3), pred_coords.reshape(-1, 3)], axis=0)
    min_vals = np.min(all_coords, axis=0)
    max_vals = np.max(all_coords, axis=0)
    center = (min_vals + max_vals) / 2
    max_range = np.max(max_vals - min_vals) / 2 * 1.2
    
    fig = plt.figure(figsize=(24, 6))
    gs = GridSpec(1, 4, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_title('Ground Truth', fontsize=14, fontweight='bold')
    
    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.set_title('Prediction', fontsize=14, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.set_title('GT (Blue) vs Pred (Red)', fontsize=14, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_title('RGB Video', fontsize=14, fontweight='bold')
    ax4.axis('off')
    
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim([center[0] - max_range, center[0] + max_range])
        ax.set_ylim([center[1] - max_range, center[1] + max_range])
        ax.set_zlim([center[2] - max_range, center[2] + max_range])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    if len(rgb_frames.shape) == 4 and rgb_frames.shape[-1] == 3:
        if rgb_frames.max() > 1.0:
            rgb_frames_normalized = rgb_frames / 255.0
        else:
            rgb_frames_normalized = rgb_frames
    else:
        print(f"Warning: RGB frames shape {rgb_frames.shape} is unexpected")
        rgb_frames_normalized = np.zeros((T, 224, 224, 3))
    
    im = ax4.imshow(rgb_frames_normalized[0])
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        ax1.set_title(f'GT (Frame {frame+1}/{T})', fontsize=14, fontweight='bold')
        ax2.set_title(f'Pred (Frame {frame+1}/{T})', fontsize=14, fontweight='bold')
        ax3.set_title(f'GT vs Pred (Frame {frame+1}/{T})', fontsize=14, fontweight='bold')
        ax4.set_title(f'RGB (Frame {frame+1}/{T})', fontsize=14, fontweight='bold')
        
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim([center[0] - max_range, center[0] + max_range])
            ax.set_ylim([center[1] - max_range, center[1] + max_range])
            ax.set_zlim([center[2] - max_range, center[2] + max_range])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        gt_coords_frame = gt_coords[frame]
        pred_coords_frame = pred_coords[frame]
        
        # GT subplot
        ax1.scatter(gt_coords_frame[:, 0], gt_coords_frame[:, 1], gt_coords_frame[:, 2], 
                   c='blue', s=2, alpha=0.6)
        
        # Prediction subplot
        ax2.scatter(pred_coords_frame[:, 0], pred_coords_frame[:, 1], pred_coords_frame[:, 2],
                   c='red', s=2, alpha=0.6)
        
        # Comparison subplot
        ax3.scatter(gt_coords_frame[:, 0], gt_coords_frame[:, 1], gt_coords_frame[:, 2],
                   c='blue', s=2, alpha=0.6, label='GT')
        ax3.scatter(pred_coords_frame[:, 0], pred_coords_frame[:, 1], pred_coords_frame[:, 2],
                   c='red', s=2, alpha=0.6, label='Pred')
        ax3.legend()
        
        im.set_array(rgb_frames_normalized[frame])
        
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=T, interval=1000//fps, blit=False)
    
    save_path = save_dir / f"{obj_name}_combined_visualization.gif"
    anim.save(str(save_path), writer='pillow', fps=fps, dpi=100)
    plt.close(fig)

    mse_per_frame = np.mean((gt_coords - pred_coords) ** 2, axis=(1, 2))
    avg_mse = np.mean(mse_per_frame)
    
    return {
        'visualization_path': str(save_path),
    }


def visualize_point_cloud_motion(ref_pcd, pcd_moved, trajs, save_path):
    """
    Visualize point cloud positions before and after movement with motion trajectory arrows.

    Args:
        ref_pcd: Original point cloud, shape (N, 3)
        pcd_moved: Moved point cloud, shape (N, 3)
        trajs: Motion trajectory vectors, shape (N, 3)
        save_path: Path to save the image
    """
    ref_pcd_np = ref_pcd.cpu().numpy()
    pcd_moved_np = pcd_moved.cpu().numpy()
    trajs_np = trajs.cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ref_pcd_np[:, 0], ref_pcd_np[:, 1], ref_pcd_np[:, 2], s=5, c='b', label='ref_pcd')
    ax.scatter(pcd_moved_np[:, 0], pcd_moved_np[:, 1], pcd_moved_np[:, 2], s=5, c='r', label='pcd_moved')
    ax.quiver(
        ref_pcd_np[:, 0], ref_pcd_np[:, 1], ref_pcd_np[:, 2],
        trajs_np[:, 0], trajs_np[:, 1], trajs_np[:, 2],
        length=1.0, normalize=False, color='g', linewidth=0.5, alpha=0.7
    )
    ax.set_title("Point Cloud Motion Comparison")
    ax.legend()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def plot_smoothing_comparison(displacements_before, displacements_after, 
                               motion_threshold, method, visualization_dir):
    """
    Plot comparison of trajectory smoothing before and after.
    
    Args:
        displacements_before: List of displacement arrays before smoothing
        displacements_after: List of displacement arrays after smoothing
        motion_threshold: Motion threshold used for filtering
        method: Smoothing method name
        visualization_dir: Directory to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    point_mean_before = np.mean(np.stack(displacements_before, axis=0), axis=0)
    point_mean_after = np.mean(np.stack(displacements_after, axis=0), axis=0)
    
    axes[0, 0].scatter(range(len(point_mean_before)), point_mean_before, s=1, alpha=0.5, color='b', label='Before')
    axes[0, 0].scatter(range(len(point_mean_after)), point_mean_after, s=1, alpha=0.5, color='r', label='After')
    axes[0, 0].set_xlabel('Point Index')
    axes[0, 0].set_ylabel('Mean Displacement Across Frames')
    axes[0, 0].set_title('Per-Point Average Displacement')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    all_before = np.concatenate(displacements_before)
    axes[0, 1].hist(all_before, bins=100, color='b', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Displacement Magnitude')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Displacement Distribution (Before)')
    axes[0, 1].axvline(x=motion_threshold, color='r', linestyle='--', label=f'Threshold={motion_threshold}')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    all_after = np.concatenate(displacements_after)
    axes[1, 0].hist(all_after, bins=100, color='r', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Displacement Magnitude')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Displacement Distribution (After)')
    axes[1, 0].axvline(x=motion_threshold, color='r', linestyle='--', label=f'Threshold={motion_threshold}')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    point_std_before = np.std(np.stack(displacements_before, axis=0), axis=0)
    point_std_after = np.std(np.stack(displacements_after, axis=0), axis=0)
    
    axes[1, 1].scatter(range(len(point_std_before)), point_std_before, s=1, alpha=0.5, color='b', label='Before')
    axes[1, 1].scatter(range(len(point_std_after)), point_std_after, s=1, alpha=0.5, color='r', label='After')
    axes[1, 1].set_xlabel('Point Index')
    axes[1, 1].set_ylabel('Std Displacement Across Frames')
    axes[1, 1].set_title('Per-Point Displacement Std')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    smooth_plot_path = os.path.join(visualization_dir, f'smooth_comparison_{method}.png')
    plt.savefig(smooth_plot_path, dpi=150)
    plt.close()
    print(f"Smoothing comparison plot saved to: {smooth_plot_path}")
    
    print(f"\n=== Smoothing Statistics ===")
    print(f"Average displacement before: {np.mean(all_before):.6f} ± {np.std(all_before):.6f}")
    print(f"Average displacement after: {np.mean(all_after):.6f} ± {np.std(all_after):.6f}")
    print(f"Jitter reduction rate: {(1 - np.std(all_after) / np.std(all_before)) * 100:.2f}%")
    print(f"Per-point average displacement - before: {np.mean(point_mean_before):.6f} ± {np.std(point_mean_before):.6f}")
    print(f"Per-point average displacement - after: {np.mean(point_mean_after):.6f} ± {np.std(point_mean_after):.6f}")
