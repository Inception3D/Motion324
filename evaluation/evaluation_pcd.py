import argparse
import numpy as np
import trimesh
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
import os

# Try to import bpy for GLB/FBX loading
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    print("Warning: bpy not available. GLB/FBX loading will not work.")


def load_pred_mesh_from_file(pred_file_path, frame_idx, pred_mesh_cache=None):
    """
    Load predicted mesh from GLB/FBX file using Blender.
    pred_mesh_cache should be a dict containing 'obj', 'depsgraph', 'scene', 'file_ext', 'frames'
    """
    if not BLENDER_AVAILABLE:
        raise RuntimeError("bpy is not available. Cannot load GLB/FBX files.")
    
    if pred_mesh_cache is None:
        raise ValueError("pred_mesh_cache must be provided for file loading")
    
    obj = pred_mesh_cache['obj']
    depsgraph = pred_mesh_cache['depsgraph']
    scene = pred_mesh_cache['scene']
    file_ext = pred_mesh_cache['file_ext']
    frames = pred_mesh_cache['frames']
    
    # Get the actual frame number (handle FBX frame offset)
    if frame_idx < len(frames):
        actual_frame = frames[frame_idx]
    else:
        # If frame_idx exceeds available frames, use last frame
        actual_frame = frames[-1]
    
    # Set scene to the target frame
    scene.frame_set(actual_frame)
    obj_eval = obj.evaluated_get(depsgraph)
    me = bpy.data.meshes.new_from_object(
        obj_eval,
        preserve_all_data_layers=True,
        depsgraph=depsgraph,
    )
    
    # Extract vertices
    vertices = np.zeros((len(me.vertices), 3), dtype=np.float32)
    for i, v in enumerate(me.vertices):
        vertices[i] = v.co
    
    # Extract faces
    faces = []
    for poly in me.polygons:
        faces.append(list(poly.vertices))
    faces = np.array(faces, dtype=np.int32)
    
    # Clean up temporary mesh
    bpy.data.meshes.remove(me)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def load_pred_mesh_from_dir(pred_path, frame_idx):
    """Load predicted mesh from faces.npy and frame_{frame_idx:04d}.npy"""
    faces = np.load(Path(pred_path) / "faces.npy")
    vertices = np.load(Path(pred_path) / f"frame_{frame_idx:04d}.npy")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def load_pred_mesh(pred_path, frame_idx, pred_mesh_cache=None):
    """
    Load predicted mesh. Supports both:
    1. Directory with faces.npy and frame_*.npy files
    2. GLB/FBX file (requires Blender)
    """
    pred_path_obj = Path(pred_path)
    
    # Check if it's a file (GLB/FBX) or directory (npy files)
    if pred_path_obj.is_file():
        # It's a file, use Blender to load
        if pred_mesh_cache is None:
            raise ValueError("pred_mesh_cache must be provided when loading from GLB/FBX file")
        return load_pred_mesh_from_file(pred_path, frame_idx, pred_mesh_cache)
    else:
        # It's a directory, load from npy files
        return load_pred_mesh_from_dir(pred_path, frame_idx)


def initialize_pred_mesh_cache(pred_path):
    """
    Initialize Blender scene and load predicted mesh file.
    Returns a cache dict containing necessary Blender objects and frame information.
    Returns None if pred_path is not a file or Blender is not available.
    """
    pred_path_obj = Path(pred_path)
    
    # If it's not a file, return None (will use directory loading)
    if not pred_path_obj.is_file():
        return None
    
    if not BLENDER_AVAILABLE:
        raise RuntimeError("bpy is not available. Cannot load GLB/FBX files.")
    
    file_ext = pred_path_obj.suffix.lower()
    if file_ext not in ['.glb', '.gltf', '.fbx']:
        return None  # Not a supported file format, will use directory loading
    
    print(f"Loading predicted mesh from {pred_path_obj}...")
    
    # Clear scene and import file
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    if file_ext == '.glb' or file_ext == '.gltf':
        bpy.ops.import_scene.gltf(filepath=str(pred_path_obj))
    elif file_ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=str(pred_path_obj))
    
    # Get the mesh object
    obj = None
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            obj = o
            break
    
    if obj is None:
        raise ValueError("No mesh object found in the imported file")
    
    # Gather keyframes
    def gather_keyframes(o):
        frames = set()
        ad = o.animation_data
        if ad and ad.action:
            for fc in ad.action.fcurves:
                for kp in fc.keyframe_points:
                    frames.add(int(round(kp.co.x)))
        sk = getattr(o.data, "shape_keys", None)
        if sk and sk.animation_data and sk.animation_data.action:
            for fc in sk.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    frames.add(int(round(kp.co.x)))
        return sorted(frames)
    
    frames = gather_keyframes(obj)
    
    # If no keyframes, use scene frame range
    if not frames:
        frames = list(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1))
    
    print(f"Found {len(frames)} frames in predicted mesh file: {frames[:10]}..." if len(frames) > 10 else f"Found {len(frames)} frames in predicted mesh file: {frames}")
    
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    
    return {
        'obj': obj,
        'depsgraph': depsgraph,
        'scene': scene,
        'file_ext': file_ext,
        'frames': frames
    }

def normalize_mesh(mesh):
    """
    Normalize mesh similar to bpyrenderer's normalize_scene method.
    Uses CUBE range type normalization.
    """
    vertices = mesh.vertices
    
    # Calculate bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    # Calculate offset to center (similar to normalize_scene)
    offset = -(bbox_min + bbox_max) / 2
    center = -offset  # Store original center for later use
    
    # Calculate scale (CUBE range type: normalize_range / max(bbox_max - bbox_min))
    normalize_range = 2.0
    scale = normalize_range / np.max(bbox_max - bbox_min)
    
    # Apply normalization: first translate to center, then scale
    vertices_centered = vertices + offset
    vertices_normalized = vertices_centered * scale
    
    return vertices_normalized, center, 1.0 / scale

def apply_normalization(vertices, center, scale):
    """Apply pre-computed normalization parameters"""
    return (vertices - center) / scale

def apply_icp_alignment(vertices_norm, R, t, s):
    """Apply ICP alignment transformation: s * (vertices_norm @ R.T) + t"""
    return s * (vertices_norm @ R.T) + t


def icp_alignment(source_points, target_points, max_iterations=1000, tolerance=1e-7, output_dir=None, optimize_scale=False, visualize=False):
    """
    Use standard Point-to-Point ICP algorithm to align source to target.
    Returns: rotation matrix R, translation vector t, scale factor s
    such that: aligned = s * (source @ R.T) + t
    
    Args:
        source_points: source point cloud
        target_points: target point cloud
        max_iterations: maximum number of iterations
        tolerance: convergence threshold
        output_dir: output directory (for saving visualizations)
        optimize_scale: whether to optimize scale factor (default False, scale limited to 0.95-1.05 range)
        visualize: whether to generate visualizations
    """
    # Initialize
    source = source_points.copy()
    target = target_points.copy()
    
    # Calculate initial scale (based on bbox max range)
    source_bbox_min = np.min(source, axis=0)
    source_bbox_max = np.max(source, axis=0)
    source_max_range = np.max((source_bbox_max - source_bbox_min)[:2])  # Only consider x and y axes
    
    target_bbox_min = np.min(target, axis=0)
    target_bbox_max = np.max(target, axis=0)
    target_max_range = np.max((target_bbox_max - target_bbox_min)[:2])  # Only consider x and y axes
    
    print(f"Source max range: {source_max_range}, Target max range: {target_max_range}")
    # Calculate scale factor based on bbox max range
    if source_max_range > 1e-10:
        scale = target_max_range / source_max_range
        # Limit initial scale to 0.95-1.05 range
        scale = np.clip(scale, 0.95, 1.05)
    else:
        print(f"Warning: source_max_range is too small: {source_max_range}")
        scale = 1.0
    
    # ICP iteration
    R = np.eye(3)
    t = np.zeros(3)
    prev_error = float('inf')
    
    # List to save intermediate results
    iteration_data = []
    
    # Create output directory (only when visualizing)
    if visualize and output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = output_dir / "icp_iterations"
        temp_dir.mkdir(exist_ok=True)
    
    for iteration in range(max_iterations):
        # Apply current transformation: scale * (source @ R.T) + t
        source_transformed = scale * (source @ R.T) + t
        
        # Find nearest neighbors
        tree = cKDTree(target)
        distances, indices = tree.query(source_transformed)
        
        # Get corresponding target points
        matched_target = target[indices]
        
        # Calculate point-to-point distance
        error = np.mean(distances)
        
        if visualize:
            print(f"ICP Iteration {iteration+1}: Mean Distance = {error:.6f}, Scale = {scale:.6f}")
        
        # Save iteration data (only when visualizing)
        if visualize:
            iteration_data.append({
                'iteration': iteration,
                'error': error,
                'R': R.copy(),
                't': t.copy(),
                'scale': scale,
                'source_transformed': source_transformed.copy()
            })
        
        # Visualize current iteration state (only when visualizing)
        if visualize and output_dir is not None and (iteration % 5 == 0 or iteration == 0 or iteration == max_iterations - 1):
            fig = plt.figure(figsize=(15, 5))
            
            # Downsample for visualization
            sample_rate = max(1, len(source_transformed) // 5000)
            source_sample = source_transformed[::sample_rate]
            target_sample = target[::sample_rate]
            
            # 3D view - front
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(source_sample[:, 0], source_sample[:, 1], source_sample[:, 2], 
                       c='red', s=1, alpha=0.5, label='Source')
            ax1.scatter(target_sample[:, 0], target_sample[:, 1], target_sample[:, 2], 
                       c='blue', s=1, alpha=0.5, label='Target')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title(f'3D View - Iter {iteration+1}')
            ax1.legend()
            ax1.view_init(elev=20, azim=45)
            
            # Set same axis range
            all_points = np.vstack([source_transformed, target])
            max_range = np.array([all_points[:, 0].max()-all_points[:, 0].min(),
                                 all_points[:, 1].max()-all_points[:, 1].min(),
                                 all_points[:, 2].max()-all_points[:, 2].min()]).max() / 2.0
            mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
            ax1.set_xlim(mid_x - max_range, mid_x + max_range)
            ax1.set_ylim(mid_y - max_range, mid_y + max_range)
            ax1.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # 3D view - side
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(source_sample[:, 0], source_sample[:, 1], source_sample[:, 2], 
                       c='red', s=1, alpha=0.5, label='Source')
            ax2.scatter(target_sample[:, 0], target_sample[:, 1], target_sample[:, 2], 
                       c='blue', s=1, alpha=0.5, label='Target')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_title(f'Side View - Iter {iteration+1}')
            ax2.legend()
            ax2.view_init(elev=0, azim=0)
            ax2.set_xlim(mid_x - max_range, mid_x + max_range)
            ax2.set_ylim(mid_y - max_range, mid_y + max_range)
            ax2.set_zlim(mid_z - max_range, mid_z + max_range)
            
            # Error curve
            ax3 = fig.add_subplot(133)
            errors = [d['error'] for d in iteration_data]
            ax3.plot(range(len(errors)), errors, 'b-', linewidth=2)
            ax3.scatter(len(errors)-1, errors[-1], c='red', s=50, zorder=5)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Mean Distance Error')
            ax3.set_title(f'Convergence Curve\nCurrent Error: {error:.6f}')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save image
            img_path = temp_dir / f"iteration_{iteration:04d}.png"
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  Saved visualization to {img_path}")
        
        # Check convergence
        if abs(prev_error - error) < tolerance:
            if visualize:
                print(f"ICP converged at iteration {iteration+1}")
            break
        prev_error = error
        
        # Calculate centroids
        source_centroid = np.mean(source_transformed, axis=0)
        target_centroid = np.mean(matched_target, axis=0)
        
        # Center
        source_centered_iter = source_transformed - source_centroid
        target_centered_iter = matched_target - target_centroid
        
        # Calculate covariance matrix
        H = source_centered_iter.T @ target_centered_iter
        
        # SVD decomposition
        U, S, Vt = np.linalg.svd(H)
        
        # Calculate rotation matrix
        R_delta = Vt.T @ U.T
        
        # Ensure right-handed coordinate system (determinant = 1)
        if np.linalg.det(R_delta) < 0:
            Vt[-1, :] *= -1
            R_delta = Vt.T @ U.T
        
        # Calculate translation increment
        t_delta = target_centroid - source_centroid @ R_delta.T
        
        # Update rotation and translation
        R = R @ R_delta
        t = t @ R_delta.T + t_delta
        
        # Orthogonalize rotation matrix (ensure numerical stability)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        
        # Optimize scale factor (if enabled)
        if optimize_scale:
            # After applying current R and t, optimize scale
            source_rotated_translated = source @ R.T + t
            # Optimal scale = sum(target * source) / sum(source * source)
            numerator = np.sum(target[indices] * source_rotated_translated)
            denominator = np.sum(source_rotated_translated * source_rotated_translated)
            if denominator > 1e-10:
                scale_new = numerator / denominator
                # Limit scale to 0.95-1.05 range
                scale_new = np.clip(scale_new, 0.95, 1.05)
                # Smooth scale update to avoid oscillation
                scale = 0.8 * scale + 0.2 * scale_new
                # Ensure scale is in range again
                scale = np.clip(scale, 0.95, 1.05)
    
    # Print final error
    print(f"Final Mean Distance: {error:.6f}")
    
    # Create final comparison plot (only when visualizing)
    if visualize and output_dir is not None:
        print("\nCreating final comparison visualization...")
        
        # Select key frames for display
        key_iterations = [0]
        step = max(1, len(iteration_data) // 8)
        for i in range(step, len(iteration_data), step):
            key_iterations.append(i)
        if len(iteration_data) - 1 not in key_iterations:
            key_iterations.append(len(iteration_data) - 1)
        
        n_images = len(key_iterations)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(5*n_cols, 5*n_rows))
        
        for idx, iter_idx in enumerate(key_iterations):
            ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
            
            data = iteration_data[iter_idx]
            source_transformed = data['source_transformed']
            
            sample_rate = max(1, len(source_transformed) // 3000)
            source_sample = source_transformed[::sample_rate]
            target_sample = target[::sample_rate]
            
            ax.scatter(source_sample[:, 0], source_sample[:, 1], source_sample[:, 2], 
                      c='red', s=1, alpha=0.5, label='Source')
            ax.scatter(target_sample[:, 0], target_sample[:, 1], target_sample[:, 2], 
                      c='blue', s=1, alpha=0.5, label='Target')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"Iter {data['iteration']+1}\nError: {data['error']:.6f}, Scale: {data['scale']:.6f}")
            ax.legend()
            ax.view_init(elev=20, azim=45)
            
            # Set same axis range
            all_points = np.vstack([source_transformed, target])
            max_range = np.array([all_points[:, 0].max()-all_points[:, 0].min(),
                                 all_points[:, 1].max()-all_points[:, 1].min(),
                                 all_points[:, 2].max()-all_points[:, 2].min()]).max() / 2.0
            mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
            mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
            mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        final_grid_path = output_dir / "icp_iterations_grid.png"
        plt.savefig(final_grid_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Final grid saved to {final_grid_path}")
        
        # Save error curve and scale curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        errors = [d['error'] for d in iteration_data]
        ax1.plot(range(len(errors)), errors, 'b-', linewidth=2, label='Error')
        ax1.scatter(range(len(errors)), errors, c='blue', s=20, alpha=0.5)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Mean Distance Error', fontsize=12)
        ax1.set_title('ICP Convergence Curve', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        if optimize_scale:
            scales = [d['scale'] for d in iteration_data]
            ax2.plot(range(len(scales)), scales, 'r-', linewidth=2, label='Scale')
            ax2.scatter(range(len(scales)), scales, c='red', s=20, alpha=0.5)
            ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='Scale limits')
            ax2.axhline(y=1.05, color='g', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Scale Factor', fontsize=12)
            ax2.set_title('Scale Optimization (limited to 0.95-1.05)', fontsize=14)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        error_curve_path = output_dir / "icp_error_curve.png"
        plt.savefig(error_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Error curve saved to {error_curve_path}")
    
    return R, t, scale


def compute_alignment_from_first_frame(gt_path, pred_path, pred_mesh_cache=None, visualize=False):
    """
    Compute ICP alignment parameters from the first frame.
    Returns: rotation matrix R, translation vector t, scale factor s, and normalization parameters.
    
    Args:
        gt_path: path to GT data directory (contains faces.npy and frame_*.npy)
        pred_path: path to predicted data (can be directory with frame_*.npy or GLB/FBX file)
        pred_mesh_cache: cache for predicted mesh (if loading from file)
        visualize: whether to generate visualizations
    """
    gt_path = Path(gt_path)
    pred_path = Path(pred_path)
    
    # Load GT first frame
    print(f"Loading GT frame 0 from {gt_path}...")
    gt_faces_path = gt_path / "faces.npy"
    if gt_faces_path.exists():
        gt_faces = np.load(gt_faces_path)
    else:
        gt_faces = None
    
    gt_frame_0 = np.load(gt_path / "frame_0000.npy")
    if gt_faces is not None:
        gt_mesh_0 = trimesh.Trimesh(vertices=gt_frame_0, faces=gt_faces, process=False)
    else:
        gt_mesh_0 = trimesh.Trimesh(vertices=gt_frame_0, process=False)
    
    # Normalize GT
    gt_vertices_0_norm, gt_center, gt_scale = normalize_mesh(gt_mesh_0)
    print(f"GT Normalization - Center: {gt_center}, Scale: {gt_scale}")
    
    # Sample 10000 points from GT mesh surface for ICP
    print("Sampling 10000 points from GT mesh surface...")
    gt_sampled_points, _ = trimesh.sample.sample_surface(gt_mesh_0, 10000)
    # Apply same normalization to sampled points
    gt_sampled_points_norm = (gt_sampled_points - gt_center) / gt_scale
    
    # Load predicted first frame
    print(f"Loading predicted frame 0 from {pred_path}...")
    pred_mesh_0 = load_pred_mesh(pred_path, 0, pred_mesh_cache)
    
    # Normalize prediction
    pred_vertices_0_norm, pred_center, pred_scale = normalize_mesh(pred_mesh_0)
    print(f"Pred Normalization - Center: {pred_center}, Scale: {pred_scale}")
    
    # Use standard Point-to-Point ICP alignment (using GT sampled points)
    print("\nRunning Point-to-Point ICP alignment...")
    output_dir = gt_path / "icp_visualization" if visualize else None
    R, t, s = icp_alignment(pred_vertices_0_norm, gt_sampled_points_norm,
                           output_dir=output_dir, visualize=visualize)
    print(f"\nICP Alignment Results:")
    print(f"Rotation Matrix R:\n{R}")
    print(f"Translation t: {t}")
    print(f"Scale s: {s}")
    
    # Save alignment parameters
    # Save to GT path (as npz format for easy loading)
    np.savez(gt_path / 'icp_alignment_params.npz',
             R=R, t=t, s=s)
    print(f"Alignment parameters saved to: {gt_path / 'icp_alignment_params.npz'}")
    
    return R, t, s


def sample_points_from_mesh(mesh, num_points=50000):
    """Sample points uniformly from mesh surface"""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def compute_chamfer_distance(points1, points2):
    """Compute bidirectional Chamfer distance"""
    # Build KD-trees
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Compute distances from points1 to points2
    dist1, _ = tree1.query(points2, k=1)
    # Compute distances from points2 to points1
    dist2, _ = tree2.query(points1, k=1)
    
    # Chamfer distance is the mean of both directions
    chamfer_dist = (np.mean(dist1) + np.mean(dist2))
    return chamfer_dist


def compute_fscore(points1, points2, threshold=0.02):
    """Compute F-score at given threshold"""
    # Build KD-trees
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # Compute distances
    dist1, _ = tree1.query(points2, k=1)
    dist2, _ = tree2.query(points1, k=1)
    
    # Compute precision and recall
    precision = np.mean(dist1 < threshold)
    recall = np.mean(dist2 < threshold)
    
    # Compute F-score
    if precision + recall == 0:
        return 0.0
    fscore = 2 * precision * recall / (precision + recall)
    return fscore


def compute_iou_voxel(mesh1, mesh2, resolution=128):
    """Compute IoU on voxel grids"""
    # Voxelize both meshes
    voxel1 = mesh1.voxelized(pitch=1.0/resolution)
    voxel2 = mesh2.voxelized(pitch=1.0/resolution)
    
    # Get filled voxel grids
    grid1 = voxel1.matrix
    grid2 = voxel2.matrix
    
    # Ensure same shape
    max_shape = np.maximum(grid1.shape, grid2.shape)
    padded_grid1 = np.zeros(max_shape, dtype=bool)
    padded_grid2 = np.zeros(max_shape, dtype=bool)
    
    padded_grid1[:grid1.shape[0], :grid1.shape[1], :grid1.shape[2]] = grid1
    padded_grid2[:grid2.shape[0], :grid2.shape[1], :grid2.shape[2]] = grid2
    
    # Compute IoU
    intersection = np.logical_and(padded_grid1, padded_grid2).sum()
    union = np.logical_or(padded_grid1, padded_grid2).sum()
    
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou


def visualize_mesh_comparison(gt_mesh, pred_mesh, frame_idx, viz_dir):
    """Visualize ground truth and predicted meshes"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot GT mesh
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_trisurf(gt_mesh.vertices[:, 0], gt_mesh.vertices[:, 1], gt_mesh.vertices[:, 2],
                     triangles=gt_mesh.faces, alpha=0.7, color='blue', edgecolor='none')
    ax1.set_title(f'GT Mesh (Frame {frame_idx})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    
    # Plot predicted mesh
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_trisurf(pred_mesh.vertices[:, 0], pred_mesh.vertices[:, 1], pred_mesh.vertices[:, 2],
                     triangles=pred_mesh.faces, alpha=0.7, color='red', edgecolor='none')
    ax2.set_title(f'Predicted Mesh (Frame {frame_idx})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    
    # Plot both together (wireframe for better visibility)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_trisurf(gt_mesh.vertices[:, 0], gt_mesh.vertices[:, 1], gt_mesh.vertices[:, 2],
                     triangles=gt_mesh.faces, alpha=0.3, color='blue', edgecolor='none')
    ax3.plot_trisurf(pred_mesh.vertices[:, 0], pred_mesh.vertices[:, 1], pred_mesh.vertices[:, 2],
                     triangles=pred_mesh.faces, alpha=0.3, color='red', edgecolor='none')
    ax3.set_title(f'Mesh Overlay (Frame {frame_idx})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    # Set viewing angle to 45 degrees
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    save_path = os.path.join(viz_dir, f'mesh_comparison_frame_{frame_idx:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mesh comparison to {save_path}")


def visualize_pointcloud_comparison(gt_points, pred_points, frame_idx, viz_dir):
    """Visualize ground truth and predicted point clouds"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot GT points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                c='blue', s=1, alpha=0.5, label='GT Points')
    ax1.set_title(f'GT Point Cloud (Frame {frame_idx})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_zlim(-1, 1)
    ax1.legend()
    
    # Plot predicted points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                c='red', s=1, alpha=0.5, label='Pred Points')
    ax2.set_title(f'Predicted Point Cloud (Frame {frame_idx})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.set_zlim(-1, 1)
    ax2.legend()
    
    # Plot both together
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                c='blue', s=1, alpha=0.3, label='GT')
    ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                c='red', s=1, alpha=0.3, label='Pred')
    ax3.set_title(f'Point Cloud Overlay (Frame {frame_idx})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    ax3.legend()

    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    save_path = os.path.join(viz_dir, f'pointcloud_comparison_frame_{frame_idx:04d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved point cloud comparison to {save_path}")


def evaluate_sequence(gt_path, pred_path, num_samples=2048, viz=False, icp_viz=False):
    """Evaluate entire sequence"""
    pred_path = Path(pred_path)
    gt_path = Path(gt_path)
    
    # Create visualization directory
    viz_dir = Path(pred_path).parent.parent / 'viz'
    viz_dir.mkdir(exist_ok=True)
    print(f"Visualization directory: {viz_dir}")
    
    # Initialize predicted mesh cache if loading from file
    pred_mesh_cache = initialize_pred_mesh_cache(pred_path)
    is_pred_file = pred_mesh_cache is not None
    
    # Load GT frames from point cloud directory
    print(f"Loading GT frames from {gt_path}...")
    gt_frame_files = sorted(gt_path.glob("frame_*.npy"))
    gt_frames = [np.load(f) for f in gt_frame_files]
    print(f"Loaded {len(gt_frames)} frames from GT point clouds")
    
    # Load GT faces
    gt_faces_path = gt_path / "faces.npy"
    if gt_faces_path.exists():
        gt_faces = np.load(gt_faces_path)
        print(f"Loaded GT faces from {gt_faces_path}")
    else:
        print(f"Warning: No faces.npy found in {gt_path}")
        gt_faces = None
    
    # Get number of frames from predicted mesh
    if is_pred_file:
        # Get frame count from Blender cache
        num_frames = len(pred_mesh_cache['frames'])
        print(f"Found {num_frames} frames in predicted mesh file")
        # Extract predicted faces from first frame (they should be consistent)
        pred_mesh_first = load_pred_mesh_from_file(pred_path, 0, pred_mesh_cache)
        pred_faces = pred_mesh_first.faces
        print(f"Loaded predicted faces from first frame: {len(pred_faces)} faces")
    else:
        # Get frame count from directory
        frame_files = sorted(pred_path.glob("frame_*.npy"))
        num_frames = len(frame_files)
        print(f"Found {num_frames} frames in predicted data directory")
        
        # Load predicted faces
        pred_faces_path = pred_path / "faces.npy"
        if pred_faces_path.exists():
            pred_faces = np.load(pred_faces_path)
            print(f"Loaded predicted faces from {pred_faces_path}")
        else:
            print(f"Warning: No faces.npy found in {pred_path}")
            pred_faces = None
    
    # Ensure we have enough GT frames
    if len(gt_frames) < num_frames:
        print(f"Warning: GT data has {len(gt_frames)} frames but prediction has {num_frames} frames")
        print(f"Will repeat last frame to match predicted length")
        while len(gt_frames) < num_frames:
            gt_frames.append(gt_frames[-1].copy())
    
    # Load first predicted frame for normalization
    pred_mesh_0 = load_pred_mesh(pred_path, 0, pred_mesh_cache)
    _, pred_center, pred_scale = normalize_mesh(pred_mesh_0)

    print(f"Pred Normalization params - Center: {pred_center}, Scale: {pred_scale}")
    
    # Normalize predicted frame 0
    pred_vertices_0_norm = apply_normalization(pred_mesh_0.vertices, pred_center, pred_scale)

    # Normalize GT frame 0
    if gt_faces is not None:
        gt_mesh_0 = trimesh.Trimesh(vertices=gt_frames[0], faces=gt_faces, process=False)
    else:
        gt_mesh_0 = trimesh.Trimesh(vertices=gt_frames[0], process=False)
    _, gt_center, gt_scale = normalize_mesh(gt_mesh_0)

    gt_vertices_0_norm = apply_normalization(gt_frames[0], gt_center, gt_scale)
    print(f"GT Normalization params - Center: {gt_center}, Scale: {gt_scale}")
    print(f"GT Normalization params - Min: {min(gt_vertices_0_norm[:, 0])}, Max: {max(gt_vertices_0_norm[:, 0])}")
    print(f"GT Normalization params - Min: {min(gt_vertices_0_norm[:, 1])}, Max: {max(gt_vertices_0_norm[:, 1])}")
    print(f"GT Normalization params - Min: {min(gt_vertices_0_norm[:, 2])}, Max: {max(gt_vertices_0_norm[:, 2])}")
    # Load ICP alignment parameters
    icp_params_path = viz_dir / 'icp_alignment_params.npz'
    if icp_params_path.exists():
        icp_params = np.load(icp_params_path)
        R = icp_params['R']
        t = icp_params['t']
        s = icp_params['s']
        print(f"Loaded ICP alignment parameters from {icp_params_path}")
        print(f"R shape: {R.shape}, t shape: {t.shape}, s: {s}")
    else:
        print(f"No icp_alignment_params.npz found in {gt_path}")
        print("Computing ICP alignment from first frame...")
        # Make visualize for compute_alignment_from_first_frame configurable via arg
        R, t, s = compute_alignment_from_first_frame(gt_path, pred_path, pred_mesh_cache, visualize=icp_viz)
        print(f"ICP alignment computed - R shape: {R.shape}, t shape: {t.shape}, s: {s}")
    
    # Store metrics for all frames
    chamfer_distances = []
    fscores = []
    ious = []
    if viz:
        print(f"Visualization directory: {viz_dir}")
        os.makedirs(viz_dir, exist_ok=True)

    for frame_idx in range(num_frames):
        print(f"\nProcessing frame {frame_idx}/{num_frames-1}...")
        
        # Load predicted mesh and apply prediction normalization
        pred_mesh = load_pred_mesh(pred_path, frame_idx, pred_mesh_cache)
        pred_vertices_norm = apply_normalization(pred_mesh.vertices, pred_center, pred_scale)
        # Use predicted faces from cache or from mesh
        pred_faces_to_use = pred_faces if pred_faces is not None else pred_mesh.faces
        pred_mesh_norm = trimesh.Trimesh(vertices=pred_vertices_norm, faces=pred_faces_to_use, process=False)
        
        # Get GT frame, apply GT normalization, then apply ICP alignment
        gt_vertices = gt_frames[frame_idx]
        gt_vertices_norm = apply_normalization(gt_vertices, gt_center, gt_scale)
        gt_vertices_aligned = apply_icp_alignment(gt_vertices_norm, R, t, s)
        if gt_faces is not None:
            gt_mesh_aligned = trimesh.Trimesh(vertices=gt_vertices_aligned, faces=gt_faces, process=False)
        else:
            gt_mesh_aligned = trimesh.Trimesh(vertices=gt_vertices_aligned, process=False)
        
        # Visualize mesh comparison for every frame
        #print(f"Visualizing mesh comparison for frame {frame_idx}...")
        if viz:
            visualize_mesh_comparison(gt_mesh_aligned, pred_mesh_norm, frame_idx, viz_dir)
        
        # Sample points
        gt_points = sample_points_from_mesh(gt_mesh_aligned, num_samples)
        pred_points = sample_points_from_mesh(pred_mesh_norm, num_samples)
        
        # Visualize point cloud comparison for every frame
        #print(f"Visualizing point cloud comparison for frame {frame_idx}...")
        
        #visualize_pointcloud_comparison(gt_points, pred_points, frame_idx, viz_dir)
        # Compute metrics
        chamfer = compute_chamfer_distance(gt_points, pred_points)
        fscore = compute_fscore(gt_points, pred_points, threshold=0.02)
        # iou = compute_iou_voxel(gt_mesh_aligned, pred_mesh_norm, resolution=128)
        
        chamfer_distances.append(chamfer)
        fscores.append(fscore)
        # ious.append(iou)
        
        print(f"Frame {frame_idx} - Chamfer: {chamfer:.6f}, F-score: {fscore:.4f}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("Summary Statistics:")
    print("="*50)
    print(f"Chamfer Distance - Mean: {np.mean(chamfer_distances):.6f}, Std: {np.std(chamfer_distances):.6f}")
    print(f"F-score - Mean: {np.mean(fscores):.4f}, Std: {np.std(fscores):.4f}")
    # print(f"IoU - Mean: {np.mean(ious):.4f}, Std: {np.std(ious):.4f}")
    
    # Save results to txt file in gt_path
    gt_path_obj = Path(gt_path)
    os.makedirs(gt_path_obj, exist_ok=True)
    results_txt_path = gt_path_obj / 'evaluation_results.txt'
    with open(results_txt_path, 'w') as f:
        f.write(f"{gt_path_obj.name}\n")
        f.write(f"cd_mean_{np.mean(chamfer_distances):.6f}\n")
        f.write(f"fs_mean_{np.mean(fscores):.6f}\n")
    print(f"\nResults saved to {results_txt_path}")
    
    return {
        'chamfer_distances': chamfer_distances,
        'fscores': fscores,
        # 'ious': ious
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate mesh reconstruction quality')
    parser.add_argument('--gt_path', type=str, required=True,
                    help='Path to ground-truth point cloud directory (contains faces.npy and frame_*.npy)')
    parser.add_argument('--pred_path', type=str, required=True,
                        help='Path to predicted data (contains faces.npy and frame_*.npy)')
    parser.add_argument('--num_samples', type=int, default=50000,
                        help='Number of points to sample from each mesh')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize point clouds and meshes')
    parser.add_argument('--icp_viz', action='store_true',
                        help='Visualize ICP alignment procedure for first frame')
    
    args = parser.parse_args()
    
    results = evaluate_sequence(args.gt_path, args.pred_path, args.num_samples, args.viz, icp_viz=args.icp_viz)
    
    print(f"\nResults saved to {args.pred_path}")


if __name__ == '__main__':
    main()
