# Animate existing mesh with existing video.

import os
import sys
# Add project root directory to Python path to enable imports from root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.append(project_root)

import importlib
import torch
import numpy as np
import imageio
from natsort import natsorted
from PIL import Image

from setup import init_config, init_distributed
from utils.render import clear_scene, import_glb, drive_mesh_with_trajs_frames_gt
from utils.mesh_processing import convert_fbx_to_glb_with_blender, sample_pointcloud_with_albedo
from utils.inference_utils import seed_everything, load_checkpoint, smooth_trajectories, load_u2net_model, segment_foreground_with_u2net
from utils.visualization import visualize_input_data

os.environ['SPCONV_ALGO'] = 'native'
config = init_config()

def load_video_from_path(video_path):
    """
    Load video from file or image directory.
    
    Args:
        video_path: Path to video file (.mp4, .avi, .mov) or directory containing images
    
    Returns:
        video_np: numpy array of shape (T, H, W, C), range [0, 255]
    """
    if video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        print(f"Loading video file: {video_path}")
        reader = imageio.get_reader(video_path)
        frames = [frame for frame in reader]
        reader.close()
        video_np = np.stack(frames, axis=0)
        print(f"✓ Loaded {len(frames)} frames from video, shape: {video_np.shape}")
        
    elif os.path.isdir(video_path):
        print(f"Loading images from directory: {video_path}")
        image_paths = natsorted([
            os.path.join(video_path, f)
            for f in os.listdir(video_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        frames = [np.array(Image.open(img_path).convert("RGB")) for img_path in image_paths]
        video_np = np.stack(frames, axis=0)
        print(f"✓ Loaded {len(frames)} images from directory, shape: {video_np.shape}")
    else:
        raise ValueError(f"video_path must be a video file or image directory: {video_path}")
    
    return video_np


def prepare_mesh_data(config, glb_path, device):
    """
    Load and prepare mesh data for inference.
    
    Args:
        config: Configuration object
        glb_path: Path to GLB file
        device: Device to load data on
    
    Returns:
        input_data: Dictionary containing mesh data
        mesh: trimesh object
        faces: Face indices
    """
    import trimesh
    from scipy.spatial import cKDTree
    
    print(f"Loading GLB file: {glb_path}")
    mesh = trimesh.load(glb_path, force='mesh')
    mesh.fix_normals()
    
    if isinstance(mesh, trimesh.Scene):
        print(f"Detected scene with {len(mesh.geometry)} geometries")
        for name in mesh.geometry.keys():
            print(f"  - {name}")
        mesh = trimesh.util.concatenate(mesh.geometry.values())
    else:
        print(f"Detected single geometry, vertices shape: {mesh.vertices.shape}")
    
    # Normalize mesh
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int64)
    vertex_normals_np = mesh.vertex_normals.astype(np.float32)
    
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = vertices - center
    v_max = np.abs(vertices).max()
    vertices = vertices / (2 * (v_max + 1e-8))
    
    R_example = np.eye(3, dtype=np.float32)
    vertices = vertices @ R_example.T
    vertex_normals_np = vertex_normals_np @ R_example.T
    
    mesh.vertices = mesh.vertices - center
    mesh.vertices = mesh.vertices / (2 * (v_max + 1e-8))
    mesh.vertices = mesh.vertices @ R_example.T
    
    # Sample surface points
    num_shape_samples = getattr(config.training, "num_shape_samples", 16384)
    print(f"Sampling {num_shape_samples} surface points with texture interpolation...")
    samples_xyz, samples_normals, samples_rgb = sample_pointcloud_with_albedo(mesh, num=num_shape_samples)
    print(f"✓ Sampling complete: points {samples_xyz.shape}, normals {samples_normals.shape}, colors {samples_rgb.shape}")
    
    # Assign colors to vertices
    tree = cKDTree(samples_xyz.numpy())
    _, nearest_indices = tree.query(vertices, k=1)
    vert_rgb = samples_rgb[nearest_indices].numpy()
    
    # Prepare input data
    input_data = {
        'ref_shape_pcd': samples_xyz[None].float().to(device),
        'ref_shape_normals': samples_normals[None].float().to(device),
        'ref_shape_rgbs': samples_rgb[None].float().to(device),
        'ref_pcd': torch.from_numpy(vertices)[None].float().to(device),
        'ref_normal': torch.from_numpy(vertex_normals_np)[None].float().to(device),
        'ref_rgb': torch.from_numpy(vert_rgb)[None].float().to(device),
        'faces': torch.from_numpy(faces)[None].long().to(device)
    }
    
    return input_data, mesh, faces


def run_model_inference(model, input_data, video_tensor, config, device):
    """
    Run model inference on video with chunking support.
    
    Args:
        model: Loaded model
        input_data: Dictionary containing mesh data
        video_tensor: Video frames tensor
        config: Configuration object
        device: Device to run on
    
    Returns:
        trajs: Predicted trajectories (1, T, N, 3)
    """
    chunk_size = config.training.get('frames', 12)
    total_T = video_tensor.shape[0]
    print(f"Total frames: {total_T}, chunk size: {chunk_size}")
    
    amp_dtype_mapping = {
        "fp16": torch.float16, 
        "bf16": torch.bfloat16, 
        "fp32": torch.float32, 
        'tf32': torch.float32
    }
    
    if total_T <= chunk_size:
        print(f"Total frames ({total_T}) <= chunk size ({chunk_size}), processing all frames at once")
        input_data_chunk = input_data.copy()
        input_data_chunk['rgb_video'] = video_tensor[None].float().to(device)
        
        with torch.autocast(
            enabled=config.training.get('use_amp', False),
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.get('amp_dtype', 'bf16')],
        ):
            output_chunk = model(input_data_chunk)
        
        if isinstance(output_chunk, dict) and 'pcd_moved' in output_chunk:
            trajs = output_chunk['pcd_moved'].float()
            print(f"Trajectories shape: {trajs.shape}")
        else:
            trajs = None
            print("Warning: No pcd_moved in output")
    else:
        # Process with overlapping chunks
        slide_size = chunk_size - 1
        chunk_start_indices = list(range(0, total_T - chunk_size + 1, slide_size))
        if chunk_start_indices and (chunk_start_indices[-1] + chunk_size < total_T):
            chunk_start_indices.append(total_T - chunk_size)
        print(f"Chunk start indices: {chunk_start_indices}")
        
        out_trajs_lst = []
        
        for i, start_idx in enumerate(chunk_start_indices):
            end_idx = start_idx + chunk_size
            
            if i == 0:
                rgb_window = video_tensor[0:chunk_size]
            else:
                rgb_window = torch.cat([
                    video_tensor[0:1],
                    video_tensor[start_idx+1:end_idx]
                ], dim=0)
            
            input_data_chunk = input_data.copy()
            input_data_chunk['rgb_video'] = rgb_window[None].float().to(device)
            
            if rgb_window.shape[0] != chunk_size:
                print(f"Warning: Chunk rgb_video frames != chunk_size, skipping")
                continue
            
            print(f"Processing chunk {i+1}/{len(chunk_start_indices)}: frames {start_idx}-{end_idx}")
            
            with torch.autocast(
                enabled=config.training.get('use_amp', False),
                device_type="cuda",
                dtype=amp_dtype_mapping[config.training.get('amp_dtype', 'fp16')],
            ):
                output_chunk = model(input_data_chunk)
            
            if isinstance(output_chunk, dict) and 'pcd_moved' in output_chunk:
                trajs_chunk = output_chunk['pcd_moved'].float()
                out_trajs_lst.append(trajs_chunk)
            else:
                print("Warning: No pcd_moved in chunk output, skipping")
        
        # Merge trajectories
        if len(out_trajs_lst) > 0:
            if len(chunk_start_indices) >= 2:
                merged_trajs_lst = []
                for i in range(len(out_trajs_lst)):
                    if i == 0 and i != len(out_trajs_lst) - 2: # if len==2 should keep first N frames
                        chunk_trajs = out_trajs_lst[i].clone()
                        chunk_trajs[:, 0, :, :] = input_data['ref_pcd']
                        merged_trajs_lst.append(chunk_trajs)
                    elif i < len(out_trajs_lst) - 2:
                        merged_trajs_lst.append(out_trajs_lst[i][:, 1:, ...])
                    elif i == len(out_trajs_lst) - 2:
                        start_a = chunk_start_indices[-2]
                        start_b = chunk_start_indices[-1]
                        keep_len = max(start_b - start_a, 0)
                        if keep_len > 0 and len(out_trajs_lst) != 2:
                            merged_trajs_lst.append(out_trajs_lst[i][:, 1:1+keep_len, ...])
                        elif keep_len > 0 and i == 0 and len(out_trajs_lst) == 2:
                            chunk_trajs = out_trajs_lst[i].clone()
                            chunk_trajs[:, 0, :, :] = input_data['ref_pcd']
                            merged_trajs_lst.append(chunk_trajs[:, :1+keep_len, ...])
                    elif i == len(out_trajs_lst) - 1:
                        merged_trajs_lst.append(out_trajs_lst[i][:, 1:, ...])
                
                if len(merged_trajs_lst) > 0:
                    trajs = torch.cat(merged_trajs_lst, dim=1)
                    print(f"Merged trajectories shape: {trajs.shape}")
                else:
                    trajs = None
                    print("No trajectories obtained")
            else:
                trajs = out_trajs_lst[0].clone()
                trajs[:, 0, :, :] = input_data['ref_pcd']
                print(f"Trajectories shape: {trajs.shape}")
        else:
            trajs = None
            print("No trajectories obtained")
    
    return trajs


def save_segmented_videos(video_tensor, mask_tensor, output_dir, fps=12):
    """
    Save segmented videos (black background, white background) and mask video.
    
    Args:
        video_tensor: Video frames tensor (T, H, W, C), range [0, 1]
        mask_tensor: Mask tensor (T, H, W, 1), range [0, 1]
        output_dir: Output directory to save videos
        fps: Frames per second for output videos
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save segmented video with black background
    output_black_path = os.path.join(output_dir, "input_rgb_video_segmented_black.mp4")
    with imageio.get_writer(output_black_path, fps=fps) as writer:
        for i in range(video_tensor.shape[0]):
            frame_uint8 = (video_tensor[i].numpy() * 255).astype(np.uint8)
            writer.append_data(frame_uint8)
    print(f"✓ Saved segmented video (black bg): {output_black_path}")
    
    # Save segmented video with white background
    output_white_path = os.path.join(output_dir, "input_rgb_video_segmented_white.mp4")
    with imageio.get_writer(output_white_path, fps=fps) as writer:
        for i in range(video_tensor.shape[0]):
            frame = video_tensor[i].numpy()
            mask = mask_tensor[i].numpy()
            frame_white = frame * mask + (1 - mask)
            frame_uint8 = (frame_white * 255).astype(np.uint8)
            writer.append_data(frame_uint8)
    print(f"✓ Saved segmented video (white bg): {output_white_path}")
    
    # Save foreground mask video
    output_mask_path = os.path.join(output_dir, "input_foreground_mask.mp4")
    with imageio.get_writer(output_mask_path, fps=fps) as writer:
        for i in range(mask_tensor.shape[0]):
            mask = mask_tensor[i].numpy()
            mask_gray = (mask.squeeze(-1) * 255).astype(np.uint8)
            mask_rgb = np.stack([mask_gray] * 3, axis=-1)
            writer.append_data(mask_rgb)
    print(f"✓ Saved foreground mask video: {output_mask_path}")


def run_inference_on_video(config):
    """Main inference function for custom mesh and video input."""
    seed_everything(777)
    ddp_info = init_distributed(seed=777)
    torch.backends.cuda.matmul.allow_tf32 = config.training.get('use_tf32', True)
    torch.backends.cudnn.allow_tf32 = config.training.get('use_tf32', True)
    
    # Load model
    module, class_name = config.model.class_name.rsplit(".", 1)
    Motion324 = importlib.import_module(module).__dict__[class_name]
    model = Motion324(config).to(ddp_info.device)
    
    ckpt_path = config.training.get("resume_ckpt", "")
    if not ckpt_path:
        ckpt_path = os.path.join(config.training.checkpoint_dir, "latest.pt")
    
    step_info = load_checkpoint(ckpt_path, model, ddp_info.device)
    print(f"Loaded checkpoint from step {step_info['param_update_step']}")
    model.eval()
    
    # Load U2Net for segmentation if enabled
    use_segmentation = getattr(config, 'use_segmentation', True)
    u2net_model = None
    if use_segmentation:
        print("Loading U2Net model for foreground segmentation...")
        u2net_model = load_u2net_model(device=ddp_info.device)
        if u2net_model is None:
            print("Warning: U2Net model loading failed, will not perform foreground segmentation")
    else:
        print("Foreground segmentation disabled (use_segmentation=False)")
    
    # Prepare mesh
    filepath = config.data_dir
    if filepath.lower().endswith('.fbx'):
        print(f"Detected FBX file, converting to GLB...")
        output_dir = config.output_dir if hasattr(config, 'output_dir') else "./"
        os.makedirs(output_dir, exist_ok=True)
        glb_path = os.path.join(output_dir, "converted_mesh.glb")
        convert_fbx_to_glb_with_blender(filepath, glb_path)
        print(f"✓ GLB file saved to: {glb_path}")
    elif filepath.lower().endswith('.glb'):
        print(f"Detected GLB file, using directly")
        glb_path = filepath
    else:
        raise ValueError(f"Unsupported file format: {filepath}, only .fbx and .glb are supported")
    
    with torch.no_grad():
        input_data, mesh, faces = prepare_mesh_data(config, glb_path, ddp_info.device)
        
        # Load video
        if not hasattr(config, 'video_path') or config.video_path is None:
            raise ValueError("config.video_path not specified, cannot load video")
        
        video_np = load_video_from_path(config.video_path)
        
        # Apply foreground segmentation
        if u2net_model is not None:
            masked_frames, masks = segment_foreground_with_u2net(video_np, u2net_model, device=ddp_info.device)
            print("✓ Applied foreground segmentation, background as black")
        else:
            masked_frames = video_np
            masks = np.ones(video_np.shape[:3] + (1,), dtype=np.float32)
        
        video_tensor = torch.from_numpy(masked_frames.astype(np.float32)).float() / 255.0
        mask_tensor = torch.from_numpy(masks.astype(np.float32)).float()
        
        # Select frames
        T = config.training.frames
        start_frame = getattr(config, 'start_frame', 0)
        
        if start_frame + T <= video_tensor.shape[0]:
            video_tensor = video_tensor[start_frame:start_frame + T]
            mask_tensor = mask_tensor[start_frame:start_frame + T]
        else:
            print(f"Warning: Insufficient frames, using {video_tensor.shape[0] - start_frame} frames from frame {start_frame}")
            video_tensor = video_tensor[start_frame:]
            mask_tensor = mask_tensor[start_frame:]
        
        # Save input videos
        output_video_dir = config.output_dir if hasattr(config, 'output_dir') else "./"
        save_segmented_videos(video_tensor, mask_tensor, output_video_dir, fps=12)
        
        total_T = video_tensor.shape[0]
        input_data['rgb_video'] = video_tensor
        
        # Visualize input data
        visualize_input_data(input_data, save_path=os.path.join(output_video_dir, "input_data_vis.png"))
        
        # Run inference
        trajs = run_model_inference(model, input_data, video_tensor, config, ddp_info.device)
        
        # Apply smoothing
        if trajs is not None:
            # apply smoothing to trajectories to avoid jitter for static vertices
            trajs = smooth_trajectories(
                trajs, 
                method='combined',
                motion_threshold=0.002,
                window_size=3,
                sigma=1.0,
                savgol_polyorder=2,
                oneeuro_mincutoff=1.0,
                oneeuro_beta=0.007,
                visualization_dir=None
            )
            print(f"Trajectory smoothing complete")
        
        # Generate animation
        if trajs is not None:
            trajs_to_use = trajs[:, :total_T, :, :]
            # convert coords to blender coordinate system
            trajs_b = trajs_to_use.clone()
            trajs_b[..., 0], trajs_b[..., 1], trajs_b[..., 2] = (
                trajs_to_use[..., 0],     # x -> x
                -trajs_to_use[..., 2],    # y -> -z
                trajs_to_use[..., 1]      # z -> y
            )
            trajs_to_use = trajs_b
            clear_scene()
            mesh_objects = import_glb(glb_path)
            
            drive_mesh_with_trajs_frames_gt(
                mesh_objects, 
                trajs_to_use.cpu(), 
                os.path.join(output_video_dir, 'output_animation'),
                azi=0, 
                ele=0, 
                export_format='glb'
            )
            print(f"✓ Animation saved to: {os.path.join(output_video_dir, 'output_animation.glb')}")


if __name__ == "__main__":
    run_inference_on_video(config)
