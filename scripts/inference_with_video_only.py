# Animate hunyuan genereated fbx mesh with input video.

import os
import sys
# Add project root directory to Python path to enable imports from root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.append(project_root)

os.environ['SPCONV_ALGO'] = 'native'

import importlib
import torch
import numpy as np
import imageio
from natsort import natsorted
from PIL import Image
import trimesh

from setup import init_config, init_distributed
from utils.render import (clear_scene, import_fbx, get_all_vertices, 
                        get_all_faces, drive_mesh_with_trajs_frames)
from utils.visualization import visualize_input_data
from utils.inference_utils import seed_everything, load_checkpoint, smooth_trajectories

config = init_config()
seed_everything(777)


def load_model(config, device):
    """
    Load model and checkpoint.
    
    Args:
        config: Model configuration
        device: Device to load model on
    
    Returns:
        model: Loaded model
    """
    module, class_name = config.model.class_name.rsplit(".", 1)
    Motion324 = importlib.import_module(module).__dict__[class_name]
    model = Motion324(config).to(device)

    ckpt_path = config.training.get("resume_ckpt", "")
    if not ckpt_path:
        ckpt_path = os.path.join(config.training.checkpoint_dir, "latest.pt")

    step_info = load_checkpoint(ckpt_path, model, device)
    print(f"Loaded checkpoint from step {step_info['param_update_step']}")
    model.eval()
    
    return model


def prepare_mesh_data_fbx(filepath, config):
    """
    Load and prepare mesh data from FBX file.
    
    Args:
        filepath: Path to FBX file
        config: Configuration object
    
    Returns:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        samples_xyz: Sampled surface points (num_samples, 3)
        samples_normals: Normals at sampled points (num_samples, 3)
        samples_rgb: RGB colors at sampled points (num_samples, 3)
        vertex_normals_np: Vertex normals (N, 3)
        vert_rgb: Vertex RGB colors (N, 3)
        mesh_objects: Blender mesh objects
    """
    clear_scene()
    mesh_objects = import_fbx(filepath)
    
    all_vertices = np.array(get_all_vertices(mesh_objects))[0]
    all_faces = np.array(get_all_faces(mesh_objects))[0]
    faces = all_faces
    
    # Normalize mesh
    center = (all_vertices.max(axis=0) + all_vertices.min(axis=0)) / 2
    all_vertices = all_vertices - center
    v_max = np.abs(all_vertices).max()
    all_vertices = all_vertices / (2 * (v_max + 1e-8))
    mesh_combined = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    
    # Sample surface points
    num_samples = getattr(config.training, "num_shape_samples", 4096)
    samples_xyz, face_indices = trimesh.sample.sample_surface(mesh_combined, num_samples)
    samples_normals = mesh_combined.face_normals[face_indices]  # (num_samples, 3)
    
    vertex_normals_np = mesh_combined.vertex_normals.astype(np.float32)
    
    # Initialize colors
    samples_rgb = np.full(samples_xyz.shape, [128, 128, 128], dtype=np.uint8)
    vert_rgb = np.full(all_vertices.shape, [128, 128, 128], dtype=np.uint8)
    
    # Extract texture colors
    for mesh_obj in mesh_objects:
        if mesh_obj.data.materials:
            mat = mesh_obj.data.materials[0]
            if mat and mat.use_nodes:
                nodes = mat.node_tree.nodes
                for node in nodes:
                    if node.type == 'TEX_IMAGE' and node.image:
                        try:
                            from PIL import Image as PILImage
                            temp_path = "/tmp/temp_texture.png"
                            node.image.save_render(temp_path)
                            texture = PILImage.open(temp_path)
                            if texture.mode != 'RGB':
                                texture_np = np.array(texture.convert('RGB'))
                            else:
                                texture_np = np.array(texture)
                            
                            if mesh_obj.data.uv_layers:
                                uv_layer = mesh_obj.data.uv_layers.active.data
                                tex_height, tex_width, _ = texture_np.shape
                                
                                # Sample colors for surface points
                                for i, face_idx in enumerate(face_indices):
                                    poly = mesh_obj.data.polygons[face_idx]
                                    face_uvs = []
                                    for loop_idx in poly.loop_indices:
                                        face_uvs.append(uv_layer[loop_idx].uv[:])
                                    face_uvs = np.array(face_uvs)  # (3, 2)
                                    
                                    triangle = all_vertices[all_faces[face_idx]]
                                    point = samples_xyz[i]
                                    bary = trimesh.triangles.points_to_barycentric([triangle], [point])[0]
                                    
                                    interpolated_uv = np.dot(bary, face_uvs)
                                    interpolated_uv[1] = 1.0 - interpolated_uv[1]
                                    
                                    pixel_x = int(np.clip(interpolated_uv[0] * (tex_width-1), 0, tex_width-1))
                                    pixel_y = int(np.clip(interpolated_uv[1] * (tex_height-1), 0, tex_height-1))
                                    samples_rgb[i] = texture_np[pixel_y, pixel_x]
                                
                                # Sample colors for vertices
                                vert_to_faces = {}
                                for face_idx, face in enumerate(all_faces):
                                    for local_idx, vert_idx in enumerate(face):
                                        if vert_idx not in vert_to_faces:
                                            vert_to_faces[vert_idx] = []
                                        vert_to_faces[vert_idx].append((face_idx, local_idx))
                                
                                for vert_idx in range(len(all_vertices)):
                                    if vert_idx in vert_to_faces:
                                        face_idx, local_idx = vert_to_faces[vert_idx][0]
                                        poly = mesh_obj.data.polygons[face_idx]
                                        loop_idx = poly.loop_indices[local_idx]
                                        uv = uv_layer[loop_idx].uv[:]
                                        uv_flipped = [uv[0], 1.0 - uv[1]]
                                        pixel_x = int(np.clip(uv_flipped[0] * (tex_width-1), 0, tex_width-1))
                                        pixel_y = int(np.clip(uv_flipped[1] * (tex_height-1), 0, tex_height-1))
                                        vert_rgb[vert_idx] = texture_np[pixel_y, pixel_x]
                                break
                        except Exception as e:
                            print(f"Failed to load texture: {e}")
                            pass
    
    # Ensure RGB has 3 channels
    if samples_rgb.shape[1] == 4:
        samples_rgb = samples_rgb[:, :3]
    if vert_rgb.shape[1] == 4:
        vert_rgb = vert_rgb[:, :3]
    
    # Convert coordinates
    R_example = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ])
    vertices = all_vertices @ R_example.T
    vertex_normals_np = vertex_normals_np @ R_example.T
    samples_xyz = samples_xyz @ R_example.T
    samples_normals = samples_normals @ R_example.T
    
    return vertices, faces, samples_xyz, samples_normals, samples_rgb, vertex_normals_np, vert_rgb, mesh_objects


def prepare_mesh_data_trimesh(filepath, config):
    """
    Load and prepare mesh data from trimesh-compatible file.
    
    Args:
        filepath: Path to mesh file
        config: Configuration object
    
    Returns:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        samples_xyz: Sampled surface points (num_samples, 3)
        samples_normals: Normals at sampled points (num_samples, 3)
        samples_rgb: RGB colors at sampled points (num_samples, 3)
        vertex_normals_np: Vertex normals (N, 3)
        vert_rgb: Vertex RGB colors (N, 3)
    """
    mesh = trimesh.load(filepath, process=False)

    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.geometry.values())
    
    vertices = mesh.vertices.astype(np.float32)               # (N, 3)
    faces = mesh.faces.astype(np.int64)                       # (M, 3)
    vertex_normals_np = mesh.vertex_normals.astype(np.float32) # (N, 3)
    
    # Normalize mesh
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = vertices - center
    v_max = np.abs(vertices).max()
    vertices = vertices / (2 * (v_max + 1e-8))

    # Sample surface points
    num_samples = getattr(config.training, "num_shape_samples", 2048)
    samples_xyz, face_indices = trimesh.sample.sample_surface(mesh, num_samples)
    samples_xyz = samples_xyz - center
    samples_xyz = samples_xyz / (2 * (v_max + 1e-8))
    samples_normals = mesh.face_normals[face_indices]  # (num_samples, 3)

    # Initialize colors
    samples_rgb = np.full(samples_xyz.shape, [128, 128, 128], dtype=np.uint8)
    vert_rgb = np.full(vertices.shape, [128, 128, 128], dtype=np.uint8)
    
    # Extract texture or vertex colors
    print(mesh.visual.kind)
    if mesh.visual.kind == 'texture' and hasattr(mesh.visual, 'uv'):
        Image = None
        try:
            from PIL import Image as PILImage
            Image = PILImage
        except ImportError:
            pass
        
        if Image is not None:
            if hasattr(mesh.visual.material, 'baseColorTexture'):
                texture = mesh.visual.material.baseColorTexture
            elif hasattr(mesh.visual.material, 'image'):
                texture = mesh.visual.material.image
            else:
                texture = None
            
            if texture is not None:
                if texture.mode != 'RGB':
                    texture_np = np.array(texture.convert('RGB'))
                else:
                    texture_np = np.array(texture)
                tex_height, tex_width, _ = texture_np.shape
                
                # Sample colors for surface points
                triangles = mesh.vertices[mesh.faces[face_indices]]
                bary_coords = trimesh.triangles.points_to_barycentric(triangles, samples_xyz)
                face_uvs = mesh.visual.uv[mesh.faces[face_indices]]
                interpolated_uvs = np.einsum('ij,ijk->ik', bary_coords, face_uvs)
                interpolated_uvs[:, 1] = 1.0 - interpolated_uvs[:, 1]
                pixel_coords_f = interpolated_uvs * [tex_width-1, tex_height-1]
                pixel_indices = np.round(pixel_coords_f).astype(int)
                pixel_indices[:, 0] = np.clip(pixel_indices[:, 0], 0, tex_width-1)
                pixel_indices[:, 1] = np.clip(pixel_indices[:, 1], 0, tex_height-1)
                samples_rgb = texture_np[pixel_indices[:, 1], pixel_indices[:, 0]]  # (num_samples, 3)

                # Sample colors for vertices
                vertex_uvs = mesh.visual.uv
                vertex_uvs_flipped = vertex_uvs.copy()
                vertex_uvs_flipped[:, 1] = 1.0 - vertex_uvs_flipped[:, 1]
                pixel_coords_f = vertex_uvs_flipped * [tex_width-1, tex_height-1]
                pixel_indices = np.round(pixel_coords_f).astype(int)
                pixel_indices[:, 0] = np.clip(pixel_indices[:, 0], 0, tex_width-1)
                pixel_indices[:, 1] = np.clip(pixel_indices[:, 1], 0, tex_height-1)
                vert_rgb = texture_np[pixel_indices[:, 1], pixel_indices[:, 0]]
    
    elif mesh.visual.kind == 'vertex' and hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors
        if vertex_colors is not None:
            triangles = mesh.vertices[mesh.faces[face_indices]]
            bary_coords = trimesh.triangles.points_to_barycentric(triangles, samples_xyz)
            face_vertex_colors = vertex_colors[mesh.faces[face_indices]]  # (num_samples, 3, 3/4)
            interpolated_colors = np.einsum('ij,ijk->ik', bary_coords, face_vertex_colors)
            samples_rgb = interpolated_colors[:, :3]
            
            vert_rgb = vertex_colors[:, :3]
    
    # Ensure RGB has 3 channels
    if samples_rgb.shape[1] == 4:
        samples_rgb = samples_rgb[:, :3]
    if vert_rgb.shape[1] == 4:
        vert_rgb = vert_rgb[:, :3]

    # Convert coordinates
    R_example = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ])
    vertices = vertices @ R_example.T
    vertex_normals_np = vertex_normals_np @ R_example.T
    samples_xyz = samples_xyz @ R_example.T
    samples_normals = samples_normals @ R_example.T
    
    return vertices, faces, samples_xyz, samples_normals, samples_rgb, vertex_normals_np, vert_rgb


def load_images_from_path(image_path, start_frame=0):
    """
    Load images from directory or list of paths.
    
    Args:
        image_path: Path to image directory or list of image paths
        start_frame: Starting frame index
    
    Returns:
        video_np: numpy array of shape (T, H, W, C), range [0, 255]
        video_tensor: torch tensor of shape (T, H, W, C), range [0, 1]
    """
    if isinstance(image_path, list):
        image_paths = image_path
    elif os.path.isdir(image_path):
        image_paths = natsorted([
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
    else:
        raise ValueError("image_path must be a list of image paths or a directory")
    
    frames = []
    sel_image_paths = image_paths[start_frame:]
    for img_path in sel_image_paths:
        img = Image.open(img_path).convert("RGB")
        frames.append(np.array(img))
    
    video_np = np.stack(frames, axis=0)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video_np).float() / 255.0
    
    return video_np, video_tensor


def prepare_input_data(vertices, faces, samples_xyz, samples_normals, samples_rgb, 
                       vertex_normals_np, vert_rgb, device):
    """
    Prepare input data dictionary for model inference.
    
    Args:
        vertices: Vertex positions (N, 3)
        faces: Face indices (M, 3)
        samples_xyz: Sampled surface points (num_samples, 3)
        samples_normals: Normals at sampled points (num_samples, 3)
        samples_rgb: RGB colors at sampled points (num_samples, 3)
        vertex_normals_np: Vertex normals (N, 3)
        vert_rgb: Vertex RGB colors (N, 3)
        device: Device to load data on
    
    Returns:
        input_data: Dictionary containing mesh data
    """
    vertices_torch = torch.tensor(vertices, dtype=torch.float32)
    faces_torch = torch.tensor(faces, dtype=torch.int64)

    ref_shape_pcd = torch.from_numpy(samples_xyz.astype(np.float32))              # (num_samples, 3)
    ref_shape_normals = torch.from_numpy(samples_normals.astype(np.float32))      # (num_samples, 3)
    ref_shape_rgbs = torch.from_numpy(samples_rgb.astype(np.float32) / 255.0)      # (num_samples, 3)
    ref_rgb = torch.from_numpy(vert_rgb.astype(np.float32) / 255.0)        # (N, 3)
    ref_normal = torch.from_numpy(vertex_normals_np.astype(np.float32))
    ref_pcd = vertices_torch
    faces = faces_torch
    
    input_data = {}
    input_data['faces'] = torch.from_numpy(faces).unsqueeze(0).to(device) if isinstance(faces, np.ndarray) else faces[None].to(device)  # (1, F, 3)
    input_data['ref_shape_pcd'] = ref_shape_pcd[None].to(device)         # (1, num_shape_samples, 3)
    input_data['ref_shape_normals'] = ref_shape_normals[None].to(device) # (1, num_shape_samples, 3)
    input_data['ref_shape_rgbs'] = ref_shape_rgbs[None].to(device)       # (1, num_shape_samples, 3)
    input_data['ref_pcd'] = ref_pcd[None].to(device)                     # (1, N_points, 3)
    input_data['ref_normal'] = ref_normal[None].to(device)               # (1, N_points, 3)
    input_data['ref_rgb'] = ref_rgb[None].to(device)                     # (1, N_points, 3)
    
    return input_data


def run_model_inference(model, input_data, video_tensor, config, device):
    """
    Run model inference on video with chunking support.
    
    Args:
        model: Loaded model
        input_data: Dictionary containing mesh data
        video_tensor: Video frames tensor (T, H, W, C)
        config: Configuration object
        device: Device to run on
    
    Returns:
        trajs: Predicted trajectories (1, T, N, 3)
    """
    total_T = video_tensor.shape[0]
    chunk_size = config.training.frames
    rgb_video = video_tensor  # (total_T, H, W, C)
    
    amp_dtype_mapping = {
        "fp16": torch.float16, 
        "bf16": torch.bfloat16, 
        "fp32": torch.float32, 
        'tf32': torch.float32
    }
    
    print(f"Total frames: {total_T}, frames per chunk: {chunk_size}")
    
    # Process video in chunks
    if total_T <= chunk_size:
        print(f"Total frames ({total_T}) <= chunk_size ({chunk_size}), processing all frames directly")
        input_data_chunk = input_data.copy()
        input_data_chunk['rgb_video'] = rgb_video[None].to(device)
        
        with torch.autocast(
            enabled=config.training.get('use_amp', False),
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.get('amp_dtype', 'fp16')],
        ):
            output_chunk = model(input_data_chunk)
        
        if isinstance(output_chunk, dict) and 'pcd_moved' in output_chunk:
            trajs = output_chunk['pcd_moved'].float()  # (1, total_T, N, 3)
            print(f"trajs.shape: {trajs.shape}")
        else:
            trajs = None
            print("Warning: pcd_moved not in output")
    else:
        slide_size = chunk_size - 1
        chunk_start_indices = list(range(0, total_T - chunk_size + 1, slide_size))
        print(chunk_start_indices)
        if chunk_start_indices and (chunk_start_indices[-1] + chunk_size < total_T):
            chunk_start_indices.append(total_T - chunk_size)
        print(chunk_start_indices)
        out_trajs_lst = []

        for i, start_idx in enumerate(chunk_start_indices):
            end_idx = start_idx + chunk_size

            if i == 0:
                rgb_window = rgb_video[0:chunk_size]
            else:
                rgb_window = torch.cat(
                    [
                        rgb_video[0:1],
                        rgb_video[start_idx+1:end_idx]
                    ],
                    dim=0
                )
            input_data_chunk = input_data.copy()
            input_data_chunk['rgb_video'] = rgb_window[None].to(device)
            if rgb_window.shape[0] != chunk_size:
                print(f"Warning: chunk rgb_video frame count ({rgb_window.shape[0]}) != T ({chunk_size}), skipping")
                continue
            print(f"Inferring chunk: {start_idx}-{end_idx}, rgb_video.shape: {input_data_chunk['rgb_video'].shape}")

            with torch.autocast(
                enabled=config.training.get('use_amp', False),
                device_type="cuda",
                dtype=amp_dtype_mapping[config.training.get('amp_dtype', 'fp16')],
            ):
                output_chunk = model(input_data_chunk)

            if isinstance(output_chunk, dict) and 'pcd_moved' in output_chunk:
                trajs_chunk = output_chunk['pcd_moved'].float()  # (1, chunk, N, 3)
                print(f"trajs_chunk.shape: {trajs_chunk.shape}")
                out_trajs_lst.append(trajs_chunk)
            else:
                print("Warning: chunk did not output pcd_moved, skipping")
        
        # Merge chunks
        if len(out_trajs_lst) > 0:
            if len(chunk_start_indices) >= 2:
                merged_trajs_lst = []
                for i in range(len(out_trajs_lst)):
                    if i == 0 and len(out_trajs_lst) != 2: # if len==2 should keep first N frames
                        chunk_trajs = out_trajs_lst[i].clone()  # (1, chunk_size, N, 3)
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
                    print(f"Concatenated trajs.shape: {trajs.shape}")
                else:
                    trajs = None
                    print("No trajs obtained")
            else:
                trajs = out_trajs_lst[0].clone()
                trajs[:, 0, :, :] = input_data['ref_pcd']
                print(f"Concatenated trajs.shape: {trajs.shape}")
        else:
            trajs = None
            print("No trajs obtained")
    
    return trajs


def run_inference_on_video(config):
    """
    Main inference function for video with mesh data.
    
    Args:
        config: Model configuration
    Returns:
        Model output
    """
    ddp_info = init_distributed(seed=777)
    torch.backends.cuda.matmul.allow_tf32 = config.training.get('use_tf32', True)
    torch.backends.cudnn.allow_tf32 = config.training.get('use_tf32', True)
    
    # Load model
    model = load_model(config, ddp_info.device)
    
    with torch.no_grad():
        # Prepare mesh data
        filepath = config.data_dir
        use_fbx = config.training.get('use_fbx', True)
        
        if use_fbx:
            vertices, faces, samples_xyz, samples_normals, samples_rgb, \
                vertex_normals_np, vert_rgb, mesh_objects = prepare_mesh_data_fbx(filepath, config)
        else:
            vertices, faces, samples_xyz, samples_normals, samples_rgb, \
                vertex_normals_np, vert_rgb = prepare_mesh_data_trimesh(filepath, config)
            mesh_objects = None
        
        # Prepare input data
        input_data = prepare_input_data(
            vertices, faces, samples_xyz, samples_normals, samples_rgb,
            vertex_normals_np, vert_rgb, ddp_info.device
        )
        
        # Load images
        if not hasattr(config, 'image_path') or config.image_path is None:
            raise ValueError("config.image_path not specified, cannot load rgb_video")
        
        start_frame = getattr(config, 'start_frame', 0)
        video_np, video_tensor = load_images_from_path(config.image_path, start_frame)
        
        # Save input video
        output_video_dir = config.output_dir if hasattr(config, 'output_dir') else "./"
        os.makedirs(output_video_dir, exist_ok=True)
        output_video_path = os.path.join(output_video_dir, "input_rgb_video.mp4")
        T = config.training.frames
        with imageio.get_writer(output_video_path, fps=12) as writer:
            for frame in video_np[:T]:
                writer.append_data(frame)
        print(f"Saved first {T} frames to video file: {output_video_path}")
        
        # Visualize input data
        visualize_input_data(input_data, save_path=os.path.join(output_video_dir, "input_data_vis.png"))
        
        # Run inference
        trajs = run_model_inference(model, input_data, video_tensor, config, ddp_info.device)
        
        # Apply smoothing
        if trajs is not None:
            trajs = smooth_trajectories(
                trajs, 
                method='combined',
                motion_threshold=0.002,
                window_size=3,
                sigma=1.0,
                savgol_polyorder=2,
                oneeuro_mincutoff=1.0,
                oneeuro_beta=0.007,
                #visualization_dir=config.output_dir
            )
            print(f"Trajectory smoothing completed")
        
        # Generate animation
        if trajs is not None:
            total_T = video_tensor.shape[0]
            trajs_to_use = trajs[:, :total_T, :, :]
            
            # Convert coords to blender coordinate system
            trajs_b = trajs_to_use.clone()
            trajs_b[..., 0], trajs_b[..., 1], trajs_b[..., 2] = (
                trajs_to_use[..., 0],     # x -> x
                -trajs_to_use[..., 2],    # y -> -z
                trajs_to_use[..., 1]      # z -> y
            )
            trajs_to_use = trajs_b
            
            if use_fbx and mesh_objects is not None:
                drive_mesh_with_trajs_frames(
                    mesh_objects, 
                    trajs_to_use.cpu(), 
                    os.path.join(output_video_dir, 'output_animation'),
                    azi=0, 
                    ele=0, 
                    export_format='fbx'
                )
            else:
                print("Warning: mesh_objects not available, cannot generate animation")
        
        return trajs


if __name__ == "__main__":
    run_inference_on_video(config)
