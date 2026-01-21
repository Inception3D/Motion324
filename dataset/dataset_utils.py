import os
from typing import Dict, Tuple

import numpy as np
import torch
import trimesh


def load_uv_preprocessing_data(save_dir: str) -> Dict[str, np.ndarray]:
    """Load preprocessed UV data."""
    
    data = np.load(save_dir)
    return {
        'face_uvs': data['face_uvs'],
        'texture_array': data['texture_array']
    }


def sample_texture_color_vectorized(
    uvs: np.ndarray, 
    texture_array: np.ndarray
) -> np.ndarray:
    """
    Sample colors from texture map in batch based on UV coordinates.

    Args:
        uvs (np.ndarray): UV coordinate array with shape (N, 2).
        texture_array (np.ndarray): Texture map with shape (H, W, 3).

    Returns:
        np.ndarray: Sampled RGB color array with shape (N, 3).
    """
    u, v = uvs[:, 0], uvs[:, 1]
    
    x = (u * (texture_array.shape[1] - 1)).astype(int)
    y = ((1 - v) * (texture_array.shape[0] - 1)).astype(int)
    
    x = np.clip(x, 0, texture_array.shape[1] - 1)
    y = np.clip(y, 0, texture_array.shape[0] - 1)
    
    return texture_array[y, x]


def track_with_normal_rgb(
    init_mesh: trimesh.Trimesh,
    vertex_frames: np.ndarray,
    faces: np.ndarray,
    num_samples: int,
    face_uvs: np.ndarray,
    texture_array: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Sample points on a deforming mesh sequence and track their positions, normals, and RGB colors.

    This method first samples points on the first frame (t=0) and computes their barycentric coordinates.
    Then it uses these fixed coordinates to compute the positions of sampled points in all subsequent frames,
    interpolated normals, and RGB colors based on UV mapping.

    Args:
        init_mesh (trimesh.Trimesh): Trimesh mesh object of the first frame, used for initial sampling.
        vertex_frames (np.ndarray): Mesh vertex sequence with shape (T, N, 3).
        faces (np.ndarray): Mesh faces with shape (F, 3), shared across all frames.
        num_samples (int): Number of points to sample and track.
        face_uvs (np.ndarray): UV coordinates for three vertices of each face, with shape (F, 3, 2).
        texture_array (np.ndarray): Texture map with shape (H, W, 3).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        - tracked_points (torch.Tensor): Tracked point cloud sequence with shape (T, num_samples, 3).
        - tracked_normals (torch.Tensor): Corresponding normal vector sequence with shape (T, num_samples, 3).
        - tracked_rgbs (torch.Tensor): Corresponding RGB color sequence with shape (T, num_samples, 3).
                                      Colors are fixed across all frames.
    """
    num_frames = vertex_frames.shape[0]

    
    sampled_points_t0, face_indices = trimesh.sample.sample_surface(init_mesh, num_samples)

    triangles_t0 = init_mesh.vertices[faces[face_indices]]

    barycentric_coords = trimesh.triangles.points_to_barycentric(
        triangles=triangles_t0, points=sampled_points_t0
    )

    # face_uvs: (F, 3, 2) -> sampled_face_uvs: (num_samples, 3, 2)
    sampled_face_uvs = face_uvs[face_indices]

    # barycentric_coords: (num_samples, 3), sampled_face_uvs: (num_samples, 3, 2)
    interpolated_uvs = np.einsum(
        'ij,ijk->ik',
        barycentric_coords,
        sampled_face_uvs
    )

    tracked_rgbs_np = sample_texture_color_vectorized(interpolated_uvs, texture_array)
    tracked_rgbs_np = tracked_rgbs_np / 255.0

    all_tracked_points = []
    all_tracked_normals = []

    tracking_mesh = trimesh.Trimesh(faces=faces, process=False)

    for t in range(num_frames):
        tracking_mesh.vertices = vertex_frames[t]

        current_triangles = tracking_mesh.vertices[faces[face_indices]]
        current_points = trimesh.triangles.barycentric_to_points(current_triangles, barycentric_coords)
        all_tracked_points.append(current_points)

        vertex_normals_of_triangles = tracking_mesh.vertex_normals[faces[face_indices]]
        
        interpolated_normals = np.einsum(
            'ij,ijk->ik',
            barycentric_coords,
            vertex_normals_of_triangles
        )
        
        norms = np.linalg.norm(interpolated_normals, axis=1, keepdims=True)
        non_zero_norms = np.where(norms == 0, 1.0, norms)
        interpolated_normals /= non_zero_norms
        
        all_tracked_normals.append(interpolated_normals)

    tracked_points_np = np.stack(all_tracked_points, axis=0)
    tracked_normals_np = np.stack(all_tracked_normals, axis=0)
    
    tracked_rgbs_np_expanded = np.tile(
        tracked_rgbs_np[np.newaxis, :, :], 
        (num_frames, 1, 1)
    )

    tracked_points = torch.from_numpy(tracked_points_np).float()
    tracked_normals = torch.from_numpy(tracked_normals_np).float()
    tracked_rgbs = torch.from_numpy(tracked_rgbs_np_expanded).float()
    
    return tracked_points, tracked_normals, tracked_rgbs, face_indices
