import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from dataset.dataset_utils import load_uv_preprocessing_data, track_with_normal_rgb


class Dyscene16k_Dataset(Dataset):
    def __init__(self, config, pcd_subdir="pcds", transform=None):
        """
        Args:
            root_dir (string): Directory with all the .glb files, .mp4 files, and the pcd_subdir.
                               e.g., /home/share/Dataset/3d_object/dyobjs/Arti-XL/glbs/tracks/
            pcd_subdir (string): Subdirectory under root_dir where object-named folders
                                 containing .npy point clouds are located. Default is "pcds".
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = Path(config.dataset_path)
        self.pcd_base_dir = self.root_dir / pcd_subdir
        self.image_base_dir = self.root_dir / "all_images"
        if hasattr(config, 'train_lst') and config.train_lst:
            train_lst_path = config.train_lst
        else:
            train_lst_path = 'dataset/train.lst'
        self.transform = transform
        self.frames = config.frames
        self.replica = config.replica
        self.num_shape_samples = config.num_shape_samples
        self.num_pcd_samples = config.num_pcd_samples

        # Find all .glb files and derive object names all_train
        with open(train_lst_path, 'r') as f: #valid all_unique_fliter 2kagain.lst
            self.obj_names = [line.strip() for line in f if line.strip()]
        
        self.obj_names = self.obj_names[config.dataset_begin:config.dataset_end]

        if not self.obj_names:
            raise RuntimeError(f"No .glb files found in {self.root_dir}")

        print(f"Found {len(self.obj_names)} objects: {self.obj_names[0]}...") # Print first 5 for brevity

    def __len__(self):
        return len(self.obj_names) * self.replica

    def _get_sequence_length(self, obj_name):
        """Get the total number of frames for a given object"""
        # Check point cloud files
        pcd_path = obj_name + '_pointclouds'
        obj_pcd_dir = self.pcd_base_dir / pcd_path
        pcd_files = []
        if obj_pcd_dir.is_dir():
            pcd_files = sorted(
                list(obj_pcd_dir.glob("frame_*.npy")),
                key=self._extract_frame_number
            )
        
        # Check image files
        image_path = obj_name + '_images'
        obj_image_dir = self.image_base_dir / image_path / 'camera_0'
        image_files = []
        if obj_image_dir.is_dir():
            image_files = sorted(
                list(obj_image_dir.glob("frame_*.jpg")) + list(obj_image_dir.glob("frame_*.png")),
                key=self._extract_frame_number
            )
        
        # Return the minimum length to ensure both modalities have data
        return max(len(pcd_files), len(image_files)) if pcd_files and image_files else 0

    def _generate_frame_indices(self, T):
        """Generate frame indices based on sampling strategies"""
        if T < self.frames:
            return None
            
        possible_options = []
        if T >= self.frames:
            possible_options.append({
                'name': 'skip1',
                'skip_interval': 1,
                'probability_weight': 0.4,
                'span': self.frames
            })
        required_span_skip2 = (self.frames - 1) * 2 + 1
        if T >= required_span_skip2:
            possible_options.append({
                'name': 'skip2',
                'skip_interval': 2,
                'probability_weight': 0.4,
                'span': required_span_skip2
            })
        required_span_skip4 = (self.frames - 1) * 4 + 1
        if T >= required_span_skip4:
            possible_options.append({
                'name': 'skip4',
                'skip_interval': 4,
                'probability_weight': 0.2,
                'span': required_span_skip4
            })
        
        if not possible_options:
            return None
            
        total_weight = sum(opt['probability_weight'] for opt in possible_options)
        probabilities = [opt['probability_weight'] / total_weight for opt in possible_options]
        
        chosen_opt_index = np.random.choice(len(possible_options), p=probabilities)
        chosen_strategy = possible_options[chosen_opt_index]

        skip = chosen_strategy['skip_interval']
        span_needed = chosen_strategy['span']

        max_start_idx = T - span_needed
        if max_start_idx <= 0:
            start_idx = np.random.randint(0, T - self.frames + 1)
            return list(range(start_idx, start_idx + self.frames))
        else:
            start_idx = np.random.randint(0, max_start_idx + 1)
            if skip == 1:
                return list(range(start_idx, start_idx + self.frames))
            else:
                return [start_idx + i * skip for i in range(self.frames)]

    def _load_single_image(self, image_path):
        """Load a single image file"""
        try:
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image).astype(np.float32) / 255.0
            return image_array
        except Exception as e:
            #print(f"Warning: Could not load image {image_path}: {e}")
            return None

    def _load_single_pointcloud(self, pcd_path):
        """Load a single point cloud file"""
        try:
            point_cloud = np.load(pcd_path)
            return point_cloud
        except Exception as e:
            print(f"Warning: Could not load point cloud {pcd_path}: {e}")
            return None

    def _extract_frame_number(self, filename):
        """Extracts frame number from filenames like frame_0001.npy"""
        match = re.search(r'frame_(\d+)\.npy', filename.name)
        if not match:
            match = re.search(r'frame_(\d+)\.(jpg|png)', filename.name)
        return int(match.group(1)) if match else -1

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        obj_name = self.obj_names[idx % len(self.obj_names)]
        
        # Get total sequence length
        T = self._get_sequence_length(obj_name)
        
        if T < self.frames:
            #print(f"Warning: No data found for object {obj_name}. Returning empty sample.")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]
        
        # Generate frame indices
        frame_indices = self._generate_frame_indices(T)
        
        if frame_indices is None:
            print(f"Warning: Cannot generate valid frame indices for {obj_name} with T={T}")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        if len(frame_indices) != self.frames:
            print(f"Warning: Cannot generate valid frame indices for {obj_name} with T={T}")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        # assert len(frame_indices) == self.frames
        # Load images and point clouds based on frame indices
        image_path = obj_name + '_images'
        camera = random.choice([f'camera_{i}' for i in range(15)])
        obj_image_dir = self.image_base_dir / image_path / camera

        pcd_path = obj_name + '_pointclouds'
        obj_pcd_dir = self.pcd_base_dir / pcd_path

        faces_path = obj_pcd_dir / 'faces.npy'
        faces_np = np.load(faces_path)
        faces = torch.from_numpy(faces_np)

        # Get sorted file lists
        image_files = sorted(
            list(obj_image_dir.glob("frame_*.jpg")) + list(obj_image_dir.glob("frame_*.png")),
            key=self._extract_frame_number
        )
        pcd_files = sorted(
            list(obj_pcd_dir.glob("frame_*.npy")),
            key=self._extract_frame_number
        )
        
        # Load selected frames
        rgb_frames = []
        point_clouds_list = []

        #assert len(frame_indices) == self.frames

        for frame_idx in frame_indices:
            # Load image
            if frame_idx < len(image_files):
                image_data = self._load_single_image(image_files[frame_idx])
                if image_data is not None:
                    rgb_frames.append(image_data)
                else:
                    #print(f"Warning: Failed to load image frame {frame_idx} for {obj_name}")
                    continue
            else:
                #print(f"Warning: Image frame {frame_idx} not found for {obj_name}")
                continue
                
            # Load point cloud
            if frame_idx < len(pcd_files):
                pcd_data = self._load_single_pointcloud(pcd_files[frame_idx])
                if pcd_data is not None:
                    point_clouds_list.append(pcd_data)
                else:
                    print(f"Warning: Failed to load point cloud frame {frame_idx} for {obj_name}")
                    continue
            else:
                print(f"Warning: Point cloud frame {frame_idx} not found for {obj_name}")
                continue
        
        if len(point_clouds_list) != self.frames or len(rgb_frames) !=self.frames:
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        # Convert to tensors
        video_np = np.stack(rgb_frames, axis=0).astype(np.float32)
        vertex_np = np.stack(point_clouds_list, axis=0)

        if video_np.shape[0] != vertex_np.shape[0]:
            #print(f"Warning: video_np.shape[0] ({video_np.shape[0]}) != vertex_np.shape[0] ({vertex_np.shape[0]}), resampling for {obj_name}")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        uv_data_dir = obj_pcd_dir
        data_path = os.path.join(uv_data_dir, 'uv_face_texture.npz')
        
        if not os.path.exists(data_path):
            print(f"Warning: UV data not found for {obj_name}")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]
        
        uv_data = load_uv_preprocessing_data(data_path)
        face_uvs = uv_data['face_uvs']
        texture_array = uv_data['texture_array']

        try:
            final_rgb_frames = torch.from_numpy(video_np).float()

            ref_shape_pcd, ref_shape_normals, ref_shape_rgbs, ref_shape_face_indices = track_with_normal_rgb(
                init_mesh=trimesh.Trimesh(vertices=vertex_np[0], faces=faces_np, process=False),
                vertex_frames=vertex_np[0][None, ...],
                faces=faces,
                num_samples=self.num_shape_samples,
                face_uvs=face_uvs,
                texture_array=texture_array
            )
            ref_shape_pcd = ref_shape_pcd[0]
            ref_shape_normals = ref_shape_normals[0]
            ref_shape_rgbs = ref_shape_rgbs[0]

            if (
                torch.isnan(ref_shape_pcd).any() or torch.isinf(ref_shape_pcd).any() or
                torch.isnan(ref_shape_normals).any() or torch.isinf(ref_shape_normals).any()
            ):
                new_idx = random.randint(0, len(self) - 1)
                return self[new_idx]

            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()
            else:
                faces = faces

            initial_vertices = vertex_np[0]
            initial_mesh = trimesh.Trimesh(vertices=initial_vertices, faces=faces_np, process=False)

            final_point_clouds, final_point_normals, final_point_rgbs, face_indices = track_with_normal_rgb(
                init_mesh=initial_mesh,
                vertex_frames=vertex_np,
                faces=faces,
                num_samples=self.num_pcd_samples,
                face_uvs=face_uvs,
                texture_array=texture_array
            )


            if (
                torch.isnan(final_point_clouds).any() or torch.isinf(final_point_clouds).any() or
                torch.isnan(final_point_normals).any() or torch.isinf(final_point_normals).any()
            ):
                new_idx = random.randint(0, len(self) - 1)
                return self[new_idx]

        except Exception as e:
            #print(f"Error processing object {obj_name} at index {idx}: {str(e)}. Skipping and trying a new one.")
            new_idx = random.randint(0, len(self) - 1)
            return self[new_idx]

        sample = {
            'obj_name': obj_name,
            'rgb_video': final_rgb_frames,
            'point_clouds': final_point_clouds,
            'point_rgbs': final_point_rgbs,
            'ref_shape_pcd': ref_shape_pcd,
            'ref_shape_normals': ref_shape_normals,
            'ref_shape_rgbs': ref_shape_rgbs,
            'ref_pcd': final_point_clouds[0],
            'ref_normal': final_point_normals[0],
            'ref_rgb': final_point_rgbs[0],

        }

        return sample

def collate_fn_with_topology(batch):
    """
    Batch processing function supporting topology information.
    """
    collated = {}
    
    tensor_keys = [
        'rgb_video',
        'point_clouds',
        'point_rgbs',
        'ref_shape_pcd',
        'ref_shape_normals',
        'ref_shape_rgbs',
        'ref_pcd',
        'ref_normal',
        'ref_rgb',
    ]
    
    for key in tensor_keys:
        if key in batch[0]:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except RuntimeError as e:
                # shapes = [item[key].shape for item in batch]
                for i, item in enumerate(batch):
                    if item[key].shape != batch[0][key].shape:
                        expected_shape = batch[0][key].shape
                        actual_shape = item[key].shape
                        raise RuntimeError(
                            f"Shape mismatch in tensor '{key}' for object {item['obj_name']}. "
                            f"Expected {expected_shape} (from {batch[0]['obj_name']}), "
                            f"got {actual_shape}"
                        )
                raise RuntimeError(f"Failed to stack tensor '{key}' for batch. Original error: {str(e)}")
    
    collated['obj_name'] = [item['obj_name'] for item in batch]

    edge_indices = []
    num_nodes_offset = 0
    num_nodes_per_sample = batch[0]['point_clouds'].size(1)
    if 'edge_indices' in batch[0]:
        for sample in batch:
            edge_index_offset = sample['edge_indices'] + num_nodes_offset
            edge_indices.append(edge_index_offset)
            
            num_nodes_offset += num_nodes_per_sample
            
        # final shape: (2, B * E)
        batched_edge_index = torch.cat(edge_indices, dim=1)

        collated['edge_indices'] = batched_edge_index

    return collated
