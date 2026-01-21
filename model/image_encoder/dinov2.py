import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid."""

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Generate spatial positions for a batch of patches.

        Args:
            batch_size: Number of samples in the batch
            height: Height of the grid in patches
            width: Width of the grid in patches
            device: Target device for the position tensor

        Returns:
            Tensor of shape (batch_size, height*width, 2) containing y,x coordinates
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

class DinoEncoder(nn.Module):
    """DINOv2 vision transformer encoder for extracting image features."""
    
    def __init__(self, patch_size=14, model_name='dinov2_vitb14'):
        super(DinoEncoder, self).__init__()
        #self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = torch.hub.load('/home/hongyuan/.cache/torch/hub/facebookresearch_dinov2_main', model_name, source = 'local')
        if hasattr(self.model, 'patch_size'):
            assert self.model.patch_size == patch_size, \
                f"DINOv2 model's patch size ({self.model.patch_size}) must match provided patch_size ({patch_size})"
        else:
            print(f"Warning: DINOv2 model does not expose patch_size. Assuming it is {patch_size}.")

        self.patch_size = patch_size
        self.position_getter = PositionGetter()

        # For fixed 224x224 input with patch_size=14
        self.image_size = 224
        self.num_patches_per_dim = self.image_size // self.patch_size  # 16
        self.num_patches_total = self.num_patches_per_dim * self.num_patches_per_dim  # 256

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        
    def forward(self, image):
        """
        Forward pass through DINOv2 encoder.
        
        Args:
            image: Input images of shape (B, C, H, W), e.g., (B, 3, 224, 224)
        
        Returns:
            encoded_feats: Patch features of shape (B, N_patches, D_embed)
        """
        B, C, H, W = image.shape
        device = image.device

        resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
        resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
        images_resnet_norm = (image - resnet_mean) / resnet_std
        
        assert H == self.image_size and W == self.image_size, \
            f"Input image size must be {self.image_size}x{self.image_size}, but got {H}x{W}"

        encoded_feats = self._process_images(images_resnet_norm)
        return encoded_feats

    def _process_images(self, images):
        """
        Process a batch of images through DINO encoder.
        
        Args:
            images: Input images of shape (B, C, H, W), e.g., (B, 3, 224, 224)
        
        Returns:
            encoded_features: Patch features
        """
        B, C, H, W = images.shape
        features_dict = self.model.forward_features(images)
        
        # Get patch tokens
        if 'x_norm_patchtokens' in features_dict:
            encoded_features = features_dict['x_norm_patchtokens']
        elif 'x_patchtokens' in features_dict:
            encoded_features = features_dict['x_patchtokens']
        else:
            if isinstance(features_dict, torch.Tensor) and features_dict.ndim == 3 and features_dict.shape[0] == B:
                num_expected_patches = (H // self.patch_size) * (W // self.patch_size)
                if features_dict.shape[1] == num_expected_patches:
                    encoded_features = features_dict
                else:
                    raise KeyError(f"DINOv2 output shape {features_dict.shape} does not match expected patch tokens")
            else:
                available_keys = list(features_dict.keys()) if isinstance(features_dict, dict) else 'N/A'
                raise KeyError(f"Could not find patch tokens in DINOv2 output. Available keys: {available_keys}")

        if hasattr(self.model, 'embed_dim'):
            assert encoded_features.shape[-1] == self.model.embed_dim, \
                f"Feature dimension {encoded_features.shape[-1]} does not match model embed_dim {self.model.embed_dim}"

        assert encoded_features.shape[1] == self.num_patches_total, \
            f"Number of patches {encoded_features.shape[1]} does not match expected {self.num_patches_total}"

        return encoded_features
    
    def train(self, mode: bool = False):
        """Override train method to keep DINO model in eval mode."""
        super(DinoEncoder, self).train(mode)
        if self.model is not None:
            self.model.eval()
        return self