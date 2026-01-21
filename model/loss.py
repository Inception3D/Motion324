import torch.nn as nn
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict

class MSELossComputer(nn.Module):
    def __init__(self, config):
        """
        Initializes the LossComputer.

        Args:
            config: Configuration object. Expected to have config.training.coord_mse_loss_weight.
        """
        super().__init__()
        self.config = config

        # Ensure the config has 'coord_mse_loss_weight' defined.
        if not hasattr(self.config, 'training') or \
           not hasattr(self.config.training, 'coord_mse_loss_weight'):
            raise ValueError(
                "Configuration must have 'config.training.coord_mse_loss_weight' defined."
            )

    def forward(
        self,
        coords_pred: torch.Tensor,
        coords_target: torch.Tensor,
    ):
        """
        Calculate Mean Squared Error (MSE) loss for coordinates.

        Args:
            coords_pred (torch.Tensor): Predicted coordinates. Shape [B, T, N, C].
            coords_target (torch.Tensor): Ground truth coordinates. Shape [B, T, N, C].
        
        Returns:
            edict: Dictionary of loss metrics, containing 'loss' and 'coord_mse_loss'.
        """

        # Determine device from input tensor
        current_device = coords_pred.device

        total_loss = torch.tensor(0.0, device=current_device)
        loss_metrics = edict()

        # --- Coordinate MSE Loss ---
        coord_mse_loss_value = torch.tensor(0.0, device=current_device)

        # Validate shapes
        if not (coords_pred.ndim == 4 and \
                coords_target.ndim == 4 and \
                coords_pred.shape == coords_target.shape):
            raise ValueError(
                f"Shape mismatch or invalid shape for coordinate MSE. "
                f"Expected both tensors of shape (B, T, N, C). "
                f"Got pred: {coords_pred.shape}, target: {coords_target.shape}"
            )

        if self.config.training.coord_mse_loss_weight > 0.0:
            coord_mse_loss_value = F.mse_loss(coords_pred, coords_target)
            total_loss += self.config.training.coord_mse_loss_weight * coord_mse_loss_value
        
        loss_metrics.coord_mse_loss = coord_mse_loss_value
        loss_metrics.loss = total_loss # The final combined loss (which is just coord_mse_loss weighted)
        
        return loss_metrics
