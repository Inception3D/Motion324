import os
import random
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from .visualization import plot_smoothing_comparison


def seed_everything(seed=0):
    """Set random seed for reproducibility."""
    print(f"Setting random seed: {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_checkpoint(ckpt_path, model, device):
    """
    Load checkpoint and return model state.
    
    Args:
        ckpt_path: Path to checkpoint file
        model: Model to load weights into
        device: Device to load checkpoint on
    
    Returns:
        step_info: Dictionary containing fwdbwd_pass_step and param_update_step
    """
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_state_dict = checkpoint['model']

    if 'pos_embed' in model_state_dict:
        del model_state_dict['pos_embed']
    
    try:
        model.load_state_dict(model_state_dict, strict=True)
        print("✓ Model loaded successfully with strict=True")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed: {e}")
        print("Attempting to load with strict=False...")
        model.load_state_dict(model_state_dict, strict=False)
        print("✓ Model loaded with strict=False")
    
    step_info = {
        'fwdbwd_pass_step': checkpoint.get('fwdbwd_pass_step', 0),
        'param_update_step': checkpoint.get('param_update_step', 0)
    }
    return step_info


class OneEuroFilter:
    """
    One Euro Filter for smooth trajectory tracking.
    
    Args:
        mincutoff: Minimum cutoff frequency
        beta: Speed coefficient
        dcutoff: Cutoff frequency for derivative
    """
    def __init__(self, mincutoff=1.0, beta=0.007, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        
    def __call__(self, x, alpha_d=None):
        if self.x_prev is None:
            self.x_prev = x
            return x
        
        dx = x - self.x_prev
        if alpha_d is None:
            alpha_d = self.smoothing_factor(1.0, self.dcutoff)
        dx_hat = alpha_d * dx + (1 - alpha_d) * self.dx_prev
        
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = self.smoothing_factor(1.0, cutoff)
        
        x_hat = alpha * x + (1 - alpha) * self.x_prev
        
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat
    
    @staticmethod
    def smoothing_factor(te, cutoff):
        r = 2 * np.pi * cutoff * te
        return r / (r + 1)


def smooth_trajectories(trajs, method='combined', motion_threshold=0.005, 
                       window_size=3, sigma=1.0, savgol_polyorder=2,
                       oneeuro_mincutoff=1.0, oneeuro_beta=0.007,
                       visualization_dir=None):
    """
    Apply smoothing to trajectories to reduce jitter.
    
    Args:
        trajs: (B, T, N, 3) trajectory tensor
        method: Smoothing method ('threshold', 'gaussian', 'savgol', 'oneeuro', 'combined')
        motion_threshold: Threshold for motion filtering
        window_size: Window size for gaussian and moving average
        sigma: Standard deviation for gaussian smoothing
        savgol_polyorder: Polynomial order for Savitzky-Golay filter
        oneeuro_mincutoff: Minimum cutoff frequency for One Euro Filter
        oneeuro_beta: Speed coefficient for One Euro Filter
        visualization_dir: Directory to save visualization plots
    
    Returns:
        Smoothed trajectories (B, T, N, 3)
    """
    trajs_smoothed = trajs.clone()
    B, T, N, _ = trajs.shape
    
    # Threshold filtering
    if method in ['threshold', 'combined']:
        print(f"Applying threshold filtering, threshold={motion_threshold}")
        for b in range(B):
            for t in range(1, T):
                displacement = trajs[b, t] - trajs[b, t-1]
                displacement_magnitude = torch.norm(displacement, dim=-1)
                mask = displacement_magnitude < motion_threshold
                trajs_smoothed[b, t][mask] = trajs_smoothed[b, t-1][mask]
    
    # Gaussian smoothing
    if method in ['gaussian', 'combined']:
        print(f"Applying gaussian smoothing, window_size={window_size}, sigma={sigma}")
        trajs_np = trajs_smoothed.cpu().numpy()
        for b in range(B):
            for n in range(N):
                for dim in range(3):
                    trajs_np[b, :, n, dim] = gaussian_filter1d(
                        trajs_np[b, :, n, dim], 
                        sigma=sigma, 
                        mode='nearest'
                    )
        trajs_smoothed = torch.from_numpy(trajs_np).to(trajs.device).type(trajs.dtype)
    
    # Savitzky-Golay filter
    if method == 'savgol':
        print(f"Applying Savitzky-Golay filter, window_size={window_size}, polyorder={savgol_polyorder}")
        if window_size % 2 == 0:
            window_size += 1
        if T >= window_size:
            trajs_np = trajs_smoothed.cpu().numpy()
            for b in range(B):
                for n in range(N):
                    for dim in range(3):
                        trajs_np[b, :, n, dim] = savgol_filter(
                            trajs_np[b, :, n, dim], 
                            window_length=window_size, 
                            polyorder=min(savgol_polyorder, window_size-1),
                            mode='nearest'
                        )
            trajs_smoothed = torch.from_numpy(trajs_np).to(trajs.device).type(trajs.dtype)
    
    # One Euro Filter
    if method == 'oneeuro':
        print(f"Applying One Euro Filter, mincutoff={oneeuro_mincutoff}, beta={oneeuro_beta}")
        trajs_np = trajs_smoothed.cpu().numpy()
        for b in range(B):
            for n in range(N):
                for dim in range(3):
                    one_euro = OneEuroFilter(mincutoff=oneeuro_mincutoff, beta=oneeuro_beta)
                    for t in range(T):
                        trajs_np[b, t, n, dim] = one_euro(trajs_np[b, t, n, dim])
        trajs_smoothed = torch.from_numpy(trajs_np).to(trajs.device).type(trajs.dtype)
    
    # Visualization
    if visualization_dir is not None:
        displacements_before = []
        displacements_after = []
        
        for b in range(B):
            for t in range(1, T):
                disp_before = trajs[b, t] - trajs[b, t-1]
                mag_before = torch.norm(disp_before, dim=-1).cpu().numpy()
                displacements_before.append(mag_before)
                
                disp_after = trajs_smoothed[b, t] - trajs_smoothed[b, t-1]
                mag_after = torch.norm(disp_after, dim=-1).cpu().numpy()
                displacements_after.append(mag_after)
        
        plot_smoothing_comparison(displacements_before, displacements_after, 
                                 motion_threshold, method, visualization_dir)
    
    return trajs_smoothed


def load_u2net_model(model_path=None, device='cuda'):
    """
    Load U2Net model for foreground segmentation.
    
    Args:
        model_path: Path to U2Net model weights (if None, will try to auto-download)
        device: Device to run on
    
    Returns:
        U2Net model session or None if loading fails
    """
    try:
        import sys
        import importlib
        
        setup_module_backup = sys.modules.get('setup')
        if 'setup' in sys.modules:
            del sys.modules['setup']
        
        try:
            rembg = importlib.import_module('rembg')
            print("Using rembg library for foreground segmentation")
            session = rembg.new_session('u2net')
            print("✓ U2Net model loaded successfully")
            return session
        finally:
            if setup_module_backup is not None:
                sys.modules['setup'] = setup_module_backup
    except ImportError as e:
        print(f"rembg not installed or import failed: {e}")
        print("Hint: Install rembg with: pip install rembg")
        return None
    except Exception as e:
        print(f"Error loading U2Net model: {e}")
        import traceback
        traceback.print_exc()
        return None


def segment_foreground_with_u2net(frames, u2net_model=None, device='cuda'):
    """
    Use U2Net to perform foreground segmentation on video frames.
    
    Args:
        frames: numpy array, shape (T, H, W, 3), range [0, 255]
        u2net_model: U2Net model or rembg session
        device: Device to run on
    
    Returns:
        masked_frames: numpy array, shape (T, H, W, 3), background as black
        masks: numpy array, shape (T, H, W, 1), segmentation mask, range [0,1]
    """
    if u2net_model is None:
        print("No U2Net model provided, skipping foreground segmentation")
        return frames, np.ones(frames.shape[:-1] + (1,), dtype=np.float32)
    
    print(f"Applying U2Net foreground segmentation to {len(frames)} frames...")
    masked_frames = []
    masks = []
    
    try:
        import importlib
        rembg_module = importlib.import_module('rembg')
        remove_func = rembg_module.remove
        
        for i, frame in enumerate(frames):
            if i % 10 == 0:
                print(f"Processing frame {i+1}/{len(frames)}...")
            
            from PIL import Image as PILImage
            frame_pil = PILImage.fromarray(frame.astype(np.uint8))
            output_pil = remove_func(frame_pil, session=u2net_model)
            output_np = np.array(output_pil)
            
            if output_np.shape[2] == 4:  # RGBA
                mask = output_np[:, :, 3:4] / 255.0
                rgb = output_np[:, :, :3]
            else:
                mask = np.ones((output_np.shape[0], output_np.shape[1], 1), dtype=np.float32)
                rgb = output_np[:, :, :3]
            
            masked_frame = (rgb * mask).astype(np.uint8)
            masked_frames.append(masked_frame)
            masks.append(mask.astype(np.float32))
        
        print("✓ Foreground segmentation complete")
        masked_frames = np.stack(masked_frames, axis=0)
        masks = np.stack(masks, axis=0)
        return masked_frames, masks
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return frames, np.ones(frames.shape[:-1] + (1,), dtype=np.float32)

