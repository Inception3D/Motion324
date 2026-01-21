import torch
import torchvision.transforms as transforms
import cv2
import os
import json
import numpy as np
from calculate_fvd import calculate_fvd
from calculate_lpips import calculate_lpips, calculate_dreamsim_loss, calculate_clip_loss
import argparse
from glob import glob
from PIL import Image
import lpips
import open_clip
from dreamsim import dreamsim

# Constants
FRAME_SIZE = 512  # Resize all frames to 512x512
TIMESTAMP_LIMIT = 32  # Split videos into 32-frame subvideos

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
    transforms.ToTensor()  # Converts to [0,1] range
])

def load_video_as_tensor(video_path):
    """Load a video and convert it into a tensor [timestamp, channel, width, height]."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = transform(frame)  # Resize & normalize
        frames.append(frame)
    
    cap.release()
    
    #if len(frames) < TIMESTAMP_LIMIT:
    #    return None  # Ignore videos with fewer than 32 frames
    
    video_tensor = torch.stack(frames)  # Shape: [timestamp, channel, width, height]
    return video_tensor

def load_images_as_tensor(image_dir):
    """Load images from a directory and convert them into a tensor [timestamp, channel, width, height]."""
    # Find all image files (png, jpg, jpeg)
    image_files = sorted(glob(os.path.join(image_dir, "*.png")) + 
                        glob(os.path.join(image_dir, "*.jpg")) + 
                        glob(os.path.join(image_dir, "*.jpeg")))
    
    #if len(image_files) < TIMESTAMP_LIMIT:
    #    return None  # Ignore if fewer than required frames
    
    frames = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        frame = transform(img_array)  # Resize & normalize
        frames.append(frame)
    
    video_tensor = torch.stack(frames)  # Shape: [timestamp, channel, width, height]
    return video_tensor

def is_video_file(path):
    """Check if path is a video file"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    return os.path.isfile(path) and any(path.lower().endswith(ext) for ext in video_extensions)

def load_input(path):
    """Load input from video file or image directory"""
    if is_video_file(path):
        return load_video_as_tensor(path)
    elif os.path.isdir(path):
        return load_images_as_tensor(path)
    else:
        raise ValueError(f"Path {path} is neither a video file nor a directory")

def get_frame_count(path):
    """Get frame count from video file or image directory"""
    if is_video_file(path):
        cap = cv2.VideoCapture(path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return count
    elif os.path.isdir(path):
        image_files = sorted(glob(os.path.join(path, "*.png")) + 
                            glob(os.path.join(path, "*.jpg")) + 
                            glob(os.path.join(path, "*.jpeg")))
        return len(image_files)
    else:
        return 0

def process_single_video(video_path_or_dir):
    """Load a single video/image directory and convert it into a tensor [timestamp, channel, width, height]."""
    tensor = load_input(video_path_or_dir)
    if tensor.shape[0] == 0:
        raise ValueError(f"Input {video_path_or_dir} has no frames")
    # If video has fewer frames than TIMESTAMP_LIMIT, pad by repeating the last frame
    if tensor.shape[0] < TIMESTAMP_LIMIT:
        num_frames_needed = TIMESTAMP_LIMIT - tensor.shape[0]
        # Take the last num_frames_needed frames and reverse them for padding
        padding = tensor[-num_frames_needed:].flip(0)  # Reverse the last num_frames_needed frames
        tensor = torch.cat([tensor, padding], dim=0)  # Concatenate along timestamp dimension
        print(f"Input {video_path_or_dir} padded from {tensor.shape[0] - num_frames_needed} to {tensor.shape[0]} frames")
    # Split into 32-frame subvideos
    subvideos = [tensor[i:i+TIMESTAMP_LIMIT] for i in range(0, tensor.shape[0] - TIMESTAMP_LIMIT + 1, TIMESTAMP_LIMIT)]
    print(f"Loaded {len(subvideos)} subvideos from {video_path_or_dir}")
    
    # Stack into batch: [batch, timestamp, channel, width, height]
    return torch.stack(subvideos)

parser = argparse.ArgumentParser(description="Evaluate video metrics between videos/images.")
parser.add_argument('--gt_paths', type=str, required=True, nargs='+', help='Path(s) to ground truth video(s) or image directory(ies)')
parser.add_argument('--result_paths', type=str, required=True, nargs='+', help='Path(s) to generated/result video(s) or image directory(ies)')
parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results (optional)')
args = parser.parse_args()

gt_paths = args.gt_paths
result_paths = args.result_paths
output_dir = args.output_dir

# Ensure same number of paths
if len(gt_paths) != len(result_paths):
    raise ValueError(f"Number of GT paths ({len(gt_paths)}) must match number of result paths ({len(result_paths)})")

# Move tensors to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load all models once before the loop
print("Loading all models...")
print("  - Loading LPIPS model...")
lpips_model = lpips.LPIPS(net='vgg', spatial=True)
lpips_model = lpips_model.to(device)

print("  - Loading I3D model for FVD...")
from fvd.styleganv.fvd import load_i3d_pretrained
i3d_model = load_i3d_pretrained(device=device)

print("  - Loading DreamSim model...")
dreamsim_model, _ = dreamsim(pretrained=True, device=device)

print("  - Loading OpenCLIP model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2B_s39B_b160k")
clip_model = clip_model.to(device).eval()

print("All models loaded successfully!\n")

# Create output directory if specified
if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Process each pair of paths
all_results = []
for idx, (gt_path, result_path) in enumerate(zip(gt_paths, result_paths)):
    print(f"\n{'='*60}")
    print(f"Processing pair {idx+1}/{len(gt_paths)}")
    print(f"{'='*60}")
    
    # Check if paths exist
    if not os.path.exists(gt_path):
        print(f"Warning: GT path not found at {gt_path}, skipping...")
        continue
    if not os.path.exists(result_path):
        print(f"Warning: Result path not found at {result_path}, skipping...")
        continue
    
    print(f"GT: {gt_path}")
    print(f"Result: {result_path}")
    
    try:
        # Get frame counts
        gt_frame_count = get_frame_count(gt_path)
        result_frame_count = get_frame_count(result_path)
        
        if gt_frame_count == 0:
            print(f"Warning: GT path {gt_path} has no frames, skipping...")
            continue
        if result_frame_count == 0:
            print(f"Warning: Result path {result_path} has no frames, skipping...")
            continue
        
        if gt_frame_count != result_frame_count:
            print(f"Warning: Frame count mismatch: GT has {gt_frame_count} frames, but result has {result_frame_count} frames.")
        
        print(f"GT has {gt_frame_count} frames")
        print(f"Result has {result_frame_count} frames")
        print(f"Processing...")
        
        videos1 = process_single_video(gt_path)
        videos2 = process_single_video(result_path)
        
        # Compute metrics
        only_final = True
        
        print("Computing metrics...")
        lpips_result, lpips_eachvideo = calculate_lpips(videos1, videos2, device, only_final=only_final, lpips_model=lpips_model)
        fvd_result = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=only_final, i3d_model=i3d_model)
        dreamsim_result, dreamsim_eachvideo = calculate_dreamsim_loss(videos1, videos2, device, only_final=only_final, dreamsim_model=dreamsim_model)
        clip_loss_result, clip_loss_eachvideo = calculate_clip_loss(videos1, videos2, device, only_final=only_final, clip_model=clip_model, clip_preprocess=clip_preprocess)
        
        # Extract scalar values from result dictionaries
        # When only_final=True, value is a list with single element
        fvd_value = fvd_result['value'][0] if isinstance(fvd_result['value'], list) else fvd_result['value']
        lpips_value = lpips_result['value'][0] if isinstance(lpips_result['value'], list) else lpips_result['value']
        dreamsim_value = dreamsim_result['value'][0] if isinstance(dreamsim_result['value'], list) else dreamsim_result['value']
        clip_loss_value = clip_loss_result['value'] if isinstance(clip_loss_result['value'], (int, float)) else clip_loss_result['value'][0]
        
        result = {
            'gt_path': gt_path,
            'result_path': result_path,
            'fvd': fvd_value,
            'lpips': lpips_value,
            'dreamsim': dreamsim_value,
            'clip_loss': clip_loss_value,
        }
        all_results.append(result)
        
        # Print results
        print(f"\nResults:")
        print(json.dumps({k: v for k, v in result.items() if k not in ['gt_path', 'result_path']}, indent=4))
        
        # Save results to file
        # Default: save to result_path's dirname with basename.txt
        result_basename = os.path.basename(result_path.rstrip('/'))
        # Remove extension if it's a file
        if os.path.isfile(result_path):
            result_basename = os.path.splitext(result_basename)[0]
        
        if output_dir:
            # If output_dir is specified, save there
            output_file = os.path.join(output_dir, f'{result_basename}.txt')
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Otherwise, save to result_path's directory
            result_dir = os.path.dirname(result_path)
            if result_dir:
                output_file = os.path.join(result_dir, f'{result_basename}.txt')
                os.makedirs(result_dir, exist_ok=True)
            else:
                # If result_path has no dirname (current directory), save to current directory
                output_file = f'{result_basename}.txt'
            with open(output_file, 'w') as f:
                f.write(f"GT Source: {gt_path}\n")
                f.write(f"Result Source: {result_path}\n")
                f.write(f"FVD: {fvd_value}\n")
                f.write(f"LPIPS: {lpips_value}\n")
                f.write(f"DreamSim: {dreamsim_value}\n")
                f.write(f"CLIP Loss: {clip_loss_value}\n")
                f.write("-" * 50 + "\n")
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing pair {idx+1}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

# Print summary
print(f"\n{'='*60}")
print(f"Summary: Processed {len(all_results)}/{len(gt_paths)} pairs")
if len(all_results) > 0:
    avg_fvd = np.mean([r['fvd'] for r in all_results])
    avg_lpips = np.mean([r['lpips'] for r in all_results])
    avg_dreamsim = np.mean([r['dreamsim'] for r in all_results])
    avg_clip = np.mean([r['clip_loss'] for r in all_results])
    print(f"Average FVD: {avg_fvd:.6f}")
    print(f"Average LPIPS: {avg_lpips:.6f}")
    print(f"Average DreamSim: {avg_dreamsim:.6f}")
    print(f"Average CLIP Loss: {avg_clip:.6f}")
print(f"{'='*60}")