import os
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from PIL import Image
import rembg
import cv2  # for video processing
from pathlib import Path  # for path handling

# rembg session models (sorted by quality/speed trade-off):
# 
# Highest Quality (slower):
#   - 'birefnet-massive'  : Best quality, slowest, large model
#   - 'isnet-general-use' : Very high quality, balanced speed
#   - 'birefnet'          : High quality, good balance

#
# Fast:
#   - 'u2net'             : Good quality, fast (default choice)
#   - 'u2net_human_seg'   : Optimized for human segmentation
#   - 'u2net_cloth_seg'   : Optimized for clothing segmentation

# rembg_session = rembg.new_session('birefnet-massive')
# rembg_session = rembg.new_session('u2net')
rembg_session = rembg.new_session('isnet-general-use')
# rembg_session = rembg.new_session('u2net_human_seg')

def remove_background(image):
    # Remove background from given PIL Image. Returns:
    # - RGBA image with black background on transparent regions
    # - mask as numpy array (float [0, 1])
    # - mask image as PIL Image (mode 'L')
    # - bounding box (min_x, min_y, max_x+1, max_y+1)

    # Step 1: Generate the mask using rembg
    if image.mode != 'RGBA':
        image_rgb = image.convert('RGB')
        out = rembg.remove(image_rgb, session=rembg_session)
    else:
        arr = np.array(image)
        if arr.shape[2] != 4 or np.all(arr[..., 3] == 255):
            image_rgb = image.convert('RGB')
            out = rembg.remove(image_rgb, session=rembg_session)
        else:
            out = image

    arr = np.array(out)
    alpha = arr[..., 3]
    mask = (alpha > int(0.8 * 255)).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask, mode='L')

    # Calculate tight bounding box of the foreground (do not expand)
    pos = np.argwhere(mask > 0)
    min_x, min_y = np.min(pos[:, 1]), np.min(pos[:, 0])
    max_x, max_y = np.max(pos[:, 1]), np.max(pos[:, 0])
    bbox = (min_x, min_y, max_x + 1, max_y + 1)  # left, top, right (open), bottom (open)

    # Step 2: Apply mask to original image, set background to black and keep alpha = mask
    img_rgba = image.convert('RGBA')
    img_np = np.array(img_rgba).astype(np.float32) / 255.0         # shape H,W,4
    mask_np = (mask / 255.0).astype(np.float32)                    # shape H,W

    result_np = np.zeros_like(img_np)
    result_np[..., :3] = img_np[..., :3] * mask_np[..., None]
    result_np[..., 3] = mask_np

    new_image_blackbg = Image.fromarray((result_np * 255).astype(np.uint8), mode='RGBA')

    return new_image_blackbg, mask_np, mask_img, bbox

def compute_mask_bbox(mask_img):
    mask_arr = np.array(mask_img)
    pos = np.argwhere(mask_arr > 0)
    if pos.size == 0:
        return None
    min_y, min_x = pos.min(axis=0)
    max_y, max_x = pos.max(axis=0)
    return (int(min_x), int(min_y), int(max_x) + 1, int(max_y) + 1)

def merge_bbox(global_bbox, new_bbox):
    if new_bbox is None:
        return global_bbox
    if global_bbox is None:
        return list(new_bbox)
    g_left, g_top, g_right, g_bottom = global_bbox
    n_left, n_top, n_right, n_bottom = new_bbox
    return [
        min(g_left, n_left),
        min(g_top, n_top),
        max(g_right, n_right),
        max(g_bottom, n_bottom),
    ]

def crop_and_center_to_512(img, bbox, fill_value):
    # crop by bbox, resize to fit within 512 keeping aspect, then center-pad
    left, top, right, bottom = bbox
    cropped = img.crop((left, top, right, bottom))
    w, h = cropped.size
    if w == 0 or h == 0:
        return Image.new(img.mode, (512, 512), fill_value)
    scale = 512 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new(img.mode, (512, 512), fill_value)
    offset_x = (512 - new_w) // 2
    offset_y = (512 - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas

def process_images_in_folder(folder_path, out_base):
    """
    Process all images (png/jpg/jpeg) in a folder and save outputs to subfolders within out_base.
    - Saves full original image to /origin
    - Saves masked RGBA image to /masked
    - Saves binary mask image to /mask
    - Saves resized (512x512) binary mask to /mask_512
    - Saves RGB masked (512x512) image to /masked_rgb and /frames/masked_rgb
    """
    os.makedirs(out_base, exist_ok=True)
    orig_out_base = os.path.join(out_base, "origin")
    masked_out_base = os.path.join(out_base, "masked")
    mask_out_base = os.path.join(out_base, "mask")
    mask512_out_base = os.path.join(out_base, "mask_512")
    masked_rgb_out_base = os.path.join(out_base, "masked_rgb")
    masked_rgb_out_base2 = os.path.join(out_base, "frames", "masked_rgb")
    for subdir in [orig_out_base, masked_out_base, mask_out_base, mask512_out_base, masked_rgb_out_base, masked_rgb_out_base2]:
        os.makedirs(subdir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if len(image_files) == 0:
        return
    bbox = None
    processed_items = []
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image_name = os.path.splitext(os.path.basename(image_file))[0]
        image = Image.open(image_path)
        # Save the original image
        orig_save_path = os.path.join(orig_out_base, image_file)
        image.save(orig_save_path)
        if bbox is None:
            image_masked, mask_np, mask_img, first_bbox = remove_background(image)
            curr_bbox = first_bbox
        else:
            # Only generate mask, apply it to current image to get RGBA black bkg
            if image.mode != 'RGBA':
                image = image.convert('RGB')
                out = rembg.remove(image, session=rembg_session)
            else:
                arr = np.array(image)
                if arr.shape[2] != 4 or np.all(arr[..., 3] == 255):
                    image = image.convert('RGB')
                    out = rembg.remove(image, session=rembg_session)
                else:
                    out = image
            arr = np.array(out)
            alpha = arr[..., 3]
            mask_img = Image.fromarray((alpha > int(0.8 * 255)).astype(np.uint8) * 255, mode='L')

            # Apply mask to out, output RGBA, black where mask==0
            out_np = np.array(out).astype(np.float32) / 255.0
            mask_np = np.array(mask_img).astype(np.float32) / 255.0
            if mask_np.ndim == 2:
                mask_np = mask_np[..., None]
            image_masked_np = out_np.copy()
            image_masked_np[..., :3] = image_masked_np[..., :3] * mask_np
            image_masked_np[..., 3] = mask_np[..., 0]
            image_masked = Image.fromarray((image_masked_np * 255).astype(np.uint8), mode='RGBA')
            curr_bbox = compute_mask_bbox(mask_img)

        bbox = merge_bbox(bbox, curr_bbox)

        # Save masked RGBA image
        masked_save_path = os.path.join(masked_out_base, f'{image_name}_masked.png')
        image_masked.save(masked_save_path)

        # Save mask as image (L)
        mask_save_path = os.path.join(mask_out_base, f'{image_name}_mask.png')
        mask_img.save(mask_save_path)

        # Record data for later global bbox crop + center to 512
        processed_items.append((image_name, mask_img, image_masked, image_path))

    if bbox is None:
        print("No valid masks found; skip 512 outputs.")
        return
    for image_name, mask_img, image_masked, image_path in processed_items:
        mask512 = crop_and_center_to_512(mask_img, bbox, fill_value=0)
        mask512_save_path = os.path.join(mask512_out_base, f'{image_name}_mask_512.png')
        mask512.save(mask512_save_path)

        rgb_np = np.array(image_masked)[..., :3]
        image_masked_rgb = Image.fromarray(rgb_np)
        image_masked_512 = crop_and_center_to_512(image_masked_rgb, bbox, fill_value=(0, 0, 0))
        masked_rgb_save_path = os.path.join(masked_rgb_out_base, f'{image_name}_masked_rgb.png')
        image_masked_512.save(masked_rgb_save_path)
        masked_rgb_save_path2 = os.path.join(masked_rgb_out_base2, f'{image_name}_masked_rgb.png')
        image_masked_512.save(masked_rgb_save_path2)
    
    # Also crop and center the original images to 512x512 using the computed global bbox
    for idx, (image_name, mask_img, image_masked, image_path) in enumerate(processed_items):
        image_file = image_name + ".png"
        if os.path.exists(image_path):
            orig_image = Image.open(image_path)
            # Crop and center to 512
            orig512 = crop_and_center_to_512(orig_image, bbox, fill_value=(0, 0, 0))
            # Save to separate output folder for origin_cropped results
            orig_cropped_save_path = os.path.join(orig_out_base + "_cropped", f'{image_name}_origin_cropped_512.png')
            os.makedirs(os.path.dirname(orig_cropped_save_path), exist_ok=True)
            orig512.save(orig_cropped_save_path)
        else:
            print(f"Warning: original image not found: {image_path}, cannot save cropped origin 512.")

def extract_frames_from_video(video_path, output_dir):
    """
    Extracts frames from a video and saves them to output_dir as PNG images.
    Returns True on success, False if fails.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False

    frame_count = 0
    print(f"Extracting frames from video: {video_path}")

    with tqdm(desc="Extracting frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_path = os.path.join(output_dir, f"{frame_count:06d}.png")
            frame_pil.save(frame_path)
            frame_count += 1
            pbar.update(1)
    cap.release()
    print(f"Extracted {frame_count} frames to {output_dir}")
    return True

def process_recursively(root_dir, out_base):
    """
    Recursively process all images in all subfolders of root_dir (ignores images directly in root_dir).
    Skips subfolders named 'resized_images' or 'masked_images'.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the root dir itself (only process subfolders)
        if os.path.abspath(dirpath) == os.path.abspath(root_dir):
            continue
        # Skip 'resized_images' and 'masked_images' subtrees
        rel_path = os.path.relpath(dirpath, root_dir)
        if rel_path.split(os.sep)[0] == "resized_images" or rel_path.split(os.sep)[0] == "masked_images":
            continue
        out_folder = os.path.join(out_base, rel_path)
        process_images_in_folder(dirpath, out_folder)

def main_split_only(opt):
    """
    Only extract frames from input video and save to output directory.
    No background removal is performed.
    """
    input_path = opt.input
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Error: {input_path} is not a supported video format")
            return

        video_name = Path(input_path).stem
        if opt.output_dir:
            base_dir = opt.output_dir
        else:
            base_dir = os.path.join(os.path.dirname(input_path), f"{video_name}_processed")
        
        # Skip if output directory already exists
        if os.path.exists(base_dir) and os.path.isdir(base_dir):
            print(f"Output directory already exists: {base_dir}")
            print(f"Skipping processing for {input_path}")
            return
        
        # Output structure: base_dir/masked_images/video/origin/
        masked_rgb_dir = os.path.join(base_dir, "masked_images", "video", "origin")
        if not extract_frames_from_video(input_path, masked_rgb_dir):
            return

        # Also save each frame to base_dir/masked_rgb/ for compatibility
        masked_rgb_dir2 = os.path.join(base_dir, "masked_rgb")
        os.makedirs(masked_rgb_dir2, exist_ok=True)
        for f in os.listdir(masked_rgb_dir):
            if f.endswith('.png'):
                src = os.path.join(masked_rgb_dir, f)
                dst = os.path.join(masked_rgb_dir2, f)
                Image.open(src).save(dst)

        masked_rgb_dir3 = os.path.join(base_dir, "frames", "video")
        os.makedirs(masked_rgb_dir3, exist_ok=True)
        for f in os.listdir(masked_rgb_dir):
            if f.endswith('.png'):
                src = os.path.join(masked_rgb_dir, f)
                dst = os.path.join(masked_rgb_dir3, f)
                Image.open(src).save(dst)

        print(f"Video splitting complete.")
        print(f"Frames saved in: {masked_rgb_dir}")
        print(f"Frames also saved in: {masked_rgb_dir3}")
    else:
        print(f"Error: {input_path} is not a video file")

def main(opt):
    input_path = opt.input

    # If split_only is specified, only extract frames from video
    if opt.split_only:
        main_split_only(opt)
        return

    # Check if input is a video file or a directory
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Error: {input_path} is not a supported video format")
            return

        video_name = Path(input_path).stem
        if opt.output_dir:
            base_dir = opt.output_dir
        else:
            base_dir = os.path.join(os.path.dirname(input_path), f"{video_name}_processed")
        
        # Skip if output directory already exists
        if os.path.exists(base_dir) and os.path.isdir(base_dir):
            print(f"Output directory already exists: {base_dir}")
            print(f"Skipping processing for {input_path}")
            return
        
        # Extract frames
        frames_dir = os.path.join(base_dir, "frames")
        if not extract_frames_from_video(input_path, frames_dir):
            return

        # Move extracted frames into a "video" subfolder for processing convention
        out_base = os.path.join(base_dir, "masked_images")
        frames_subdir = os.path.join(frames_dir, "video")
        os.makedirs(frames_subdir, exist_ok=True)
        for f in os.listdir(frames_dir):
            if f.endswith('.png'):
                src = os.path.join(frames_dir, f)
                dst = os.path.join(frames_subdir, f)
                os.rename(src, dst)

        # Also save to base_dir/masked_rgb/ for compatibility after mask processing
        masked_rgb_dir = os.path.join(base_dir, "masked_rgb")
        os.makedirs(masked_rgb_dir, exist_ok=True)

        # Copy processed masked RGB files to masked_rgb_dir (strip "_masked_rgb" suffix)
        import shutil
        masked_rgb_video_dir = os.path.join(out_base, "video", "masked_rgb")
        if os.path.exists(masked_rgb_video_dir):
            for fname in os.listdir(masked_rgb_video_dir):
                if fname.endswith("_masked_rgb.png"):
                    src_path = os.path.join(masked_rgb_video_dir, fname)
                    new_name = fname.replace("_masked_rgb", "")
                    dst_path = os.path.join(masked_rgb_dir, new_name)
                    shutil.copy(src_path, dst_path)
            print(f'copied {masked_rgb_video_dir} to {masked_rgb_dir}')
        else:
            process_recursively(frames_dir, out_base)
            if os.path.exists(masked_rgb_video_dir):
                for fname in os.listdir(masked_rgb_video_dir):
                    if fname.endswith("_masked_rgb.png"):
                        src_path = os.path.join(masked_rgb_video_dir, fname)
                        new_name = fname.replace("_masked_rgb", "")
                        dst_path = os.path.join(masked_rgb_dir, new_name)
                        shutil.copy(src_path, dst_path)
            print(f'copied {masked_rgb_video_dir} to {masked_rgb_dir}')
        print(f"Video processing complete. Output in: {base_dir}")
    else:
        print(f"Error: {input_path} is not a video file")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input video file or directory containing images')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (optional, will auto-generate if not provided)')
    parser.add_argument('--split_only', action='store_true',
                        help='Only extract frames from video without background removal (split_only mode)')
    opt = parser.parse_args()
    opt = edict(vars(opt))
    main(opt)
# python ./utils/rmbg_for_black_bg.py --input /path/to/video.mp4
# python ./utils/rmbg_for_black_bg.py --input /path/to/video.mp4 --split_only