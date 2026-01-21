"""
Simple script to convert images in a directory to a video.
Usage: python images2video.py <image_dir> <output_dir> [--fps FPS]
The output video name will be the basename of image_dir with .mp4 suffix, placed inside output_dir.
"""

import os
import sys
import argparse
import imageio
from PIL import Image
from natsort import natsorted
""" 
python scripts/images2video.py /path/to/your/image_name_folder /path/to/your/video_folder
"""
def images_to_video(image_dir, output_dir, fps=12):
    """
    Convert images in a directory to a video.
    
    Args:
        image_dir: Directory containing images
        output_dir: Directory to save the output video
        fps: Frames per second (default: 12)
    """
    if not os.path.isdir(image_dir):
        print(f"Error: Directory not found: {image_dir}")
        return False

    if not os.path.isdir(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return False

    # Get all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        image_files.extend([f for f in os.listdir(image_dir) if f.endswith(ext)])
    
    if not image_files:
        print(f"Error: No image files found in {image_dir}")
        return False
    
    # Sort images naturally
    image_files = natsorted(image_files)
    
    if len(image_files) < 12:
        print(f"Warning: Found only {len(image_files)} images, will create video with {len(image_files)} frames")

    # Set output path: <output_dir>/<basename(image_dir)>.mp4
    base_name = os.path.basename(os.path.abspath(image_dir.rstrip("/\\")))
    output_path = os.path.join(output_dir, f"{base_name}.mp4")

    print(f"Converting {len(image_files)} images to video: {output_path}")
    print(f"FPS: {fps}")

    # Read images
    frames = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            frames.append(img)
            print(f"  Loaded: {img_file}")
        except Exception as e:
            print(f"  Error loading {img_file}: {e}")
            continue

    if not frames:
        print("Error: No valid images loaded")
        return False

    # Write video
    try:
        imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8)
        print(f"✓ Video saved successfully: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving video: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert images to video")
    parser.add_argument("image_dir", type=str, help="Directory containing images")
    parser.add_argument("output_dir", type=str, help="Directory to save the output video")
    parser.add_argument("--fps", type=int, default=12, help="Frames per second (default: 12)")
    
    args = parser.parse_args()
    
    images_to_video(args.image_dir, args.output_dir, args.fps)

if __name__ == "__main__":
    main()
""" 
python scripts/images2video.py /path/to/your/image_name_folder /path/to/your/video_folder
"""