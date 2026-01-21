# Modified from V2M4
import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import imageio
import torch
import random
import numpy as np
import rembg
from PIL import Image
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from natsort import ns, natsorted
import argparse
import trimesh
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover

def preprocess_image(input: Image.Image, return_rgba=False, return_all_rbga=False) -> Image.Image:
    """
    Preprocess the input image.
    """
    # If the input has an alpha channel, use it directly; otherwise, remove background
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        # Use birefnet-massive instead of u2net for better performance
        rembg_session = rembg.new_session('birefnet-massive')
        output = rembg.remove(input, session=rembg_session)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2

    # Uncropped image
    before_crop = output.resize((518, 518), Image.Resampling.LANCZOS)
    before_crop = np.array(before_crop).astype(np.float32) / 255

    # If the crop region is out of the image, it will be automatically padded with black.
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255

    if return_all_rbga:
        before_crop_rgba = Image.fromarray((before_crop * 255).astype(np.uint8), mode='RGBA')
        
        before_crop = before_crop[:, :, :3] * before_crop[:, :, 3:4]
        before_crop = Image.fromarray((before_crop * 255).astype(np.uint8))

        output = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
        return output, before_crop_rgba, before_crop

    before_crop = before_crop[:, :, :3] * before_crop[:, :, 3:4]
    before_crop = Image.fromarray((before_crop * 255).astype(np.uint8))

    if return_rgba:
        output = Image.fromarray((output * 255).astype(np.uint8), mode='RGBA')
        return output, before_crop

    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output, before_crop

def seed_torch(seed=0):
    print("Seed Fixed!")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description='HunyuanGen Pipeline')
    parser.add_argument('--root', type=str, default='', help='Root directory of the dataset')
    parser.add_argument('--output', type=str, default='', help='Output directory of the results')
    parser.add_argument('--N', type=int, default=1, help='Total number of parallel processes')
    parser.add_argument('--n', type=int, default=0, help='Index of the current process')
    parser.add_argument('--model', type=str, default='Hunyuan', help='Base model', choices=['Hunyuan'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip', type=int, default=256, help='Skip every N frames for large movement of the object (default: 5)')
    parser.add_argument('--max_faces', type=int, default=10000, help='Maximum number of faces for the generated mesh (default: 10000)')
    return parser.parse_args()

def get_folder_size(folder):
    return len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

if __name__ == "__main__":
    args = parse_args()
    root = "/home/hongyuan/obj_motion/Eval/1/viz" if args.root == "" else args.root
    default_output_root = "results_examples"
    print(f"Root viz dir: {root}")

    output_root = args.output if args.output != "" else default_output_root

    processed_dirs = []
    for name in os.listdir(root):
        sub_path = os.path.join(root, name)
        if not os.path.isdir(sub_path):
            continue
        processed_path = os.path.join(sub_path, "masked_rgb")
        animation_path = os.path.join(sub_path, "animation")
        if os.path.isdir(processed_path) and not os.path.isdir(animation_path):
            processed_dirs.append(processed_path)
    print("Found processed masked_rgb folders:")
    print(processed_dirs)
    animations = natsorted(processed_dirs, alg=ns.PATH)

    folder_sizes = [(anim, get_folder_size(anim)) for anim in animations]
    folder_sizes.sort(key=lambda x: x[1], reverse=True)

    assignments = [[] for _ in range(args.N)]
    workload = [0] * args.N

    for folder, size in folder_sizes:
        min_index = workload.index(min(workload))
        assignments[min_index].append(folder)
        workload[min_index] += size

    assigned_animations = assignments[args.n]

    total_imgs = sum(get_folder_size(f) for f in assigned_animations)
    print(f"Process {args.n} assigned {len(assigned_animations)} folders with total images: {total_imgs}")


    # Only support Hunyuan model branch
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
    pipeline_paint = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

    pipeline.model.requires_grad = False
    pipeline.vae.requires_grad = False
    pipeline.conditioner.requires_grad = False
    pipeline_paint.models['delight_model'].pipeline.feature_extractor.requires_grad = False
    pipeline_paint.models['delight_model'].pipeline.text_encoder.requires_grad = False
    pipeline_paint.models['delight_model'].pipeline.unet.requires_grad = False
    pipeline_paint.models['delight_model'].pipeline.vae.requires_grad = False
    pipeline_paint.models['multiview_model'].pipeline.unet.requires_grad = False
    pipeline_paint.models['multiview_model'].pipeline.vae.requires_grad = False
    pipeline_paint.models['multiview_model'].pipeline.text_encoder.requires_grad = False
    pipeline_paint.models['multiview_model'].pipeline.feature_extractor.requires_grad = False

    for animation in assigned_animations:
        source_path = animation
        output_path = source_path if args.output == "" else os.path.join(args.output, source_path.split("/")[-2], 'Hunyuan_Gen_Input')

        print("/n/n ============= Start processing: ", animation, " =============/n")
        os.makedirs(output_path, exist_ok=True)

        seed = args.seed
        seed_torch(seed)
        
        imgs_list = os.listdir(source_path)
        imgs_list = [img for img in imgs_list if not os.path.isdir(source_path + "/" + img)]
        imgs_list = natsorted(imgs_list, alg=ns.PATH)

        outputs_list = []
        base_name_list = []
        extrinsics_list = []
        visual_list = []
        params = None

        for ind, img in enumerate(imgs_list):            
            if ind % args.skip != 0:
                continue

            image = Image.open(source_path + "/" + img)
            base_name = image.filename.split("/")[-1].split(".")[0]

            # Hunyuan only logic
            save_path = output_path + "/" + base_name + "_rmbg.png"
            cropped_image, rmbg_image = preprocess_image(image, return_rgba=True)
            rmbg_image.save(save_path)
            cropped_image.save(save_path.replace(".png", "_cropped.png"))
            import time
            start_time = time.time()
            
            torch.manual_seed(seed)
            mesh = pipeline(image=cropped_image)[0]

            for cleaner in [FloaterRemover(), DegenerateFaceRemover()]:
                mesh = cleaner(mesh)

            mesh = FaceReducer()(mesh, max_facenum=args.max_faces)

            vertices_watertight = mesh.vertices @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            faces_watertight = mesh.faces
            mean_point = mesh.vertices.mean(axis=0)
            vertices_watertight = (vertices_watertight - mean_point) * 0.5 + mean_point
            mesh_watertight = trimesh.Trimesh(vertices_watertight, faces_watertight, process=False)
            mesh_watertight.export(output_path + "/" + base_name + "_hunyuan_original_watertight.glb")
            print("mesh_watertight exported:", output_path + "/" + base_name + "_hunyuan_original_watertight.glb")

            mesh, vmapping = pipeline_paint(mesh, image=cropped_image)
            # Save vmapping for later use
            vmapping_path = output_path + "/" + base_name + "_vmapping.npy"
            np.save(vmapping_path, vmapping)
            print(f"vmapping saved to: {vmapping_path}")
            
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"Hunyuan generation time: {generation_time:.4f} seconds")
            # Save the original Hunyuan mesh before further processing
            mesh_original = mesh.copy()
            mesh_original.export(output_path + "/" + base_name + "_hunyuan_original_wo_remap.glb")
            print("mesh_original exported:", output_path + "/" + base_name + "_hunyuan_original_wo_remap.glb")