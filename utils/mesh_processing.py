import os
import numpy as np
import torch
import subprocess
import tempfile
from PIL import Image
import trimesh


def convert_fbx_to_glb_with_blender(fbx_path, output_glb_path):
    """
    Convert FBX file to GLB using Blender.
    
    Args:
        fbx_path: Input FBX file path
        output_glb_path: Output GLB file path
    """
    python_script = f"""
import bpy
import bmesh

def reset_scene():
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

if __name__ == "__main__":
    reset_scene()
    bpy.ops.import_scene.fbx(filepath='{fbx_path}')
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    for obj in mesh_objs:
        for mat_slot in obj.data.materials:
            if not mat_slot.use_nodes:
                continue
            nodes = mat_slot.node_tree.nodes
            img_nodes = [n for n in nodes if n.type == 'TEX_IMAGE']
            if img_nodes:
                base_img = img_nodes[0].image
                new_mat = bpy.data.materials.new(name="MergedMat")
                new_mat.use_nodes = True
                new_nodes = new_mat.node_tree.nodes
                new_nodes.clear()
                bsdf = new_nodes.new('ShaderNodeBsdfPrincipled')
                tex_node = new_nodes.new('ShaderNodeTexImage')
                tex_node.image = base_img
                output = new_nodes.new('ShaderNodeOutputMaterial')
                new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
                new_mat.node_tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
                obj.data.materials[0] = new_mat

    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH':
            continue
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        if obj.parent:
            bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        for mod in obj.modifiers:
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except:
                pass
        obj.select_set(False)

    bpy.ops.export_scene.gltf(
        filepath='{output_glb_path}',
        export_format='GLB',
        export_materials='EXPORT',
        use_selection=False,
        export_yup=True,
        export_apply=True,
    )
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        temp_script = f.name
        f.write(python_script)
    
    try:
        subprocess.run(
            f'python "{temp_script}"',
            shell=True,
            check=True
        )
        print(f"✓ Successfully converted FBX to GLB: {output_glb_path}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Blender conversion failed: {e}")
        raise
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def barycentric_coords(tri, p):
    """
    Calculate barycentric coordinates of point p in triangle tri.
    
    Args:
        tri: (3, 3) triangle vertex coordinates
        p: (3,) point coordinates
    
    Returns:
        (3,) barycentric coordinates [u, v, w] where u+v+w=1
    """
    v0 = tri[1] - tri[0]
    v1 = tri[2] - tri[0]
    v2 = p - tri[0]
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return np.array([1/3, 1/3, 1/3])
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])


def sample_pointcloud_with_albedo(mesh, num=200000):
    """
    Sample point cloud from mesh with positions, normals, and colors using barycentric interpolation.
    
    Args:
        mesh: trimesh.Trimesh object
        num: Number of points to sample
    
    Returns:
        points: (num, 3) point cloud coordinates
        normals: (num, 3) normals
        colors: (num, 3) RGB colors in range [0, 1]
    """
    points, face_idx = mesh.sample(num, return_index=True)
    normals = mesh.face_normals[face_idx]
    colors = None

    # Try vertex colors first
    if hasattr(mesh.visual, "vertex_colors") and len(mesh.visual.vertex_colors) == len(mesh.vertices):
        vertex_colors = mesh.visual.vertex_colors
        if vertex_colors.ndim == 2 and vertex_colors.shape[1] >= 3:
            vertex_colors = vertex_colors[:, :3] / 255.0
            tri_colors = vertex_colors[mesh.faces[face_idx]]
            colors = tri_colors.mean(axis=1)

    # Try texture if no vertex colors
    if colors is None and hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "baseColorTexture") and mesh.visual.material.baseColorTexture:
        tex = mesh.visual.material.baseColorTexture
        image = tex.image if hasattr(tex, "image") else tex
        
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode not in ["RGB", "RGBA"]:
            image = image.convert("RGB")
        tex_np = np.array(image).astype(np.float32) / 255.0
        if tex_np.ndim == 2:
            tex_np = np.stack([tex_np] * 3, axis=-1)
        elif tex_np.shape[2] > 3:
            tex_np = tex_np[:, :, :3]
        tex_w, tex_h = image.size

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uv = mesh.visual.uv % 1.0
            tri_uvs = uv[mesh.faces[face_idx]]
            colors = np.zeros((num, 3), dtype=np.float32)
            for i in range(num):
                tri = mesh.triangles[face_idx[i]]
                p = points[i]
                w = barycentric_coords(tri, p)
                uv_sample = w[0] * tri_uvs[i, 0] + w[1] * tri_uvs[i, 1] + w[2] * tri_uvs[i, 2]
                u = int(np.clip(uv_sample[0] * tex_w, 0, tex_w - 1))
                v = int(np.clip((1 - uv_sample[1]) * tex_h, 0, tex_h - 1))
                colors[i] = tex_np[v, u, :]

    if colors is None:
        colors = np.full((num, 3), 0.5, dtype=np.float32)

    points = torch.from_numpy(points.astype(np.float32))
    normals = torch.from_numpy(normals.astype(np.float32))
    colors = torch.from_numpy(colors.astype(np.float32))

    return points, normals, colors


def normalize_mesh(mesh, return_params=False):
    """
    Normalize mesh to fit in a unit cube centered at origin.
    
    Args:
        mesh: trimesh.Trimesh object
        return_params: If True, return normalization parameters
    
    Returns:
        mesh: Normalized mesh (or vertices if return_params=True)
        center: Center point (only if return_params=True)
        scale: Scale factor (only if return_params=True)
    """
    vertices = mesh.vertices.astype(np.float32)
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    vertices = vertices - center
    v_max = np.abs(vertices).max()
    scale = 2 * (v_max + 1e-8)
    vertices = vertices / scale
    
    if return_params:
        return vertices, center, scale
    else:
        mesh.vertices = vertices
        return mesh

