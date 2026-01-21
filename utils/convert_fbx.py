"""
Create FBX files directly using FBX SDK, keeping original vertices and faces unchanged
Also supports independent UV indices (ByPolygonVertex mapping mode)
"""
import os
import sys
import numpy as np
from pathlib import Path
import trimesh
import subprocess
import glob

def load_obj_with_independent_uv(obj_path):
    """
    Load an OBJ file with independent vertex and UV indices (v/vt format).
    
    Args:
        obj_path: Path to the OBJ file
        
    Returns:
        vertices: numpy array of vertex positions (N, 3)
        faces: numpy array of face vertex indices (F, 3)
        uvs: numpy array of UV coordinates (M, 2)
        uv_indices: numpy array of face UV indices (F, 3)
        texture_path: path to the texture image (if found in MTL)
    """
    vertices = []
    uvs = []
    faces = []
    uv_indices = []
    texture_path = None
    mtl_file = None
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            parts = line.split()
            if not parts:
                continue
                
            if parts[0] == 'v':
                # Vertex position
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt':
                # UV coordinate
                uvs.append([float(parts[1]), float(parts[2])])
            elif parts[0] == 'f':
                # Face with format v/vt or v/vt/vn
                face_verts = []
                face_uvs = []
                for i in range(1, len(parts)):
                    indices = parts[i].split('/')
                    # Vertex index (1-based in OBJ, convert to 0-based)
                    face_verts.append(int(indices[0]) - 1)
                    # UV index (1-based in OBJ, convert to 0-based)
                    if len(indices) > 1 and indices[1]:
                        face_uvs.append(int(indices[1]) - 1)
                
                # Only add triangular faces
                if len(face_verts) == 3:
                    faces.append(face_verts)
                    if len(face_uvs) == 3:
                        uv_indices.append(face_uvs)
            elif parts[0] == 'mtllib':
                # Material library file
                mtl_file = parts[1]
    
    # Try to load texture from MTL file
    if mtl_file:
        mtl_path = os.path.join(os.path.dirname(obj_path), mtl_file)
        if os.path.exists(mtl_path):
            with open(mtl_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('map_Kd'):
                        texture_filename = line.split()[1]
                        texture_path = os.path.join(os.path.dirname(obj_path), texture_filename)
                        break
    
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)
    uvs = np.array(uvs, dtype=np.float32) if uvs else None
    uv_indices = np.array(uv_indices, dtype=np.int32) if uv_indices else None
    
    return vertices, faces, uvs, uv_indices, texture_path


def create_fbx_with_independent_uv_via_sdk(obj_path, output_fbx_path):
    """
    Create FBX using Blender's low-level API via Python script
    Ensures vertex and face structure remains completely unchanged
    """
    
    # First load OBJ data
    vertices, faces, uvs, uv_indices, texture_path = load_obj_with_independent_uv(obj_path)
    
    print("=== Original OBJ Data ===")
    print(f"Vertex count: {len(vertices)}")
    print(f"Face count: {len(faces)}")  
    print(f"UV count: {len(uvs)}")
    print(f"UV index shape: {uv_indices.shape}")
    print(f"Texture path: {texture_path}")
    
    # Create Blender script, manually build mesh to preserve topology
    blender_script = f"""
import bpy
import bmesh
import numpy as np
import os

# Clear scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Load data (from numpy arrays)
vertices = {vertices.tolist()}
faces = {faces.tolist()}
uvs = {uvs.tolist()}
uv_indices = {uv_indices.tolist()}
texture_path = "{texture_path}"

print(f"Blender: Loaded {{len(vertices)}} vertices, {{len(faces)}} faces, {{len(uvs)}} UVs")
print(f"Blender: Texture path: {{texture_path}}")

# Create new mesh data
mesh_data = bpy.data.meshes.new("PreservedMesh")

# Use from_pydata to create mesh (this preserves original vertex indices)
mesh_data.from_pydata(vertices, [], faces)
mesh_data.update()

# Create object
obj = bpy.data.objects.new("PreservedObject", mesh_data)
bpy.context.collection.objects.link(obj)
bpy.context.view_layer.objects.active = obj

# Ensure mesh data is updated
mesh_data.update()
mesh_data.calc_loop_triangles()

# Create UV layer
uv_layer = mesh_data.uv_layers.new(name="UVMap")

# Set UV coordinates (per-loop, i.e., per-face-vertex)
# Blender's loop corresponds to face corner
loop_index = 0
for face_idx, face in enumerate(faces):
    for corner_idx in range(len(face)):
        # Get UV index for this corner
        uv_idx = uv_indices[face_idx][corner_idx]
        
        # Set UV (Blender's loops are sequential)
        uv_layer.data[loop_index].uv = uvs[uv_idx]
        loop_index += 1

print(f"Blender: Set {{loop_index}} UV loops")

# Create material and load texture
if texture_path and os.path.exists(texture_path):
    print(f"Blender: Loading texture {{texture_path}}")
    
    # Create material
    mat = bpy.data.materials.new(name="TexturedMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Create image texture node
    tex_image = nodes.new(type='ShaderNodeTexImage')
    tex_image.location = (-300, 0)
    
    # Load texture image
    try:
        img = bpy.data.images.load(texture_path)
        tex_image.image = img
        print(f"Blender: Successfully loaded texture image")
    except Exception as e:
        print(f"Blender: Failed to load texture: {{e}}")
    
    # Connect texture to Base Color
    links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])
    
    # Create output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    print(f"Blender: Material created and assigned")
else:
    print(f"Blender: Texture file not found or path is empty")

# Verify vertex and face counts
print(f"Blender: Mesh has {{len(mesh_data.vertices)}} vertices, {{len(mesh_data.polygons)}} faces")
assert len(mesh_data.vertices) == len(vertices), "Vertex count mismatch!"
assert len(mesh_data.polygons) == len(faces), "Face count mismatch!"

# Export as FBX
bpy.ops.export_scene.fbx(
    filepath="{output_fbx_path}",
    use_selection=False,
    use_mesh_modifiers=False,  # Don't apply modifiers
    mesh_smooth_type='OFF',    # No smoothing
    use_tspace=True,           # Use tangent space
    path_mode='COPY',          # Copy textures
    embed_textures=True,       # Embed textures
    bake_space_transform=False # Don't bake transform
)

print(f"Blender: Successfully exported to {{'{output_fbx_path}'}}")
print(f"Blender: Final vertex count: {{len(mesh_data.vertices)}}")
print(f"Blender: Final face count: {{len(mesh_data.polygons)}}")
    """
    
    # Save Blender script
    script_path = obj_path.replace('.obj', '_fbx_export.py')
    with open(script_path, 'w') as f:
        f.write(blender_script)
    
    print(f"\nSaved Blender script: {script_path}")
    
    # Run Blender
    print("\nRunning Blender to export FBX...")
    try: #modify the path to the blender executable HERE
        result = subprocess.run(
            ['/home/hongyuan/obj_motion/Data_process/bpy-renderer/blender-4.0.0-linux-x64/blender', '--background', '--python', script_path],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Find key information in output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            print(line)
        
        os.remove(script_path)
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("✗ Blender not found, please ensure it is installed")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def export_mesh_with_texture(mesh, output_path):
    """
    Export mesh with texture as OBJ format
    
    Args:
        mesh: trimesh.Trimesh object with visual attributes
        output_path: Path to save OBJ file (without extension)
    """
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Get mesh components
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Get UV coordinates and indices
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uvs = mesh.visual.uv
        if hasattr(mesh.visual, 'uv_indices') and mesh.visual.uv_indices is not None:
            uv_indices = mesh.visual.uv_indices
        else:
            uv_indices = faces
    else:
        print("Warning: No UV coordinates found")
        uvs = None
        uv_indices = None
    
    # Write OBJ file
    obj_path = output_path if output_path.endswith('.obj') else output_path + '.obj'
    mtl_path = obj_path.replace('.obj', '.mtl')
    texture_path = obj_path.replace('.obj', '.png')
    
    with open(obj_path, 'w') as f:
        # Write MTL reference
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        
        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        # Write UV coordinates
        if uvs is not None:
            for uv in uvs:
                f.write(f"vt {uv[0]} {uv[1]}\n")

        f.write("usemtl material_0\n")
        
        # Write faces
        for i, face in enumerate(faces):
            if uv_indices is not None:
                uv_face = uv_indices[i]
                f.write(f"f {face[0]+1}/{uv_face[0]+1} {face[1]+1}/{uv_face[1]+1} {face[2]+1}/{uv_face[2]+1}\n")
            else:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    # Write MTL file
    with open(mtl_path, 'w') as f:
        f.write("newmtl material_0\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd material_0.png\n")
    
    print(f"Exported mesh to: {obj_path}")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Faces: {len(faces)}")
    if uvs is not None:
        print(f"  UV coordinates: {len(uvs)}")
    
    return obj_path


def process_mesh_conversion(base_path, non_water_filename, water_filename, vmapping_filename):
    """
    Main function for processing mesh conversion
    
    Args:
        base_path: Base path
        non_water_filename: Non-watertight mesh filename (e.g., 000000_hunyuan_original_wo_remap.glb)
        water_filename: Watertight mesh filename (e.g., 000000_hunyuan_original_watertight.glb)
        vmapping_filename: Vmapping filename (e.g., 000000_vmapping.npy)
    
    Returns:
        success: Whether conversion was successful
    """
    # Build full paths
    non_water_path = os.path.join(base_path, non_water_filename)
    water_path = os.path.join(base_path, water_filename)
    vmapping_path = os.path.join(base_path, vmapping_filename)
    
    # Check if files exist
    if not os.path.exists(non_water_path):
        print(f"Error: Non-watertight mesh file not found {non_water_path}")
        return False
    if not os.path.exists(water_path):
        print(f"Error: Watertight mesh file not found {water_path}")
        return False
    if not os.path.exists(vmapping_path):
        print(f"Error: Vmapping file not found {vmapping_path}")
        return False
    
    print(f"=== Starting Mesh Conversion ===")
    print(f"Base path: {base_path}")
    print(f"Non-watertight mesh: {non_water_filename}")
    print(f"Watertight mesh: {water_filename}")
    print(f"Vmapping: {vmapping_filename}")
    
    # Load vmapping file
    vmapping = np.load(vmapping_path)
    print(f"Vmapping shape: {vmapping.shape}")
    
    # Load watertight mesh (before xatlas)
    original_mesh = trimesh.load_mesh(water_path, process=False)
    print(f"Original watertight mesh: {len(original_mesh.vertices)} vertices, {len(original_mesh.faces)} faces")
    
    # Load non-watertight mesh (after xatlas, contains UV)
    unwrapped_mesh = trimesh.load_mesh(non_water_path, process=False)
    print(f"Unwrapped mesh: {len(unwrapped_mesh.vertices)} vertices, {len(unwrapped_mesh.faces)} faces")
    # Export the unwrapped mesh for comparison
    unwrapped_mesh_export_path = os.path.join(base_path, f"{os.path.splitext(non_water_filename)[0]}_unwrapped.obj")
    unwrapped_mesh.export(unwrapped_mesh_export_path)
    print(f"Unwrapped mesh exported: {unwrapped_mesh_export_path}")
    # Use vmapping to map unwrapped mesh back to original mesh structure
    # Create new mesh with original vertices but containing unwrapped faces
    remapped_mesh = trimesh.Trimesh(
        vertices=original_mesh.vertices,
        faces=vmapping[unwrapped_mesh.faces],
        process=False
    )
    
    # Copy texture and UV information
    remapped_mesh.visual.material = unwrapped_mesh.visual.material
    remapped_mesh.visual.uv = unwrapped_mesh.visual.uv
    remapped_mesh.visual.uv_indices = unwrapped_mesh.faces
    
    print(f"Remapped mesh: {len(remapped_mesh.vertices)} vertices, {len(remapped_mesh.faces)} faces")
    
    # Extract base filename (remove extension)
    base_filename = os.path.splitext(non_water_filename)[0]
    
    # Export intermediate OBJ file (in same directory)
    obj_output_path = os.path.join(base_path, f"{base_filename}_converted")
    obj_path = export_mesh_with_texture(remapped_mesh, obj_output_path)
    
    print(f"\n=== Intermediate OBJ File Generated ===")
    print(f"OBJ path: {obj_path}")
    
    # Convert to FBX
    fbx_output_path = os.path.join(base_path, f"{base_filename}_converted.fbx")
    print(f"\n=== Starting FBX Conversion ===")
    print(f"FBX output path: {fbx_output_path}")
    
    success = create_fbx_with_independent_uv_via_sdk(obj_path, fbx_output_path)
    
    if success:
        print("\n✓ Conversion completed successfully!")
        print(f"Output files:")
        print(f"  - OBJ: {obj_path}")
        print(f"  - FBX: {fbx_output_path}")
    else:
        print("\n✗ Conversion failed!")
    
    return success


def main():
    """Main function - example usage"""
    
    # If arguments passed from command line
    if len(sys.argv) == 2:
        base_path = sys.argv[1]
        print(f"base_path: {base_path}")
        # Extract filename prefix from base_path
        # Find matching files
        non_water_pattern = os.path.join(base_path, "*hunyuan_original_wo_remap.glb")
        water_pattern = os.path.join(base_path, "*_hunyuan_original_watertight.glb")
        vmapping_pattern = os.path.join(base_path, "*vmapping.npy")
        
        non_water_files = glob.glob(non_water_pattern)
        water_files = glob.glob(water_pattern)
        vmapping_files = glob.glob(vmapping_pattern)
        
        if not non_water_files:
            print(f"Error: No matching files found: {non_water_pattern}")
            sys.exit(1)
        if not water_files:
            print(f"Error: No matching files found: {water_pattern}")
            sys.exit(1)
        if not vmapping_files:
            print(f"Error: No matching files found: {vmapping_pattern}")
            sys.exit(1)
            
        non_water_filename = os.path.basename(non_water_files[0])
        water_filename = os.path.basename(water_files[0])
        vmapping_filename = os.path.basename(vmapping_files[0])
    else:
        # Default parameters (example)
        base_path = "./example/wild_processed/Hunyuan_Gen_Input"
        non_water_filename = "000000_hunyuan_original_wo_remap.glb"
        water_filename = "000000_hunyuan_original_watertight.glb"
        vmapping_filename = "000000_vmapping.npy"
        
        print("Usage:")
        print("  python convert_fbx.py <base_path> <non_water_filename> <water_filename> <vmapping_filename>")
        print("\nExample:")
        print(f"  python convert_fbx.py {base_path} {non_water_filename} {water_filename} {vmapping_filename}")
        print("\nUsing default parameters...\n")
    
    # Execute conversion
    success = process_mesh_conversion(base_path, non_water_filename, water_filename, vmapping_filename)
    
    return success

if __name__ == "__main__":
    main()
