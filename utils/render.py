import bpy
import bmesh
import os
import numpy as np
import torch
import math
import mathutils
from mathutils import Vector
import imageio
import imageio.v3 as iio
import glob

def clear_scene():
    """Clear the current scene."""
    bpy.ops.wm.read_homefile(use_empty=True)
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

def import_glb(filepath):
    """Import GLB/GLTF file and return mesh objects."""
    bpy.ops.import_scene.gltf(filepath=filepath)
    mesh_objects = [obj for obj in bpy.context.scene.objects if (obj.type == 'MESH' and obj.name != 'Cube' and obj.name != 'Icosphere')]
    if not mesh_objects:
        raise ValueError("No mesh objects found")
    for obj in mesh_objects:
        if obj.animation_data:
            obj.animation_data_clear()
        if obj.data.shape_keys:
            obj.shape_key_clear()
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"Loaded {len(mesh_objects)} mesh objects")
    for mesh_obj in mesh_objects:
        print(f"Mesh: {mesh_obj.name}, Vertices: {len(mesh_obj.data.vertices)}")
    return mesh_objects

def import_fbx(filepath):
    """Import FBX file and align to trimesh coordinate system."""
    bpy.ops.import_scene.fbx(filepath=filepath)
    mesh_objects = [obj for obj in bpy.context.scene.objects if (obj.type == 'MESH' and obj.name != 'Cube' and obj.name != 'Icosphere')]
    if not mesh_objects:
        raise ValueError("No mesh objects found")
    for obj in mesh_objects:
        if obj.animation_data:
            obj.animation_data_clear()
        if obj.data.shape_keys:
            obj.shape_key_clear()
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"Loaded {len(mesh_objects)} mesh objects from FBX")
    for mesh_obj in mesh_objects:
        print(f"Mesh: {mesh_obj.name}, Vertices: {len(mesh_obj.data.vertices)}")
    return mesh_objects

def get_mesh_vertices(mesh_obj):
    """Extract vertex coordinates from a mesh object."""
    vertices = []
    for vertex in mesh_obj.data.vertices:
        vertices.append([vertex.co[0], vertex.co[1], vertex.co[2]])
    return np.array(vertices)

def get_all_vertices(mesh_objects):
    """Extract vertex coordinates from multiple mesh objects."""
    all_vertices = []
    for mesh_obj in mesh_objects:
        vertices = get_mesh_vertices(mesh_obj)
        all_vertices.append(torch.from_numpy(vertices))
    return all_vertices

def get_mesh_faces(mesh_obj):
    """Extract face indices from a mesh object."""
    faces = []
    for face in mesh_obj.data.polygons:
        faces.append(list(face.vertices))
    return np.array(faces)

def get_all_faces(mesh_objects):
    """Extract face indices from multiple mesh objects, adjusting vertex indices for merged meshes."""
    all_faces = []
    vertex_count = 0
    for mesh_obj in mesh_objects:
        faces = get_mesh_faces(mesh_obj)
        if len(all_faces) > 0:
            faces = faces + vertex_count
        all_faces.append(torch.from_numpy(faces))
        vertex_count += len(mesh_obj.data.vertices)
    return all_faces

def move_vertices_with_trajectory(mesh_obj, frame, trajectories):
    """Move mesh vertices according to trajectory data for a specific frame."""
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    if mesh_obj.data.shape_keys is None:
        basis = mesh_obj.shape_key_add(name='Basis')
        basis.interpolation = 'KEY_LINEAR'  
    
    bpy.context.scene.frame_set(frame)
    shape_key_name = f"Frame_{frame}"
    shape_key = mesh_obj.data.shape_keys.key_blocks.get(shape_key_name)
    if not shape_key:
        shape_key = mesh_obj.shape_key_add(name=shape_key_name)
        shape_key.interpolation = 'KEY_LINEAR'
    if hasattr(trajectories[frame], 'numpy'):
        positions = trajectories[frame].numpy()
    else:
        positions = trajectories[frame]
    for idx, pos in enumerate(positions):
        shape_key.data[idx].co = Vector(pos)
    
    shape_key.value = 1.0
    shape_key.keyframe_insert(data_path="value", frame=frame)
    if frame > 0:
        shape_key.value = 0.0
        shape_key.keyframe_insert(data_path="value", frame=frame - 1)

def drive_mesh_with_trajs_frames(mesh_objects, trajs, output_dir, azi=0.0, ele=0.0, export_format="fbx"):
    """Drive mesh animation with trajectory data and optionally export to various formats."""
    try:
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("Setting up animations for all meshes...")
        for i in range(len(mesh_objects)):
            mesh_obj = mesh_objects[i]
            traj = trajs[i]
            num_frames = traj.shape[0]
            print(traj.shape)
            print(traj.max(), traj.min())
            for frame in range(num_frames):
                move_vertices_with_trajectory(mesh_obj, frame, traj)
            # Insert all keyframes first
            for frame in range(num_frames):
                for shape_key in mesh_obj.data.shape_keys.key_blocks[1:]:
                    shape_key.value = 0
                    shape_key.keyframe_insert("value", frame=frame)
                
                current_shape_key = mesh_obj.data.shape_keys.key_blocks[f"Frame_{frame}"]
                current_shape_key.value = 1
                current_shape_key.keyframe_insert("value", frame=frame)
            
            # Set interpolation to CONSTANT for all keyframes at once (AFTER all keyframes are inserted)
            if mesh_obj.data.shape_keys.animation_data and mesh_obj.data.shape_keys.animation_data.action:
                fcurves = mesh_obj.data.shape_keys.animation_data.action.fcurves
                for fc in fcurves:
                    for kf in fc.keyframe_points:
                        kf.interpolation = 'CONSTANT'
            
        print("Setting up scene...")
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = trajs.shape[1] - 1
        if export_format != "none":
            print(f"Exporting animated mesh to {export_format.upper()} format...")
            bpy.ops.object.select_all(action='DESELECT')
            for obj in mesh_objects:
                obj.select_set(True)
            filename = os.path.basename(output_dir)
            export_path = os.path.join(os.path.dirname(output_dir), f"{filename}.{export_format}")
            if export_format == "abc":
                bpy.ops.wm.alembic_export(
                    filepath=export_path,
                    start=bpy.context.scene.frame_start,
                    end=bpy.context.scene.frame_end,
                    selected=True,
                    uvs=True,           
                    face_sets=True      
                )
            elif export_format == "fbx":
                # Optimized FBX export for shape key animations
                bpy.ops.export_scene.fbx(
                    filepath=export_path,
                    use_selection=True,
                    bake_anim=True,
                    bake_anim_use_all_bones=False,
                    bake_anim_use_nla_strips=False,
                    bake_anim_use_all_actions=False,
                    bake_anim_force_startend_keying=False,  # Disabled: not needed for shape keys
                    bake_anim_step=1.0,  # Set to 1.0 to bake only existing keyframes
                    bake_anim_simplify_factor=0.0,  # Disable simplification for speed
                    add_leaf_bones=False,
                    path_mode='COPY',
                    embed_textures=True,
                    use_mesh_modifiers=True,
                    use_mesh_edges=True,
                    use_tspace=True,
                    use_custom_props=True,
                    use_active_collection=False,
                )
            elif export_format == "glb":
                bpy.ops.export_scene.gltf(
                    filepath=export_path,
                    use_selection=True,
                    export_format='GLB',
                    export_animations=True,
                    export_frame_range=True,
                )
            print(f"Successfully exported animated mesh to: {export_path}")
            bpy.ops.object.select_all(action='DESELECT')
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

def move_vertices_with_trajectory_gt(mesh_obj, frame, trajectories):
    """Move mesh vertices according to ground truth trajectory data for a specific frame."""
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='OBJECT')
    if mesh_obj.data.shape_keys is None:
        basis = mesh_obj.shape_key_add(name='Basis')
        basis.interpolation = 'KEY_LINEAR'  
    
    shape_key_name = f"Frame_{frame}"
    shape_key = mesh_obj.data.shape_keys.key_blocks.get(shape_key_name)
    if not shape_key:
        shape_key = mesh_obj.shape_key_add(name=shape_key_name)
        shape_key.interpolation = 'KEY_LINEAR'
    if hasattr(trajectories[frame], 'numpy'):
        positions = trajectories[frame].numpy()
    else:
        positions = trajectories[frame]
    for idx, pos in enumerate(positions):
        shape_key.data[idx].co = Vector(pos)

def drive_mesh_with_trajs_frames_gt(mesh_objects, trajs, output_dir, azi=0.0, ele=0.0, export_format="glb"):
    """
    Drive mesh animation by merging multiple meshes into one and using shape keys.
    This approach is most compatible with GLB export as it only requires handling a single object.
    
    Args:
        mesh_objects: List of Blender mesh objects
        trajs: Trajectory data with shape (1, T, N, 3), where N is the total number of vertices
        output_dir: Output directory
        export_format: Export format ('glb', 'fbx', 'abc', 'none')
    """
    try:
        print(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print("Setting up animations for all meshes (V2 - Merged Mesh with Shape Keys)...")
        
        vertex_counts = []
        for mesh_obj in mesh_objects:
            vertex_counts.append(len(mesh_obj.data.vertices))
        
        print(f"Mesh objects count: {len(mesh_objects)}")
        print(f"Vertex counts per mesh: {vertex_counts}")
        print(f"Total vertices: {sum(vertex_counts)}")
        print(f"Trajectories shape: {trajs.shape}")
        
        total_vertices = sum(vertex_counts)
        if trajs.shape[2] != total_vertices:
            print(f"Warning: Total vertices ({total_vertices}) does not match trajectory points ({trajs.shape[2]})")
        
        num_frames = trajs.shape[1]
        trajs = trajs[0]
        
        print("Merging all meshes into one...")
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_objects[0]
        
        bpy.ops.object.join()
        merged_mesh = bpy.context.active_object
        merged_mesh.name = "MergedAnimatedMesh"
        print(f"Merged mesh created: {merged_mesh.name} with {len(merged_mesh.data.vertices)} vertices")
        
        print("Creating shape keys for merged mesh...")
        bpy.ops.object.mode_set(mode='OBJECT')
        
        basis = merged_mesh.shape_key_add(name='Basis')
        basis.interpolation = 'KEY_LINEAR'
        
        for frame_idx in range(num_frames):
            shape_key_name = f"Frame_{frame_idx}"
            shape_key = merged_mesh.shape_key_add(name=shape_key_name)
            shape_key.interpolation = 'KEY_LINEAR'
            
            if hasattr(trajs[frame_idx], 'numpy'):
                positions = trajs[frame_idx].numpy()
            else:
                positions = trajs[frame_idx]
            
            for vert_idx, pos in enumerate(positions):
                shape_key.data[vert_idx].co = Vector(pos)
        
        print(f"Created {num_frames} shape keys")
        
        print("Setting up shape key animation...")
        for frame_idx in range(num_frames):
            for shape_key in merged_mesh.data.shape_keys.key_blocks[1:]:
                shape_key.value = 0
                shape_key.keyframe_insert("value", frame=frame_idx)
            
            current_shape_key = merged_mesh.data.shape_keys.key_blocks[f"Frame_{frame_idx}"]
            current_shape_key.value = 1
            current_shape_key.keyframe_insert("value", frame=frame_idx)
        
        if merged_mesh.data.shape_keys.animation_data and merged_mesh.data.shape_keys.animation_data.action:
            fcurves = merged_mesh.data.shape_keys.animation_data.action.fcurves
            for fc in fcurves:
                for kf in fc.keyframe_points:
                    kf.interpolation = 'CONSTANT'
        
        print("Shape key animation setup complete")
        
        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = num_frames - 1
        
        print("Setting up scene...")
        
        if export_format != "none":
            print(f"Exporting animated mesh to {export_format.upper()} format...")
            bpy.ops.object.select_all(action='DESELECT')
            merged_mesh.select_set(True)
            filename = os.path.basename(output_dir)
            export_path = os.path.join(os.path.dirname(output_dir), f"{filename}.{export_format}")
            
            if export_format == "abc":
                bpy.ops.wm.alembic_export(
                    filepath=export_path,
                    start=bpy.context.scene.frame_start,
                    end=bpy.context.scene.frame_end,
                    selected=True,
                    uvs=True,
                    face_sets=True
                )
            elif export_format == "glb":
                bpy.ops.export_scene.gltf(
                    filepath=export_path,
                    use_selection=True,
                    export_format='GLB',
                    export_animations=True,
                    export_frame_range=True,
                    export_morph=True,
                    export_morph_animation=True,
                )
            
            print(f"Successfully exported animated mesh to: {export_path}")
            bpy.ops.object.select_all(action='DESELECT')
        
        print("Animation setup complete!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
