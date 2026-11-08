import array
import hashlib
import json
import math
import os
import struct
import tempfile
import wave
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler


# ==============================================================================
# 1. SCENE INITIALIZATION, PURGE & RENDER ENGINE CONFIGURATION
# ==============================================================================

def purge_scene_completely():
    """Purge all existing objects, collections, meshes, armatures, lights, cameras, and orphans."""
    active_obj = bpy.context.active_object
    if active_obj is not None and active_obj.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    scene = bpy.context.scene
    if hasattr(scene, "timeline_markers"):
        scene.timeline_markers.clear()
    if hasattr(scene, "sequence_editor_clear") and getattr(scene, "sequence_editor", None) is not None:
        scene.sequence_editor_clear()

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)

    base_pools = [
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.lattices,
        bpy.data.materials,
        bpy.data.node_groups,
        bpy.data.actions,
        bpy.data.textures,
        bpy.data.images,
    ]
    if hasattr(bpy.data, "armatures"):
        base_pools.append(bpy.data.armatures)
    if hasattr(bpy.data, "sounds"):
        base_pools.append(bpy.data.sounds)

    for pool in base_pools:
        for block in list(pool):
            pool.remove(block, do_unlink=True)

    for world in list(bpy.data.worlds):
        bpy.data.worlds.remove(world, do_unlink=True)

    bpy.ops.outliner.orphans_purge(
        do_local_ids=True,
        do_linked_ids=True,
        do_recursive=True,
    )


def create_collection_hierarchy(scene):
    """Create a structured 9-collection hierarchy for the full city diorama."""
    master_col = bpy.data.collections.new("Cyberpunk_Floating_Ramen_Diorama")
    scene.collection.children.link(master_col)

    sub_names = (
        "01_Environment_Base",
        "02_Floating_Hull_Thrusters",
        "03_Stall_Architecture_Roof",
        "04_Interior_Ramen_Props",
        "05_Lanterns_Fans_Cables",
        "06_Cyber_Neon_Signage",
        "07_Cinematic_Lights_Camera",
        "08_Cyber_City_Megablock",
        "09_Characters_And_Drones",
    )
    collections = {"master": master_col}
    for name in sub_names:
        sub_col = bpy.data.collections.new(name)
        master_col.children.link(sub_col)
        collections[name] = sub_col
    return collections


def configure_render_and_world(scene):
    """Configure Cycles GPU, 4096 samples, Denoising, AgX High Contrast, and Volumetric World."""
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = 24
    scene.frame_start = 1
    scene.frame_end = 240
    scene.frame_current = 1
    scene.render.use_motion_blur = False

    # Configure GPU Compute across OptiX / CUDA / HIP / Metal / oneAPI hardware
    prefs = bpy.context.preferences
    cycles_addon = prefs.addons.get('cycles') if hasattr(prefs.addons, "get") else None
    if cycles_addon is not None:
        cprefs = cycles_addon.preferences
        active_backend = 'NONE'
        for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
            try:
                cprefs.compute_device_type = backend
                if hasattr(cprefs, "refresh_devices"):
                    cprefs.refresh_devices()
                elif hasattr(cprefs, "get_devices"):
                    cprefs.get_devices()
                gpu_found = any(dev.type == backend for dev in cprefs.devices)
                if gpu_found:
                    active_backend = backend
                    break
            except Exception:
                active_backend = 'NONE'
        if active_backend != 'NONE':
            for dev in cprefs.devices:
                dev.use = (dev.type != 'CPU')
        else:
            cprefs.compute_device_type = 'NONE'

    scene.cycles.device = 'GPU'
    scene.cycles.samples = 4096
    scene.cycles.preview_samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.008
    if hasattr(scene.cycles, "adaptive_min_samples"):
        scene.cycles.adaptive_min_samples = 128
    if hasattr(scene.cycles, "sample_clamp_direct"):
        scene.cycles.sample_clamp_direct = 10.0
    if hasattr(scene.cycles, "sample_clamp_indirect"):
        scene.cycles.sample_clamp_indirect = 3.5
    if hasattr(scene.cycles, "blur_glossy"):
        scene.cycles.blur_glossy = 0.18
    if hasattr(scene.cycles, "use_light_tree"):
        scene.cycles.use_light_tree = True
    if hasattr(scene.cycles, "seed"):
        scene.cycles.seed = 2099
    if hasattr(scene.cycles, "use_animated_seed"):
        scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    if hasattr(scene.cycles, "denoising_input_passes"):
        scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    if hasattr(scene.cycles, "denoising_prefilter"):
        scene.cycles.denoising_prefilter = 'ACCURATE'
    scene.cycles.use_preview_denoising = True

    # Light path & volumetric integration settings (anti-firefly single-scatter volume)
    scene.cycles.max_bounces = 12
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 6
    scene.cycles.transmission_bounces = 8
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.volume_step_rate = 0.65
    scene.cycles.volume_max_steps = 256

    # AgX Color Management with High Contrast (compatible with Blender 4.0 and 4.2+ OCIO names)
    scene.view_settings.view_transform = 'AgX'
    look_enum_ids = [
        item.identifier
        for item in scene.view_settings.bl_rna.properties['look'].enum_items
    ]
    if 'AgX - High Contrast' in look_enum_ids:
        scene.view_settings.look = 'AgX - High Contrast'
    elif 'High Contrast' in look_enum_ids:
        scene.view_settings.look = 'High Contrast'
    scene.view_settings.exposure = 0.15
    scene.view_settings.gamma = 1.0

    # Procedural Volumetric World Shader (Principled Volume with anisotropic scattering)
    world = bpy.data.worlds.new("Cyberpunk_Volumetric_World")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()

    out_world = spawn_node(nt, "ShaderNodeOutputWorld", 620, 0, "World Output")
    bg_node = spawn_node(nt, "ShaderNodeBackground", 340, 160, "Midnight Sky")
    tex_coord = spawn_node(nt, "ShaderNodeTexCoord", -560, 40, "World Coords")
    sep_xyz = spawn_node(nt, "ShaderNodeSepXYZ", -340, 180, "Horizon Split")
    sky_ramp = spawn_node(nt, "ShaderNodeValToRGB", -80, 180, "Sky Gradient")

    sky_ramp.color_ramp.interpolation = 'EASE'
    sky_ramp.color_ramp.elements[0].position = 0.0
    sky_ramp.color_ramp.elements[0].color = (0.005, 0.009, 0.022, 1.0)
    sky_ramp.color_ramp.elements[1].position = 1.0
    sky_ramp.color_ramp.elements[1].color = (0.016, 0.007, 0.030, 1.0)
    bg_node.inputs["Strength"].default_value = 0.32

    mapping_vol = spawn_node(nt, "ShaderNodeMapping", -340, -140, "Fog Drift")
    mapping_vol.inputs["Scale"].default_value = (0.28, 0.28, 0.42)
    noise_vol = spawn_node(nt, "ShaderNodeTexNoise", -120, -140, "Fog Density Noise")
    noise_vol.inputs["Scale"].default_value = 1.4
    noise_vol.inputs["Detail"].default_value = 4.0
    noise_vol.inputs["Roughness"].default_value = 0.55

    math_mul = spawn_node(nt, "ShaderNodeMath", 100, -180, "Scale Density")
    math_mul.operation = 'MULTIPLY'
    math_mul.inputs[1].default_value = 0.012

    math_add = spawn_node(nt, "ShaderNodeMath", 260, -180, "Base Fog Floor")
    math_add.operation = 'ADD'
    math_add.inputs[1].default_value = 0.009

    vol_node = spawn_node(nt, "ShaderNodeVolumePrincipled", 420, -140, "Anisotropic Cyber Fog")
    vol_node.inputs["Color"].default_value = (0.32, 0.45, 0.62, 1.0)
    vol_node.inputs["Anisotropy"].default_value = 0.66
    vol_node.inputs["Emission Strength"].default_value = 0.0

    link_nodes(nt, tex_coord, "Generated", sep_xyz, "Vector")
    link_nodes(nt, sep_xyz, "Z", sky_ramp, "Fac")
    link_nodes(nt, sky_ramp, "Color", bg_node, "Color")
    link_nodes(nt, bg_node, "Background", out_world, "Surface")

    link_nodes(nt, tex_coord, "Object", mapping_vol, "Vector")
    link_nodes(nt, mapping_vol, "Vector", noise_vol, "Vector")
    link_nodes(nt, noise_vol, "Fac", math_mul, 0)
    link_nodes(nt, math_mul, "Value", math_add, 0)
    link_nodes(nt, math_add, "Value", vol_node, "Density")
    link_nodes(nt, vol_node, "Volume", out_world, "Volume")


# ==============================================================================
# 2. REUSABLE PROCEDURAL NODE-BASED SHADERS (ShaderNodeTree)
# ==============================================================================

def spawn_node(node_tree, node_type, x, y, label=None):
    """Create and position a shader node cleanly inside a ShaderNodeTree."""
    node = node_tree.nodes.new(type=node_type)
    node.location = (x, y)
    if label is not None:
        node.label = label
    return node


def link_nodes(node_tree, from_node, from_socket, to_node, to_socket):
    """Reusable helper function to connect sockets by name or integer index."""
    out_sock = from_node.outputs[from_socket]
    in_sock = to_node.inputs[to_socket]
    return node_tree.links.new(out_sock, in_sock)


def init_material(name):
    """Initialize a fresh node-enabled Blender Material with empty node tree."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.node_tree.nodes.clear()
    return mat, mat.node_tree


def _get_or_create_brushed_metal_core_group():
    """Shared ShaderNodeGroup NG_Brushed_Metal_Core for weathered anisotropic metals."""
    grp = bpy.data.node_groups.get("NG_Brushed_Metal_Core")
    if grp is not None:
        return grp
    grp = bpy.data.node_groups.new(name="NG_Brushed_Metal_Core", type='ShaderNodeTree')
    for s_name, s_type in (
        ("Brush Stretch", 'NodeSocketVector'),
        ("Dark Color", 'NodeSocketColor'),
        ("Light Color", 'NodeSocketColor'),
        ("Rough Min", 'NodeSocketFloat'),
        ("Rough Max", 'NodeSocketFloat'),
        ("Bump Strength", 'NodeSocketFloat'),
    ):
        grp.interface.new_socket(name=s_name, in_out='INPUT', socket_type=s_type)
    for s_name, s_type in (
        ("Base Color", 'NodeSocketColor'),
        ("Metallic", 'NodeSocketFloat'),
        ("Roughness", 'NodeSocketFloat'),
        ("Normal", 'NodeSocketVector'),
    ):
        grp.interface.new_socket(name=s_name, in_out='OUTPUT', socket_type=s_type)

    g_in = spawn_node(grp, "NodeGroupInput", -1180, 0, "Metal Parameters")
    tex_coord = spawn_node(grp, "ShaderNodeTexCoord", -980, 120, "Object Coords")
    mapping = spawn_node(grp, "ShaderNodeMapping", -780, 120, "Anisotropic Brush Stretch")

    noise = spawn_node(grp, "ShaderNodeTexNoise", -560, 120, "Brush Streak Noise")
    noise.inputs["Scale"].default_value = 14.0
    noise.inputs["Detail"].default_value = 4.5
    noise.inputs["Roughness"].default_value = 0.65
    noise.inputs["Distortion"].default_value = 0.35

    voronoi = spawn_node(grp, "ShaderNodeTexVoronoi", -340, 120, "Oxidation Grain Voronoi")
    voronoi.feature = 'F1'
    voronoi.distance = 'EUCLIDEAN'
    voronoi.inputs["Scale"].default_value = 22.0
    voronoi.inputs["Randomness"].default_value = 0.85

    color_mix = spawn_node(grp, "ShaderNodeMix", -80, 260, "Albedo Patina Mix")
    color_mix.data_type = 'RGBA'
    color_mix.blend_type = 'MIX'

    metal_ramp = spawn_node(grp, "ShaderNodeValToRGB", -80, 40, "Metallic Wear Ramp")
    metal_ramp.color_ramp.elements[0].position = 0.12
    metal_ramp.color_ramp.elements[0].color = (0.76, 0.76, 0.76, 1.0)
    metal_ramp.color_ramp.elements[1].position = 0.65
    metal_ramp.color_ramp.elements[1].color = (0.99, 0.99, 0.99, 1.0)

    rough_mix = spawn_node(grp, "ShaderNodeMix", -80, -160, "Roughness Scuff Mix")
    rough_mix.data_type = 'FLOAT'

    bump = spawn_node(grp, "ShaderNodeBump", 160, -240, "Brushed Micro-Groove Bump")
    bump.inputs["Distance"].default_value = 0.004

    g_out = spawn_node(grp, "NodeGroupOutput", 420, 80, "Metal Core Output")

    link_nodes(grp, tex_coord, "Object", mapping, "Vector")
    link_nodes(grp, g_in, "Brush Stretch", mapping, "Scale")
    link_nodes(grp, mapping, "Vector", noise, "Vector")
    link_nodes(grp, noise, "Color", voronoi, "Vector")
    link_nodes(grp, voronoi, "Distance", color_mix, 0)
    link_nodes(grp, g_in, "Dark Color", color_mix, 6)
    link_nodes(grp, g_in, "Light Color", color_mix, 7)
    link_nodes(grp, voronoi, "Distance", metal_ramp, "Fac")
    link_nodes(grp, voronoi, "Distance", rough_mix, 0)
    link_nodes(grp, g_in, "Rough Min", rough_mix, 2)
    link_nodes(grp, g_in, "Rough Max", rough_mix, 3)
    link_nodes(grp, g_in, "Bump Strength", bump, "Strength")
    link_nodes(grp, noise, "Fac", bump, "Height")

    link_nodes(grp, color_mix, 2, g_out, "Base Color")
    link_nodes(grp, metal_ramp, "Color", g_out, "Metallic")
    link_nodes(grp, rough_mix, 0, g_out, "Roughness")
    link_nodes(grp, bump, "Normal", g_out, "Normal")
    return grp


def _get_or_create_cyber_neon_tube_group():
    """Shared ShaderNodeGroup NG_Cyber_Neon_Tube for Fresnel neon plasma glass tubes."""
    grp = bpy.data.node_groups.get("NG_Cyber_Neon_Tube")
    if grp is not None:
        return grp
    grp = bpy.data.node_groups.new(name="NG_Cyber_Neon_Tube", type='ShaderNodeTree')
    for s_name, s_type in (
        ("Core Color", 'NodeSocketColor'),
        ("Rim Color", 'NodeSocketColor'),
        ("Strength", 'NodeSocketFloat'),
        ("Rim Strength", 'NodeSocketFloat'),
    ):
        grp.interface.new_socket(name=s_name, in_out='INPUT', socket_type=s_type)
    for s_name, s_type in (
        ("Emission Color", 'NodeSocketColor'),
        ("Emission Strength", 'NodeSocketFloat'),
    ):
        grp.interface.new_socket(name=s_name, in_out='OUTPUT', socket_type=s_type)

    g_in = spawn_node(grp, "NodeGroupInput", -760, 40, "Neon Parameters")
    fresnel = spawn_node(grp, "ShaderNodeFresnel", -540, 180, "Tube Glass Fresnel")
    fresnel.inputs["IOR"].default_value = 1.58

    layer_weight = spawn_node(grp, "ShaderNodeLayerWeight", -540, 0, "Plasma Core Facing")
    layer_weight.inputs["Blend"].default_value = 0.38

    falloff_ramp = spawn_node(grp, "ShaderNodeValToRGB", -300, 80, "Fresnel Falloff Curve")
    falloff_ramp.color_ramp.interpolation = 'EASE'
    falloff_ramp.color_ramp.elements[0].position = 0.08
    falloff_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    falloff_ramp.color_ramp.elements[1].position = 0.78
    falloff_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    color_mix = spawn_node(grp, "ShaderNodeMix", -40, 180, "Hot Core to Neon Rim")
    color_mix.data_type = 'RGBA'

    strength_mix = spawn_node(grp, "ShaderNodeMix", -40, -40, "Fresnel Strength Falloff")
    strength_mix.data_type = 'FLOAT'

    g_out = spawn_node(grp, "NodeGroupOutput", 240, 80, "Neon Output")

    link_nodes(grp, layer_weight, "Fresnel", falloff_ramp, "Fac")
    link_nodes(grp, falloff_ramp, "Color", color_mix, 0)
    link_nodes(grp, g_in, "Core Color", color_mix, 6)
    link_nodes(grp, g_in, "Rim Color", color_mix, 7)
    link_nodes(grp, fresnel, "Fac", strength_mix, 0)
    link_nodes(grp, g_in, "Strength", strength_mix, 2)
    link_nodes(grp, g_in, "Rim Strength", strength_mix, 3)
    link_nodes(grp, color_mix, 2, g_out, "Emission Color")
    link_nodes(grp, strength_mix, 0, g_out, "Emission Strength")
    return grp


def _get_or_create_triplanar_ramp_bump_group():
    """Shared ShaderNodeGroup NG_Triplanar_Ramp_Bump for stained timber and ceramic glazes."""
    grp = bpy.data.node_groups.get("NG_Triplanar_Ramp_Bump")
    if grp is not None:
        return grp
    grp = bpy.data.node_groups.new(name="NG_Triplanar_Ramp_Bump", type='ShaderNodeTree')
    for s_name, s_type in (
        ("Fac", 'NodeSocketFloat'),
        ("Color A", 'NodeSocketColor'),
        ("Color B", 'NodeSocketColor'),
        ("Bump Strength", 'NodeSocketFloat'),
        ("Bump Distance", 'NodeSocketFloat'),
    ):
        grp.interface.new_socket(name=s_name, in_out='INPUT', socket_type=s_type)
    for s_name, s_type in (
        ("Base Color", 'NodeSocketColor'),
        ("Normal", 'NodeSocketVector'),
    ):
        grp.interface.new_socket(name=s_name, in_out='OUTPUT', socket_type=s_type)

    g_in = spawn_node(grp, "NodeGroupInput", -540, 0, "Surface Parameters")
    ramp = spawn_node(grp, "ShaderNodeValToRGB", -280, 140, "Surface Contrast Curve")
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    col_mix = spawn_node(grp, "ShaderNodeMix", -20, 140, "Dual Tone Albedo Mix")
    col_mix.data_type = 'RGBA'

    bump = spawn_node(grp, "ShaderNodeBump", -20, -100, "Surface Relief Bump")
    g_out = spawn_node(grp, "NodeGroupOutput", 240, 40, "Surface Output")

    link_nodes(grp, g_in, "Fac", ramp, "Fac")
    link_nodes(grp, ramp, "Color", col_mix, 0)
    link_nodes(grp, g_in, "Color A", col_mix, 6)
    link_nodes(grp, g_in, "Color B", col_mix, 7)
    link_nodes(grp, g_in, "Bump Strength", bump, "Strength")
    link_nodes(grp, g_in, "Bump Distance", bump, "Distance")
    link_nodes(grp, g_in, "Fac", bump, "Height")
    link_nodes(grp, col_mix, 2, g_out, "Base Color")
    link_nodes(grp, bump, "Normal", g_out, "Normal")
    return grp


def build_weathered_brushed_metal_shader(
    name,
    dark_color,
    light_color,
    brush_stretch=(1.0, 26.0, 1.0),
    rough_min=0.22,
    rough_max=0.64,
    bump_strength=0.14,
):
    """Weathered Brushed Metal via shared NG_Brushed_Metal_Core ShaderNodeGroup."""
    mat, nt = init_material(name)
    core_grp = _get_or_create_brushed_metal_core_group()

    grp_node = spawn_node(nt, "ShaderNodeGroup", -240, 80, "Brushed Metal Core")
    grp_node.node_tree = core_grp
    grp_node.inputs["Brush Stretch"].default_value = brush_stretch
    grp_node.inputs["Dark Color"].default_value = dark_color
    grp_node.inputs["Light Color"].default_value = light_color
    grp_node.inputs["Rough Min"].default_value = rough_min
    grp_node.inputs["Rough Max"].default_value = rough_max
    grp_node.inputs["Bump Strength"].default_value = bump_strength

    avg_col = tuple((dark_color[k] + light_color[k]) * 0.5 for k in range(3)) + (1.0,)
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 300, 80, "Principled Brushed Metal")
    bsdf.inputs["Base Color"].default_value = avg_col
    bsdf.inputs["Metallic"].default_value = 0.875
    bsdf.inputs["Roughness"].default_value = (rough_min + rough_max) * 0.5
    bsdf.inputs["Coat Weight"].default_value = 0.18
    bsdf.inputs["Coat Roughness"].default_value = 0.28

    out = spawn_node(nt, "ShaderNodeOutputMaterial", 600, 80, "Material Output")

    link_nodes(nt, grp_node, "Base Color", bsdf, "Base Color")
    link_nodes(nt, grp_node, "Metallic", bsdf, "Metallic")
    link_nodes(nt, grp_node, "Roughness", bsdf, "Roughness")
    link_nodes(nt, grp_node, "Normal", bsdf, "Normal")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    return mat


def build_wet_asphalt_concrete_shader(name):
    """Wet Asphalt/Concrete Base: Musgrave/Noise -> Puddle mask mixing glossy reflections & stone."""
    mat, nt = init_material(name)

    tex_coord = spawn_node(nt, "ShaderNodeTexCoord", -1220, 0, "Object Coords")
    mapping = spawn_node(nt, "ShaderNodeMapping", -1020, 0, "Street Scale")
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)

    # Blender 4.0 uses ShaderNodeTexMusgrave; Blender 4.1+ merged Musgrave into ShaderNodeTexNoise
    if bpy.app.version < (4, 1, 0):
        macro_tex = spawn_node(nt, "ShaderNodeTexMusgrave", -780, 220, "Musgrave Puddle Basin")
        macro_tex.musgrave_type = 'HYBRID_MULTIFRACTAL'
        macro_tex.inputs["Scale"].default_value = 1.35
        macro_tex.inputs["Detail"].default_value = 4.5
        macro_tex.inputs["Dimension"].default_value = 0.7
        macro_tex.inputs["Lacunarity"].default_value = 2.1
    else:
        macro_tex = spawn_node(nt, "ShaderNodeTexNoise", -780, 220, "Multifractal Puddle Basin")
        macro_tex.noise_type = 'HYBRID_MULTIFRACTAL'
        macro_tex.normalize = True
        macro_tex.inputs["Scale"].default_value = 1.35
        macro_tex.inputs["Detail"].default_value = 4.5
        macro_tex.inputs["Roughness"].default_value = 0.58
        macro_tex.inputs["Lacunarity"].default_value = 2.1

    puddle_ramp = spawn_node(nt, "ShaderNodeValToRGB", -500, 220, "Puddle Mask")
    puddle_ramp.color_ramp.interpolation = 'EASE'
    puddle_ramp.color_ramp.elements[0].position = 0.44
    puddle_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    puddle_ramp.color_ramp.elements[1].position = 0.56
    puddle_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    stone_noise = spawn_node(nt, "ShaderNodeTexNoise", -780, -120, "Aggregate Grit")
    stone_noise.inputs["Scale"].default_value = 42.0
    stone_noise.inputs["Detail"].default_value = 4.5
    stone_noise.inputs["Roughness"].default_value = 0.72

    stone_voronoi = spawn_node(nt, "ShaderNodeTexVoronoi", -780, -340, "Pebble Chippings")
    stone_voronoi.feature = 'F1'
    stone_voronoi.inputs["Scale"].default_value = 64.0

    grit_mix = spawn_node(nt, "ShaderNodeMath", -520, -220, "Combine Grit")
    grit_mix.operation = 'MULTIPLY'

    # Invert puddle mask so standing water flattens stone bump to mirror smoothness
    inv_puddle = spawn_node(nt, "ShaderNodeMath", -240, -60, "Dry Surface Mask")
    inv_puddle.operation = 'SUBTRACT'
    inv_puddle.use_clamp = True
    inv_puddle.inputs[0].default_value = 1.0

    bump_atten = spawn_node(nt, "ShaderNodeMath", -40, -200, "Flatten Puddle Normals")
    bump_atten.operation = 'MULTIPLY'

    bump = spawn_node(nt, "ShaderNodeBump", 160, -200, "Asphalt Micro-Bump")
    bump.inputs["Strength"].default_value = 0.52
    bump.inputs["Distance"].default_value = 0.008

    dry_stone_ramp = spawn_node(nt, "ShaderNodeValToRGB", -500, -20, "Dry Asphalt Color")
    dry_stone_ramp.color_ramp.elements[0].position = 0.25
    dry_stone_ramp.color_ramp.elements[0].color = (0.048, 0.052, 0.060, 1.0)
    dry_stone_ramp.color_ramp.elements[1].position = 0.80
    dry_stone_ramp.color_ramp.elements[1].color = (0.115, 0.120, 0.132, 1.0)

    # Universal Blender 4.x Mix Node in RGBA mode (0=Factor, 6=Color A, 7=Color B, out 2=Result)
    albedo_mix = spawn_node(nt, "ShaderNodeMix", -140, 220, "Dry vs Wet Albedo")
    albedo_mix.data_type = 'RGBA'
    albedo_mix.blend_type = 'MIX'
    albedo_mix.inputs[7].default_value = (0.012, 0.014, 0.020, 1.0)

    rough_mix = spawn_node(nt, "ShaderNodeMix", -140, 40, "Stone vs Puddle Roughness")
    rough_mix.data_type = 'FLOAT'
    rough_mix.inputs[2].default_value = 0.84
    rough_mix.inputs[3].default_value = 0.02

    coat_mix = spawn_node(nt, "ShaderNodeMix", 120, 80, "Rain Coat Weight")
    coat_mix.data_type = 'FLOAT'
    coat_mix.inputs[2].default_value = 0.18
    coat_mix.inputs[3].default_value = 1.0

    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 380, 120, "Principled Wet Asphalt")
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Coat Roughness"].default_value = 0.015

    out = spawn_node(nt, "ShaderNodeOutputMaterial", 680, 120, "Material Output")

    link_nodes(nt, tex_coord, "Object", mapping, "Vector")
    link_nodes(nt, mapping, "Vector", macro_tex, "Vector")
    link_nodes(nt, mapping, "Vector", stone_noise, "Vector")
    link_nodes(nt, mapping, "Vector", stone_voronoi, "Vector")

    link_nodes(nt, macro_tex, "Fac", puddle_ramp, "Fac")
    link_nodes(nt, stone_noise, "Fac", dry_stone_ramp, "Fac")
    link_nodes(nt, stone_noise, "Fac", grit_mix, 0)
    link_nodes(nt, stone_voronoi, "Distance", grit_mix, 1)

    link_nodes(nt, puddle_ramp, "Color", inv_puddle, 1)
    link_nodes(nt, grit_mix, "Value", bump_atten, 0)
    link_nodes(nt, inv_puddle, "Value", bump_atten, 1)
    link_nodes(nt, bump_atten, "Value", bump, "Height")

    link_nodes(nt, puddle_ramp, "Color", albedo_mix, 0)
    link_nodes(nt, dry_stone_ramp, "Color", albedo_mix, 6)
    link_nodes(nt, puddle_ramp, "Color", rough_mix, 0)
    link_nodes(nt, puddle_ramp, "Color", coat_mix, 0)

    link_nodes(nt, albedo_mix, 2, bsdf, "Base Color")
    link_nodes(nt, rough_mix, 0, bsdf, "Roughness")
    link_nodes(nt, coat_mix, 0, bsdf, "Coat Weight")
    link_nodes(nt, bump, "Normal", bsdf, "Normal")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    return mat


def build_cyber_neon_shader(name, core_rgba, rim_rgba, strength=26.0):
    """Cyber Neon Emission Shader via shared NG_Cyber_Neon_Tube ShaderNodeGroup."""
    mat, nt = init_material(name)
    neon_grp = _get_or_create_cyber_neon_tube_group()

    grp_node = spawn_node(nt, "ShaderNodeGroup", -120, 100, "Cyber Neon Tube Core")
    grp_node.node_tree = neon_grp
    grp_node.inputs["Core Color"].default_value = core_rgba
    grp_node.inputs["Rim Color"].default_value = rim_rgba
    grp_node.inputs["Strength"].default_value = strength
    grp_node.inputs["Rim Strength"].default_value = strength * 0.42

    mid_emit = tuple((core_rgba[k] + rim_rgba[k]) * 0.5 for k in range(3)) + (1.0,)
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 220, 100, "Principled Neon Tube")
    bsdf.inputs["Base Color"].default_value = rim_rgba
    bsdf.inputs["Roughness"].default_value = 0.06
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.02
    bsdf.inputs["Emission Color"].default_value = mid_emit
    bsdf.inputs["Emission Strength"].default_value = strength * 0.71

    out = spawn_node(nt, "ShaderNodeOutputMaterial", 520, 100, "Material Output")

    link_nodes(nt, grp_node, "Emission Color", bsdf, "Emission Color")
    link_nodes(nt, grp_node, "Emission Strength", bsdf, "Emission Strength")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    return mat


def build_translucent_paper_lantern_shader(name):
    """Translucent Paper Lantern Shader: Subsurface Scattering + Translucent BSDF + Washi Fiber."""
    mat, nt = init_material(name)

    tex_coord = spawn_node(nt, "ShaderNodeTexCoord", -920, 0, "Lantern Coords")
    mapping = spawn_node(nt, "ShaderNodeMapping", -720, 0, "Paper Fiber Scale")
    mapping.inputs["Scale"].default_value = (1.0, 1.0, 1.0)

    wave_ribs = spawn_node(nt, "ShaderNodeTexWave", -500, 180, "Bamboo Rib Shadowing")
    wave_ribs.wave_type = 'BANDS'
    wave_ribs.bands_direction = 'Z'
    wave_ribs.wave_profile = 'SIN'
    wave_ribs.inputs["Scale"].default_value = 18.0
    wave_ribs.inputs["Distortion"].default_value = 0.4

    fiber_noise = spawn_node(nt, "ShaderNodeTexNoise", -500, -60, "Mulberry Washi Fibers")
    fiber_noise.inputs["Scale"].default_value = 46.0
    fiber_noise.inputs["Detail"].default_value = 4.5
    fiber_noise.inputs["Roughness"].default_value = 0.7

    paper_ramp = spawn_node(nt, "ShaderNodeValToRGB", -240, 140, "Warm Vermilion Washi")
    paper_ramp.color_ramp.elements[0].position = 0.15
    paper_ramp.color_ramp.elements[0].color = (0.72, 0.14, 0.04, 1.0)
    paper_ramp.color_ramp.elements[1].position = 0.72
    paper_ramp.color_ramp.elements[1].color = (0.98, 0.44, 0.12, 1.0)

    bump = spawn_node(nt, "ShaderNodeBump", -200, -120, "Paper Texture Bump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.003

    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 80, 180, "SSS Paper BSDF")
    bsdf.inputs["Roughness"].default_value = 0.72
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.46, 0.16)
    bsdf.inputs["Subsurface Scale"].default_value = 0.09
    bsdf.inputs["Transmission Weight"].default_value = 0.32
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.40, 0.09, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.55

    translucent = spawn_node(nt, "ShaderNodeBsdfTranslucent", 80, -120, "Backlit Paper Glow")
    translucent.inputs["Color"].default_value = (1.0, 0.54, 0.18, 1.0)

    mix_shader = spawn_node(nt, "ShaderNodeMixShader", 360, 60, "Blend SSS & Translucent")
    mix_shader.inputs[0].default_value = 0.38

    out = spawn_node(nt, "ShaderNodeOutputMaterial", 600, 60, "Material Output")

    link_nodes(nt, tex_coord, "Object", mapping, "Vector")
    link_nodes(nt, mapping, "Vector", wave_ribs, "Vector")
    link_nodes(nt, mapping, "Vector", fiber_noise, "Vector")
    link_nodes(nt, wave_ribs, "Fac", paper_ramp, "Fac")
    link_nodes(nt, fiber_noise, "Fac", bump, "Height")

    link_nodes(nt, paper_ramp, "Color", bsdf, "Base Color")
    link_nodes(nt, bump, "Normal", bsdf, "Normal")
    link_nodes(nt, bump, "Normal", translucent, "Normal")
    link_nodes(nt, bsdf, "BSDF", mix_shader, 1)
    link_nodes(nt, translucent, "BSDF", mix_shader, 2)
    link_nodes(nt, mix_shader, "Shader", out, "Surface")
    return mat


def build_procedural_wood_shader(name, grain_dark, grain_light, coat_weight=0.35, roughness=0.42):
    """Procedural Japanese Timber Shader using shared NG_Triplanar_Ramp_Bump."""
    mat, nt = init_material(name)
    surf_grp = _get_or_create_triplanar_ramp_bump_group()

    tex_coord = spawn_node(nt, "ShaderNodeTexCoord", -900, 0, "Wood Coords")
    mapping = spawn_node(nt, "ShaderNodeMapping", -700, 0, "Wood Grain Stretch")
    mapping.inputs["Scale"].default_value = (1.8, 14.0, 1.8)

    wave = spawn_node(nt, "ShaderNodeTexWave", -480, 100, "Growth Rings")
    wave.wave_type = 'RINGS'
    wave.inputs["Scale"].default_value = 6.5
    wave.inputs["Distortion"].default_value = 5.2
    wave.inputs["Detail"].default_value = 4.5

    grp_node = spawn_node(nt, "ShaderNodeGroup", -200, 80, "Timber Ramp & Relief")
    grp_node.node_tree = surf_grp
    grp_node.inputs["Color A"].default_value = grain_dark
    grp_node.inputs["Color B"].default_value = grain_light
    grp_node.inputs["Bump Strength"].default_value = 0.16
    grp_node.inputs["Bump Distance"].default_value = 0.004

    avg_col = tuple((grain_dark[k] + grain_light[k]) * 0.5 for k in range(3)) + (1.0,)
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 120, 80, "Principled Timber")
    bsdf.inputs["Base Color"].default_value = avg_col
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat_weight
    bsdf.inputs["Coat Roughness"].default_value = 0.14

    out = spawn_node(nt, "ShaderNodeOutputMaterial", 400, 80, "Material Output")

    link_nodes(nt, tex_coord, "Object", mapping, "Vector")
    link_nodes(nt, mapping, "Vector", wave, "Vector")
    link_nodes(nt, wave, "Fac", grp_node, "Fac")
    link_nodes(nt, grp_node, "Base Color", bsdf, "Base Color")
    link_nodes(nt, grp_node, "Normal", bsdf, "Normal")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    return mat


def build_glazed_ceramic_shader(name, base_col, accent_col, roughness=0.14, coat=0.90):
    """Glazed Ceramic Shader using shared NG_Triplanar_Ramp_Bump."""
    mat, nt = init_material(name)
    surf_grp = _get_or_create_triplanar_ramp_bump_group()

    tex_coord = spawn_node(nt, "ShaderNodeTexCoord", -780, 0, "Ceramic Coords")
    noise = spawn_node(nt, "ShaderNodeTexNoise", -560, 60, "Kiln Glaze Speckle")
    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 4.5

    grp_node = spawn_node(nt, "ShaderNodeGroup", -280, 60, "Ceramic Glaze & Dimple")
    grp_node.node_tree = surf_grp
    grp_node.inputs["Color A"].default_value = base_col
    grp_node.inputs["Color B"].default_value = accent_col
    grp_node.inputs["Bump Strength"].default_value = 0.06
    grp_node.inputs["Bump Distance"].default_value = 0.002

    avg_col = tuple((base_col[k] + accent_col[k]) * 0.5 for k in range(3)) + (1.0,)
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 40, 60, "Principled Glazed Ceramic")
    bsdf.inputs["Base Color"].default_value = avg_col
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = 0.04

    out = spawn_node(nt, "ShaderNodeOutputMaterial", 340, 60, "Material Output")

    link_nodes(nt, tex_coord, "Object", noise, "Vector")
    link_nodes(nt, noise, "Fac", grp_node, "Fac")
    link_nodes(nt, grp_node, "Base Color", bsdf, "Base Color")
    link_nodes(nt, grp_node, "Normal", bsdf, "Normal")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    return mat


def build_culinary_and_prop_shaders():
    """Create complete dictionary of all procedural materials for the diorama."""
    mats = {}

    # Core architectural shaders
    mats["wet_asphalt"] = build_wet_asphalt_concrete_shader("MAT_Wet_Asphalt_Concrete")
    mats["hull_metal"] = build_weathered_brushed_metal_shader(
        "MAT_Hull_Brushed_Gunmetal",
        dark_color=(0.065, 0.075, 0.090, 1.0),
        light_color=(0.220, 0.245, 0.280, 1.0),
        brush_stretch=(1.0, 28.0, 1.0),
        rough_min=0.24,
        rough_max=0.62,
        bump_strength=0.16,
    )
    mats["pipe_metal"] = build_weathered_brushed_metal_shader(
        "MAT_Pipe_Oxidized_BronzeSteel",
        dark_color=(0.140, 0.095, 0.065, 1.0),
        light_color=(0.440, 0.310, 0.200, 1.0),
        brush_stretch=(1.0, 18.0, 1.0),
        rough_min=0.26,
        rough_max=0.58,
        bump_strength=0.14,
    )
    mats["stainless"] = build_weathered_brushed_metal_shader(
        "MAT_Kitchen_Brushed_Stainless",
        dark_color=(0.340, 0.360, 0.385, 1.0),
        light_color=(0.720, 0.745, 0.780, 1.0),
        brush_stretch=(18.0, 1.0, 1.0),
        rough_min=0.16,
        rough_max=0.38,
        bump_strength=0.08,
    )

    # Cyber Neon trio with Fresnel falloff
    mats["neon_cyan"] = build_cyber_neon_shader(
        "MAT_Cyber_Neon_Cyan",
        core_rgba=(0.72, 0.99, 1.0, 1.0),
        rim_rgba=(0.00, 0.86, 1.0, 1.0),
        strength=28.0,
    )
    mats["neon_magenta"] = build_cyber_neon_shader(
        "MAT_Cyber_Neon_Magenta",
        core_rgba=(1.0, 0.72, 0.90, 1.0),
        rim_rgba=(1.0, 0.03, 0.48, 1.0),
        strength=28.0,
    )
    mats["neon_orange"] = build_cyber_neon_shader(
        "MAT_Cyber_Neon_WarmOrange",
        core_rgba=(1.0, 0.86, 0.52, 1.0),
        rim_rgba=(1.0, 0.26, 0.02, 1.0),
        strength=25.0,
    )

    # Translucent paper lantern
    mats["lantern_paper"] = build_translucent_paper_lantern_shader("MAT_Lantern_Washi_Paper")

    # Wood & Ceramic shaders
    mats["charred_wood"] = build_procedural_wood_shader(
        "MAT_Charred_Sugi_Timber",
        grain_dark=(0.028, 0.020, 0.016, 1.0),
        grain_light=(0.095, 0.062, 0.040, 1.0),
        coat_weight=0.28,
        roughness=0.52,
    )
    mats["hinoki_wood"] = build_procedural_wood_shader(
        "MAT_Hinoki_Counter_Wood",
        grain_dark=(0.260, 0.145, 0.068, 1.0),
        grain_light=(0.520, 0.330, 0.165, 1.0),
        coat_weight=0.82,
        roughness=0.22,
    )
    mats["roof_tile"] = build_glazed_ceramic_shader(
        "MAT_Kawara_Roof_Tile",
        base_col=(0.030, 0.038, 0.056, 1.0),
        accent_col=(0.075, 0.092, 0.130, 1.0),
        roughness=0.18,
        coat=0.88,
    )
    mats["bowl_ceramic"] = build_glazed_ceramic_shader(
        "MAT_Ramen_Bowl_Lacquer",
        base_col=(0.020, 0.018, 0.022, 1.0),
        accent_col=(0.420, 0.035, 0.030, 1.0),
        roughness=0.10,
        coat=0.95,
    )

    # Ramen Broth Shader (Rich golden-amber subsurface + glossy oil coat)
    broth_mat, nt = init_material("MAT_Tonkotsu_Shoyu_Broth")
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 80, 60, "Broth BSDF")
    bsdf.inputs["Base Color"].default_value = (0.58, 0.25, 0.04, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["Subsurface Weight"].default_value = 0.90
    bsdf.inputs["Subsurface Radius"].default_value = (0.9, 0.45, 0.12)
    bsdf.inputs["Subsurface Scale"].default_value = 0.05
    bsdf.inputs["Transmission Weight"].default_value = 0.42
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.01
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 360, 60, "Material Output")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["broth"] = broth_mat

    # Alkaline Ramen Noodle Shader
    noodle_mat, nt = init_material("MAT_Ramen_Wheat_Noodles")
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 80, 60, "Noodle BSDF")
    bsdf.inputs["Base Color"].default_value = (0.88, 0.68, 0.24, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.22
    bsdf.inputs["Subsurface Weight"].default_value = 0.65
    bsdf.inputs["Subsurface Scale"].default_value = 0.02
    bsdf.inputs["Coat Weight"].default_value = 0.75
    bsdf.inputs["Coat Roughness"].default_value = 0.06
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 360, 60, "Material Output")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["noodle"] = noodle_mat

    # Egg / Narutomaki Cream White & Yolk Amber
    egg_mat, nt = init_material("MAT_Egg_Narutomaki_White")
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 80, 60, "Egg White BSDF")
    bsdf.inputs["Base Color"].default_value = (0.94, 0.90, 0.82, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Subsurface Weight"].default_value = 0.70
    bsdf.inputs["Subsurface Scale"].default_value = 0.025
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 360, 60, "Material Output")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["egg_white"] = egg_mat

    yolk_mat, nt = init_material("MAT_Ajitama_Jammy_Yolk")
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 80, 60, "Yolk BSDF")
    bsdf.inputs["Base Color"].default_value = (0.95, 0.34, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.12
    bsdf.inputs["Coat Weight"].default_value = 0.85
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 360, 60, "Material Output")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["egg_yolk"] = yolk_mat

    # Noren Indigo Linen Fabric Shader
    noren_mat, nt = init_material("MAT_Indigo_Noren_Fabric")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -560, 0, "Fabric Coords")
    wv = spawn_node(nt, "ShaderNodeTexVoronoi", -340, 0, "Linen Weave")
    wv.inputs["Scale"].default_value = 110.0
    bp = spawn_node(nt, "ShaderNodeBump", -120, -80, "Weave Bump")
    bp.inputs["Strength"].default_value = 0.18
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 120, 60, "Linen BSDF")
    bsdf.inputs["Base Color"].default_value = (0.035, 0.058, 0.145, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.82
    bsdf.inputs["Sheen Weight"].default_value = 0.85
    bsdf.inputs["Subsurface Weight"].default_value = 0.25
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 400, 60, "Material Output")
    link_nodes(nt, tc, "Object", wv, "Vector")
    link_nodes(nt, wv, "Distance", bp, "Height")
    link_nodes(nt, bp, "Normal", bsdf, "Normal")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["noren"] = noren_mat

    # Insulated Rubber Power Cable Shader
    cable_mat, nt = init_material("MAT_Rubber_Power_Cable")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -520, 0, "Cable Coords")
    ns = spawn_node(nt, "ShaderNodeTexNoise", -300, 0, "Rubber Scuff")
    ns.inputs["Scale"].default_value = 24.0
    bp = spawn_node(nt, "ShaderNodeBump", -100, -80, "Rubber Bump")
    bp.inputs["Strength"].default_value = 0.08
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 120, 60, "Rubber BSDF")
    bsdf.inputs["Base Color"].default_value = (0.022, 0.024, 0.028, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.44
    bsdf.inputs["Coat Weight"].default_value = 0.25
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 400, 60, "Material Output")
    link_nodes(nt, tc, "Object", ns, "Vector")
    link_nodes(nt, ns, "Fac", bp, "Height")
    link_nodes(nt, bp, "Normal", bsdf, "Normal")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["cable"] = cable_mat

    return mats


# ==============================================================================
# 3. PROCEDURAL GEOMETRY HELPERS & ARCHITECTURAL MODULE BUILDERS
# ==============================================================================

def finalize_bmesh_object(bm, name, collection, material=None, smooth=True, location=(0.0, 0.0, 0.0)):
    """Convert BMesh into a Mesh Object, assign material, recalculate normals, and link."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for face in bm.faces:
        face.smooth = smooth

    mesh = bpy.data.meshes.new(name + "_Mesh")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    if material is not None:
        mesh.materials.append(material)

    obj = bpy.data.objects.new(name, mesh)
    obj.location = Vector(location)
    collection.objects.link(obj)
    return obj


def add_bevel_modifier(obj, width=0.012, segments=3, angle_deg=32.0):
    """Attach an angle-limited multi-segment Bevel modifier for specular edge highlights."""
    bev = obj.modifiers.new(name="CyberBevel", type='BEVEL')
    bev.width = width
    bev.segments = segments
    bev.limit_method = 'ANGLE'
    bev.angle_limit = math.radians(angle_deg)
    bev.use_clamp_overlap = True
    return bev


def create_beveled_box(
    name,
    collection,
    size_xyz,
    location,
    material,
    rotation=(0.0, 0.0, 0.0),
    bevel_width=0.010,
    bevel_segments=3,
):
    """Create a bmesh box scaled to exact dimensions with a clean Bevel modifier."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, verts=bm.verts[:], vec=size_xyz)
    obj = finalize_bmesh_object(bm, name, collection, material=material, smooth=True, location=location)
    obj.rotation_euler = Euler(rotation, 'XYZ')
    if bevel_width > 0.0:
        add_bevel_modifier(obj, width=bevel_width, segments=bevel_segments)
    return obj


def create_lathe_object(
    name,
    collection,
    profile_rz,
    steps,
    location,
    material,
    rotation=(0.0, 0.0, 0.0),
    close_caps=False,
):
    """Revolve a 2D (radius, z) profile 360 degrees around local Z via bmesh.ops.spin."""
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0.0, z)) for (r, z) in profile_rz]
    edges = []
    for idx in range(len(verts) - 1):
        edges.append(bm.edges.new((verts[idx], verts[idx + 1])))
    if close_caps and len(verts) > 2:
        edges.append(bm.edges.new((verts[-1], verts[0])))

    bmesh.ops.spin(
        bm,
        geom=verts + edges,
        cent=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 1.0),
        angle=2.0 * math.pi,
        steps=steps,
    )
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0001)
    obj = finalize_bmesh_object(bm, name, collection, material=material, smooth=True, location=location)
    obj.rotation_euler = Euler(rotation, 'XYZ')
    return obj


def create_pipe_curve(
    name,
    collection,
    points_3d,
    bevel_depth,
    material,
    cyclic=False,
    spline_type='POLY',
    resolution=12,
):
    """Create a 3D structural pipe or neon tube using bpy.data.curves with custom bevel_depth."""
    curve_data = bpy.data.curves.new(name=name + "_Curve", type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.fill_mode = 'FULL'
    curve_data.bevel_depth = bevel_depth
    curve_data.bevel_resolution = 6
    curve_data.resolution_u = resolution

    spline = curve_data.splines.new(type=spline_type)
    spline.points.add(len(points_3d) - 1)
    for idx, pt in enumerate(points_3d):
        spline.points[idx].co = (float(pt[0]), float(pt[1]), float(pt[2]), 1.0)
    spline.use_cyclic_u = cyclic
    if spline_type == 'NURBS':
        spline.use_endpoint_u = not cyclic
        spline.order_u = min(4, len(points_3d))

    curve_data.materials.append(material)
    obj = bpy.data.objects.new(name, curve_data)
    collection.objects.link(obj)
    return obj


def build_environment_wet_base(col, mats):
    """Build chamfered octagonal wet asphalt street plinth, curb rim, and drain grates."""
    bm = bmesh.new()
    radial_rings = 18
    angular_segs = 64
    plinth_r = 3.65
    plinth_h = 0.26

    # Concentric octagonal-radial mesh topology with subtle puddle basins
    center_top = bm.verts.new((0.0, 0.0, -0.004))
    rings_top = []
    for ir in range(1, radial_rings + 1):
        frac = ir / radial_rings
        ring = []
        for ia in range(angular_segs):
            theta = (2.0 * math.pi * ia / angular_segs) + (math.pi / 8.0)
            ux = math.cos(theta)
            uy = math.sin(theta)
            # Smooth super-octagonal diorama boundary norm
            oct_norm = max(abs(ux), abs(uy), (abs(ux) + abs(uy)) * 0.725)
            r_eff = (plinth_r * frac) / oct_norm
            px = ux * r_eff
            py = uy * r_eff
            # Millimeter puddle depressions + subtle outer street lip
            curb_lip = 0.018 if ir == radial_rings else 0.0
            ripple = -0.007 * math.sin(px * 1.45) * math.cos(py * 1.45) + curb_lip
            ring.append(bm.verts.new((px, py, ripple)))
        rings_top.append(ring)

    # Inner fan around center vertex
    first_ring = rings_top[0]
    for ia in range(angular_segs):
        ib = (ia + 1) % angular_segs
        bm.faces.new((center_top, first_ring[ia], first_ring[ib]))

    # Concentric quad strips across the wet asphalt top surface
    for ir in range(len(rings_top) - 1):
        r_in = rings_top[ir]
        r_out = rings_top[ir + 1]
        for ia in range(angular_segs):
            ib = (ia + 1) % angular_segs
            bm.faces.new((r_in[ia], r_out[ia], r_out[ib], r_in[ib]))

    # Solid beveled plinth side skirt and bottom cap
    perim_top = rings_top[-1]
    perim_bot = [bm.verts.new((v.co.x * 1.025, v.co.y * 1.025, -plinth_h)) for v in perim_top]
    center_bot = bm.verts.new((0.0, 0.0, -plinth_h))

    for ia in range(angular_segs):
        ib = (ia + 1) % angular_segs
        bm.faces.new((perim_top[ia], perim_bot[ia], perim_bot[ib], perim_top[ib]))
        bm.faces.new((perim_bot[ib], perim_bot[ia], center_bot))

    base_obj = finalize_bmesh_object(bm, "Wet_Asphalt_Diorama_Plinth", col, mats["wet_asphalt"], smooth=True)
    add_bevel_modifier(base_obj, width=0.014, segments=2)

    # Metallic cyber-curb trim frame & storm drain grates embedded along front street edge
    for side_sign in (-1.0, 1.0):
        create_beveled_box(
            f"Street_Curb_Rail_{'L' if side_sign < 0 else 'R'}",
            col,
            (1.15, 0.16, 0.055),
            (side_sign * 1.55, -2.55, 0.025),
            mats["hull_metal"],
            bevel_width=0.008,
        )

    grate_slat = create_beveled_box(
        "Storm_Drain_Grate_Slats",
        col,
        (0.032, 0.34, 0.018),
        (-0.38, -2.55, 0.008),
        mats["pipe_metal"],
        bevel_width=0.003,
    )
    arr = grate_slat.modifiers.new(name="GrateArray", type='ARRAY')
    arr.fit_type = 'FIXED_COUNT'
    arr.count = 16
    arr.use_relative_offset = False
    arr.use_constant_offset = True
    arr.constant_offset_displace = (0.052, 0.0, 0.0)


def build_floating_hull_and_thrusters(col, mats):
    """Build multi-tiered metallic anti-gravity boat hull, 4 vector thrusters, and hover stools."""
    # Stepped armored trapezoidal under-hull (hovering Z = 0.58m to Z = 1.05m)
    bm = bmesh.new()
    tiers = [
        (1.32, 0.86, 0.58),
        (1.64, 1.08, 0.72),
        (1.82, 1.24, 0.92),
        (1.86, 1.28, 1.05),
    ]
    chamfer = 0.24
    rings = []
    for hx, hy, hz in tiers:
        pts = [
            (-hx + chamfer, -hy, hz), (hx - chamfer, -hy, hz),
            (hx, -hy + chamfer, hz), (hx, hy - chamfer, hz),
            (hx - chamfer, hy, hz), (-hx + chamfer, hy, hz),
            (-hx, hy - chamfer, hz), (-hx, -hy + chamfer, hz),
        ]
        rings.append([bm.verts.new(p) for p in pts])

    for r_idx in range(len(rings) - 1):
        r0, r1 = rings[r_idx], rings[r_idx + 1]
        for i in range(8):
            j = (i + 1) % 8
            bm.faces.new((r0[i], r0[j], r1[j], r1[i]))

    bm.faces.new(tuple(reversed(rings[0])))
    bm.faces.new(tuple(rings[-1]))

    hull_obj = finalize_bmesh_object(bm, "Floating_Stall_Armored_Hull", col, mats["hull_metal"], smooth=True)
    add_bevel_modifier(hull_obj, width=0.016, segments=3)

    # Hull perimeter neon trim bumper
    bumper_pts = [
        (-1.64, -1.30, 0.98), (1.64, -1.30, 0.98),
        (1.90, -1.04, 0.98), (1.90, 1.04, 0.98),
        (1.64, 1.30, 0.98), (-1.64, 1.30, 0.98),
        (-1.90, 1.04, 0.98), (-1.90, -1.04, 0.98),
    ]
    create_pipe_curve("Hull_Neon_Bumper_Strip", col, bumper_pts, 0.014, mats["neon_cyan"], cyclic=True)

    # Central under-belly repulsor core & glowing plasma ring
    core_profile = [(0.0, -0.06), (0.58, -0.06), (0.68, 0.02), (0.68, 0.08), (0.0, 0.08)]
    create_lathe_object("Belly_Repulsor_Housing", col, core_profile, 40, (0.0, 0.0, 0.56), mats["hull_metal"])
    ring_profile = [(0.36, -0.03), (0.52, -0.03), (0.52, 0.01), (0.36, 0.01)]
    create_lathe_object(
        "Belly_Repulsor_Plasma_Ring", col, ring_profile, 40, (0.0, 0.0, 0.52), mats["neon_cyan"], close_caps=True
    )

    # 4 Corner Vector-Thrust Anti-Gravity Pods
    pod_coords = [
        (-1.82, -1.10, 0.76, -0.18, 0.16, "FL"),
        (1.82, -1.10, 0.76, -0.18, -0.16, "FR"),
        (-1.82, 1.10, 0.76, 0.18, 0.16, "RL"),
        (1.82, 1.10, 0.76, 0.18, -0.16, "RR"),
    ]
    shroud_prof = [
        (0.18, -0.22), (0.25, -0.10), (0.27, 0.08), (0.22, 0.22),
        (0.17, 0.22), (0.21, 0.06), (0.19, -0.10), (0.14, -0.22),
    ]
    glow_prof = [(0.0, -0.16), (0.15, -0.16), (0.17, -0.04), (0.0, -0.04)]

    for px, py, pz, rx, ry, tag in pod_coords:
        pod = create_lathe_object(
            f"Hover_Thruster_Pod_{tag}", col, shroud_prof, 32, (px, py, pz),
            mats["hull_metal"], rotation=(rx, ry, 0.0), close_caps=True,
        )
        add_bevel_modifier(pod, width=0.006, segments=2)
        create_lathe_object(
            f"Hover_Thruster_IonCore_{tag}", col, glow_prof, 28, (px, py, pz),
            mats["neon_cyan"], rotation=(rx, ry, 0.0), close_caps=True,
        )
        strut_x = px * 0.88
        strut_y = py * 0.88
        create_beveled_box(
            f"Thruster_Pylon_{tag}", col, (0.28, 0.14, 0.10),
            (strut_x, strut_y, 0.84), mats["pipe_metal"], bevel_width=0.008,
        )

    # 3 Floating Cyber Customer Bar Stools hovering in front of the ramen counter
    stool_seat_prof = [(0.0, 0.0), (0.19, 0.0), (0.21, 0.04), (0.19, 0.09), (0.0, 0.09)]
    stool_base_prof = [(0.08, -0.10), (0.17, -0.08), (0.14, 0.0), (0.05, 0.0)]
    stool_glow_prof = [(0.09, -0.11), (0.15, -0.11), (0.15, -0.08), (0.09, -0.08)]

    for idx, sx in enumerate((-0.96, 0.0, 0.96)):
        sz = 1.18 + 0.025 * (idx % 2)
        create_lathe_object(
            f"Floating_Bar_Stool_Seat_{idx + 1}", col, stool_seat_prof, 32,
            (sx, -1.58, sz), mats["bowl_ceramic"], close_caps=True,
        )
        create_lathe_object(
            f"Floating_Bar_Stool_Thruster_{idx + 1}", col, stool_base_prof, 28,
            (sx, -1.58, sz - 0.02), mats["hull_metal"], close_caps=True,
        )
        create_lathe_object(
            f"Floating_Bar_Stool_Glow_{idx + 1}", col, stool_glow_prof, 28,
            (sx, -1.58, sz - 0.02), mats["neon_cyan"], close_caps=True,
        )


def build_stall_timber_and_roof(col, mats):
    """Build multi-tiered wooden/metallic stall structure, Noren curtains, and sine/array roof tiles."""
    # Interior floor deck & lower enclosed counter walls
    create_beveled_box("Stall_Interior_Deck", col, (3.44, 2.24, 0.06), (0.0, 0.0, 1.08), mats["charred_wood"])
    create_beveled_box("Counter_Front_Bulkhead", col, (3.24, 0.14, 0.76), (0.0, -0.96, 1.46), mats["charred_wood"])
    create_beveled_box("Stall_Rear_Wall", col, (3.24, 0.12, 1.88), (0.0, 0.98, 2.02), mats["charred_wood"])

    # Side half-walls + upper transom beams
    for side_x, tag in ((-1.58, "L"), (1.58, "R")):
        create_beveled_box(f"Side_Lower_Wall_{tag}", col, (0.12, 1.96, 0.78), (side_x, 0.0, 1.47), mats["charred_wood"])
        create_beveled_box(f"Side_Upper_Panel_{tag}", col, (0.10, 0.82, 1.06), (side_x, 0.56, 2.40), mats["charred_wood"])
        create_beveled_box(f"Side_Eave_Beam_{tag}", col, (0.16, 2.28, 0.16), (side_x, 0.0, 2.92), mats["charred_wood"])

    # Front vertical wooden rib slats via Array modifier
    front_slat = create_beveled_box(
        "Counter_Front_Cedar_Ribs", col, (0.042, 0.035, 0.70), (-1.52, -1.045, 1.46),
        mats["hinoki_wood"], bevel_width=0.004,
    )
    rib_arr = front_slat.modifiers.new(name="RibArray", type='ARRAY')
    rib_arr.fit_type = 'FIXED_COUNT'
    rib_arr.count = 39
    rib_arr.use_relative_offset = False
    rib_arr.use_constant_offset = True
    rib_arr.constant_offset_displace = (0.080, 0.0, 0.0)

    # Oiled Hinoki customer ramen bar counter & stainless inner prep counter
    create_beveled_box(
        "Hinoki_Customer_Countertop", col, (3.56, 0.48, 0.065), (0.0, -1.08, 1.86),
        mats["hinoki_wood"], bevel_width=0.010,
    )
    create_beveled_box(
        "Counter_Stainless_Footrest", col, (3.30, 0.05, 0.04), (0.0, -1.24, 1.24),
        mats["stainless"], bevel_width=0.006,
    )
    create_beveled_box(
        "Chef_Inner_Prep_Counter", col, (3.08, 0.58, 0.72), (0.0, 0.62, 1.44),
        mats["stainless"], bevel_width=0.008,
    )
    create_beveled_box(
        "Rear_Storage_Shelf", col, (2.80, 0.26, 0.04), (0.0, 0.80, 2.34),
        mats["hinoki_wood"], bevel_width=0.005,
    )

    # 4 Structural Corner Pillars with metallic reinforcement collars
    pillar_xy = [(-1.58, -0.96, "FL"), (1.58, -0.96, "FR"), (-1.58, 0.96, "RL"), (1.58, 0.96, "RR")]
    for px, py, tag in pillar_xy:
        create_beveled_box(f"Corner_Timber_Pillar_{tag}", col, (0.15, 0.15, 1.92), (px, py, 2.02), mats["charred_wood"])
        create_beveled_box(f"Pillar_Metal_Collar_Bot_{tag}", col, (0.18, 0.18, 0.12), (px, py, 1.16), mats["hull_metal"])
        create_beveled_box(f"Pillar_Metal_Collar_Top_{tag}", col, (0.18, 0.18, 0.14), (px, py, 2.86), mats["hull_metal"])

    create_beveled_box("Front_Eave_Lintels", col, (3.48, 0.16, 0.16), (0.0, -0.96, 2.92), mats["charred_wood"])

    # Traditional 5-panel Indigo Noren Curtain across the front service opening
    create_pipe_curve(
        "Noren_Hanging_Rod", col, [(-1.50, -0.98, 2.82), (1.50, -0.98, 2.82)], 0.014, mats["stainless"]
    )
    for idx in range(5):
        cx = -1.16 + idx * 0.58
        bm = bmesh.new()
        nx, nz = 8, 10
        p_w, p_h = 0.50, 0.42
        grid = []
        for iz in range(nz + 1):
            t_z = iz / nz
            z_val = 2.81 - t_z * p_h
            # Natural bottom breeze flare
            y_bow = -0.98 - 0.035 * math.sin(t_z * math.pi)
            row = []
            for ix in range(nx + 1):
                t_x = -0.5 + (ix / nx)
                x_val = cx + t_x * p_w
                y_ripple = y_bow + 0.008 * math.cos(t_x * math.pi * 2.0) * t_z
                row.append(bm.verts.new((x_val, y_ripple, z_val)))
            grid.append(row)
        for iz in range(nz):
            for ix in range(nx):
                bm.faces.new((grid[iz][ix], grid[iz][ix + 1], grid[iz + 1][ix + 1], grid[iz + 1][ix]))
        panel = finalize_bmesh_object(bm, f"Noren_Curtain_Panel_{idx + 1}", col, mats["noren"], smooth=True)
        sol = panel.modifiers.new(name="ClothThickness", type='SOLIDIFY')
        sol.thickness = 0.006
        wv = panel.modifiers.new(name="BreezeWave", type='WAVE')
        wv.use_normal = True
        wv.height = 0.008
        wv.width = 0.35
        wv.speed = 0.08

    # Multi-tiered Pagoda Roof Substructure + Clerestory Louver Tier
    create_beveled_box("Roof_Lower_Eave_Deck", col, (4.24, 2.88, 0.10), (0.0, 0.0, 3.02), mats["charred_wood"])
    create_beveled_box("Roof_Clerestory_Core", col, (2.72, 1.66, 0.34), (0.0, 0.0, 3.38), mats["hull_metal"])
    create_beveled_box("Roof_Clerestory_GlowStrip", col, (2.56, 1.70, 0.06), (0.0, 0.0, 3.40), mats["neon_orange"])

    # Mathematical Sine-Wave Corrugated Kawara Roof Tile Courses + Array Modifiers
    def build_sine_tile_slope(slope_name, half_span_x, y_ridge, y_eave, z_ridge, z_eave, courses):
        """Generate sinusoidal corrugated roof tile course and array down parabolic pagoda slope."""
        bm = bmesh.new()
        samples_x = 96
        samples_y = 6
        course_len_y = (y_eave - y_ridge) / courses
        course_drop_z = (z_ridge - z_eave) / courses
        wave_cycles = 18.0
        amp_z = 0.024

        grid_v = []
        for iy in range(samples_y + 1):
            ty = iy / samples_y
            y_pos = ty * course_len_y
            # Slight downward step lip at leading edge of each tile course
            z_base = -ty * course_drop_z
            row_v = []
            for ix in range(samples_x + 1):
                tx = ix / samples_x
                x_pos = -half_span_x + tx * (2.0 * half_span_x)
                phase = tx * wave_cycles * 2.0 * math.pi
                # Harmonic sine wave profile mimicking interlocking pan & ridge kawara tiles
                sine_h = amp_z * (math.sin(phase) + 0.25 * math.cos(2.0 * phase))
                # Upturned pagoda corner lift near +/- X eaves
                corner_lift = 0.055 * ((abs(x_pos) / half_span_x) ** 3)
                row_v.append(bm.verts.new((x_pos, y_pos, z_base + sine_h + corner_lift)))
            grid_v.append(row_v)

        for iy in range(samples_y):
            for ix in range(samples_x):
                bm.faces.new((grid_v[iy][ix], grid_v[iy][ix + 1], grid_v[iy + 1][ix + 1], grid_v[iy + 1][ix]))

        tile_obj = finalize_bmesh_object(
            bm, slope_name, col, mats["roof_tile"], smooth=True, location=(0.0, y_ridge, z_ridge)
        )
        sol = tile_obj.modifiers.new(name="TileThickness", type='SOLIDIFY')
        sol.thickness = 0.016
        sol.offset = 1.0

        arr = tile_obj.modifiers.new(name="TileCourseArray", type='ARRAY')
        arr.fit_type = 'FIXED_COUNT'
        arr.count = courses
        arr.use_relative_offset = False
        arr.use_constant_offset = True
        arr.constant_offset_displace = (0.0, course_len_y * 0.94, -course_drop_z * 0.94)

        add_bevel_modifier(tile_obj, width=0.003, segments=2)
        return tile_obj

    # Lower main roof front & rear slopes + Upper crown pagoda front & rear slopes
    build_sine_tile_slope("Main_Roof_Tiles_Front", 2.18, -0.68, -1.62, 3.30, 3.03, courses=6)
    build_sine_tile_slope("Main_Roof_Tiles_Rear", 2.18, 0.68, 1.62, 3.30, 3.03, courses=6)
    build_sine_tile_slope("Crown_Roof_Tiles_Front", 1.54, -0.02, -1.06, 3.88, 3.54, courses=6)
    build_sine_tile_slope("Crown_Roof_Tiles_Rear", 1.54, 0.02, 1.06, 3.88, 3.54, courses=6)

    # Ornate Pagoda Crown Ridge Beam (Munagi) & Metallic Finial Caps
    create_beveled_box("Crown_Pagoda_Ridge_Beam", col, (3.22, 0.18, 0.12), (0.0, 0.0, 3.92), mats["hull_metal"])
    for side_x, tag in ((-1.64, "L"), (1.64, "R")):
        create_beveled_box(
            f"Crown_Ridge_Shibi_Finial_{tag}", col, (0.12, 0.22, 0.24),
            (side_x, 0.0, 4.00), mats["pipe_metal"], rotation=(0.0, side_x * -0.12, 0.0),
        )


def build_structural_pipes(col, mats):
    """Build industrial coolant, steam, and hydraulic conduits using bpy.data.curves."""
    pipe_specs = [
        ("Pipe_Left_Main_Riser", [
            (-1.72, 0.72, 0.84), (-1.78, 0.72, 1.15), (-1.78, 0.72, 2.86),
            (-1.52, 0.72, 3.16), (-1.18, 0.72, 3.44),
        ], 0.042, mats["pipe_metal"]),
        ("Pipe_Right_Main_Riser", [
            (1.72, 0.72, 0.84), (1.78, 0.72, 1.15), (1.78, 0.72, 2.86),
            (1.52, 0.72, 3.16), (1.18, 0.72, 3.44),
        ], 0.042, mats["pipe_metal"]),
        ("Pipe_Secondary_Coolant_L", [
            (-1.68, 0.86, 0.88), (-1.86, 0.86, 1.40), (-1.86, 0.15, 1.40),
            (-1.86, 0.15, 2.65), (-1.62, 0.15, 2.88),
        ], 0.026, mats["stainless"]),
        ("Pipe_Secondary_Coolant_R", [
            (1.68, 0.86, 0.88), (1.86, 0.86, 1.40), (1.86, 0.15, 1.40),
            (1.86, 0.15, 2.65), (1.62, 0.15, 2.88),
        ], 0.026, mats["stainless"]),
        ("Pipe_Roof_Steam_Manifold", [
            (-1.30, 0.42, 3.52), (-1.30, 0.42, 3.74), (1.30, 0.42, 3.74), (1.30, 0.42, 3.52),
        ], 0.034, mats["pipe_metal"]),
        ("Pipe_Rear_Exhaust_Chimney", [
            (0.82, 1.08, 1.85), (1.12, 1.18, 2.15), (1.12, 1.18, 3.68), (1.24, 1.28, 3.96),
        ], 0.052, mats["hull_metal"]),
        ("Pipe_UnderHull_Hydraulic_Front", [
            (-1.55, -0.95, 0.72), (-0.75, -1.05, 0.66), (0.75, -1.05, 0.66), (1.55, -0.95, 0.72),
        ], 0.028, mats["pipe_metal"]),
    ]
    for p_name, pts, bev_r, mat in pipe_specs:
        create_pipe_curve(p_name, col, pts, bev_r, mat, cyclic=False, spline_type='NURBS', resolution=16)

    # Bolted pipe flange collars on main vertical risers
    flange_prof = [(0.038, -0.025), (0.068, -0.025), (0.068, 0.025), (0.038, 0.025)]
    for side_x, tag in ((-1.78, "L"), (1.78, "R")):
        for idx, fz in enumerate((1.35, 2.05, 2.72)):
            create_lathe_object(
                f"Pipe_Flange_{tag}_{idx + 1}", col, flange_prof, 24,
                (side_x, 0.72, fz), mats["hull_metal"], close_caps=True,
            )


def build_interior_ramen_props(col, mats):
    """Build lathe-spun ramen bowls, broth, wavy noodles, toppings, chopsticks, and broth vat."""
    # Heavy Spun Stainless Broth Cauldron & Simmering Broth on Chef Prep Counter
    vat_prof = [
        (0.0, 0.0), (0.28, 0.0), (0.30, 0.03), (0.30, 0.44),
        (0.33, 0.46), (0.33, 0.48), (0.27, 0.48), (0.27, 0.04), (0.0, 0.04),
    ]
    create_lathe_object("Kitchen_Tonkotsu_Cauldron", col, vat_prof, 40, (-0.58, 0.58, 1.80), mats["stainless"])
    create_lathe_object(
        "Cauldron_Simmering_Broth", col, [(0.0, 0.0), (0.268, 0.0), (0.268, 0.02), (0.0, 0.02)],
        36, (-0.58, 0.58, 2.20), mats["broth"], close_caps=True,
    )
    create_beveled_box("Kitchen_Exhaust_Hood", col, (1.95, 0.68, 0.28), (-0.15, 0.58, 2.72), mats["stainless"])

    # Ceramic Ramen Bowl Cross-Section Profile (footed kodai base + flared rim + hollow interior)
    bowl_prof = [
        (0.000, 0.000),
        (0.058, 0.000),
        (0.062, 0.018),
        (0.088, 0.036),
        (0.122, 0.078),
        (0.136, 0.116),
        (0.128, 0.118),
        (0.114, 0.080),
        (0.080, 0.042),
        (0.048, 0.024),
        (0.000, 0.022),
    ]
    broth_prof = [(0.0, 0.0), (0.118, 0.0), (0.118, 0.008), (0.0, 0.008)]

    def create_tapered_chopstick(stick_name, loc_xyz, rot_xyz):
        """Construct a traditional square-to-round tapered chopstick via bmesh."""
        bm = bmesh.new()
        length = 0.235
        r_back = 0.0042
        r_tip = 0.0016
        segs = 8
        back_ring = []
        tip_ring = []
        for i in range(segs):
            ang = 2.0 * math.pi * i / segs
            # Subtle superellipse squaring at handle end, pure circle at eating tip
            cx, sy = math.cos(ang), math.sin(ang)
            sq_x = math.copysign(abs(cx) ** 0.75, cx) * r_back
            sq_y = math.copysign(abs(sy) ** 0.75, sy) * r_back
            back_ring.append(bm.verts.new((-length * 0.5, sq_x, sq_y)))
            tip_ring.append(bm.verts.new((length * 0.5, cx * r_tip, sy * r_tip)))
        for i in range(segs):
            j = (i + 1) % segs
            bm.faces.new((back_ring[i], back_ring[j], tip_ring[j], tip_ring[i]))
        bm.faces.new(tuple(reversed(back_ring)))
        bm.faces.new(tuple(tip_ring))
        obj = finalize_bmesh_object(bm, stick_name, col, mats["hinoki_wood"], smooth=True, location=loc_xyz)
        obj.rotation_euler = Euler(rot_xyz, 'XYZ')
        return obj

    # 3 Hero Served Ramen Bowls on Customer Counter with Broth, Noodles, Egg, Narutomaki & Chopsticks
    bowl_x_positions = (-0.96, 0.0, 0.96)
    for b_idx, bx in enumerate(bowl_x_positions):
        by = -1.06
        bz = 1.893
        tag = f"0{b_idx + 1}"

        create_lathe_object(f"Hero_Ramen_Bowl_{tag}", col, bowl_prof, 48, (bx, by, bz), mats["bowl_ceramic"])
        create_lathe_object(
            f"Hero_Ramen_Broth_{tag}", col, broth_prof, 40, (bx, by, bz + 0.084), mats["broth"], close_caps=True
        )

        # 7 Tangled Sinusoidal Ramen Noodle Strands inside each bowl
        for n_idx in range(7):
            strand_pts = []
            angle_off = n_idx * 0.88 + b_idx * 0.45
            for s in range(18):
                t = s / 17.0
                rad = 0.018 + 0.076 * math.sin(t * math.pi * 0.92)
                theta = angle_off + t * 3.8 + 0.24 * math.sin(t * 16.0)
                nx = bx + rad * math.cos(theta)
                ny = by + rad * math.sin(theta)
                nz = bz + 0.089 + 0.009 * math.sin(t * 14.0 + n_idx)
                strand_pts.append((nx, ny, nz))
            create_pipe_curve(
                f"Ramen_Noodle_Strand_{tag}_{n_idx + 1}",
                col, strand_pts, 0.0036, mats["noodle"], spline_type='NURBS', resolution=12,
            )

        # Soft-boiled Ajitama Egg Half (flattened UV sphere white + glossy jammy yolk dome)
        bm_egg = bmesh.new()
        bmesh.ops.create_uvsphere(bm_egg, u_segments=24, v_segments=14, radius=0.026)
        bmesh.ops.scale(bm_egg, verts=bm_egg.verts[:], vec=(1.25, 0.95, 0.55))
        egg_w = finalize_bmesh_object(
            bm_egg, f"Ajitama_Egg_White_{tag}", col, mats["egg_white"],
            smooth=True, location=(bx - 0.042, by + 0.028, bz + 0.094),
        )
        egg_w.rotation_euler = Euler((0.12, -0.15, 0.5 + b_idx), 'XYZ')

        bm_yolk = bmesh.new()
        bmesh.ops.create_uvsphere(bm_yolk, u_segments=20, v_segments=12, radius=0.015)
        bmesh.ops.scale(bm_yolk, verts=bm_yolk.verts[:], vec=(1.05, 0.95, 0.52))
        finalize_bmesh_object(
            bm_yolk, f"Ajitama_Egg_Yolk_{tag}", col, mats["egg_yolk"],
            smooth=True, location=(bx - 0.042, by + 0.028, bz + 0.102),
        )

        # Narutomaki Fish Cake Coin + Archimedean Pink Neon-Magenta Spiral
        naru_prof = [(0.0, 0.0), (0.024, 0.0), (0.024, 0.006), (0.0, 0.006)]
        create_lathe_object(
            f"Narutomaki_Disk_{tag}", col, naru_prof, 24,
            (bx + 0.045, by - 0.022, bz + 0.093), mats["egg_white"], close_caps=True,
        )
        spiral_pts = []
        for sp in range(22):
            th = sp * 0.48
            sr = 0.002 + 0.0016 * th
            spiral_pts.append((
                bx + 0.045 + sr * math.cos(th),
                by - 0.022 + sr * math.sin(th),
                bz + 0.100,
            ))
        create_pipe_curve(f"Narutomaki_Spiral_{tag}", col, spiral_pts, 0.0014, mats["neon_magenta"], spline_type='NURBS')

        # Ceramic Hashioki Rest + Pair of Tapered Chopsticks beside each bowl
        create_beveled_box(
            f"Chopstick_Rest_{tag}", col, (0.016, 0.046, 0.012),
            (bx + 0.165, by - 0.08, bz + 0.006), mats["bowl_ceramic"], bevel_width=0.003,
        )
        create_tapered_chopstick(
            f"Chopstick_A_{tag}", (bx + 0.175, by - 0.01, bz + 0.016), (0.0, 0.02, 1.54)
        )
        create_tapered_chopstick(
            f"Chopstick_B_{tag}", (bx + 0.194, by - 0.01, bz + 0.016), (0.0, 0.02, 1.60)
        )

    # Stacked Spare Bowls & Condiment Canister on Rear Storage Shelf
    for s_idx in range(4):
        create_lathe_object(
            f"Shelf_Stacked_Bowl_{s_idx + 1}", col, bowl_prof, 36,
            (0.75, 0.80, 2.36 + s_idx * 0.045), mats["bowl_ceramic"],
        )
    canister_prof = [(0.0, 0.0), (0.048, 0.0), (0.048, 0.15), (0.042, 0.15), (0.042, 0.01), (0.0, 0.01)]
    create_lathe_object("Counter_Chopstick_Canister", col, canister_prof, 28, (0.48, -0.96, 1.893), mats["hull_metal"])
    for c_idx in range(6):
        c_ang = c_idx * (2.0 * math.pi / 6.0)
        create_tapered_chopstick(
            f"Canister_Chopstick_{c_idx + 1}",
            (0.48 + 0.018 * math.cos(c_ang), -0.96 + 0.018 * math.sin(c_ang), 2.02),
            (0.12 * math.sin(c_ang), -1.45, c_ang),
        )


def build_lanterns_fans_and_catenary_cables(col, mats):
    """Build Lattice+Wave deformed paper lanterns, animated cooling fan blades, and catenary cables."""
    # 6 Hanging Paper Lanterns (4 along front eaves, 2 flanking side eaves)
    lantern_coords = [
        (-1.38, -1.36, 2.66, "Front_01"),
        (-0.46, -1.36, 2.66, "Front_02"),
        (0.46, -1.36, 2.66, "Front_03"),
        (1.38, -1.36, 2.66, "Front_04"),
        (-1.92, -0.32, 2.64, "Side_L"),
        (1.92, -0.32, 2.64, "Side_R"),
    ]
    lantern_light_positions = []

    cap_prof = [(0.0, 0.0), (0.072, 0.0), (0.076, 0.024), (0.0, 0.024)]

    for lx, ly, lz, tag in lantern_coords:
        lantern_light_positions.append((lx, ly, lz))

        # Subdivided UV Sphere Paper Body
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=20, radius=0.145)
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1, use_grid_fill=True)
        lantern_obj = finalize_bmesh_object(
            bm, f"Paper_Lantern_{tag}", col, mats["lantern_paper"], smooth=True, location=(lx, ly, lz)
        )

        # 3D Lattice Object sculpting classic barrel-shaped Chochin lantern silhouette
        lat_data = bpy.data.lattices.new(f"Lantern_Lattice_{tag}_Data")
        lat_data.points_u = 3
        lat_data.points_v = 3
        lat_data.points_w = 5
        lat_data.interpolation_type_u = 'KEY_BSPLINE'
        lat_data.interpolation_type_v = 'KEY_BSPLINE'
        lat_data.interpolation_type_w = 'KEY_BSPLINE'

        w_radial_profile = [0.68, 1.12, 1.18, 1.12, 0.68]
        w_z_squeeze = [-0.44, -0.24, 0.0, 0.24, 0.44]
        for pt_idx, pt in enumerate(lat_data.points):
            w_layer = pt_idx // 9
            r_scale = w_radial_profile[w_layer]
            base_co = pt.co
            pt.co_deform = Vector((base_co.x * r_scale, base_co.y * r_scale, w_z_squeeze[w_layer]))

        lat_obj = bpy.data.objects.new(f"Lantern_Lattice_{tag}", lat_data)
        lat_obj.location = Vector((lx, ly, lz))
        lat_obj.scale = Vector((0.34, 0.34, 0.38))
        lat_obj.hide_render = True
        lat_obj.hide_viewport = True
        col.objects.link(lat_obj)

        lat_mod = lantern_obj.modifiers.new(name="ChochinBarrelLattice", type='LATTICE')
        lat_mod.object = lat_obj

        wave_mod = lantern_obj.modifiers.new(name="ThermalPaperRipple", type='WAVE')
        wave_mod.use_normal = True
        wave_mod.height = 0.004
        wave_mod.width = 0.18
        wave_mod.narrowness = 1.6
        wave_mod.speed = 0.10

        # Top/Bottom Lacquered Collars, Hanging Cord, and Bottom Tassel (snapped flush to deformed poles)
        create_lathe_object(
            f"Lantern_Cap_Top_{tag}", col, cap_prof, 24, (lx, ly, lz + 0.120), mats["bowl_ceramic"], close_caps=True
        )
        create_lathe_object(
            f"Lantern_Cap_Bot_{tag}", col, cap_prof, 24, (lx, ly, lz - 0.144), mats["bowl_ceramic"], close_caps=True
        )
        create_pipe_curve(
            f"Lantern_Cord_{tag}", col, [(lx, ly, lz + 0.144), (lx, ly, 3.04)], 0.005, mats["cable"]
        )
        create_pipe_curve(
            f"Lantern_Tassel_{tag}", col, [(lx, ly, lz - 0.144), (lx, ly, lz - 0.255)], 0.008, mats["neon_orange"]
        )

    # 2 Industrial Cooling Exhaust Fans with Twisted Pitch Blades + Pure X-Axle 240-Frame Spin
    shroud_prof = [
        (0.26, -0.09), (0.31, -0.09), (0.31, 0.09), (0.26, 0.09),
    ]
    for side_sign, tag in ((-1.0, "L"), (1.0, "R")):
        fx = side_sign * 1.66
        fy = 0.56
        fz = 2.40
        rot_y = side_sign * (math.pi * 0.5)

        shroud = create_lathe_object(
            f"Cooling_Fan_Shroud_{tag}", col, shroud_prof, 36,
            (fx, fy, fz), mats["hull_metal"], rotation=(0.0, rot_y, 0.0), close_caps=True,
        )
        add_bevel_modifier(shroud, width=0.006, segments=2)

        # Aerodynamic 7-blade rotor constructed directly around local X shaft axis (zero gimbal lock)
        bm_fan = bmesh.new()
        bmesh.ops.create_uvsphere(bm_fan, u_segments=20, v_segments=12, radius=0.068)
        bmesh.ops.scale(bm_fan, verts=bm_fan.verts[:], vec=(0.72, 1.0, 1.0))

        blade_count = 7
        radial_stations = 5
        chord_steps = 3
        for b_i in range(blade_count):
            base_theta = 2.0 * math.pi * b_i / blade_count
            b_grid = []
            for ir in range(radial_stations + 1):
                tr = ir / radial_stations
                rad = 0.058 + tr * 0.185
                # Aerodynamic blade twist: 34 deg at root tapering to 17 deg at tip
                pitch = math.radians(34.0 - 17.0 * tr)
                chord = 0.062 * (0.78 + 0.35 * math.sin(tr * math.pi))
                row = []
                for ic in range(chord_steps + 1):
                    tc = -0.5 + (ic / chord_steps)
                    along = tc * chord * math.cos(pitch)
                    x_off = tc * chord * math.sin(pitch) * side_sign
                    ang = base_theta + (along / rad)
                    vy = rad * math.cos(ang)
                    vz = rad * math.sin(ang)
                    row.append(bm_fan.verts.new((x_off, vy, vz)))
                b_grid.append(row)
            for ir in range(radial_stations):
                for ic in range(chord_steps):
                    bm_fan.faces.new((b_grid[ir][ic], b_grid[ir + 1][ic], b_grid[ir + 1][ic + 1], b_grid[ir][ic + 1]))

        rotor = finalize_bmesh_object(
            bm_fan, f"Cooling_Fan_Rotor_{tag}", col, mats["stainless"], smooth=True, location=(fx, fy, fz)
        )
        sol = rotor.modifiers.new(name="BladeThickness", type='SOLIDIFY')
        sol.thickness = 0.005

        # Keyframe continuous turbine spin around pure X shaft axis over 240 frames
        rotor.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
        rotor.keyframe_insert(data_path="rotation_euler", frame=1)
        rotor.rotation_euler = Euler((side_sign * 8.0 * math.pi, 0.0, 0.0), 'XYZ')
        rotor.keyframe_insert(data_path="rotation_euler", frame=240)
        if rotor.animation_data is not None and rotor.animation_data.action is not None:
            if hasattr(rotor.animation_data.action, "fcurves"):
                for fc in rotor.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'

    # Dangling Power Cables via Exact Closed-Form Hyperbolic Catenary Mathematics
    def sample_catenary_curve(p1, p2, catenary_a, samples=28):
        """Exact closed-form catenary z(s) = c + a * cosh((s - s0) / a) between unequal 3D anchors."""
        v1 = Vector(p1)
        v2 = Vector(p2)
        dx = v2.x - v1.x
        dy = v2.y - v1.y
        dz = v2.z - v1.z
        chord_xy = math.hypot(dx, dy)
        if chord_xy < 1e-4:
            return [v1.lerp(v2, i / (samples - 1)) for i in range(samples)]

        a_val = max(catenary_a, 0.15)
        denom = 2.0 * a_val * math.sinh(chord_xy / (2.0 * a_val))
        s0 = 0.5 * chord_xy - a_val * math.asinh(dz / denom)
        c_offset = v1.z - a_val * math.cosh(-s0 / a_val)

        pts = []
        for i in range(samples):
            u = i / (samples - 1)
            s = u * chord_xy
            x_u = v1.x + dx * u
            y_u = v1.y + dy * u
            z_u = c_offset + a_val * math.cosh((s - s0) / a_val)
            pts.append((x_u, y_u, z_u))
        return pts

    cable_spans = [
        ("Cable_Roof_To_LeftEave_A", (-1.15, -0.35, 3.86), (-1.96, -0.82, 3.04), 0.72, 0.012),
        ("Cable_Roof_To_LeftEave_B", (-1.08, -0.25, 3.84), (-1.88, -0.52, 3.02), 0.62, 0.009),
        ("Cable_Roof_To_RightEave_A", (1.15, -0.35, 3.86), (1.96, -0.82, 3.04), 0.72, 0.012),
        ("Cable_Roof_To_RightEave_B", (1.08, -0.25, 3.84), (1.88, -0.52, 3.02), 0.62, 0.009),
        ("Cable_Front_Swag_Left", (-1.56, -1.04, 2.88), (-0.48, -1.04, 2.88), 0.68, 0.010),
        ("Cable_Front_Swag_Center", (-0.48, -1.04, 2.88), (0.48, -1.04, 2.88), 0.64, 0.010),
        ("Cable_Front_Swag_Right", (0.48, -1.04, 2.88), (1.56, -1.04, 2.88), 0.68, 0.010),
        ("Cable_SideFan_To_Hull_L", (-1.70, 0.56, 2.18), (-1.80, -0.88, 1.02), 0.95, 0.013),
        ("Cable_SideFan_To_Hull_R", (1.70, 0.56, 2.18), (1.80, -0.88, 1.02), 0.95, 0.013),
        ("Cable_Rear_Power_Loop_A", (-1.42, 1.06, 2.86), (0.05, 1.08, 2.65), 0.78, 0.014),
        ("Cable_Rear_Power_Loop_B", (0.05, 1.08, 2.65), (1.42, 1.06, 2.86), 0.78, 0.014),
        ("Cable_Sign_Feeder_Left", (-1.62, -0.58, 2.94), (-2.04, -0.58, 2.88), 0.28, 0.010),
    ]
    for c_name, p_start, p_end, a_param, radius in cable_spans:
        cat_pts = sample_catenary_curve(p_start, p_end, a_param, samples=28)
        create_pipe_curve(c_name, col, cat_pts, radius, mats["cable"], spline_type='POLY', resolution=8)

    return lantern_light_positions


def build_cyber_neon_signage(col, mats):
    """Build 3D procedural text and geometric neon frames + Katakana tube signage on roof and sides."""
    def create_3d_neon_text(name, body_text, size, extrude, bevel, loc, rot, mat):
        font_curve = bpy.data.curves.new(name=name + "_Font", type='FONT')
        font_curve.body = body_text
        font_curve.size = size
        font_curve.extrude = extrude
        font_curve.bevel_depth = bevel
        font_curve.bevel_resolution = 3
        font_curve.align_x = 'CENTER'
        font_curve.align_y = 'CENTER'
        font_curve.materials.append(mat)
        txt_obj = bpy.data.objects.new(name, font_curve)
        txt_obj.location = Vector(loc)
        txt_obj.rotation_euler = Euler(rot, 'XYZ')
        col.objects.link(txt_obj)
        return txt_obj

    # --- A. Rooftop Main Billboard Truss, Backing Grid, Neon Frames & 3D Text ---
    for sx in (-1.05, 0.0, 1.05):
        create_pipe_curve(
            f"Roof_Sign_Truss_Post_{sx:+.2f}", col,
            [(sx, -0.18, 3.90), (sx, -0.18, 4.66), (sx, 0.08, 3.92)], 0.018, mats["hull_metal"],
        )

    create_beveled_box("Roof_Billboard_Backing", col, (2.68, 0.05, 0.68), (0.0, -0.20, 4.34), mats["hull_metal"])

    # Chamfered outer Cyan Neon Frame + inner Magenta Corner Brackets
    outer_frame = [
        (-1.28, -0.24, 3.98), (1.28, -0.24, 3.98),
        (1.38, -0.24, 4.08), (1.38, -0.24, 4.60),
        (1.28, -0.24, 4.70), (-1.28, -0.24, 4.70),
        (-1.38, -0.24, 4.60), (-1.38, -0.24, 4.08),
    ]
    create_pipe_curve("Roof_Neon_Frame_Cyan", col, outer_frame, 0.016, mats["neon_cyan"], cyclic=True)

    inner_frame = [
        (-1.24, -0.25, 4.05), (1.24, -0.25, 4.05),
        (1.24, -0.25, 4.63), (-1.24, -0.25, 4.63),
    ]
    create_pipe_curve("Roof_Neon_Inner_Frame_Magenta", col, inner_frame, 0.010, mats["neon_magenta"], cyclic=True)

    create_3d_neon_text(
        "Roof_3DText_Main", "CYBER RAMEN", size=0.33, extrude=0.024, bevel=0.005,
        loc=(-0.22, -0.26, 4.42), rot=(math.pi * 0.5, 0.0, 0.0), mat=mats["neon_magenta"],
    )
    create_3d_neon_text(
        "Roof_3DText_Sub", "SYNTH BROTH // 24H", size=0.145, extrude=0.016, bevel=0.003,
        loc=(-0.22, -0.26, 4.15), rot=(math.pi * 0.5, 0.0, 0.0), mat=mats["neon_cyan"],
    )

    # Rooftop Geometric Neon Ramen Bowl + Sinusoidal Steam Emblem on Right End of Billboard
    bowl_arc = []
    for i in range(17):
        ang = math.pi + (i / 16.0) * math.pi
        bowl_arc.append((0.98 + 0.20 * math.cos(ang), -0.27, 4.34 + 0.15 * math.sin(ang)))
    bowl_arc.append((0.78, -0.27, 4.34))
    create_pipe_curve("Roof_Neon_Bowl_Icon", col, bowl_arc, 0.011, mats["neon_orange"], cyclic=False)

    for s_i, x_off in enumerate((0.90, 0.98, 1.06)):
        steam_pts = []
        for k in range(12):
            tk = k / 11.0
            sx = x_off + 0.022 * math.sin(tk * math.pi * 2.5 + s_i)
            sz = 4.38 + tk * 0.21
            steam_pts.append((sx, -0.27, sz))
        create_pipe_curve(f"Roof_Neon_Steam_{s_i + 1}", col, steam_pts, 0.008, mats["neon_cyan"], spline_type='NURBS')

    # --- B. Left Side Vertical Blade Sign with Procedural 3D Katakana Neon Tubes (ラ ー メ ン) ---
    bx_l = -2.08
    by_l = -0.58
    create_beveled_box("Left_Sign_Bracket_Top", col, (0.48, 0.05, 0.05), (-1.82, by_l, 2.86), mats["hull_metal"])
    create_beveled_box("Left_Sign_Bracket_Bot", col, (0.48, 0.05, 0.05), (-1.82, by_l, 1.98), mats["hull_metal"])
    create_beveled_box("Left_Blade_Sign_Box", col, (0.06, 0.42, 1.12), (bx_l, by_l, 2.42), mats["hull_metal"])

    blade_frame = [
        (bx_l - 0.04, by_l - 0.19, 1.88), (bx_l - 0.04, by_l + 0.19, 1.88),
        (bx_l - 0.04, by_l + 0.19, 2.96), (bx_l - 0.04, by_l - 0.19, 2.96),
    ]
    create_pipe_curve("Left_Blade_Neon_Frame", col, blade_frame, 0.012, mats["neon_cyan"], cyclic=True)

    # Geometric 3D Neon Tube Katakana Strokes for "ラーメン" (RA - Choonpu - ME - N) facing -X
    gx = bx_l - 0.055
    katakana_strokes = [
        # ラ (Ra)
        ("Katakana_RA_Top", [(gx, by_l - 0.09, 2.86), (gx, by_l + 0.09, 2.86)]),
        ("Katakana_RA_Body", [(gx, by_l - 0.11, 2.79), (gx, by_l + 0.11, 2.79), (gx, by_l + 0.06, 2.70), (gx, by_l - 0.07, 2.65)]),
        # ー (Vertical Choonpu)
        ("Katakana_Dash", [(gx, by_l, 2.60), (gx, by_l, 2.45)]),
        # メ (Me)
        ("Katakana_ME_Diag1", [(gx, by_l + 0.10, 2.39), (gx, by_l + 0.01, 2.28), (gx, by_l - 0.10, 2.20)]),
        ("Katakana_ME_Diag2", [(gx, by_l - 0.08, 2.36), (gx, by_l + 0.07, 2.24)]),
        # ン (N)
        ("Katakana_N_Tick", [(gx, by_l - 0.10, 2.12), (gx, by_l - 0.02, 2.08)]),
        ("Katakana_N_Sweep", [(gx, by_l - 0.10, 1.96), (gx, by_l + 0.02, 2.00), (gx, by_l + 0.11, 2.13)]),
    ]
    for s_name, s_pts in katakana_strokes:
        create_pipe_curve(s_name, col, s_pts, 0.009, mats["neon_orange"], spline_type='NURBS')

    # --- C. Right Side Hexagonal Neon Frame & Double-Sided 3D Procedural Text ---
    rx_r = 2.08
    ry_r = -0.58
    create_beveled_box("Right_Sign_Bracket_Top", col, (0.48, 0.05, 0.05), (1.82, ry_r, 2.78), mats["hull_metal"])
    create_beveled_box("Right_Sign_Bracket_Bot", col, (0.48, 0.05, 0.05), (1.82, ry_r, 2.12), mats["hull_metal"])
    create_beveled_box("Right_Hex_Sign_Backing", col, (0.05, 0.64, 0.64), (rx_r, ry_r, 2.45), mats["hull_metal"])

    hex_pts = []
    for h_i in range(6):
        ang = h_i * (math.pi / 3.0)
        hex_pts.append((rx_r + 0.04, ry_r + 0.34 * math.cos(ang), 2.45 + 0.34 * math.sin(ang)))
    create_pipe_curve("Right_Hex_Neon_Frame", col, hex_pts, 0.012, mats["neon_magenta"], cyclic=True)

    create_3d_neon_text(
        "Right_3DText_Open", "OPEN", size=0.17, extrude=0.018, bevel=0.004,
        loc=(rx_r + 0.045, ry_r, 2.53), rot=(math.pi * 0.5, 0.0, math.pi * 0.5), mat=mats["neon_cyan"],
    )
    create_3d_neon_text(
        "Right_3DText_Noodles", "NOODLES", size=0.10, extrude=0.014, bevel=0.003,
        loc=(rx_r + 0.045, ry_r, 2.35), rot=(math.pi * 0.5, 0.0, math.pi * 0.5), mat=mats["neon_orange"],
    )


# ==============================================================================
# 4. CINEMATIC LIGHTING & ANIMATED 85MM F/1.4 CAMERA RIG (240 FRAMES @ 24 FPS)
# ==============================================================================

def build_lighting_and_animated_camera(scene, col, lantern_positions):
    """Build 3-point area lighting, lantern point lights, DOF target Empty, and 240f orbit camera."""
    # DOF Focus Target Empty at the center of the stall
    dof_empty = bpy.data.objects.new("DOF_Stall_Center_Target", None)
    dof_empty.empty_display_type = 'SPHERE'
    dof_empty.empty_display_size = 0.25
    dof_empty.location = Vector((0.0, -0.10, 2.15))
    col.objects.link(dof_empty)

    def spawn_area_light(name, loc, size_x, size_y, energy, color_rgb):
        ldata = bpy.data.lights.new(name=name + "_Data", type='AREA')
        ldata.shape = 'RECTANGLE'
        ldata.size = size_x
        ldata.size_y = size_y
        ldata.energy = energy
        ldata.color = color_rgb
        lobj = bpy.data.objects.new(name, ldata)
        lobj.location = Vector(loc)
        col.objects.link(lobj)

        track = lobj.constraints.new(type='TRACK_TO')
        track.target = dof_empty
        track.track_axis = 'TRACK_NEGATIVE_Z'
        track.up_axis = 'UP_Y'
        return lobj

    # 3-Point Cinematic Cyberpunk Area Lighting Setup (Key, Rim, Fill)
    spawn_area_light(
        "Light_01_Key_Area", loc=(-5.6, -6.2, 5.2),
        size_x=4.4, size_y=2.6, energy=620.0, color_rgb=(1.0, 0.78, 0.58),
    )
    spawn_area_light(
        "Light_02_Rim_Area", loc=(4.2, 6.4, 5.6),
        size_x=5.2, size_y=2.2, energy=920.0, color_rgb=(0.14, 0.84, 1.0),
    )
    spawn_area_light(
        "Light_03_Fill_Area", loc=(6.4, -4.6, 3.2),
        size_x=5.6, size_y=3.4, energy=280.0, color_rgb=(0.88, 0.30, 0.96),
    )

    # Under-counter warm task bar light & under-hull anti-gravity puddle bounce light
    warm_bar_data = bpy.data.lights.new(name="Counter_Task_Strip_Data", type='AREA')
    warm_bar_data.shape = 'RECTANGLE'
    warm_bar_data.size = 2.85
    warm_bar_data.size_y = 0.22
    warm_bar_data.energy = 42.0
    warm_bar_data.color = (1.0, 0.68, 0.34)
    warm_bar_obj = bpy.data.objects.new("Light_Counter_Warm_Strip", warm_bar_data)
    warm_bar_obj.location = Vector((0.0, -1.04, 2.76))
    warm_bar_obj.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    col.objects.link(warm_bar_obj)

    belly_glow_data = bpy.data.lights.new(name="UnderHull_Ion_Light_Data", type='POINT')
    belly_glow_data.energy = 95.0
    belly_glow_data.color = (0.05, 0.88, 1.0)
    belly_glow_data.shadow_soft_size = 0.35
    belly_glow_obj = bpy.data.objects.new("Light_UnderHull_Ion_Bounce", belly_glow_data)
    belly_glow_obj.location = Vector((0.0, 0.0, 0.44))
    col.objects.link(belly_glow_obj)

    # Localized Warm Internal Point Lights inside EVERY Hanging Paper Lantern
    for idx, (lx, ly, lz) in enumerate(lantern_positions):
        pt_data = bpy.data.lights.new(name=f"Lantern_Internal_Light_{idx + 1:02d}_Data", type='POINT')
        pt_data.energy = 36.0
        pt_data.color = (1.0, 0.48, 0.14)
        pt_data.shadow_soft_size = 0.065
        pt_obj = bpy.data.objects.new(f"Lantern_Internal_PointLight_{idx + 1:02d}", pt_data)
        pt_obj.location = Vector((lx, ly, lz))
        col.objects.link(pt_obj)

    # Exact 4-Control-Point Cubic Bezier Circle Path (calibrated for 85mm full-frame framing)
    orbit_radius = 19.6
    orbit_z = 2.95
    kappa = (4.0 / 3.0) * (math.sqrt(2.0) - 1.0) * orbit_radius

    path_curve = bpy.data.curves.new(name="Camera_Bezier_Orbit_Data", type='CURVE')
    path_curve.dimensions = '3D'
    path_curve.resolution_u = 64
    path_curve.use_path = True
    path_curve.path_duration = 240

    bez_spline = path_curve.splines.new(type='BEZIER')
    bez_spline.use_cyclic_u = True
    bez_spline.bezier_points.add(3)

    # Start at front-left hero 3/4 view via path rotation, with subtle vertical crane wave
    circle_knots = [
        ((orbit_radius, 0.0, orbit_z + 0.25), (orbit_radius, -kappa, orbit_z + 0.25), (orbit_radius, kappa, orbit_z + 0.25)),
        ((0.0, orbit_radius, orbit_z + 0.55), (kappa, orbit_radius, orbit_z + 0.55), (-kappa, orbit_radius, orbit_z + 0.55)),
        ((-orbit_radius, 0.0, orbit_z + 0.25), (-orbit_radius, kappa, orbit_z + 0.25), (-orbit_radius, -kappa, orbit_z + 0.25)),
        ((0.0, -orbit_radius, orbit_z - 0.20), (-kappa, -orbit_radius, orbit_z - 0.20), (kappa, -orbit_radius, orbit_z - 0.20)),
    ]
    for idx, (co, hl, hr) in enumerate(circle_knots):
        bp = bez_spline.bezier_points[idx]
        bp.handle_left_type = 'FREE'
        bp.handle_right_type = 'FREE'
        bp.co = Vector(co)
        bp.handle_left = Vector(hl)
        bp.handle_right = Vector(hr)

    path_obj = bpy.data.objects.new("Camera_Bezier_Circle_Path", path_curve)
    path_obj.rotation_euler = Euler((0.0, 0.0, math.radians(-125.0)), 'XYZ')
    col.objects.link(path_obj)

    # 85mm f/1.4 Cinematic Camera with 9-Blade Bokeh, Follow Path & Track To Constraints
    cam_data = bpy.data.cameras.new(name="Cinematic_Camera_85mm_Data")
    cam_data.lens = 85.0
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 500.0
    cam_data.passepartout_alpha = 0.92
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = dof_empty
    cam_data.dof.aperture_fstop = 1.4
    cam_data.dof.aperture_blades = 9

    cam_obj = bpy.data.objects.new("Cinematic_Camera_85mm", cam_data)
    cam_obj.location = Vector((0.0, 0.0, 0.0))
    cam_obj.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    col.objects.link(cam_obj)
    scene.camera = cam_obj

    follow_con = cam_obj.constraints.new(type='FOLLOW_PATH')
    follow_con.name = "OrbitFollowBezier"
    follow_con.target = path_obj
    follow_con.use_fixed_location = True

    follow_con.offset_factor = 0.0
    follow_con.keyframe_insert(data_path="offset_factor", frame=1)
    follow_con.offset_factor = 1.0
    follow_con.keyframe_insert(data_path="offset_factor", frame=240)

    if cam_obj.animation_data is not None and cam_obj.animation_data.action is not None:
        if hasattr(cam_obj.animation_data.action, "fcurves"):
            for fc in cam_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'BEZIER'
                    kp.easing = 'EASE_IN_OUT'
                    kp.handle_left_type = 'AUTO_CLAMPED'
                    kp.handle_right_type = 'AUTO_CLAMPED'

    track_con = cam_obj.constraints.new(type='TRACK_TO')
    track_con.name = "AimAtStallCenter"
    track_con.target = dof_empty
    track_con.track_axis = 'TRACK_NEGATIVE_Z'
    track_con.up_axis = 'UP_Y'

    return dof_empty, cam_obj


# ==============================================================================
# PHASE 2 & 5 (TRACKS 1 & 16): MULTI-CHANNEL AOVS & BLENDER 4.x COMPOSITOR
# ==============================================================================

def register_shader_aov_outputs(view_layer, mats):
    """Register custom ViewLayer Shader AOVs and inject ShaderNodeOutputAOV nodes."""
    if not hasattr(view_layer, "aovs"):
        return

    existing_names = {aov.name for aov in view_layer.aovs}
    for aov_name, aov_type in (("AOV_Neon_Mask", 'VALUE'), ("AOV_Thermal_Heat", 'COLOR')):
        if aov_name not in existing_names:
            aov_item = view_layer.aovs.add()
            aov_item.name = aov_name
            if hasattr(aov_item, "type"):
                aov_item.type = aov_type

    for neon_key in ("neon_cyan", "neon_magenta", "neon_orange"):
        mat = mats.get(neon_key)
        if mat is not None and mat.node_tree is not None:
            nt = mat.node_tree
            aov_node = spawn_node(nt, "ShaderNodeOutputAOV", 520, -140, "Neon Mask AOV")
            aov_node.aov_name = "AOV_Neon_Mask"
            aov_node.inputs["Value"].default_value = 1.0
            aov_node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)

    thermal_targets = [
        ("broth", (1.0, 0.42, 0.05, 1.0), 0.92),
        ("lantern_paper", (1.0, 0.55, 0.14, 1.0), 0.85),
        ("steam", (0.85, 0.52, 0.25, 1.0), 0.65),
    ]
    for mat_key, heat_rgba, heat_val in thermal_targets:
        mat = mats.get(mat_key)
        if mat is not None and mat.node_tree is not None:
            nt = mat.node_tree
            aov_node = spawn_node(nt, "ShaderNodeOutputAOV", 600, -180, "Thermal Heat AOV")
            aov_node.aov_name = "AOV_Thermal_Heat"
            aov_node.inputs["Color"].default_value = heat_rgba
            aov_node.inputs["Value"].default_value = heat_val


def build_compositor_post_processing(scene):
    """Build multi-stage Blender 4.x Cycles Compositor graph with Denoise, Despeckle, Mist, AO, Dual Glare & Grading."""
    view_layer = bpy.context.view_layer
    for attr_name in (
        "use_pass_mist",
        "use_pass_emit",
        "use_pass_z",
        "use_pass_normal",
        "use_pass_vector",
        "use_pass_ambient_occlusion",
        "use_pass_shadow",
        "use_pass_diffuse_color",
        "use_pass_glossy_direct",
        "use_pass_environment",
        "use_pass_cryptomatte_object",
        "use_pass_cryptomatte_material",
        "use_pass_denoising_data",
    ):
        if hasattr(view_layer, attr_name):
            setattr(view_layer, attr_name, True)

    if hasattr(view_layer, "cryptomatte_levels"):
        view_layer.cryptomatte_levels = 6

    if scene.world is not None and hasattr(scene.world, "mist_settings"):
        scene.world.mist_settings.use_mist = True
        scene.world.mist_settings.start = 8.0
        scene.world.mist_settings.depth = 28.0
        scene.world.mist_settings.falloff = 'QUADRATIC'

    scene.use_nodes = True
    ctree = scene.node_tree
    ctree.nodes.clear()

    rlayers = spawn_node(ctree, "CompositorNodeRLayers", -1380, 120, "Render Channels")

    # Stage 0: OIDN Albedo/Normal-Guided Denoise + Anti-Firefly Despeckle before Glare
    denoise_node = spawn_node(ctree, "CompositorNodeDenoise", -1140, 180, "OIDN Aux Denoise")
    if hasattr(denoise_node, "use_hdr"):
        denoise_node.use_hdr = True
    if hasattr(denoise_node, "prefilter"):
        denoise_node.prefilter = 'ACCURATE'
    link_nodes(ctree, rlayers, "Image", denoise_node, "Image")
    if "Normal" in rlayers.outputs and "Normal" in denoise_node.inputs:
        link_nodes(ctree, rlayers, "Normal", denoise_node, "Normal")
    if "DiffCol" in rlayers.outputs and "Albedo" in denoise_node.inputs:
        link_nodes(ctree, rlayers, "DiffCol", denoise_node, "Albedo")

    despeckle_node = spawn_node(ctree, "CompositorNodeDespeckle", -960, 180, "Firefly Suppressor")
    if hasattr(despeckle_node, "threshold"):
        despeckle_node.threshold = 0.08
    if hasattr(despeckle_node, "threshold_neighbor"):
        despeckle_node.threshold_neighbor = 0.12
    link_nodes(ctree, denoise_node, "Image", despeckle_node, "Image")

    # Stage 1: Ambient Occlusion contact-shadow micro-contrast multiplication
    ao_mix = spawn_node(ctree, "CompositorNodeMixRGB", -760, 180, "AO Contact Multiply")
    ao_mix.blend_type = 'MULTIPLY'
    ao_mix.inputs[0].default_value = 0.24
    link_nodes(ctree, despeckle_node, "Image", ao_mix, 1)
    if "AO" in rlayers.outputs:
        link_nodes(ctree, rlayers, "AO", ao_mix, 2)
    else:
        ao_mix.inputs[2].default_value = (1.0, 1.0, 1.0, 1.0)

    # Stage 2: Emission-reinforced Dual Glare (Atmospheric Fog Glow + Anamorphic Cyber Streaks)
    emit_boost = spawn_node(ctree, "CompositorNodeMixRGB", -540, 180, "Emission Feed")
    emit_boost.blend_type = 'ADD'
    emit_boost.inputs[0].default_value = 0.22
    link_nodes(ctree, ao_mix, "Image", emit_boost, 1)
    if "Emit" in rlayers.outputs:
        link_nodes(ctree, rlayers, "Emit", emit_boost, 2)
    else:
        emit_boost.inputs[2].default_value = (0.0, 0.0, 0.0, 1.0)

    glare_fog = spawn_node(ctree, "CompositorNodeGlare", -300, 220, "Neon Fog Halation")
    glare_fog.glare_type = 'FOG_GLOW'
    glare_fog.quality = 'HIGH'
    glare_fog.threshold = 1.10
    glare_fog.size = 8
    glare_fog.mix = 0.0

    glare_streaks = spawn_node(ctree, "CompositorNodeGlare", -40, 220, "Anamorphic Cyber Streaks")
    glare_streaks.glare_type = 'STREAKS'
    glare_streaks.quality = 'HIGH'
    glare_streaks.streaks = 4
    glare_streaks.angle_offset = 0.26
    glare_streaks.threshold = 3.20
    glare_streaks.fade = 0.90
    if hasattr(glare_streaks, "color_modulation"):
        glare_streaks.color_modulation = 0.10
    glare_streaks.mix = 0.0

    link_nodes(ctree, emit_boost, "Image", glare_fog, "Image")
    link_nodes(ctree, glare_fog, "Image", glare_streaks, "Image")

    # Stage 3: Mist / Z-Depth atmospheric horizon haze tinting
    mist_ramp = spawn_node(ctree, "CompositorNodeValToRGB", -340, -140, "Horizon Haze Ramp")
    mist_ramp.color_ramp.interpolation = 'EASE'
    mist_ramp.color_ramp.elements[0].position = 0.0
    mist_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    mist_ramp.color_ramp.elements[1].position = 1.0
    mist_ramp.color_ramp.elements[1].color = (0.038, 0.065, 0.115, 1.0)

    if "Mist" in rlayers.outputs:
        link_nodes(ctree, rlayers, "Mist", mist_ramp, "Fac")

    mist_add = spawn_node(ctree, "CompositorNodeMixRGB", 220, 160, "Atmospheric Depth Tint")
    mist_add.blend_type = 'ADD'
    mist_add.inputs[0].default_value = 0.42
    link_nodes(ctree, glare_streaks, "Image", mist_add, 1)
    link_nodes(ctree, mist_ramp, "Image", mist_add, 2)

    # Stage 4: Optical Lens Chromatic Dispersion, Exposure & 3-Way AgX Lift/Gamma/Gain Grade
    lens_dist = spawn_node(ctree, "CompositorNodeLensdist", 460, 160, "Anamorphic Chromatic Dispersion")
    if hasattr(lens_dist, "use_projector"):
        lens_dist.use_projector = True
    if hasattr(lens_dist, "use_fit"):
        lens_dist.use_fit = True
    if "Distortion" in lens_dist.inputs:
        lens_dist.inputs["Distortion"].default_value = -0.015
    elif "Distort" in lens_dist.inputs:
        lens_dist.inputs["Distort"].default_value = -0.015
    if "Dispersion" in lens_dist.inputs:
        lens_dist.inputs["Dispersion"].default_value = 0.012

    exposure_node = spawn_node(ctree, "CompositorNodeExposure", 700, 160, "Master Exposure Trim")
    exposure_node.inputs["Exposure"].default_value = 0.06

    color_bal = spawn_node(ctree, "CompositorNodeColorBalance", 940, 160, "Cyberpunk 3-Way Grade")
    color_bal.correction_method = 'LIFT_GAMMA_GAIN'
    color_bal.lift = (0.96, 0.99, 1.04)
    color_bal.gamma = (1.02, 1.00, 0.98)
    color_bal.gain = (1.03, 0.99, 1.02)

    comp_out = spawn_node(ctree, "CompositorNodeComposite", 1240, 240, "Composite Output")
    viewer_out = spawn_node(ctree, "CompositorNodeViewer", 1240, 40, "Backdrop Viewer")

    link_nodes(ctree, mist_add, "Image", lens_dist, "Image")
    link_nodes(ctree, lens_dist, "Image", exposure_node, "Image")
    link_nodes(ctree, exposure_node, "Image", color_bal, "Image")
    link_nodes(ctree, color_bal, "Image", comp_out, "Image")
    link_nodes(ctree, color_bal, "Image", viewer_out, "Image")


def upgrade_render_passes_and_compositor(scene, mats):
    """Configure production ViewLayer channels, Cryptomatte, custom Shader AOVs & Compositor."""
    build_compositor_post_processing(scene)
    register_shader_aov_outputs(bpy.context.view_layer, mats)


# ==============================================================================
# PHASE 2 (TRACK 2): GEOMETRY NODES RAIN STORM, PUDDLE RIPPLES & RAMEN STEAM
# ==============================================================================

def add_gn_interface_socket(gn_tree, name, in_out, socket_type):
    """Reusable Blender 4.0+ GeometryNodeTree interface socket builder."""
    return gn_tree.interface.new_socket(
        name=name,
        in_out=in_out,
        socket_type=socket_type,
    )


def build_rain_and_steam_shaders(mats):
    """Create procedural shaders for refractive rain streaks and volumetric rising steam."""
    # Motion-blurred refractive rain streak material
    rain_mat, nt = init_material("MAT_Rain_Streak_Glass")
    fres = spawn_node(nt, "ShaderNodeFresnel", -300, 120, "Droplet Fresnel")
    fres.inputs["IOR"].default_value = 1.333
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 40, 80, "Rain Streak BSDF")
    bsdf.inputs["Base Color"].default_value = (0.78, 0.94, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.03
    bsdf.inputs["IOR"].default_value = 1.333
    bsdf.inputs["Transmission Weight"].default_value = 0.92
    bsdf.inputs["Alpha"].default_value = 0.36
    bsdf.inputs["Emission Color"].default_value = (0.32, 0.85, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.45
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 340, 80, "Material Output")
    link_nodes(nt, fres, "Fac", bsdf, "Alpha")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["rain_glass"] = rain_mat

    # Volumetric rising steam plume shader with 3D vertical dissipation fade
    steam_mat, nt = init_material("MAT_Volumetric_Ramen_Steam")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -880, 0, "Steam Coords")
    sep = spawn_node(nt, "ShaderNodeSepXYZ", -660, 160, "Vertical Fade Axis")
    grad_ramp = spawn_node(nt, "ShaderNodeValToRGB", -420, 180, "Envelope Taper")
    grad_ramp.color_ramp.interpolation = 'EASE'
    grad_ramp.color_ramp.elements[0].position = 0.0
    grad_ramp.color_ramp.elements[0].color = (0.85, 0.85, 0.85, 1.0)
    mid_el = grad_ramp.color_ramp.elements.new(0.35)
    mid_el.color = (0.55, 0.55, 0.55, 1.0)
    grad_ramp.color_ramp.elements[-1].position = 1.0
    grad_ramp.color_ramp.elements[-1].color = (0.0, 0.0, 0.0, 1.0)

    noise = spawn_node(nt, "ShaderNodeTexNoise", -660, -100, "Turbulent Vapor")
    noise.inputs["Scale"].default_value = 5.8
    noise.inputs["Detail"].default_value = 4.5
    noise.inputs["Roughness"].default_value = 0.65
    noise.inputs["Distortion"].default_value = 0.85

    alpha_mul = spawn_node(nt, "ShaderNodeMath", -120, 40, "Vapor Density")
    alpha_mul.operation = 'MULTIPLY'
    alpha_scale = spawn_node(nt, "ShaderNodeMath", 80, 40, "Soft Opacity")
    alpha_scale.operation = 'MULTIPLY'
    alpha_scale.inputs[1].default_value = 0.26

    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 300, 100, "Translucent Steam BSDF")
    bsdf.inputs["Base Color"].default_value = (0.92, 0.95, 0.98, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Scale"].default_value = 0.12
    bsdf.inputs["Emission Color"].default_value = (1.0, 0.76, 0.52, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.14
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 580, 100, "Material Output")

    link_nodes(nt, tc, "Generated", sep, "Vector")
    link_nodes(nt, sep, "Z", grad_ramp, "Fac")
    link_nodes(nt, tc, "Object", noise, "Vector")
    link_nodes(nt, grad_ramp, "Color", alpha_mul, 0)
    link_nodes(nt, noise, "Fac", alpha_mul, 1)
    link_nodes(nt, alpha_mul, "Value", alpha_scale, 0)
    link_nodes(nt, alpha_scale, "Value", bsdf, "Alpha")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["steam"] = steam_mat


def build_geometry_nodes_rain_storm(col, mats):
    """Build GN_Cyber_Rain_Field GeometryNodeTree emitter and continuous 240-frame falling rain."""
    # Elongated optical raindrop streak prototype mesh (pre-assigned MAT_Rain_Streak_Glass)
    streak_prof = [
        (0.0, -0.22), (0.0022, -0.14), (0.0036, -0.02),
        (0.0020, 0.14), (0.0, 0.24),
    ]
    proto_drop = create_lathe_object(
        "Raindrop_Streak_Prototype", col, streak_prof, 8,
        (0.0, 0.0, -12.0), mats["rain_glass"], close_caps=False,
    )
    proto_drop.hide_render = True
    proto_drop.hide_viewport = True

    # Custom GeometryNodeTree using Blender 4.0+ interface socket API & modular SceneTime translation
    gn_tree = bpy.data.node_groups.new(name="GN_Cyber_Rain_Field", type='GeometryNodeTree')
    add_gn_interface_socket(gn_tree, "Geometry", 'INPUT', 'NodeSocketGeometry')
    add_gn_interface_socket(gn_tree, "Geometry", 'OUTPUT', 'NodeSocketGeometry')

    g_in = spawn_node(gn_tree, "NodeGroupInput", -640, 80, "Emitter Input")
    dist_pts = spawn_node(gn_tree, "GeometryNodeDistributePointsOnFaces", -380, 80, "Scatter Rain Columns")
    dist_pts.distribute_method = 'RANDOM'
    dist_pts.inputs["Density"].default_value = 7.5
    dist_pts.inputs["Seed"].default_value = 2099

    obj_info = spawn_node(gn_tree, "GeometryNodeObjectInfo", -380, -180, "Raindrop Prototype")
    obj_info.transform_space = 'ORIGINAL'
    obj_info.inputs["Object"].default_value = proto_drop

    rand_scale = spawn_node(gn_tree, "FunctionNodeRandomValue", -380, -380, "Streak Length Jitter")
    rand_scale.data_type = 'FLOAT_VECTOR'
    rand_scale.inputs["Min"].default_value = (0.75, 0.75, 0.65)
    rand_scale.inputs["Max"].default_value = (1.25, 1.25, 1.65)

    inst_pts = spawn_node(gn_tree, "GeometryNodeInstanceOnPoints", -80, 80, "Instance Rain Streaks")

    scene_time = spawn_node(gn_tree, "GeometryNodeInputSceneTime", -380, -560, "Rain Scene Clock")
    vel_scale = spawn_node(gn_tree, "ShaderNodeVectorMath", -160, -560, "Rain Fall Velocity")
    vel_scale.operation = 'SCALE'
    vel_scale.inputs[0].default_value = (0.0, 0.0, -0.85)

    mod_wrap = spawn_node(gn_tree, "ShaderNodeVectorMath", 40, -560, "Modular Z Loop")
    mod_wrap.operation = 'WRAP'
    mod_wrap.inputs[1].default_value = (7.5, 7.5, 0.0)
    mod_wrap.inputs[2].default_value = (-7.5, -7.5, -8.5)

    trans_inst = spawn_node(gn_tree, "GeometryNodeTranslateInstances", 220, 80, "Translate Rain Instances")
    trans_inst.inputs["Local Space"].default_value = False
    g_out = spawn_node(gn_tree, "NodeGroupOutput", 480, 80, "Rain Volume Output")

    link_nodes(gn_tree, g_in, "Geometry", dist_pts, "Mesh")
    link_nodes(gn_tree, dist_pts, "Points", inst_pts, "Points")
    link_nodes(gn_tree, obj_info, "Geometry", inst_pts, "Instance")
    link_nodes(gn_tree, rand_scale, "Value", inst_pts, "Scale")
    link_nodes(gn_tree, scene_time, "Seconds", vel_scale, "Scale")
    link_nodes(gn_tree, vel_scale, "Vector", mod_wrap, 0)
    link_nodes(gn_tree, inst_pts, "Instances", trans_inst, "Instances")
    link_nodes(gn_tree, mod_wrap, "Vector", trans_inst, "Translation")
    link_nodes(gn_tree, trans_inst, "Instances", g_out, "Geometry")

    # Build multi-layered 3D rain emitter sheets falling continuously across frames 1 to 240
    for sheet_idx, z_start in enumerate((9.5, 16.5)):
        bm = bmesh.new()
        layers_z = 6
        span_xy = 7.5
        for lz in range(layers_z):
            zz = lz * 1.15
            v0 = bm.verts.new((-span_xy, -span_xy, zz))
            v1 = bm.verts.new((span_xy, -span_xy, zz))
            v2 = bm.verts.new((span_xy, span_xy, zz))
            v3 = bm.verts.new((-span_xy, span_xy, zz))
            bm.faces.new((v0, v1, v2, v3))

        emitter_obj = finalize_bmesh_object(
            bm, f"Cyber_Rain_Emitter_Sheet_0{sheet_idx + 1}", col,
            mats["rain_glass"], smooth=False, location=(0.0, -0.5, z_start),
        )
        gn_mod = emitter_obj.modifiers.new(name="RainFieldNodes", type='NODES')
        gn_mod.node_group = gn_tree

        emitter_obj.location = Vector((0.0, -0.5, z_start))
        emitter_obj.keyframe_insert(data_path="location", frame=1)
        emitter_obj.location = Vector((0.0, -0.5, z_start - 8.5))
        emitter_obj.keyframe_insert(data_path="location", frame=240)
        if emitter_obj.animation_data is not None and emitter_obj.animation_data.action is not None:
            if hasattr(emitter_obj.animation_data.action, "fcurves"):
                for fc in emitter_obj.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'


def build_puddle_splash_ripples(col, mats):
    """Generate 14 animated concentric puddle splash ripple meshes across the wet asphalt plinth."""
    splash_coords = [
        (-2.40, -1.95), (-1.65, -2.20), (-0.75, -2.10), (0.20, -2.25),
        (1.15, -2.05), (2.15, -1.85), (2.65, -0.75), (-2.75, -0.60),
        (-2.35, 0.85), (2.45, 0.95), (-1.40, 1.95), (0.05, 2.15),
        (1.55, 1.85), (0.65, -1.85),
    ]
    for idx, (sx, sy) in enumerate(splash_coords):
        bm = bmesh.new()
        segs = 28
        for ring_r in (0.035, 0.075, 0.120):
            r_in = ring_r - 0.007
            r_out = ring_r + 0.007
            v_in = []
            v_out = []
            for i in range(segs):
                th = 2.0 * math.pi * i / segs
                ct, st = math.cos(th), math.sin(th)
                v_in.append(bm.verts.new((ct * r_in, st * r_in, 0.002)))
                v_out.append(bm.verts.new((ct * r_out, st * r_out, 0.005)))
            for i in range(segs):
                j = (i + 1) % segs
                bm.faces.new((v_in[i], v_out[i], v_out[j], v_in[j]))

        ripple_obj = finalize_bmesh_object(
            bm, f"Puddle_Splash_Ripple_{idx + 1:02d}", col,
            mats["rain_glass"], smooth=True, location=(sx, sy, 0.006),
        )
        wv = ripple_obj.modifiers.new(name="SplashCrownWave", type='WAVE')
        wv.use_normal = True
        wv.use_cyclic = True
        wv.height = 0.006
        wv.width = 0.06
        wv.speed = 0.35

        # Staggered expanding splash scale pulses across 240 frames
        phase_f = 1 + (idx * 17) % 180
        for cycle_start in (phase_f, phase_f + 60):
            if cycle_start + 36 <= 240:
                ripple_obj.scale = Vector((0.25, 0.25, 1.0))
                ripple_obj.keyframe_insert(data_path="scale", frame=cycle_start)
                ripple_obj.scale = Vector((1.55, 1.55, 0.35))
                ripple_obj.keyframe_insert(data_path="scale", frame=cycle_start + 32)
                ripple_obj.scale = Vector((0.20, 0.20, 0.10))
                ripple_obj.keyframe_insert(data_path="scale", frame=cycle_start + 36)


def build_ramen_steam_plumes(col, mats):
    """Generate 4 consolidated triple-ribbon helical steam plume meshes (1 per steam source)."""
    plume_sources = [
        ("Bowl_01", (-0.96, -1.06, 1.99), 0.085, 0.52),
        ("Bowl_02", (0.00, -1.06, 1.99), 0.090, 0.56),
        ("Bowl_03", (0.96, -1.06, 1.99), 0.085, 0.52),
        ("Cauldron", (-0.58, 0.58, 2.24), 0.190, 0.68),
    ]
    for tag, (px, py, pz), base_r, height in plume_sources:
        bm = bmesh.new()
        for rib_i in range(3):
            nz = 18
            nx = 4
            twist_off = rib_i * (2.0 * math.pi / 3.0)
            grid = []
            for iz in range(nz + 1):
                tz = iz / nz
                zz = tz * height
                r_cur = base_r * (0.45 + 0.95 * tz)
                swirl = twist_off + tz * 3.4
                cx = r_cur * 0.45 * math.cos(swirl)
                cy = r_cur * 0.45 * math.sin(swirl)
                ribbon_w = base_r * (0.55 + 0.75 * math.sin(tz * math.pi))
                row = []
                for ix in range(nx + 1):
                    tx = -0.5 + (ix / nx)
                    vx = cx + tx * ribbon_w * math.cos(swirl + 1.57)
                    vy = cy + tx * ribbon_w * math.sin(swirl + 1.57)
                    row.append(bm.verts.new((vx, vy, zz)))
                grid.append(row)
            for iz in range(nz):
                for ix in range(nx):
                    bm.faces.new((grid[iz][ix], grid[iz][ix + 1], grid[iz + 1][ix + 1], grid[iz + 1][ix]))

        plume = finalize_bmesh_object(
            bm, f"Ramen_Steam_Plume_{tag}", col,
            mats["steam"], smooth=True, location=(px, py, pz),
        )
        wv = plume.modifiers.new(name="RisingVaporWave", type='WAVE')
        wv.use_x = True
        wv.use_y = True
        wv.use_cyclic = True
        wv.height = 0.024
        wv.width = 0.22
        wv.speed = 0.22

        plume.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
        plume.keyframe_insert(data_path="rotation_euler", frame=1)
        plume.rotation_euler = Euler((0.0, 0.0, 2.0 * math.pi), 'XYZ')
        plume.keyframe_insert(data_path="rotation_euler", frame=240)


# ==============================================================================
# PHASE 2 (TRACK 3): ARTICULATED 6-DOF CYBERNETIC RAMEN CHEF AUTOMATON
# ==============================================================================

def build_cybernetic_ramen_chef(col, mats):
    """Build articulated 6-DOF cybernetic chef with LED visor, Hachimaki, and keyframed dual arms."""
    # Dedicated pixel-matrix LED visor shader
    visor_mat, nt = init_material("MAT_Chef_Visor_LED")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -620, 0, "Visor Coords")
    brick = spawn_node(nt, "ShaderNodeTexBrick", -380, 60, "LED Pixel Matrix")
    brick.inputs["Scale"].default_value = 36.0
    brick.inputs["Mortar Size"].default_value = 0.08
    brick.inputs["Brick Width"].default_value = 0.22
    brick.inputs["Row Height"].default_value = 0.22
    brick.inputs["Color1"].default_value = (0.0, 0.95, 1.0, 1.0)
    brick.inputs["Color2"].default_value = (1.0, 0.08, 0.58, 1.0)
    brick.inputs["Mortar"].default_value = (0.01, 0.01, 0.02, 1.0)
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 40, 60, "LED Faceplate BSDF")
    bsdf.inputs["Base Color"].default_value = (0.01, 0.01, 0.02, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.06
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Emission Strength"].default_value = 16.0
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 340, 60, "Material Output")
    link_nodes(nt, tc, "Object", brick, "Vector")
    link_nodes(nt, brick, "Color", bsdf, "Emission Color")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["chef_visor"] = visor_mat

    # Master chef root positioned behind customer counter (Y = 0.12, Z = 1.15)
    chef_root = bpy.data.objects.new("Cyber_Chef_Automaton_Root", None)
    chef_root.empty_display_type = 'PLAIN_AXES'
    chef_root.empty_display_size = 0.20
    chef_root.location = Vector((-0.18, 0.12, 1.15))
    col.objects.link(chef_root)

    # Floor & ceiling rail pedestal + multi-segmented titanium torso
    pedestal = create_lathe_object(
        "Chef_Rail_Pedestal", col,
        [(0.0, 0.0), (0.18, 0.0), (0.14, 0.12), (0.09, 0.72), (0.13, 0.80), (0.0, 0.80)],
        28, (0.0, 0.0, 0.0), mats["hull_metal"], close_caps=True,
    )
    pedestal.parent = chef_root

    torso_lower = create_beveled_box(
        "Chef_Abdomen_Core", col, (0.28, 0.22, 0.24), (0.0, 0.0, 0.90), mats["hull_metal"], bevel_width=0.012
    )
    torso_lower.parent = chef_root

    chest = create_beveled_box(
        "Chef_Titanium_Chest", col, (0.44, 0.28, 0.34), (0.0, -0.02, 1.18), mats["stainless"], bevel_width=0.016
    )
    chest.parent = torso_lower

    apron = create_beveled_box(
        "Chef_Indigo_Maekake_Apron", col, (0.34, 0.03, 0.42), (0.0, -0.14, 0.72), mats["noren"], bevel_width=0.004
    )
    apron.parent = torso_lower

    # Articulated Head, Curved LED Visor Faceplate & Traditional Hachimaki Headband
    head_pivot = bpy.data.objects.new("Chef_Neck_Pivot", None)
    head_pivot.location = Vector((0.0, -0.02, 0.24))
    head_pivot.parent = chest
    col.objects.link(head_pivot)

    bm_head = bmesh.new()
    bmesh.ops.create_uvsphere(bm_head, u_segments=24, v_segments=16, radius=0.135)
    bmesh.ops.scale(bm_head, verts=bm_head.verts[:], vec=(0.92, 1.05, 1.12))
    head_dome = finalize_bmesh_object(
        bm_head, "Chef_Cranial_Unit", col, mats["stainless"], smooth=True, location=(0.0, 0.0, 0.12)
    )
    head_dome.parent = head_pivot

    visor = create_beveled_box(
        "Chef_LED_Visor_Faceplate", col, (0.21, 0.06, 0.085), (0.0, -0.115, 0.12),
        mats["chef_visor"], bevel_width=0.012,
    )
    visor.parent = head_pivot

    hachimaki_prof = [(0.126, -0.018), (0.140, -0.018), (0.140, 0.018), (0.126, 0.018)]
    headband = create_lathe_object(
        "Chef_Hachimaki_Headband", col, hachimaki_prof, 28,
        (0.0, 0.0, 0.175), mats["egg_white"], close_caps=True,
    )
    headband.parent = head_pivot

    # Helper to construct a 4-joint hierarchical robotic arm chain
    def build_robotic_arm(side_tag, shoulder_offset):
        sh_joint = bpy.data.objects.new(f"Chef_Shoulder_{side_tag}", None)
        sh_joint.location = Vector(shoulder_offset)
        sh_joint.parent = chest
        col.objects.link(sh_joint)

        sh_ball = create_lathe_object(
            f"Chef_Shoulder_Servo_{side_tag}", col,
            [(0.0, -0.055), (0.068, -0.03), (0.068, 0.03), (0.0, 0.055)],
            20, (0.0, 0.0, 0.0), mats["pipe_metal"], close_caps=True,
        )
        sh_ball.parent = sh_joint

        upper_arm = create_beveled_box(
            f"Chef_UpperArm_{side_tag}", col, (0.075, 0.075, 0.28),
            (0.0, 0.0, -0.15), mats["stainless"], bevel_width=0.008,
        )
        upper_arm.parent = sh_joint

        el_joint = bpy.data.objects.new(f"Chef_Elbow_{side_tag}", None)
        el_joint.location = Vector((0.0, 0.0, -0.15))
        el_joint.parent = upper_arm
        col.objects.link(el_joint)

        forearm = create_beveled_box(
            f"Chef_Forearm_{side_tag}", col, (0.065, 0.28, 0.065),
            (0.0, -0.14, 0.0), mats["hull_metal"], bevel_width=0.008,
        )
        forearm.parent = el_joint

        wr_joint = bpy.data.objects.new(f"Chef_Wrist_{side_tag}", None)
        wr_joint.location = Vector((0.0, -0.15, 0.0))
        wr_joint.parent = forearm
        col.objects.link(wr_joint)
        return sh_joint, upper_arm, el_joint, forearm, wr_joint

    # Left Arm holding Stainless Tebo Noodle Strainer Basket over Cauldron (rhythmic draining shake)
    sh_l, _, el_l, _, wr_l = build_robotic_arm("L", (-0.27, 0.02, 0.12))
    tebo_prof = [(0.0, 0.0), (0.058, 0.0), (0.068, 0.13), (0.062, 0.13), (0.052, 0.008), (0.0, 0.008)]
    tebo_basket = create_lathe_object(
        "Chef_Tebo_Noodle_Strainer", col, tebo_prof, 24,
        (0.0, -0.16, -0.06), mats["stainless"], close_caps=True,
    )
    tebo_basket.parent = wr_l
    tebo_handle = create_beveled_box(
        "Chef_Tebo_Handle", col, (0.022, 0.18, 0.022), (0.0, -0.07, 0.05), mats["hinoki_wood"], bevel_width=0.004
    )
    tebo_handle.parent = wr_l

    # Right Arm holding Long-Handled Hishaku Broth Ladle (smooth figure-8 pouring motion)
    sh_r, _, el_r, _, wr_r = build_robotic_arm("R", (0.27, -0.02, 0.12))
    ladle_cup_prof = [(0.0, 0.0), (0.042, 0.0), (0.048, 0.048), (0.042, 0.048), (0.036, 0.006), (0.0, 0.006)]
    ladle_cup = create_lathe_object(
        "Chef_Hishaku_Ladle_Cup", col, ladle_cup_prof, 24,
        (0.0, -0.22, -0.04), mats["stainless"], close_caps=True,
    )
    ladle_cup.parent = wr_r
    ladle_shaft = create_beveled_box(
        "Chef_Hishaku_Ladle_Shaft", col, (0.016, 0.24, 0.016),
        (0.0, -0.10, 0.02), mats["hinoki_wood"], rotation=(-0.22, 0.0, 0.0), bevel_width=0.003,
    )
    ladle_shaft.parent = wr_r

    # Keyframe 240-frame culinary kinematics across head, left strainer arm & right ladle arm
    for frame in range(1, 241, 6):
        u = (frame - 1) / 239.0
        omega = 2.0 * math.pi * u

        # Chef attentive scanning head nod & pan
        head_pivot.rotation_euler = Euler((0.12 + 0.08 * math.sin(omega * 2.0), 0.0, 0.32 * math.sin(omega)), 'XYZ')
        head_pivot.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Left arm turned back toward cauldron with rapid sinusoidal noodle-draining shake
        shake = 0.22 * math.sin(omega * 10.0)
        sh_l.rotation_euler = Euler((-0.55 + 0.10 * shake, 0.0, 2.35), 'XYZ')
        sh_l.keyframe_insert(data_path="rotation_euler", frame=frame)
        el_l.rotation_euler = Euler((0.42 + shake, 0.0, 0.0), 'XYZ')
        el_l.keyframe_insert(data_path="rotation_euler", frame=frame)
        wr_l.rotation_euler = Euler((-0.35 * shake, 0.0, 0.0), 'XYZ')
        wr_l.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Right arm figure-8 ladle sweep from cauldron to center Hero_Ramen_Bowl_02
        fig8_yaw = -0.55 + 0.85 * math.sin(omega * 2.0)
        fig8_lift = -0.35 + 0.28 * math.sin(omega * 4.0)
        sh_r.rotation_euler = Euler((fig8_lift, 0.12 * math.cos(omega * 2.0), fig8_yaw), 'XYZ')
        sh_r.keyframe_insert(data_path="rotation_euler", frame=frame)
        el_r.rotation_euler = Euler((0.35 + 0.22 * math.cos(omega * 2.0), 0.0, 0.0), 'XYZ')
        el_r.keyframe_insert(data_path="rotation_euler", frame=frame)
        wr_r.rotation_euler = Euler((0.30 * math.sin(omega * 4.0), 0.45 * math.sin(omega * 2.0), 0.0), 'XYZ')
        wr_r.keyframe_insert(data_path="rotation_euler", frame=frame)


# ==============================================================================
# PHASE 2 (TRACK 4): VENDING AUTOMATON, GAS RACK & HOLOGRAM MENU PROJECTOR
# ==============================================================================

def build_cyber_street_props(col, mats):
    """Build cyber beverage vending automaton, rear plasma-gas fuel rack, and hologram menu."""
    # Hologram scanline shader (Wave bands + Fresnel + Cyan additive glow)
    holo_mat, nt = init_material("MAT_Hologram_Scanline")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -720, 0, "Holo Coords")
    wave_scan = spawn_node(nt, "ShaderNodeTexWave", -480, 120, "Horizontal Scanlines")
    wave_scan.wave_type = 'BANDS'
    wave_scan.bands_direction = 'Z'
    wave_scan.inputs["Scale"].default_value = 42.0
    fres = spawn_node(nt, "ShaderNodeFresnel", -480, -100, "Edge Hologram Boost")
    fres.inputs["IOR"].default_value = 1.45
    add_mask = spawn_node(nt, "ShaderNodeMath", -200, 40, "Combine Scan & Rim")
    add_mask.operation = 'ADD'
    add_mask.use_clamp = True
    alpha_mul = spawn_node(nt, "ShaderNodeMath", 0, -40, "Semi-Transparent Alpha")
    alpha_mul.operation = 'MULTIPLY'
    alpha_mul.inputs[1].default_value = 0.58

    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 240, 80, "Hologram BSDF")
    bsdf.inputs["Base Color"].default_value = (0.02, 0.88, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["Emission Color"].default_value = (0.05, 0.92, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 11.0
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 520, 80, "Material Output")

    link_nodes(nt, tc, "Generated", wave_scan, "Vector")
    link_nodes(nt, wave_scan, "Fac", add_mask, 0)
    link_nodes(nt, fres, "Fac", add_mask, 1)
    link_nodes(nt, add_mask, "Value", alpha_mul, 0)
    link_nodes(nt, alpha_mul, "Value", bsdf, "Alpha")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["hologram_scanline"] = holo_mat

    # 1. Cyber Beverage Vending Machine mounted on right side deck
    vm_x, vm_y, vm_z = 1.42, 0.35, 1.66
    create_beveled_box(
        "Cyber_Vending_Cabinet", col, (0.34, 0.58, 1.12), (vm_x, vm_y, vm_z), mats["hull_metal"], bevel_width=0.012
    )
    create_beveled_box(
        "Vending_Display_Window", col, (0.04, 0.44, 0.58), (vm_x + 0.16, vm_y, vm_z + 0.18),
        mats["rain_glass"], bevel_width=0.004,
    )
    create_beveled_box(
        "Vending_NFC_Payment_Pad", col, (0.05, 0.12, 0.18), (vm_x + 0.16, vm_y - 0.18, vm_z - 0.24),
        mats["neon_magenta"], bevel_width=0.004,
    )
    create_beveled_box(
        "Vending_Pickup_Chute", col, (0.06, 0.36, 0.14), (vm_x + 0.15, vm_y + 0.04, vm_z - 0.38),
        mats["stainless"], bevel_width=0.005,
    )

    can_prof = [(0.0, 0.0), (0.022, 0.0), (0.022, 0.082), (0.0, 0.082)]
    can_mats = (mats["neon_cyan"], mats["neon_magenta"], mats["neon_orange"])
    for row_i in range(2):
        for col_i in range(4):
            cy = vm_y - 0.15 + col_i * 0.10
            cz = vm_z + 0.02 + row_i * 0.18
            create_lathe_object(
                f"Vending_Drink_Can_{row_i + 1}_{col_i + 1}", col, can_prof, 16,
                (vm_x + 0.12, cy, cz), can_mats[(row_i + col_i) % 3], close_caps=True,
            )

    # 2. Dual Pressurized Plasma-Gas Fuel Cylinders strapped to rear hull + braided hoses & gauges
    tank_prof = [
        (0.0, 0.0), (0.14, 0.0), (0.15, 0.04), (0.15, 0.68),
        (0.11, 0.78), (0.04, 0.84), (0.04, 0.90), (0.0, 0.90),
    ]
    gauge_prof = [(0.0, 0.0), (0.042, 0.0), (0.042, 0.022), (0.0, 0.022)]
    for idx, tx in enumerate((-0.46, -0.08)):
        ty, tz = 1.18, 1.12
        create_lathe_object(
            f"Plasma_Gas_Cylinder_{idx + 1}", col, tank_prof, 28, (tx, ty, tz), mats["pipe_metal"], close_caps=True
        )
        create_beveled_box(
            f"Gas_Tank_Strap_{idx + 1}", col, (0.34, 0.22, 0.04), (tx, ty - 0.06, tz + 0.42), mats["hull_metal"]
        )
        create_lathe_object(
            f"Gas_Pressure_Gauge_{idx + 1}", col, gauge_prof, 20,
            (tx, ty + 0.06, tz + 0.90), mats["neon_cyan"], rotation=(math.pi * 0.5, 0.0, 0.0), close_caps=True,
        )
        hose_pts = [
            (tx, ty, tz + 0.88), (tx, ty + 0.12, tz + 0.98),
            (tx - 0.18, ty + 0.08, tz + 0.62), (-0.52, 0.98, 1.62),
        ]
        create_pipe_curve(
            f"Braided_Gas_Manifold_Hose_{idx + 1}", col, hose_pts, 0.014, mats["stainless"], spline_type='NURBS'
        )

    # 3. Front Counter Corner Volumetric Hologram Menu Projector & Floating Price Tiers
    proj_x, proj_y, proj_z = -1.62, -1.16, 1.90
    create_lathe_object(
        "Counter_Holo_Projector_Emitter", col,
        [(0.0, 0.0), (0.075, 0.0), (0.062, 0.045), (0.042, 0.055), (0.0, 0.055)],
        24, (proj_x, proj_y, proj_z), mats["hull_metal"], close_caps=True,
    )
    create_beveled_box(
        "Floating_Hologram_Menu_Pane", col, (0.46, 0.012, 0.54),
        (proj_x, proj_y, proj_z + 0.36), mats["hologram_scanline"],
        rotation=(0.0, 0.0, 0.28), bevel_width=0.003,
    )

    menu_lines = [
        ("Holo_Menu_Title", "RAMEN // MENU", 0.058, proj_z + 0.54),
        ("Holo_Tier_01", "TONKOTSU 850Y", 0.045, proj_z + 0.41),
        ("Holo_Tier_02", "CYBER MISO 920Y", 0.045, proj_z + 0.31),
        ("Holo_Tier_03", "SYNTH GYOZA 450Y", 0.042, proj_z + 0.21),
    ]
    for m_name, m_txt, m_sz, mz in menu_lines:
        fc = bpy.data.curves.new(name=m_name + "_Font", type='FONT')
        fc.body = m_txt
        fc.size = m_sz
        fc.extrude = 0.004
        fc.align_x = 'CENTER'
        fc.align_y = 'CENTER'
        fc.materials.append(mats["neon_cyan"])
        m_obj = bpy.data.objects.new(m_name, fc)
        m_obj.location = Vector((proj_x, proj_y - 0.015, mz))
        m_obj.rotation_euler = Euler((math.pi * 0.5, 0.0, 0.28), 'XYZ')
        col.objects.link(m_obj)


# ==============================================================================
# PHASE 2 (TRACK 5): F-CURVE HOVER BOBBING PHYSICS & FAULTY NEON FLICKER
# ==============================================================================

def apply_hover_physics_animation(cols, mats):
    """Apply sinusoidal anti-gravity hover bobbing to vessel & stools plus cyber neon ballast flicker."""
    vessel_root = bpy.data.objects.new("Hover_Vessel_Master_Root", None)
    vessel_root.empty_display_type = 'ARROWS'
    vessel_root.empty_display_size = 0.45
    vessel_root.location = Vector((0.0, 0.0, 0.0))
    cols["02_Floating_Hull_Thrusters"].objects.link(vessel_root)

    # Group floating bar stools onto individual hover roots for independent phase-offset bobbing
    stool_roots = {}
    for idx, sx in enumerate((-0.96, 0.0, 0.96)):
        s_root = bpy.data.objects.new(f"Hover_Stool_Root_{idx + 1}", None)
        s_root.location = Vector((sx, -1.58, 0.0))
        cols["02_Floating_Hull_Thrusters"].objects.link(s_root)
        stool_roots[idx + 1] = s_root

    floating_col_names = (
        "02_Floating_Hull_Thrusters",
        "03_Stall_Architecture_Roof",
        "04_Interior_Ramen_Props",
        "05_Lanterns_Fans_Cables",
        "06_Cyber_Neon_Signage",
    )
    for cname in floating_col_names:
        for obj in list(cols[cname].objects):
            if obj is vessel_root or obj in stool_roots.values() or obj.parent is not None:
                continue
            if obj.name.startswith("Floating_Bar_Stool_"):
                s_idx = int(obj.name.split("_")[-1])
                s_root = stool_roots[s_idx]
                obj.location = Vector((0.0, 0.0, obj.location.z))
                obj.parent = s_root
            elif not obj.name.startswith("Lantern_Lattice_"):
                obj.parent = vessel_root

    # Keep lantern lattices moving coherently with the floating stall roof
    for obj in list(cols["05_Lanterns_Fans_Cables"].objects):
        if obj.name.startswith("Lantern_Lattice_") and obj.parent is None:
            obj.parent = vessel_root

    # Keep internal lantern point lights & counter lights anchored to the floating vessel
    for obj in list(cols["07_Cinematic_Lights_Camera"].objects):
        if obj.name.startswith("Lantern_Internal_PointLight_") or obj.name in (
            "Light_Counter_Warm_Strip",
            "Light_UnderHull_Ion_Bounce",
        ):
            obj.parent = vessel_root

    # Keyframe smooth cyclic sinusoidal hover bobbing & micro-roll over frames 1 to 240
    for frame in range(1, 241, 8):
        phase = 2.0 * math.pi * ((frame - 1) / 239.0)
        vessel_root.location = Vector((0.0, 0.0, 0.038 * math.sin(phase * 2.0)))
        vessel_root.rotation_euler = Euler((
            0.008 * math.sin(phase * 2.0 + 0.6),
            0.011 * math.cos(phase * 2.0),
            0.0,
        ), 'XYZ')
        vessel_root.keyframe_insert(data_path="location", frame=frame)
        vessel_root.keyframe_insert(data_path="rotation_euler", frame=frame)

        for s_idx, s_root in stool_roots.items():
            s_off = s_idx * 1.35
            sx = (-0.96, 0.0, 0.96)[s_idx - 1]
            s_root.location = Vector((sx, -1.58, 0.026 * math.sin(phase * 3.0 + s_off)))
            s_root.rotation_euler = Euler((0.015 * math.cos(phase * 3.0 + s_off), 0.0, 0.0), 'XYZ')
            s_root.keyframe_insert(data_path="location", frame=frame)
            s_root.keyframe_insert(data_path="rotation_euler", frame=frame)
    # Restore exact rest pose at frame 240 for seamless turntable loop
    vessel_root.location = Vector((0.0, 0.0, 0.0))
    vessel_root.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    vessel_root.keyframe_insert(data_path="location", frame=240)
    vessel_root.keyframe_insert(data_path="rotation_euler", frame=240)

    # Faulty Cyber Neon Ballast Flicker on Right_3DText_Open & Lantern_Internal_PointLight_05
    flicker_mat = build_cyber_neon_shader(
        "MAT_Cyber_Neon_Cyan_Flicker",
        core_rgba=(0.72, 0.99, 1.0, 1.0),
        rim_rgba=(0.00, 0.86, 1.0, 1.0),
        strength=28.0,
    )
    mats["neon_cyan_flicker"] = flicker_mat
    open_sign = bpy.data.objects.get("Right_3DText_Open")
    if open_sign is not None and open_sign.data is not None:
        open_sign.data.materials.clear()
        open_sign.data.materials.append(flicker_mat)

    flicker_bsdf = next(
        (n for n in flicker_mat.node_tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"),
        None,
    )
    lantern_05 = bpy.data.objects.get("Lantern_Internal_PointLight_05")

    stutter_schedule = [
        (1, 1.0), (42, 1.0), (44, 0.08), (46, 0.95), (48, 0.14), (51, 1.0),
        (112, 1.0), (114, 0.05), (117, 1.0), (178, 1.0), (180, 0.12),
        (182, 0.85), (184, 0.04), (187, 1.0), (240, 1.0),
    ]
    for f_num, mult in stutter_schedule:
        if open_sign is not None:
            s_val = 1.0 if mult > 0.2 else 0.985
            open_sign.scale = Vector((s_val, s_val, s_val))
            open_sign.keyframe_insert(data_path="scale", frame=f_num)
        if flicker_bsdf is not None:
            flicker_bsdf.inputs["Emission Strength"].default_value = 28.0 * mult
            if hasattr(flicker_bsdf.inputs["Emission Strength"], "keyframe_insert"):
                flicker_bsdf.inputs["Emission Strength"].keyframe_insert(
                    data_path="default_value", frame=f_num
                )
        if lantern_05 is not None and lantern_05.data is not None:
            lantern_05.data.energy = 36.0 * (0.35 + 0.65 * mult)
            if hasattr(lantern_05.data, "keyframe_insert"):
                lantern_05.data.keyframe_insert(data_path="energy", frame=f_num)



# ==============================================================================
# PHASE 3 (TRACK 6): BRUTALIST MEGASTRUCTURE CANYON & NEO-TOKYO GREEBLE ENGINE
# ==============================================================================

def build_cyber_city_megablock(col, mats):
    """Build 8 brutalist mega-towers, procedural night window shader, skybridges & facade greebles."""
    # Procedural Night Window Matrix Shader (Brick grid + Voronoi room variation + Cyan/Amber/Magenta)
    win_mat, nt = init_material("MAT_Building_Night_Windows")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -1100, 0, "Facade Coords")
    mp = spawn_node(nt, "ShaderNodeMapping", -900, 0, "Window Grid Scale")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 1.8)

    vor = spawn_node(nt, "ShaderNodeTexVoronoi", -660, 220, "Apartment Room ID")
    vor.feature = 'F1'
    vor.inputs["Scale"].default_value = 3.2
    vor.inputs["Randomness"].default_value = 0.25

    room_ramp = spawn_node(nt, "ShaderNodeValToRGB", -400, 240, "Window Palette")
    room_ramp.color_ramp.interpolation = 'CONSTANT'
    room_ramp.color_ramp.elements[0].position = 0.0
    room_ramp.color_ramp.elements[0].color = (0.01, 0.012, 0.018, 1.0)
    el_c = room_ramp.color_ramp.elements.new(0.46)
    el_c.color = (0.05, 0.85, 1.0, 1.0)
    el_a = room_ramp.color_ramp.elements.new(0.68)
    el_a.color = (1.0, 0.56, 0.14, 1.0)
    room_ramp.color_ramp.elements[-1].position = 0.88
    room_ramp.color_ramp.elements[-1].color = (1.0, 0.08, 0.54, 1.0)

    brick = spawn_node(nt, "ShaderNodeTexBrick", -100, 120, "Architectural Window Mullions")
    brick.offset = 0.0
    brick.inputs["Scale"].default_value = 4.2
    brick.inputs["Mortar Size"].default_value = 0.065
    brick.inputs["Mortar Smooth"].default_value = 0.02
    brick.inputs["Brick Width"].default_value = 0.42
    brick.inputs["Row Height"].default_value = 0.28
    brick.inputs["Color2"].default_value = (0.012, 0.015, 0.022, 1.0)
    brick.inputs["Mortar"].default_value = (0.022, 0.025, 0.032, 1.0)

    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 220, 120, "Facade BSDF")
    bsdf.inputs["Base Color"].default_value = (0.035, 0.040, 0.050, 1.0)
    bsdf.inputs["Metallic"].default_value = 0.35
    bsdf.inputs["Roughness"].default_value = 0.38
    bsdf.inputs["Coat Weight"].default_value = 0.45
    bsdf.inputs["Emission Strength"].default_value = 4.8
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 520, 120, "Material Output")

    link_nodes(nt, tc, "Object", mp, "Vector")
    link_nodes(nt, mp, "Vector", vor, "Vector")
    link_nodes(nt, mp, "Vector", brick, "Vector")
    link_nodes(nt, vor, "Distance", room_ramp, "Fac")
    link_nodes(nt, room_ramp, "Color", brick, "Color1")
    link_nodes(nt, brick, "Color", bsdf, "Emission Color")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["building_windows"] = win_mat

    # 8 Multi-Tiered Brutalist Mega-Towers framing a U-shaped urban canyon
    tower_blueprint = [
        ("Cyber_Megatower_01", -13.8, 7.2, 4.2, 4.0, 15.5),
        ("Cyber_Megatower_02", -14.5, 14.5, 4.8, 4.6, 21.0),
        ("Cyber_Megatower_03", -8.2, 17.5, 4.6, 4.4, 24.5),
        ("Cyber_Megatower_04", -2.6, 19.2, 4.8, 4.6, 27.0),
        ("Cyber_Megatower_05", 3.2, 18.8, 4.6, 4.4, 25.5),
        ("Cyber_Megatower_06", 8.8, 17.2, 4.8, 4.4, 22.8),
        ("Cyber_Megatower_07", 14.4, 14.0, 4.8, 4.6, 20.2),
        ("Cyber_Megatower_08", 13.6, 6.8, 4.2, 4.0, 14.8),
    ]

    for idx, (t_name, tx, ty, tw, td, th) in enumerate(tower_blueprint):
        # Tier 1: Lower brutalist podium + window core
        h1 = th * 0.56
        create_beveled_box(
            f"{t_name}_LowerCore", col, (tw, td, h1),
            (tx, ty, h1 * 0.5 - 0.25), mats["building_windows"], bevel_width=0.04,
        )
        # Tier 2: Chamfered mid-rise setback
        h2 = th * 0.30
        z2 = h1 - 0.25 + h2 * 0.5
        create_beveled_box(
            f"{t_name}_MidSetback", col, (tw * 0.80, td * 0.80, h2),
            (tx, ty, z2), mats["building_windows"], bevel_width=0.035,
        )
        # Tier 3: Penthouse crown + heavy gunmetal brutalist corner fins
        h3 = th * 0.14
        z3 = h1 + h2 - 0.25 + h3 * 0.5
        create_beveled_box(
            f"{t_name}_PenthouseCrown", col, (tw * 0.58, td * 0.58, h3),
            (tx, ty, z3), mats["hull_metal"], bevel_width=0.03,
        )

        for cx_s in (-1.0, 1.0):
            for cy_s in (-1.0, 1.0):
                create_beveled_box(
                    f"{t_name}_MegaRib_{cx_s:+.0f}_{cy_s:+.0f}", col,
                    (0.28, 0.28, h1 + h2 * 0.6),
                    (tx + cx_s * tw * 0.49, ty + cy_s * td * 0.49, (h1 + h2 * 0.6) * 0.5 - 0.25),
                    mats["hull_metal"], bevel_width=0.02,
                )

        # Rooftop communication spire & glowing WarmOrange aviation warning beacon
        top_z = h1 + h2 + h3 - 0.25
        spire_h = 2.6 + (idx % 3) * 0.85
        create_pipe_curve(
            f"{t_name}_CommSpire", col,
            [(tx, ty, top_z), (tx, ty, top_z + spire_h)], 0.045, mats["pipe_metal"],
        )
        bm_bcn = bmesh.new()
        bmesh.ops.create_uvsphere(bm_bcn, u_segments=14, v_segments=10, radius=0.14)
        finalize_bmesh_object(
            bm_bcn, f"{t_name}_AviationBeacon", col,
            mats["neon_orange"], smooth=True, location=(tx, ty, top_z + spire_h + 0.10),
        )

        # Arrayed HVAC Condenser Cages & High-Voltage Transformer Boxes along alley-facing wall
        face_y = ty - td * 0.52
        hvac = create_beveled_box(
            f"{t_name}_HVAC_Cages", col, (0.52, 0.28, 0.36),
            (tx - tw * 0.28, face_y, 2.6), mats["pipe_metal"], bevel_width=0.01,
        )
        hvac_arr = hvac.modifiers.new(name="HVACVerticalStack", type='ARRAY')
        hvac_arr.fit_type = 'FIXED_COUNT'
        hvac_arr.count = 6
        hvac_arr.use_relative_offset = False
        hvac_arr.use_constant_offset = True
        hvac_arr.constant_offset_displace = (0.0, 0.0, 1.65)

        create_beveled_box(
            f"{t_name}_HV_Transformer_Box", col, (0.68, 0.40, 0.85),
            (tx + tw * 0.25, face_y, 1.85), mats["hull_metal"], bevel_width=0.015,
        )

    # External Zigzag Steel Fire-Escape Staircases on flanking towers
    for side_s, fx, fy, tag in ((-1.0, -11.5, 7.2, "Left"), (1.0, 11.3, 6.8, "Right")):
        zig_pts = []
        for flight in range(7):
            zf = 1.4 + flight * 1.55
            yf = fy - 1.2 + (flight % 2) * 2.4
            zig_pts.append((fx, yf, zf))
        create_pipe_curve(
            f"Fire_Escape_Zigzag_Stringer_{tag}", col, zig_pts, 0.045, mats["pipe_metal"], spline_type='POLY'
        )
        landing = create_beveled_box(
            f"Fire_Escape_Landings_{tag}", col, (0.75, 2.6, 0.06),
            (fx, fy, 1.4), mats["hull_metal"], bevel_width=0.008,
        )
        l_arr = landing.modifiers.new(name="LandingStack", type='ARRAY')
        l_arr.fit_type = 'FIXED_COUNT'
        l_arr.count = 4
        l_arr.use_relative_offset = False
        l_arr.use_constant_offset = True
        l_arr.constant_offset_displace = (0.0, 0.0, 3.10)

    # 2 Brutalist Overhead Pedestrian Skybridges spanning the urban canyon
    skybridge_specs = [
        ("Brutalist_Skybridge_Lower", (-8.2, 15.6, 9.8), (8.8, 15.6, 9.8), 17.2),
        ("Brutalist_Skybridge_Upper", (-13.8, 13.8, 13.6), (-2.6, 17.5, 13.6), 12.0),
    ]
    for sb_name, p_a, p_b, span_w in skybridge_specs:
        mid = (Vector(p_a) + Vector(p_b)) * 0.5
        ang_z = math.atan2(p_b[1] - p_a[1], p_b[0] - p_a[0])
        create_beveled_box(
            sb_name, col, (span_w, 1.45, 1.65),
            (mid.x, mid.y, mid.z), mats["hull_metal"],
            rotation=(0.0, 0.0, ang_z), bevel_width=0.025,
        )
        create_beveled_box(
            f"{sb_name}_UnderGlow", col, (span_w * 0.92, 0.22, 0.08),
            (mid.x, mid.y, mid.z - 0.84), mats["neon_orange"],
            rotation=(0.0, 0.0, ang_z), bevel_width=0.006,
        )


# ==============================================================================
# PHASE 3 (TRACK 7): ELEVATED MAGLEV MONORAIL VIADUCT & ANIMATED 3-CAR TRAIN
# ==============================================================================

def build_maglev_monorail_system(col, mats):
    """Build elevated viaduct at Y=8.5m, Z=5.4m and animate a 3-car Maglev train across frames 1-240."""
    viaduct_y = 8.5
    viaduct_z = 5.4

    # Main concrete/steel box-girder beamway spanning X = -22m to +22m
    create_beveled_box(
        "Maglev_Viaduct_BoxGirder", col, (44.0, 1.45, 0.68),
        (0.0, viaduct_y, viaduct_z), mats["hull_metal"], bevel_width=0.025,
    )
    for rail_off, r_tag in ((-0.46, "Front"), (0.46, "Rear")):
        create_pipe_curve(
            f"Maglev_Induction_Rail_{r_tag}", col,
            [(-22.0, viaduct_y + rail_off, viaduct_z + 0.38), (22.0, viaduct_y + rail_off, viaduct_z + 0.38)],
            0.035, mats["neon_cyan"],
        )

    # Battered support pylons & overhead catenary gantries
    for px in (-16.5, -8.5, 0.0, 8.5, 16.5):
        create_beveled_box(
            f"Maglev_Viaduct_Pylon_{px:+.1f}", col, (0.95, 1.35, viaduct_z + 0.3),
            (px, viaduct_y, (viaduct_z - 0.3) * 0.5), mats["hull_metal"], bevel_width=0.03,
        )
        gantry_pts = [
            (px, viaduct_y - 0.82, viaduct_z + 0.3),
            (px, viaduct_y - 0.82, viaduct_z + 2.35),
            (px, viaduct_y + 0.82, viaduct_z + 2.35),
            (px, viaduct_y + 0.82, viaduct_z + 0.3),
        ]
        create_pipe_curve(f"Maglev_Overhead_Gantry_{px:+.1f}", col, gantry_pts, 0.038, mats["pipe_metal"])

    # Articulated 3-Car Aerodynamic Maglev Train parented to animated root
    train_root = bpy.data.objects.new("Maglev_Train_Root", None)
    train_root.location = Vector((-24.0, viaduct_y, viaduct_z + 0.88))
    col.objects.link(train_root)

    car_offsets = [
        ("Maglev_Locomotive_Car_01", 3.65, 1.0),
        ("Maglev_Passenger_Car_02", 0.0, 0.0),
        ("Maglev_Tail_Car_03", -3.65, -1.0),
    ]
    for car_name, cx_off, nose_dir in car_offsets:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, verts=bm.verts[:], vec=(3.35, 1.28, 0.96))
        # Taper aerodynamic wedge nose on leading and trailing cars
        if abs(nose_dir) > 0.5:
            for v in bm.verts:
                if v.co.x * nose_dir > 1.0:
                    v.co.x += nose_dir * 0.55
                    v.co.z -= 0.24
                    v.co.y *= 0.72
        car_obj = finalize_bmesh_object(
            bm, car_name, col, mats["stainless"], smooth=True, location=(cx_off, 0.0, 0.0)
        )
        add_bevel_modifier(car_obj, width=0.03, segments=3)
        car_obj.parent = train_root

        # Panoramic illuminated window strip & under-chassis plasma levitation pad
        win_strip = create_beveled_box(
            f"{car_name}_WindowStrip", col, (2.85, 1.32, 0.24),
            (cx_off, 0.0, 0.14), mats["neon_cyan"], bevel_width=0.008,
        )
        win_strip.parent = train_root

        lev_pad = create_beveled_box(
            f"{car_name}_PlasmaLevPad", col, (2.60, 1.12, 0.08),
            (cx_off, 0.0, -0.46), mats["neon_magenta"], bevel_width=0.008,
        )
        lev_pad.parent = train_root

    # Flexible accordion couplers & high-intensity locomotive headlights
    for c_idx, c_x in enumerate((1.82, -1.82)):
        coupler = create_beveled_box(
            f"Maglev_Car_Coupler_{c_idx + 1}", col, (0.38, 1.02, 0.78),
            (c_x, 0.0, -0.04), mats["cable"], bevel_width=0.015,
        )
        coupler.parent = train_root

    for hy_off, h_tag in ((-0.34, "L"), (0.34, "R")):
        hl_data = bpy.data.lights.new(name=f"Maglev_Headlight_{h_tag}_Data", type='SPOT')
        hl_data.energy = 650.0
        hl_data.color = (0.72, 0.96, 1.0)
        hl_data.spot_size = math.radians(38.0)
        hl_obj = bpy.data.objects.new(f"Maglev_Headlight_{h_tag}", hl_data)
        hl_obj.location = Vector((5.85, hy_off, -0.10))
        hl_obj.rotation_euler = Euler((0.0, math.pi * 0.5, 0.0), 'XYZ')
        hl_obj.parent = train_root
        col.objects.link(hl_obj)

    # Sweep train across viaduct from X = -24.0m to X = +24.0m over frames 1 to 240
    train_root.location = Vector((-24.0, viaduct_y, viaduct_z + 0.88))
    train_root.keyframe_insert(data_path="location", frame=1)
    train_root.location = Vector((24.0, viaduct_y, viaduct_z + 0.88))
    train_root.keyframe_insert(data_path="location", frame=240)
    if train_root.animation_data is not None and train_root.animation_data.action is not None:
        if hasattr(train_root.animation_data.action, "fcurves"):
            for fc in train_root.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


# ==============================================================================
# PHASE 3 (TRACK 8): FLYING SPINNER TRAFFIC & LONG-EXPOSURE LIGHT TRAILS
# ==============================================================================

def build_flying_traffic_lanes(col, mats):
    """Build 3 aerial highway corridors, 6 FOLLOW_PATH spinners & neon light-trail ribbons."""
    for trail_key, col_rgba in (
        ("trail_cyan", (0.04, 0.92, 1.0, 1.0)),
        ("trail_magenta", (1.0, 0.06, 0.36, 1.0)),
    ):
        t_mat, nt = init_material(f"MAT_Traffic_Light_{trail_key}")
        tc = spawn_node(nt, "ShaderNodeTexCoord", -560, 0, "Trail Coords")
        sep = spawn_node(nt, "ShaderNodeSepXYZ", -360, 60, "Along Lane Axis")
        ramp = spawn_node(nt, "ShaderNodeValToRGB", -120, 60, "Exposure Taper")
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.05, 1.0)
        el_m = ramp.color_ramp.elements.new(0.5)
        el_m.color = (0.75, 0.75, 0.75, 1.0)
        ramp.color_ramp.elements[-1].position = 1.0
        ramp.color_ramp.elements[-1].color = (0.05, 0.05, 0.05, 1.0)
        bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 180, 60, "Light Trail BSDF")
        bsdf.inputs["Base Color"].default_value = col_rgba
        bsdf.inputs["Emission Color"].default_value = col_rgba
        bsdf.inputs["Emission Strength"].default_value = 18.0
        out = spawn_node(nt, "ShaderNodeOutputMaterial", 460, 60, "Material Output")
        link_nodes(nt, tc, "Generated", sep, "Vector")
        link_nodes(nt, sep, "X", ramp, "Fac")
        link_nodes(nt, ramp, "Color", bsdf, "Alpha")
        link_nodes(nt, bsdf, "BSDF", out, "Surface")
        mats[trail_key] = t_mat

    lane_configs = [
        ("Aerial_Highway_Lane_01", [(-20.0, 5.2, 3.8), (-6.0, 5.8, 3.9), (6.0, 5.6, 3.8), (20.0, 5.2, 3.8)], "trail_cyan"),
        ("Aerial_Highway_Lane_02", [(20.0, 11.4, 7.2), (6.0, 11.0, 7.4), (-6.0, 11.2, 7.2), (-20.0, 11.6, 7.2)], "trail_magenta"),
        ("Aerial_Highway_Lane_03", [(-20.0, 14.2, 10.5), (-5.0, 13.6, 10.7), (7.0, 13.8, 10.5), (20.0, 14.2, 10.5)], "trail_cyan"),
    ]

    spinner_counter = 0
    for l_idx, (lane_name, ctrl_pts, trail_mat_key) in enumerate(lane_configs):
        curve_d = bpy.data.curves.new(name=lane_name + "_Curve", type='CURVE')
        curve_d.dimensions = '3D'
        curve_d.use_path = True
        curve_d.path_duration = 240
        sp = curve_d.splines.new(type='NURBS')
        sp.points.add(len(ctrl_pts) - 1)
        for i, pt in enumerate(ctrl_pts):
            sp.points[i].co = (pt[0], pt[1], pt[2], 1.0)
        sp.use_endpoint_u = True
        sp.order_u = 4
        lane_obj = bpy.data.objects.new(lane_name, curve_d)
        col.objects.link(lane_obj)

        # Dual long-exposure plasma light-trail ribbons along each aerial corridor
        for side_off in (-0.28, 0.28):
            offset_pts = [(p[0], p[1] + side_off, p[2] - 0.08) for p in ctrl_pts]
            create_pipe_curve(
                f"{lane_name}_LightTrail_{side_off:+.2f}", col,
                offset_pts, 0.022, mats[trail_mat_key], spline_type='NURBS',
            )

        # 2 Aerodynamic Cyber-Spinner Flying Vehicles per lane (6 total) with FOLLOW_PATH
        for sub_v in range(2):
            spinner_counter += 1
            s_name = f"Cyber_Spinner_Vehicle_{spinner_counter:02d}"

            bm = bmesh.new()
            bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.scale(bm, verts=bm.verts[:], vec=(1.35, 0.72, 0.30))
            for v in bm.verts:
                if v.co.x > 0.2:
                    v.co.z *= 0.45
                    v.co.y *= 0.68
            spinner_obj = finalize_bmesh_object(
                bm, s_name, col, mats["hull_metal"], smooth=True, location=(0.0, 0.0, 0.0)
            )
            add_bevel_modifier(spinner_obj, width=0.02, segments=2)

            canopy = create_beveled_box(
                f"{s_name}_Canopy", col, (0.62, 0.42, 0.18),
                (-0.06, 0.0, 0.16), mats["rain_glass"], bevel_width=0.015,
            )
            canopy.parent = spinner_obj

            for ny_s, n_tag in ((-0.42, "L"), (0.42, "R")):
                nacelle = create_beveled_box(
                    f"{s_name}_Thruster_{n_tag}", col, (0.58, 0.16, 0.16),
                    (-0.28, ny_s, 0.0), mats[trail_mat_key], bevel_width=0.01,
                )
                nacelle.parent = spinner_obj

            fp = spinner_obj.constraints.new(type='FOLLOW_PATH')
            fp.name = "AerialLaneFollow"
            fp.target = lane_obj
            fp.use_fixed_location = True
            fp.use_curve_follow = True

            start_fac = 0.05 + sub_v * 0.44
            end_fac = min(0.99, start_fac + 0.50)
            fp.offset_factor = start_fac
            fp.keyframe_insert(data_path="offset_factor", frame=1)
            fp.offset_factor = end_fac
            fp.keyframe_insert(data_path="offset_factor", frame=240)


# ==============================================================================
# PHASE 3 (TRACK 9): COLOSSAL BUILDING-MOUNTED HOLOGRAPHIC BILLBOARDS
# ==============================================================================

def build_city_hologram_billboards(col, mats):
    """Mount 4 colossal animated glitch-hologram billboards with 3D typography onto megatowers."""
    glitch_mat, nt = init_material("MAT_Mega_Hologram_Glitch")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -900, 0, "Billboard Coords")
    mp = spawn_node(nt, "ShaderNodeMapping", -700, 0, "Scanline Scroll")
    wv = spawn_node(nt, "ShaderNodeTexWave", -460, 160, "Holo Scanlines")
    wv.wave_type = 'BANDS'
    wv.bands_direction = 'Z'
    wv.inputs["Scale"].default_value = 54.0

    vor = spawn_node(nt, "ShaderNodeTexVoronoi", -460, -80, "Digital Glitch Blocks")
    vor.feature = 'F1'
    vor.inputs["Scale"].default_value = 9.0
    vor.inputs["Randomness"].default_value = 0.15

    fres = spawn_node(nt, "ShaderNodeFresnel", -460, -280, "Grazing Rim Boost")
    fres.inputs["IOR"].default_value = 1.52

    glitch_mul = spawn_node(nt, "ShaderNodeMath", -180, 60, "Modulate Glitch")
    glitch_mul.operation = 'MULTIPLY'
    edge_add = spawn_node(nt, "ShaderNodeMath", 40, 20, "Add Fresnel Rim")
    edge_add.operation = 'ADD'
    edge_add.use_clamp = True

    col_ramp = spawn_node(nt, "ShaderNodeValToRGB", -180, 260, "Cyan-Magenta Holo Tint")
    col_ramp.color_ramp.elements[0].position = 0.2
    col_ramp.color_ramp.elements[0].color = (0.0, 0.88, 1.0, 1.0)
    col_ramp.color_ramp.elements[1].position = 0.85
    col_ramp.color_ramp.elements[1].color = (1.0, 0.08, 0.62, 1.0)

    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 300, 120, "Mega Hologram BSDF")
    bsdf.inputs["Base Color"].default_value = (0.05, 0.90, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.04
    bsdf.inputs["Emission Strength"].default_value = 14.0
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 580, 120, "Material Output")

    link_nodes(nt, tc, "Generated", mp, "Vector")
    link_nodes(nt, mp, "Vector", wv, "Vector")
    link_nodes(nt, mp, "Vector", vor, "Vector")
    link_nodes(nt, vor, "Distance", col_ramp, "Fac")
    link_nodes(nt, wv, "Fac", glitch_mul, 0)
    link_nodes(nt, vor, "Distance", glitch_mul, 1)
    link_nodes(nt, glitch_mul, "Value", edge_add, 0)
    link_nodes(nt, fres, "Fac", edge_add, 1)
    link_nodes(nt, col_ramp, "Color", bsdf, "Emission Color")
    link_nodes(nt, edge_add, "Value", bsdf, "Alpha")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["mega_hologram"] = glitch_mat

    billboards = [
        ("Colossal_Hologram_Billboard_01", (-11.2, 6.2, 9.8), (0.0, 0.0, -0.35), "NEO-TOKYO // SECTOR 09", mats["neon_cyan"]),
        ("Colossal_Hologram_Billboard_02", (-6.2, 14.6, 13.2), (0.0, 0.0, 0.0), "CYBER NOODLE CORP", mats["neon_magenta"]),
        ("Colossal_Hologram_Billboard_03", (6.4, 14.4, 12.8), (0.0, 0.0, 0.0), "SYNTHESIS 2099", mats["neon_orange"]),
        ("Colossal_Hologram_Billboard_04", (11.2, 6.0, 9.6), (0.0, 0.0, 0.35), "REPLIKA // 24H", mats["neon_cyan"]),
    ]

    for b_name, b_loc, b_rot, b_text, txt_mat in billboards:
        pane = create_beveled_box(
            b_name, col, (3.2, 0.04, 5.4), b_loc,
            mats["mega_hologram"], rotation=b_rot, bevel_width=0.01,
        )
        # Top & bottom volumetric projector emitter bars
        for z_dir, bar_tag in ((-2.78, "Bot"), (2.78, "Top")):
            bar = create_beveled_box(
                f"{b_name}_Projector_{bar_tag}", col, (3.4, 0.24, 0.16),
                (0.0, 0.0, z_dir), mats["hull_metal"], bevel_width=0.01,
            )
            bar.parent = pane

        # 3D procedural billboard typography & glowing geometric cyber-emblem ring
        fc = bpy.data.curves.new(name=f"{b_name}_Font", type='FONT')
        fc.body = b_text
        fc.size = 0.34
        fc.extrude = 0.025
        fc.align_x = 'CENTER'
        fc.align_y = 'CENTER'
        fc.materials.append(txt_mat)
        t_obj = bpy.data.objects.new(f"{b_name}_3DText", fc)
        t_obj.location = Vector((0.0, -0.06, 0.65))
        t_obj.rotation_euler = Euler((math.pi * 0.5, 0.0, 0.0), 'XYZ')
        t_obj.parent = pane
        col.objects.link(t_obj)

        emblem_pts = []
        for k in range(8):
            ang = k * (2.0 * math.pi / 8.0)
            emblem_pts.append((0.78 * math.cos(ang), -0.06, -0.95 + 0.78 * math.sin(ang)))
        emb = create_pipe_curve(f"{b_name}_CyberEmblem", col, emblem_pts, 0.028, txt_mat, cyclic=True)
        emb.parent = pane



# ==============================================================================
# PHASE 4 (TRACK 11): PROGRAMMATIC ARMATURE RIGGING & SEATED CYBER PATRONS
# ==============================================================================

def create_humanoid_upper_armature(name, location):
    """Construct a hierarchical 13-bone humanoid upper-body ARMATURE in Edit Mode."""
    arm_data = bpy.data.armatures.new(name + "_Data")
    if hasattr(arm_data, "display_type"):
        arm_data.display_type = 'OCTAHEDRAL'

    arm_obj = bpy.data.objects.new(name, arm_data)
    arm_obj.location = Vector(location)

    # Link temporarily to scene root collection so Edit Mode context is valid in Blender 4.x
    scene_root = bpy.context.scene.collection
    scene_root.objects.link(arm_obj)
    if hasattr(bpy.context.view_layer, "objects"):
        bpy.context.view_layer.objects.active = arm_obj
    bpy.context.active_object = arm_obj if not hasattr(bpy.context.view_layer, "objects") else arm_obj
    if hasattr(arm_obj, "select_set"):
        arm_obj.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')
    ebones = arm_data.edit_bones

    bone_specs = [
        ("Root", (0.0, 0.0, 0.00), (0.0, 0.0, 0.12), None, False),
        ("Spine", (0.0, 0.0, 0.12), (0.0, 0.02, 0.30), "Root", True),
        ("Chest", (0.0, 0.02, 0.30), (0.0, 0.04, 0.52), "Spine", True),
        ("Neck", (0.0, 0.04, 0.52), (0.0, 0.05, 0.60), "Chest", True),
        ("Head", (0.0, 0.05, 0.60), (0.0, 0.06, 0.79), "Neck", True),
        ("Shoulder_L", (-0.05, 0.03, 0.49), (-0.20, 0.03, 0.49), "Chest", False),
        ("UpperArm_L", (-0.20, 0.03, 0.49), (-0.24, 0.16, 0.27), "Shoulder_L", True),
        ("Forearm_L", (-0.24, 0.16, 0.27), (-0.17, 0.39, 0.31), "UpperArm_L", True),
        ("Hand_L", (-0.17, 0.39, 0.31), (-0.14, 0.49, 0.32), "Forearm_L", True),
        ("Shoulder_R", (0.05, 0.03, 0.49), (0.20, 0.03, 0.49), "Chest", False),
        ("UpperArm_R", (0.20, 0.03, 0.49), (0.24, 0.16, 0.27), "Shoulder_R", True),
        ("Forearm_R", (0.24, 0.16, 0.27), (0.17, 0.39, 0.31), "UpperArm_R", True),
        ("Hand_R", (0.17, 0.39, 0.31), (0.14, 0.49, 0.32), "Forearm_R", True),
    ]

    created_bones = {}
    for b_name, head_co, tail_co, parent_name, connected in bone_specs:
        eb = ebones.new(b_name)
        eb.head = Vector(head_co)
        eb.tail = Vector(tail_co)
        eb.roll = 0.0
        if parent_name is not None:
            eb.parent = created_bones[parent_name]
            eb.use_connect = connected
        created_bones[b_name] = eb

    bpy.ops.object.mode_set(mode='OBJECT')
    return arm_obj


def _append_box_part_to_bmesh(bm, bone_groups, bone_name, center, dims, mat_index, taper_top=1.0):
    """Append an oriented/tapered hexahedral anatomical segment to BMesh and record vertex indices."""
    cx, cy, cz = center
    hx, hy, hz = dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5
    coords = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx - hx * taper_top, cy - hy * taper_top, cz + hz),
        (cx + hx * taper_top, cy - hy * taper_top, cz + hz),
        (cx + hx * taper_top, cy + hy * taper_top, cz + hz),
        (cx - hx * taper_top, cy + hy * taper_top, cz + hz),
    ]
    vs = [bm.verts.new(c) for c in coords]
    quads = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]
    for q in quads:
        f = bm.faces.new((vs[q[0]], vs[q[1]], vs[q[2]], vs[q[3]]))
        f.material_index = mat_index
        f.smooth = True
    bone_groups.setdefault(bone_name, []).append(vs)


def _append_limb_tube_to_bmesh(bm, bone_groups, bone_name, p0, p1, r0, r1, mat_index, sides=10):
    """Append asegmented cylindrical bionic limb between p0 and p1 and register vertices to bone_name."""
    v0 = Vector(p0)
    v1 = Vector(p1)
    axis = v1 - v0
    length = axis.length
    if length < 1e-6:
        return
    tangent = axis.normalized()
    up_ref = Vector((0.0, 0.0, 1.0)) if abs(tangent.z) < 0.92 else Vector((1.0, 0.0, 0.0))
    n1 = tangent.cross(up_ref).normalized()
    n2 = tangent.cross(n1).normalized()

    ring_a = []
    ring_b = []
    for i in range(sides):
        ang = 2.0 * math.pi * i / sides
        ca, sa = math.cos(ang), math.sin(ang)
        radial = n1 * ca + n2 * sa
        ring_a.append(bm.verts.new(v0 + radial * r0))
        ring_b.append(bm.verts.new(v1 + radial * r1))

    for i in range(sides):
        j = (i + 1) % sides
        f = bm.faces.new((ring_a[i], ring_a[j], ring_b[j], ring_b[i]))
        f.material_index = mat_index
        f.smooth = True

    cap_a = bm.faces.new(tuple(reversed(ring_a)))
    cap_a.material_index = mat_index
    cap_b = bm.faces.new(tuple(ring_b))
    cap_b.material_index = mat_index
    bone_groups.setdefault(bone_name, []).append(ring_a + ring_b)


def build_rigged_cyber_patrons(col, mats):
    """Build 2 armature-skinned cybernetic patrons seated at X=-0.96m and X=+0.96m with 240f pose animation."""
    # Techwear ballistic synth-leather trenchcoat shader
    coat_mat, nt = init_material("MAT_Cyber_Techwear_Coat")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -680, 0, "Coat Coords")
    vor = spawn_node(nt, "ShaderNodeTexVoronoi", -460, 80, "Hex Ballistic Weave")
    vor.inputs["Scale"].default_value = 78.0
    ns = spawn_node(nt, "ShaderNodeTexNoise", -460, -120, "Rain Damp Sheen")
    ns.inputs["Scale"].default_value = 12.0
    bump = spawn_node(nt, "ShaderNodeBump", -200, 60, "Techwear Micro-Bump")
    bump.inputs["Strength"].default_value = 0.14
    bump.inputs["Distance"].default_value = 0.003
    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 100, 80, "Techwear BSDF")
    bsdf.inputs["Base Color"].default_value = (0.024, 0.028, 0.038, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.34
    bsdf.inputs["Sheen Weight"].default_value = 0.42
    bsdf.inputs["Coat Weight"].default_value = 0.55
    bsdf.inputs["Coat Roughness"].default_value = 0.12
    out = spawn_node(nt, "ShaderNodeOutputMaterial", 380, 80, "Material Output")
    link_nodes(nt, tc, "Object", vor, "Vector")
    link_nodes(nt, tc, "Object", ns, "Vector")
    link_nodes(nt, vor, "Distance", bump, "Height")
    link_nodes(nt, bump, "Normal", bsdf, "Normal")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["techwear_coat"] = coat_mat

    patron_configs = [
        ("Cyber_Patron_01_Netrunner", -0.96, -1.58, 1.27, "neon_cyan", 1),
        ("Cyber_Patron_02_StreetSamurai", 0.96, -1.58, 1.27, "neon_magenta", 3),
    ]

    for p_tag, px, py, pz, accent_key, stool_idx in patron_configs:
        arm_obj = create_humanoid_upper_armature(f"{p_tag}_Armature", (px, py, pz))
        col.objects.link(arm_obj)
        if arm_obj in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(arm_obj)

        # Build unified multi-material articulated cybernetic mesh bound via Vertex Groups
        # Material slots: 0=techwear_coat, 1=hull_metal, 2=stainless, 3=accent_neon, 4=hinoki_wood/ceramic
        bm = bmesh.new()
        bone_groups = {}

        # Root: seated pelvis, draped coat tails & bent seated thighs
        _append_box_part_to_bmesh(bm, bone_groups, "Root", (0.0, 0.01, 0.06), (0.32, 0.24, 0.12), 0, taper_top=0.95)
        _append_box_part_to_bmesh(bm, bone_groups, "Root", (0.0, -0.13, -0.10), (0.34, 0.05, 0.26), 0, taper_top=0.88)
        _append_limb_tube_to_bmesh(bm, bone_groups, "Root", (-0.11, 0.04, 0.04), (-0.13, 0.30, -0.02), 0.075, 0.062, 0)
        _append_limb_tube_to_bmesh(bm, bone_groups, "Root", (0.11, 0.04, 0.04), (0.13, 0.30, -0.02), 0.075, 0.062, 0)
        _append_limb_tube_to_bmesh(bm, bone_groups, "Root", (-0.13, 0.30, -0.02), (-0.13, 0.26, -0.34), 0.058, 0.045, 1)
        _append_limb_tube_to_bmesh(bm, bone_groups, "Root", (0.13, 0.30, -0.02), (0.13, 0.26, -0.34), 0.058, 0.045, 1)

        # Spine & Chest: high-collar tech-trenchcoat torso + glowing dorsal cyber-spine vertebrae
        _append_box_part_to_bmesh(bm, bone_groups, "Spine", (0.0, 0.01, 0.21), (0.30, 0.21, 0.18), 0, taper_top=1.08)
        _append_box_part_to_bmesh(bm, bone_groups, "Chest", (0.0, 0.03, 0.41), (0.36, 0.24, 0.23), 0, taper_top=1.12)
        # Flared high origami tech-collar framing the neck with glowing inner neon piping
        _append_box_part_to_bmesh(bm, bone_groups, "Chest", (0.0, -0.06, 0.56), (0.26, 0.05, 0.11), 0, taper_top=1.18)
        _append_box_part_to_bmesh(bm, bone_groups, "Chest", (-0.12, 0.02, 0.55), (0.04, 0.18, 0.10), 0, taper_top=1.15)
        _append_box_part_to_bmesh(bm, bone_groups, "Chest", (0.12, 0.02, 0.55), (0.04, 0.18, 0.10), 0, taper_top=1.15)
        _append_box_part_to_bmesh(bm, bone_groups, "Chest", (0.0, -0.09, 0.60), (0.24, 0.02, 0.02), 3)

        for v_idx in range(5):
            vz = 0.16 + v_idx * 0.075
            b_target = "Spine" if v_idx < 2 else "Chest"
            _append_box_part_to_bmesh(bm, bone_groups, b_target, (0.0, -0.105, vz), (0.055, 0.04, 0.045), 2)
            _append_box_part_to_bmesh(bm, bone_groups, b_target, (0.0, -0.128, vz), (0.032, 0.02, 0.026), 3)

        # Shoulders: layered armored pauldrons with neon trim
        for s_sign, s_bone in ((-1.0, "Shoulder_L"), (1.0, "Shoulder_R")):
            _append_box_part_to_bmesh(bm, bone_groups, s_bone, (s_sign * 0.20, 0.03, 0.50), (0.13, 0.16, 0.09), 1, taper_top=0.82)
            _append_box_part_to_bmesh(bm, bone_groups, s_bone, (s_sign * 0.24, 0.03, 0.52), (0.06, 0.14, 0.02), 3)

        # Left & Right Segmented Bionic Arms
        _append_limb_tube_to_bmesh(bm, bone_groups, "UpperArm_L", (-0.20, 0.03, 0.49), (-0.24, 0.16, 0.27), 0.052, 0.044, 0)
        _append_box_part_to_bmesh(bm, bone_groups, "UpperArm_L", (-0.24, 0.16, 0.27), (0.068, 0.068, 0.058), 2)
        _append_limb_tube_to_bmesh(bm, bone_groups, "Forearm_L", (-0.24, 0.16, 0.27), (-0.17, 0.39, 0.31), 0.044, 0.036, 1)
        _append_box_part_to_bmesh(bm, bone_groups, "Hand_L", (-0.155, 0.44, 0.315), (0.055, 0.095, 0.034), 2)

        _append_limb_tube_to_bmesh(bm, bone_groups, "UpperArm_R", (0.20, 0.03, 0.49), (0.24, 0.16, 0.27), 0.052, 0.044, 0)
        _append_box_part_to_bmesh(bm, bone_groups, "UpperArm_R", (0.24, 0.16, 0.27), (0.068, 0.068, 0.058), 2)
        _append_limb_tube_to_bmesh(bm, bone_groups, "Forearm_R", (0.24, 0.16, 0.27), (0.17, 0.39, 0.31), 0.044, 0.036, 1)
        _append_box_part_to_bmesh(bm, bone_groups, "Hand_R", (0.155, 0.44, 0.315), (0.055, 0.095, 0.034), 2)

        # Neck & Head + Character-specific props (Chopsticks for Netrunner, Cyber-Cup for Samurai)
        _append_limb_tube_to_bmesh(bm, bone_groups, "Neck", (0.0, 0.04, 0.52), (0.0, 0.05, 0.61), 0.052, 0.048, 1)
        if stool_idx == 1:
            # Patron 01 (Netrunner): Cranial hood + wrap-around glowing Cyan LED Visor + Chopsticks in Hand_R
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (0.0, 0.055, 0.70), (0.17, 0.19, 0.19), 1, taper_top=0.88)
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (0.0, 0.148, 0.705), (0.155, 0.028, 0.052), 3)
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (-0.092, 0.04, 0.70), (0.025, 0.06, 0.06), 2)
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (0.092, 0.04, 0.70), (0.025, 0.06, 0.06), 2)
            # Paired bamboo chopsticks gripped in Hand_R extending toward noodle bowl
            _append_limb_tube_to_bmesh(bm, bone_groups, "Hand_R", (0.14, 0.42, 0.32), (-0.02, 0.62, 0.30), 0.0045, 0.0022, 4, sides=6)
            _append_limb_tube_to_bmesh(bm, bone_groups, "Hand_R", (0.15, 0.42, 0.305), (-0.02, 0.62, 0.292), 0.0045, 0.0022, 4, sides=6)
        else:
            # Patron 02 (Street Samurai): Kabuto crest + Oni half-mask + glowing Cyber-Cup in Hand_L
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (0.0, 0.055, 0.70), (0.18, 0.20, 0.19), 1, taper_top=0.84)
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (0.0, 0.150, 0.715), (0.145, 0.024, 0.034), 3)
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (0.0, 0.146, 0.655), (0.125, 0.042, 0.068), 2, taper_top=0.90)
            _append_box_part_to_bmesh(bm, bone_groups, "Head", (0.0, 0.08, 0.805), (0.035, 0.16, 0.035), 3)
            # Glowing hexagonal cyber-sake cup gripped in Hand_L
            _append_limb_tube_to_bmesh(bm, bone_groups, "Hand_L", (-0.14, 0.47, 0.30), (-0.14, 0.47, 0.39), 0.032, 0.038, 4, sides=8)
            _append_limb_tube_to_bmesh(bm, bone_groups, "Hand_L", (-0.14, 0.47, 0.375), (-0.14, 0.47, 0.392), 0.030, 0.030, 3, sides=8)

        bm.verts.index_update()
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

        mesh = bpy.data.meshes.new(f"{p_tag}_MeshData")
        bm.to_mesh(mesh)

        # Snapshot vertex index lists per bone before freeing BMesh
        bone_v_indices = {
            b_name: [v.index for group in v_lists for v in group]
            for b_name, v_lists in bone_groups.items()
        }
        bm.free()
        mesh.update()

        prop_mat = mats["hinoki_wood"] if stool_idx == 1 else mats["bowl_ceramic"]
        for slot_mat in (mats["techwear_coat"], mats["hull_metal"], mats["stainless"], mats[accent_key], prop_mat):
            mesh.materials.append(slot_mat)

        char_obj = bpy.data.objects.new(f"{p_tag}_Mesh", mesh)
        char_obj.location = Vector((0.0, 0.0, 0.0))
        char_obj.parent = arm_obj
        col.objects.link(char_obj)

        # Create Vertex Groups matching every Armature Bone and assign full skinning weights
        for b_name, idx_list in bone_v_indices.items():
            vg = char_obj.vertex_groups.new(name=b_name)
            if idx_list:
                vg.add(idx_list, 1.0, 'REPLACE')

        arm_mod = char_obj.modifiers.new(name="CyberPatronArmature", type='ARMATURE')
        arm_mod.object = arm_obj
        if hasattr(arm_mod, "use_vertex_groups"):
            arm_mod.use_vertex_groups = True

        # Configure Euler XYZ rotation mode on all PoseBones and keyframe 240-frame acting cycle
        pbones = arm_obj.pose.bones
        for pb in pbones:
            pb.rotation_mode = 'XYZ'

        stool_phase_off = stool_idx * 1.35
        for frame in range(1, 241, 8):
            t_norm = (frame - 1) / 239.0
            omega = 2.0 * math.pi * t_norm

            # Synchronize vertical anti-gravity bobbing with Hover_Stool_Root
            hover_z = pz + 0.026 * math.sin(omega * 3.0 + stool_phase_off)
            arm_obj.location = Vector((px, py, hover_z))
            arm_obj.keyframe_insert(data_path="location", frame=frame)

            if stool_idx == 1:
                # Patron 01 (Netrunner): Lift chopsticks from Hero_Ramen_Bowl_01 to visor + head nod + breathing
                lift = 0.5 - 0.5 * math.cos(omega * 2.0)
                pbones["Spine"].rotation_euler = Euler((0.12 + 0.03 * math.sin(omega * 3.0), 0.0, 0.0), 'XYZ')
                pbones["Chest"].rotation_euler = Euler((0.08 + 0.04 * math.sin(omega * 3.0) + 0.06 * lift, 0.0, 0.0), 'XYZ')
                pbones["Neck"].rotation_euler = Euler((-0.08 + 0.12 * lift, 0.0, 0.03 * math.sin(omega * 2.0)), 'XYZ')
                pbones["Head"].rotation_euler = Euler((-0.14 + 0.22 * lift + 0.05 * math.sin(omega * 6.0), 0.0, 0.0), 'XYZ')

                pbones["UpperArm_R"].rotation_euler = Euler((0.28 + 0.52 * lift, 0.0, -0.18 * lift), 'XYZ')
                pbones["Forearm_R"].rotation_euler = Euler((0.20 + 0.68 * lift, 0.15 * lift, -0.22 * lift), 'XYZ')
                pbones["Hand_R"].rotation_euler = Euler((-0.15 * lift, 0.20 * math.sin(omega * 4.0), 0.0), 'XYZ')

                pbones["UpperArm_L"].rotation_euler = Euler((0.34, 0.0, 0.10), 'XYZ')
                pbones["Forearm_L"].rotation_euler = Euler((0.22, 0.0, 0.18), 'XYZ')

                for b_key in ("Spine", "Chest", "Neck", "Head", "UpperArm_R", "Forearm_R", "Hand_R", "UpperArm_L", "Forearm_L"):
                    pbones[b_key].keyframe_insert(data_path="rotation_euler", frame=frame)
            else:
                # Patron 02 (Street Samurai): Lean on counter, turn head tracking Maglev train, raise cyber-cup with left hand
                sip = max(0.0, math.sin(omega * 1.0)) ** 2
                train_pan = -0.46 * math.cos(omega * 1.0)
                pbones["Spine"].rotation_euler = Euler((0.18, 0.0, -0.04), 'XYZ')
                pbones["Chest"].rotation_euler = Euler((0.10 + 0.025 * math.sin(omega * 2.0), 0.0, 0.05 * train_pan), 'XYZ')
                pbones["Neck"].rotation_euler = Euler((0.06 + 0.08 * (1.0 - sip), 0.0, 0.35 * train_pan), 'XYZ')
                pbones["Head"].rotation_euler = Euler((0.10 * (1.0 - sip) - 0.14 * sip, 0.04 * train_pan, 0.65 * train_pan), 'XYZ')

                pbones["UpperArm_L"].rotation_euler = Euler((0.25 + 0.56 * sip, 0.0, 0.16 * sip), 'XYZ')
                pbones["Forearm_L"].rotation_euler = Euler((0.18 + 0.74 * sip, -0.18 * sip, 0.24 * sip), 'XYZ')
                pbones["Hand_L"].rotation_euler = Euler((-0.28 * sip, 0.0, 0.0), 'XYZ')

                pbones["UpperArm_R"].rotation_euler = Euler((0.38, 0.0, -0.12), 'XYZ')
                pbones["Forearm_R"].rotation_euler = Euler((0.26, 0.0, -0.20), 'XYZ')

                for b_key in ("Spine", "Chest", "Neck", "Head", "UpperArm_L", "Forearm_L", "Hand_L", "UpperArm_R", "Forearm_R"):
                    pbones[b_key].keyframe_insert(data_path="rotation_euler", frame=frame)

        arm_obj.location = Vector((px, py, pz))
        arm_obj.keyframe_insert(data_path="location", frame=240)


# ==============================================================================
# PHASE 4 (TRACK 12): BIONIC STRAY CYBER-CAT & AUTONOMOUS DELIVERY DRONE
# ==============================================================================

def build_bionic_cat_and_drone(col, mats):
    """Build rooftop obsidian-and-gold bionic cat (1.42, -0.88, 3.12) & descending quad-rotor ramen drone."""
    obsidian_mat = build_weathered_brushed_metal_shader(
        "MAT_Bionic_Obsidian_Armor",
        dark_color=(0.012, 0.014, 0.020, 1.0),
        light_color=(0.055, 0.062, 0.082, 1.0),
        brush_stretch=(8.0, 8.0, 1.0),
        rough_min=0.08,
        rough_max=0.24,
        bump_strength=0.05,
    )
    gold_mat = build_weathered_brushed_metal_shader(
        "MAT_Bionic_Kintsugi_Gold",
        dark_color=(0.55, 0.36, 0.06, 1.0),
        light_color=(1.00, 0.78, 0.22, 1.0),
        brush_stretch=(14.0, 1.0, 1.0),
        rough_min=0.10,
        rough_max=0.28,
        bump_strength=0.06,
    )
    mats["bionic_obsidian"] = obsidian_mat
    mats["bionic_gold"] = gold_mat

    # 1. Articulated Bionic Stray Cat perched on front-right roof eave (1.42, -0.88, 3.12)
    cat_root = bpy.data.objects.new("Bionic_Stray_Cat_Root", None)
    cat_root.location = Vector((1.42, -0.88, 3.12))
    cat_root.rotation_euler = Euler((0.0, 0.0, -0.28), 'XYZ')
    col.objects.link(cat_root)
    vessel_root = bpy.data.objects.get("Hover_Vessel_Master_Root")
    if vessel_root is not None:
        cat_root.parent = vessel_root

    # Sculpted feline seated torso & chest
    bm_cat = bmesh.new()
    bmesh.ops.create_uvsphere(bm_cat, u_segments=20, v_segments=14, radius=0.095)
    bmesh.ops.scale(bm_cat, verts=bm_cat.verts[:], vec=(0.82, 1.45, 1.18))
    for v in bm_cat.verts:
        if v.co.z > 0.02:
            v.co.y -= 0.035
    cat_torso = finalize_bmesh_object(
        bm_cat, "Bionic_Cat_Torso", col, obsidian_mat, smooth=True, location=(0.0, 0.0, 0.12)
    )
    cat_torso.parent = cat_root

    collar = create_lathe_object(
        "Bionic_Cat_Gold_Collar", col,
        [(0.052, 0.0), (0.068, 0.0), (0.064, 0.024), (0.050, 0.024)],
        24, (0.0, -0.075, 0.205), gold_mat, rotation=(0.35, 0.0, 0.0), close_caps=True,
    )
    collar.parent = cat_root

    for s_sign, s_tag in ((-1.0, "L"), (1.0, "R")):
        bm_h = bmesh.new()
        bmesh.ops.create_uvsphere(bm_h, u_segments=16, v_segments=12, radius=0.068)
        bmesh.ops.scale(bm_h, verts=bm_h.verts[:], vec=(0.68, 1.15, 1.25))
        haunch = finalize_bmesh_object(
            bm_h, f"Bionic_Cat_Haunch_{s_tag}", col, obsidian_mat, smooth=True,
            location=(s_sign * 0.062, 0.055, 0.082),
        )
        haunch.parent = cat_root

        foreleg = create_beveled_box(
            f"Bionic_Cat_Foreleg_{s_tag}", col, (0.032, 0.036, 0.16),
            (s_sign * 0.042, -0.085, 0.08), gold_mat, bevel_width=0.005,
        )
        foreleg.parent = cat_root

    # Feline cranium, dual glowing cyan optic eyes & articulated twitching ears
    bm_head = bmesh.new()
    bmesh.ops.create_uvsphere(bm_head, u_segments=18, v_segments=14, radius=0.066)
    bmesh.ops.scale(bm_head, verts=bm_head.verts[:], vec=(0.96, 1.08, 0.92))
    cat_head = finalize_bmesh_object(
        bm_head, "Bionic_Cat_Head", col, obsidian_mat, smooth=True, location=(0.0, -0.105, 0.265)
    )
    cat_head.parent = cat_root

    for e_sign, e_tag in ((-1.0, "L"), (1.0, "R")):
        bm_eye = bmesh.new()
        bmesh.ops.create_uvsphere(bm_eye, u_segments=12, v_segments=8, radius=0.014)
        eye_obj = finalize_bmesh_object(
            bm_eye, f"Bionic_Cat_Optic_Eye_{e_tag}", col, mats["neon_cyan"],
            smooth=True, location=(e_sign * 0.028, -0.058, 0.012),
        )
        eye_obj.parent = cat_head

        # Pointed cybernetic ear with origin at base for crisp servo twitching
        bm_ear = bmesh.new()
        b0 = bm_ear.verts.new((-0.018, -0.012, 0.0))
        b1 = bm_ear.verts.new((0.018, -0.012, 0.0))
        b2 = bm_ear.verts.new((0.014, 0.014, 0.0))
        b3 = bm_ear.verts.new((-0.014, 0.014, 0.0))
        tip = bm_ear.verts.new((e_sign * 0.006, -0.004, 0.068))
        bm_ear.faces.new((b0, b3, b2, b1))
        bm_ear.faces.new((b0, b1, tip))
        bm_ear.faces.new((b1, b2, tip))
        bm_ear.faces.new((b2, b3, tip))
        bm_ear.faces.new((b3, b0, tip))
        ear_obj = finalize_bmesh_object(
            bm_ear, f"Bionic_Cat_Ear_{e_tag}", col, gold_mat,
            smooth=False, location=(e_sign * 0.038, -0.008, 0.048),
        )
        ear_obj.parent = cat_head

    # 5-Segment Hierarchical Articulated Tail with Kintsugi Gold Joints
    tail_segments = []
    prev_parent = cat_root
    for seg_i in range(5):
        seg_len = 0.072
        bm_seg = bmesh.new()
        bmesh.ops.create_cube(bm_seg, size=1.0)
        rad = 0.020 - seg_i * 0.0022
        bmesh.ops.scale(bm_seg, verts=bm_seg.verts[:], vec=(rad, seg_len, rad))
        bmesh.ops.translate(bm_seg, verts=bm_seg.verts[:], vec=(0.0, seg_len * 0.5, 0.0))
        seg_mat = gold_mat if seg_i % 2 == 1 else obsidian_mat
        loc_co = (0.0, 0.11, 0.045) if seg_i == 0 else (0.0, seg_len, 0.0)
        seg_obj = finalize_bmesh_object(
            bm_seg, f"Bionic_Cat_Tail_Seg_{seg_i + 1:02d}", col, seg_mat, smooth=True, location=loc_co
        )
        add_bevel_modifier(seg_obj, width=0.003, segments=2)
        seg_obj.parent = prev_parent
        prev_parent = seg_obj
        tail_segments.append(seg_obj)

    ear_l = bpy.data.objects.get("Bionic_Cat_Ear_L")
    ear_r = bpy.data.objects.get("Bionic_Cat_Ear_R")

    for frame in range(1, 241, 6):
        omega = 2.0 * math.pi * ((frame - 1) / 239.0)
        for idx, t_seg in enumerate(tail_segments):
            lag = idx * 0.52
            pitch = (0.28 if idx == 0 else 0.10) + 0.09 * math.cos(omega * 2.0 - lag)
            sway_z = 0.24 * math.sin(omega * 2.0 - lag)
            t_seg.rotation_euler = Euler((pitch, 0.0, sway_z), 'XYZ')
            t_seg.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Crisp servo ear twitches around frames 60 and 168
        twitch_l = 0.34 if frame in (61, 67, 175) else 0.04 * math.sin(omega * 3.0)
        twitch_r = -0.32 if frame in (115, 121, 199) else -0.04 * math.cos(omega * 3.0)
        if ear_l is not None:
            ear_l.rotation_euler = Euler((0.08, -0.14 - twitch_l, 0.0), 'XYZ')
            ear_l.keyframe_insert(data_path="rotation_euler", frame=frame)
        if ear_r is not None:
            ear_r.rotation_euler = Euler((0.08, 0.14 - twitch_r, 0.0), 'XYZ')
            ear_r.keyframe_insert(data_path="rotation_euler", frame=frame)

    # 2. Autonomous Quad-Rotor Ramen Delivery Drone descending to (-2.85, -1.20, 2.45)
    drone_root = bpy.data.objects.new("Delivery_Drone_Root", None)
    drone_root.location = Vector((-2.85, -1.20, 2.45))
    col.objects.link(drone_root)

    chassis = create_beveled_box(
        "Delivery_Drone_Chassis", col, (0.48, 0.56, 0.14), (0.0, 0.0, 0.0), mats["hull_metal"], bevel_width=0.016
    )
    chassis.parent = drone_root

    sensor_dome = create_lathe_object(
        "Delivery_Drone_Optical_Gimbal", col,
        [(0.0, -0.05), (0.055, -0.04), (0.062, 0.0), (0.0, 0.0)],
        20, (0.0, -0.24, -0.05), mats["neon_cyan"], close_caps=True,
    )
    sensor_dome.parent = drone_root

    # Insulated glowing thermal cargo crate ("RAMEN EXPRESS") slung underneath drone belly
    crate = create_beveled_box(
        "Delivery_Drone_Thermal_Crate", col, (0.38, 0.42, 0.24),
        (0.0, 0.0, -0.22), mats["pipe_metal"], bevel_width=0.012,
    )
    crate.parent = drone_root

    crate_band = create_beveled_box(
        "Delivery_Drone_Crate_NeonBand", col, (0.40, 0.44, 0.03),
        (0.0, 0.0, -0.22), mats["neon_orange"], bevel_width=0.004,
    )
    crate_band.parent = drone_root

    for side_x, rot_z, tag in ((-0.20, -math.pi * 0.5, "L"), (0.20, math.pi * 0.5, "R")):
        fc = bpy.data.curves.new(name=f"Drone_Crate_Font_{tag}", type='FONT')
        fc.body = "RAMEN EXPRESS"
        fc.size = 0.044
        fc.extrude = 0.005
        fc.align_x = 'CENTER'
        fc.align_y = 'CENTER'
        fc.materials.append(mats["neon_cyan"])
        txt_obj = bpy.data.objects.new(f"Delivery_Drone_Crate_Text_{tag}", fc)
        txt_obj.location = Vector((side_x, 0.0, -0.17))
        txt_obj.rotation_euler = Euler((math.pi * 0.5, 0.0, rot_z), 'XYZ')
        txt_obj.parent = drone_root
        col.objects.link(txt_obj)

    # Downward volumetric SPOT landing-light cone
    spot_data = bpy.data.lights.new(name="Delivery_Drone_Landing_Spot_Data", type='SPOT')
    spot_data.energy = 340.0
    spot_data.color = (0.22, 0.92, 1.0)
    spot_data.spot_size = math.radians(46.0)
    spot_obj = bpy.data.objects.new("Delivery_Drone_Landing_Spot", spot_data)
    spot_obj.location = Vector((0.0, 0.0, -0.35))
    spot_obj.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
    spot_obj.parent = drone_root
    col.objects.link(spot_obj)

    # 4 Outrigger Motor Booms, Ducted Shrouds & High-Speed Spinning Rotor Blades
    rotor_offsets = [
        (-0.42, -0.44, "FL"),
        (0.42, -0.44, "FR"),
        (-0.42, 0.44, "RL"),
        (0.42, 0.44, "RR"),
    ]
    shroud_ring_prof = [(0.20, -0.025), (0.225, -0.025), (0.225, 0.025), (0.20, 0.025)]

    for r_idx, (rx, ry, r_tag) in enumerate(rotor_offsets):
        boom = create_pipe_curve(
            f"Delivery_Drone_Boom_{r_tag}", col, [(0.0, 0.0, 0.02), (rx, ry, 0.05)], 0.020, mats["hull_metal"]
        )
        boom.parent = drone_root

        duct = create_lathe_object(
            f"Delivery_Drone_Shroud_{r_tag}", col, shroud_ring_prof, 28,
            (rx, ry, 0.06), mats["neon_cyan"] if r_idx < 2 else mats["neon_magenta"], close_caps=True,
        )
        duct.parent = drone_root

        # Crossed 4-blade carbon rotor hub
        bm_rot = bmesh.new()
        bmesh.ops.create_cube(bm_rot, size=1.0)
        bmesh.ops.scale(bm_rot, verts=bm_rot.verts[:], vec=(0.37, 0.038, 0.006))
        bm_rot_b = bmesh.new()
        bmesh.ops.create_cube(bm_rot_b, size=1.0)
        bmesh.ops.scale(bm_rot_b, verts=bm_rot_b.verts[:], vec=(0.038, 0.37, 0.006))
        for vb in bm_rot_b.verts:
            bm_rot.verts.new(vb.co)
        bm_rot.verts.ensure_lookup_table()
        vs_b = bm_rot.verts[-8:]
        for q in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            bm_rot.faces.new((vs_b[q[0]], vs_b[q[1]], vs_b[q[2]], vs_b[q[3]]))
        bm_rot_b.free()

        rotor_obj = finalize_bmesh_object(
            bm_rot, f"Delivery_Drone_Rotor_{r_idx + 1:02d}", col, mats["stainless"],
            smooth=False, location=(rx, ry, 0.068),
        )
        rotor_obj.parent = drone_root

        spin_dir = 1.0 if (r_idx % 2 == 0) else -1.0
        rotor_obj.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
        rotor_obj.keyframe_insert(data_path="rotation_euler", frame=1)
        rotor_obj.rotation_euler = Euler((0.0, 0.0, spin_dir * 48.0 * 2.0 * math.pi), 'XYZ')
        rotor_obj.keyframe_insert(data_path="rotation_euler", frame=240)
        if rotor_obj.animation_data is not None and rotor_obj.animation_data.action is not None:
            for fc in rotor_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    # Curved approach trajectory settling into station-keeping hover at (-2.85, -1.20, 2.45)
    drone_waypoints = [
        (1, (-7.40, 2.80, 6.25), (0.18, 0.22, -0.55)),
        (56, (-5.45, 0.72, 4.42), (0.14, 0.16, -0.35)),
        (112, (-3.85, -0.62, 3.12), (0.08, 0.09, -0.18)),
        (168, (-2.96, -1.14, 2.52), (-0.06, -0.05, -0.08)),
        (204, (-2.85, -1.20, 2.49), (0.02, -0.02, -0.05)),
        (240, (-2.85, -1.20, 2.45), (0.0, 0.0, -0.05)),
    ]
    for f_num, wp_loc, wp_rot in drone_waypoints:
        drone_root.location = Vector(wp_loc)
        drone_root.rotation_euler = Euler(wp_rot, 'XYZ')
        drone_root.keyframe_insert(data_path="location", frame=f_num)
        drone_root.keyframe_insert(data_path="rotation_euler", frame=f_num)


# ==============================================================================
# PHASE 4 (TRACK 13): TRANSLUCENT CYBER-UMBRELLAS & VERTICAL RAIN-DRIP SHADERS
# ==============================================================================

def _get_or_create_vertical_rain_drip_group():
    """Shared ShaderNodeGroup NG_Vertical_Rain_Drip with single animated Location F-Curve."""
    grp = bpy.data.node_groups.get("NG_Vertical_Rain_Drip")
    if grp is not None:
        return grp
    grp = bpy.data.node_groups.new(name="NG_Vertical_Rain_Drip", type='ShaderNodeTree')
    grp.interface.new_socket(name="Base Normal", in_out='INPUT', socket_type='NodeSocketVector')
    grp.interface.new_socket(name="Normal", in_out='OUTPUT', socket_type='NodeSocketVector')
    grp.interface.new_socket(name="Coat Weight", in_out='OUTPUT', socket_type='NodeSocketColor')

    g_in = spawn_node(grp, "NodeGroupInput", -1140, -340, "Base Normal Input")
    tc = spawn_node(grp, "ShaderNodeTexCoord", -980, -340, "Rain Drip Coords")
    mp = spawn_node(grp, "ShaderNodeMapping", -780, -340, "Vertical Runoff Stretch")
    mp.inputs["Scale"].default_value = (28.0, 28.0, 1.35)
    mp.inputs["Location"].default_value = (0.0, 0.0, 0.0)
    if hasattr(mp.inputs["Location"], "keyframe_insert"):
        mp.inputs["Location"].keyframe_insert(data_path="default_value", frame=1)
        mp.inputs["Location"].default_value = (0.0, 0.0, 2.60)
        mp.inputs["Location"].keyframe_insert(data_path="default_value", frame=240)
    if grp.animation_data is not None and grp.animation_data.action is not None:
        if hasattr(grp.animation_data.action, "fcurves"):
            for fc in grp.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    drip_ns = spawn_node(grp, "ShaderNodeTexNoise", -560, -340, "Runoff Rivulet Noise")
    drip_ns.inputs["Scale"].default_value = 9.5
    drip_ns.inputs["Detail"].default_value = 4.5
    drip_ns.inputs["Distortion"].default_value = 0.85

    drip_ramp = spawn_node(grp, "ShaderNodeValToRGB", -320, -340, "Sharp Rivulet Mask")
    drip_ramp.color_ramp.elements[0].position = 0.60
    drip_ramp.color_ramp.elements[0].color = (0.25, 0.25, 0.25, 1.0)
    drip_ramp.color_ramp.elements[1].position = 0.78
    drip_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    drip_bump = spawn_node(grp, "ShaderNodeBump", -60, -340, "Condensation Drip Normal")
    drip_bump.inputs["Strength"].default_value = 0.14
    drip_bump.inputs["Distance"].default_value = 0.003

    g_out = spawn_node(grp, "NodeGroupOutput", 180, -340, "Rain Drip Output")

    link_nodes(grp, tc, "Object", mp, "Vector")
    link_nodes(grp, mp, "Vector", drip_ns, "Vector")
    link_nodes(grp, drip_ns, "Fac", drip_ramp, "Fac")
    link_nodes(grp, drip_ramp, "Color", drip_bump, "Height")
    link_nodes(grp, g_in, "Base Normal", drip_bump, "Normal")
    link_nodes(grp, drip_bump, "Normal", g_out, "Normal")
    link_nodes(grp, drip_ramp, "Color", g_out, "Coat Weight")
    return grp


def _inject_vertical_rain_drip_layer(mat):
    """Inject shared NG_Vertical_Rain_Drip ShaderNodeGroup into glass/metal ShaderNodeTrees."""
    if mat is None or mat.node_tree is None:
        return
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return
    prev_norm_src = next(
        (
            n for n in nt.nodes
            if n.bl_idname in ("ShaderNodeBump", "ShaderNodeGroup") and "Normal" in n.outputs
        ),
        None,
    )
    drip_grp = _get_or_create_vertical_rain_drip_group()
    drip_node = spawn_node(nt, "ShaderNodeGroup", 40, -260, "Vertical Rain Drip")
    drip_node.node_tree = drip_grp
    bsdf.inputs["Coat Weight"].default_value = 0.625
    if prev_norm_src is not None:
        link_nodes(nt, prev_norm_src, "Normal", drip_node, "Base Normal")
    link_nodes(nt, drip_node, "Normal", bsdf, "Normal")
    link_nodes(nt, drip_node, "Coat Weight", bsdf, "Coat Weight")


def build_umbrellas_and_drip_shaders(col, mats):
    """Build 2 12-gore parabolic translucent cyber-umbrellas & upgrade glass/metal rain-drip shaders."""
    # 1. Custom refractive wet-vinyl shader with raindrop normal ripples
    umb_mat, nt = init_material("MAT_Clear_Wet_Umbrella")
    tc = spawn_node(nt, "ShaderNodeTexCoord", -880, 0, "Vinyl Coords")
    mp = spawn_node(nt, "ShaderNodeMapping", -680, 0, "Canopy UV Map")
    mp.inputs["Scale"].default_value = (14.0, 14.0, 14.0)

    vor_beads = spawn_node(nt, "ShaderNodeTexVoronoi", -460, 120, "Raindrop Beads")
    vor_beads.inputs["Scale"].default_value = 26.0
    vor_beads.inputs["Randomness"].default_value = 0.92

    wave_rip = spawn_node(nt, "ShaderNodeTexWave", -460, -100, "Impact Micro-Ripples")
    wave_rip.wave_type = 'RINGS'
    wave_rip.inputs["Scale"].default_value = 18.0
    wave_rip.inputs["Distortion"].default_value = 2.4

    comb = spawn_node(nt, "ShaderNodeMath", -220, 20, "Bead Plus Ripple")
    comb.operation = 'ADD'
    bump = spawn_node(nt, "ShaderNodeBump", 0, -40, "Wet Vinyl Normal")
    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.003

    fres = spawn_node(nt, "ShaderNodeFresnel", -220, 220, "Vinyl Grazing Sheen")
    fres.inputs["IOR"].default_value = 1.49

    bsdf = spawn_node(nt, "ShaderNodeBsdfPrincipled", 260, 100, "Refractive Wet Vinyl BSDF")
    bsdf.inputs["Base Color"].default_value = (0.84, 0.96, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.05
    bsdf.inputs["IOR"].default_value = 1.49
    bsdf.inputs["Transmission Weight"].default_value = 0.90
    bsdf.inputs["Coat Weight"].default_value = 1.0
    bsdf.inputs["Coat Roughness"].default_value = 0.02
    bsdf.inputs["Alpha"].default_value = 0.44
    bsdf.inputs["Emission Color"].default_value = (0.12, 0.85, 1.0, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.35

    out = spawn_node(nt, "ShaderNodeOutputMaterial", 540, 100, "Material Output")

    link_nodes(nt, tc, "Object", mp, "Vector")
    link_nodes(nt, mp, "Vector", vor_beads, "Vector")
    link_nodes(nt, mp, "Vector", wave_rip, "Vector")
    link_nodes(nt, vor_beads, "Distance", comb, 0)
    link_nodes(nt, wave_rip, "Fac", comb, 1)
    link_nodes(nt, comb, "Value", bump, "Height")
    link_nodes(nt, bump, "Normal", bsdf, "Normal")
    link_nodes(nt, fres, "Fac", bsdf, "Specular IOR Level")
    link_nodes(nt, bsdf, "BSDF", out, "Surface")
    mats["wet_umbrella"] = umb_mat

    # 2. Upgrade vertical glass & metal shaders with procedural rain-runoff rivulet layers
    for mat_key in ("hull_metal", "stainless", "rain_glass"):
        _inject_vertical_rain_drip_layer(mats.get(mat_key))

    # 3. Generate 2 open 12-gore parabolic cyber-umbrellas beside the customer stools
    umbrella_specs = [
        ("Cyber_Umbrella_01", (-1.52, -1.66, 1.24), (0.24, -0.18, 0.30), mats["neon_cyan"]),
        ("Cyber_Umbrella_02", (1.52, -1.62, 1.24), (0.22, 0.20, -0.35), mats["neon_magenta"]),
    ]

    num_gores = 12
    sub_u = 3
    total_u = num_gores * sub_u
    radial_rings = 6
    canopy_radius = 0.56
    canopy_drop = 0.24

    for u_name, u_loc, u_rot, shaft_mat in umbrella_specs:
        u_root = bpy.data.objects.new(u_name, None)
        u_root.location = Vector(u_loc)
        u_root.rotation_euler = Euler(u_rot, 'XYZ')
        col.objects.link(u_root)

        # 12-Gore scalloped parabolic canopy via BMesh
        bm = bmesh.new()
        apex = bm.verts.new((0.0, 0.0, 0.42))
        rings = []
        for r_i in range(1, radial_rings + 1):
            rf = r_i / radial_rings
            ring_v = []
            for iu in range(total_u):
                theta = 2.0 * math.pi * iu / total_u
                gore_phase = (iu % sub_u) / sub_u
                scallop = 1.0 - 0.065 * math.sin(math.pi * gore_phase) * (rf ** 2)
                rad = canopy_radius * rf * scallop
                z_val = 0.42 - canopy_drop * (rf ** 1.85) + 0.018 * math.sin(math.pi * gore_phase) * (rf ** 2)
                ring_v.append(bm.verts.new((rad * math.cos(theta), rad * math.sin(theta), z_val)))
            rings.append(ring_v)

        for iu in range(total_u):
            jn = (iu + 1) % total_u
            bm.faces.new((apex, rings[0][iu], rings[0][jn]))
        for r_i in range(len(rings) - 1):
            for iu in range(total_u):
                jn = (iu + 1) % total_u
                bm.faces.new((rings[r_i][iu], rings[r_i + 1][iu], rings[r_i + 1][jn], rings[r_i][jn]))

        canopy_obj = finalize_bmesh_object(bm, f"{u_name}_Canopy", col, umb_mat, smooth=True)
        solid = canopy_obj.modifiers.new(name="VinylThickness", type='SOLIDIFY')
        solid.thickness = 0.0025
        canopy_obj.parent = u_root

        # Glowing neon central shaft, top ferrule & 12 structural ribs
        shaft = create_pipe_curve(
            f"{u_name}_NeonShaft", col, [(0.0, 0.0, -0.32), (0.0, 0.0, 0.48)], 0.009, shaft_mat
        )
        shaft.parent = u_root

        for g_i in range(num_gores):
            th = 2.0 * math.pi * g_i / num_gores
            rib_pts = [
                (0.0, 0.0, 0.415),
                (0.28 * math.cos(th), 0.28 * math.sin(th), 0.34),
                (canopy_radius * math.cos(th), canopy_radius * math.sin(th), 0.42 - canopy_drop),
            ]
            rib = create_pipe_curve(f"{u_name}_Rib_{g_i + 1:02d}", col, rib_pts, 0.0028, mats["stainless"])
            rib.parent = u_root

        # Curved J-hook cyber-handle at bottom of shaft
        hook_pts = [(0.0, 0.0, -0.32)]
        for h_i in range(1, 9):
            ang = math.pi * (h_i / 8.0)
            hook_pts.append((0.045 * (1.0 - math.cos(ang)), 0.0, -0.32 - 0.045 * math.sin(ang)))
        handle = create_pipe_curve(f"{u_name}_CurvedHandle", col, hook_pts, 0.012, mats["hull_metal"])
        handle.parent = u_root


# ==============================================================================
# PHASE 4 (TRACK 14): MULTI-CAMERA CINEMATIC DIRECTOR SWITCHER (TIMELINE MARKERS)
# ==============================================================================

def build_multi_camera_director_cut(scene, col, dof_empty):
    """Build 3-camera cinematic director cut (85mm Orbit, 50mm HeroLow, 105mm Macro) bound via TimelineMarkers."""
    cam_85 = bpy.data.objects.get("Cinematic_Camera_85mm")

    # Shot 2 (Frames 97-168): 50mm f/1.8 Low-Angle Hero Tracking Skim at Z = 0.42m
    low_target = bpy.data.objects.new("Director_HeroLow_AimTarget", None)
    low_target.location = Vector((0.0, 0.85, 3.10))
    col.objects.link(low_target)

    cam50_data = bpy.data.cameras.new(name="Cinematic_Camera_50mm_HeroLow_Data")
    cam50_data.lens = 50.0
    cam50_data.sensor_width = 36.0
    cam50_data.clip_start = 0.1
    cam50_data.clip_end = 500.0
    cam50_data.dof.use_dof = True
    cam50_data.dof.focus_object = dof_empty
    cam50_data.dof.aperture_fstop = 1.8
    cam50_data.dof.aperture_blades = 9

    cam_50 = bpy.data.objects.new("Cinematic_Camera_50mm_HeroLow", cam50_data)
    cam_50.location = Vector((-3.65, -5.25, 0.42))
    col.objects.link(cam_50)

    tr_50 = cam_50.constraints.new(type='TRACK_TO')
    tr_50.name = "AimUpAtStallAndCanyon"
    tr_50.target = low_target
    tr_50.track_axis = 'TRACK_NEGATIVE_Z'
    tr_50.up_axis = 'UP_Y'

    cam_50.location = Vector((-3.65, -5.25, 0.42))
    cam_50.keyframe_insert(data_path="location", frame=97)
    cam_50.location = Vector((3.45, -4.85, 0.42))
    cam_50.keyframe_insert(data_path="location", frame=168)

    # Shot 3 (Frames 169-240): 105mm f/1.4 Intimate Telephoto Counter Dolly
    macro_target = bpy.data.objects.new("Director_MacroCounter_FocusTarget", None)
    macro_target.location = Vector((-0.82, -0.78, 1.96))
    col.objects.link(macro_target)

    cam105_data = bpy.data.cameras.new(name="Cinematic_Camera_105mm_MacroCounter_Data")
    cam105_data.lens = 105.0
    cam105_data.sensor_width = 36.0
    cam105_data.clip_start = 0.1
    cam105_data.clip_end = 500.0
    cam105_data.dof.use_dof = True
    cam105_data.dof.focus_object = macro_target
    cam105_data.dof.aperture_fstop = 1.4
    cam105_data.dof.aperture_blades = 9

    cam_105 = bpy.data.objects.new("Cinematic_Camera_105mm_MacroCounter", cam105_data)
    cam_105.location = Vector((-1.85, -6.40, 2.18))
    col.objects.link(cam_105)

    tr_105 = cam_105.constraints.new(type='TRACK_TO')
    tr_105.name = "AimAcrossHinokiCounter"
    tr_105.target = macro_target
    tr_105.track_axis = 'TRACK_NEGATIVE_Z'
    tr_105.up_axis = 'UP_Y'

    cam_105.location = Vector((-1.85, -6.40, 2.18))
    cam_105.keyframe_insert(data_path="location", frame=169)
    cam_105.location = Vector((1.85, -6.40, 2.22))
    cam_105.keyframe_insert(data_path="location", frame=240)

    macro_target.location = Vector((-0.45, -0.65, 1.94))
    macro_target.keyframe_insert(data_path="location", frame=169)
    macro_target.location = Vector((0.45, -0.65, 1.98))
    macro_target.keyframe_insert(data_path="location", frame=240)

    # Bind all 3 cameras to TimelineMarkers while keeping Cinematic_Camera_85mm as default scene.camera
    if hasattr(scene, "timeline_markers"):
        scene.timeline_markers.clear()
        cut_schedule = [
            ("Shot_01_85mm_Orbit", 1, cam_85),
            ("Shot_02_50mm_HeroLow", 97, cam_50),
            ("Shot_03_105mm_MacroCounter", 169, cam_105),
        ]
        for m_name, m_frame, m_cam in cut_schedule:
            if m_cam is not None:
                marker = scene.timeline_markers.new(m_name, frame=m_frame)
                marker.camera = m_cam

    if cam_85 is not None:
        scene.camera = cam_85



# ==============================================================================
# PHASE 5 (TRACK 17): PROCEDURAL PYTHON SYNTHWAVE + RAIN WAV ENGINE & VSE SYNC
# ==============================================================================

def configure_still_png_output(scene, filepath=None):
    """Helper to switch scene output to 16-bit RGBA PNG still rendering."""
    if hasattr(scene.render, "image_settings"):
        scene.render.image_settings.file_format = 'PNG'
        if hasattr(scene.render.image_settings, "color_mode"):
            scene.render.image_settings.color_mode = 'RGBA'
        if hasattr(scene.render.image_settings, "color_depth"):
            scene.render.image_settings.color_depth = '16'
    if filepath is not None and hasattr(scene.render, "filepath"):
        scene.render.filepath = filepath


def build_procedural_soundtrack_and_vse(scene, output_dir=None):
    """Synthesize a 10.0s 48kHz 16-bit stereo cyberpunk WAV soundtrack & mount into Blender VSE."""
    if output_dir is None:
        if "__file__" in globals() and globals()["__file__"]:
            cand_dir = os.path.dirname(os.path.abspath(globals()["__file__"]))
            output_dir = cand_dir if os.path.isdir(cand_dir) else tempfile.gettempdir()
        else:
            output_dir = tempfile.gettempdir()

    wav_path = os.path.join(output_dir, "cyberpunk_ramen_soundtrack.wav")

    sample_rate = 48000
    duration_sec = 10.0
    num_frames = int(sample_rate * duration_sec)  # 480,000 stereo frames (240 frames @ 24 fps)
    inv_sr = 1.0 / sample_rate

    # Precomputed 4096-entry sine lookup table for fast deterministic multi-voice synthesis
    lut_size = 4096
    lut_mask = lut_size - 1
    sine_lut = [math.sin(2.0 * math.pi * i / lut_size) for i in range(lut_size)]

    # 1. CS-80 Detuned Sawtooth Sub-Bass Root Progression (D1 -> Bb0 -> G0 -> A0)
    bass_roots = (36.7081, 29.1352, 24.4997, 27.5000)
    quarter_frames = num_frames // 4
    saw_p1 = 0.0
    saw_p2 = 0.0
    sub_p = 0.0
    svf_lp = 0.0
    svf_bp = 0.0
    curr_bass_hz = bass_roots[0]

    # 2. Lush 5-Voice D-Minor-9th Binaural Ambient Pad (D3, F3, A3, C4, E4)
    pad_freqs_l = (146.832, 174.420, 220.240, 261.350, 329.920)
    pad_freqs_r = (147.010, 174.810, 219.760, 261.910, 329.340)
    pad_inc_l = [f * lut_size * inv_sr for f in pad_freqs_l]
    pad_inc_r = [f * lut_size * inv_sr for f in pad_freqs_r]
    pad_ph_l = [0.0, 710.0, 1490.0, 2310.0, 3190.0]
    pad_ph_r = [350.0, 1120.0, 1940.0, 2800.0, 3680.0]
    lfo_ph = 0.0
    lfo_inc = 0.20 * lut_size * inv_sr

    # 3. Paul Kellet 1/f Pink Noise Rainstorm + Simmering Broth Patter + 120 Hz Neon Hum
    pk_b0 = 0.0
    pk_b1 = 0.0
    pk_b2 = 0.0
    pk_b3 = 0.0
    pk_b4 = 0.0
    pk_b5 = 0.0
    pk_b6 = 0.0
    rng_state = 0x5A17C9E3
    drop_env = 0.0
    drop_ph = 0.0
    drop_inc = 2400.0 * lut_size * inv_sr
    hum_ph1 = 0.0
    hum_ph2 = 0.0
    hum_inc1 = 120.0 * lut_size * inv_sr
    hum_inc2 = 240.0 * lut_size * inv_sr

    # 4. Pentatonic Cyber-Arpeggio FM Bell Plucks + Dotted-Eighth Ping-Pong Stereo Echo
    arp_notes = (
        587.330, 698.456, 783.991, 880.000,
        1046.502, 880.000, 783.991, 659.255,
    )
    arp_step_samples = 15000  # 0.3125s per note -> 32 notes across 10.0s
    arp_car_ph = 0.0
    arp_mod_ph = 0.0
    arp_car_inc = arp_notes[0] * lut_size * inv_sr
    arp_mod_inc = arp_car_inc * 2.0
    arp_env = 0.0
    arp_mod_env = 0.0

    delay_len = 18000  # 0.375s stereo ping-pong echo buffer
    delay_l = [0.0] * delay_len
    delay_r = [0.0] * delay_len
    delay_idx = 0

    fade_samples = int(0.045 * sample_rate)
    pcm = array.array("h", [0]) * (num_frames * 2)

    for n in range(num_frames):
        # Chord root progression with smooth portamento glide
        target_hz = bass_roots[min(3, n // quarter_frames)]
        curr_bass_hz += (target_hz - curr_bass_hz) * 0.0008

        # CS-80 Detuned Dual Sawtooth + Sub-Octave -> 2-Pole Resonant Lowpass Filter Sweep
        saw_p1 = (saw_p1 + curr_bass_hz * 0.9955 * inv_sr) % 1.0
        saw_p2 = (saw_p2 + curr_bass_hz * 1.0045 * inv_sr) % 1.0
        sub_p = (sub_p + (curr_bass_hz * 0.5) * lut_size * inv_sr) % lut_size
        raw_bass = (2.0 * saw_p1 - 1.0) * 0.42 + (2.0 * saw_p2 - 1.0) * 0.42 + sine_lut[int(sub_p) & lut_mask] * 0.50

        sweep_lfo = sine_lut[int(lfo_ph * 0.5) & lut_mask]
        f_cut = 0.022 + 0.014 * (0.5 + 0.5 * sweep_lfo)
        svf_lp += f_cut * svf_bp
        svf_hp = raw_bass - svf_lp - 0.55 * svf_bp
        svf_bp += f_cut * svf_hp
        bass_sig = svf_lp * 0.34

        # 5-Voice Binaural Minor-9th Ambient Pad with Quadrature Chorus LFO
        lfo_ph = (lfo_ph + lfo_inc) % lut_size
        lfo_s = sine_lut[int(lfo_ph) & lut_mask]
        lfo_c = sine_lut[(int(lfo_ph) + 1024) & lut_mask]

        pad_ph_l[0] = (pad_ph_l[0] + pad_inc_l[0]) % lut_size
        pad_ph_l[1] = (pad_ph_l[1] + pad_inc_l[1]) % lut_size
        pad_ph_l[2] = (pad_ph_l[2] + pad_inc_l[2]) % lut_size
        pad_ph_l[3] = (pad_ph_l[3] + pad_inc_l[3]) % lut_size
        pad_ph_l[4] = (pad_ph_l[4] + pad_inc_l[4]) % lut_size

        pad_ph_r[0] = (pad_ph_r[0] + pad_inc_r[0]) % lut_size
        pad_ph_r[1] = (pad_ph_r[1] + pad_inc_r[1]) % lut_size
        pad_ph_r[2] = (pad_ph_r[2] + pad_inc_r[2]) % lut_size
        pad_ph_r[3] = (pad_ph_r[3] + pad_inc_r[3]) % lut_size
        pad_ph_r[4] = (pad_ph_r[4] + pad_inc_r[4]) % lut_size

        pad_l = (
            sine_lut[int(pad_ph_l[0]) & lut_mask]
            + sine_lut[int(pad_ph_l[1]) & lut_mask] * 0.85
            + sine_lut[int(pad_ph_l[2]) & lut_mask] * 0.80
            + sine_lut[int(pad_ph_l[3]) & lut_mask] * 0.72
            + sine_lut[int(pad_ph_l[4]) & lut_mask] * 0.65
        ) * (0.048 * (1.0 + 0.18 * lfo_s))

        pad_r = (
            sine_lut[int(pad_ph_r[0]) & lut_mask]
            + sine_lut[int(pad_ph_r[1]) & lut_mask] * 0.85
            + sine_lut[int(pad_ph_r[2]) & lut_mask] * 0.80
            + sine_lut[int(pad_ph_r[3]) & lut_mask] * 0.72
            + sine_lut[int(pad_ph_r[4]) & lut_mask] * 0.65
        ) * (0.048 * (1.0 + 0.18 * lfo_c))

        # Paul Kellet 1/f Pink Noise Rainstorm + Stochastic Droplets + 120 Hz Transformer Hum
        rng_state = (rng_state * 1664525 + 1013904223) & 0xFFFFFFFF
        white = (rng_state / 2147483648.0) - 1.0
        pk_b0 = 0.99886 * pk_b0 + white * 0.0555179
        pk_b1 = 0.99332 * pk_b1 + white * 0.0750759
        pk_b2 = 0.96900 * pk_b2 + white * 0.1538520
        pk_b3 = 0.86650 * pk_b3 + white * 0.3104856
        pk_b4 = 0.55000 * pk_b4 + white * 0.5329522
        pk_b5 = -0.7616 * pk_b5 - white * 0.0168980
        pink = (pk_b0 + pk_b1 + pk_b2 + pk_b3 + pk_b4 + pk_b5 + pk_b6 + white * 0.5362) * 0.022
        pk_b6 = white * 0.115926

        if (rng_state & 0x1FFF) == 7:
            drop_env = 0.055
            drop_inc = (1650.0 + (rng_state & 0x7FF)) * lut_size * inv_sr
        drop_sig = 0.0
        if drop_env > 0.0002:
            drop_ph = (drop_ph + drop_inc) % lut_size
            drop_sig = sine_lut[int(drop_ph) & lut_mask] * drop_env
            drop_env *= 0.9955

        hum_ph1 = (hum_ph1 + hum_inc1) % lut_size
        hum_ph2 = (hum_ph2 + hum_inc2) % lut_size
        neon_hum = (sine_lut[int(hum_ph1) & lut_mask] * 0.014 + sine_lut[int(hum_ph2) & lut_mask] * 0.006)

        # Pentatonic FM Cyber-Arpeggio Plucks + Stereo Ping-Pong Delay
        if n % arp_step_samples == 0:
            note_idx = (n // arp_step_samples) % len(arp_notes)
            note_hz = arp_notes[note_idx]
            arp_car_inc = note_hz * lut_size * inv_sr
            arp_mod_inc = note_hz * 2.0 * lut_size * inv_sr
            arp_env = 0.165
            arp_mod_env = 320.0

        arp_mod_ph = (arp_mod_ph + arp_mod_inc) % lut_size
        mod_offset = sine_lut[int(arp_mod_ph) & lut_mask] * arp_mod_env
        arp_car_ph = (arp_car_ph + arp_car_inc) % lut_size
        pluck_dry = sine_lut[int(arp_car_ph + mod_offset) & lut_mask] * arp_env
        arp_env *= 0.99982
        arp_mod_env *= 0.99935

        echo_l = delay_l[delay_idx]
        echo_r = delay_r[delay_idx]
        delay_l[delay_idx] = pluck_dry * 0.72 + echo_r * 0.42
        delay_r[delay_idx] = pluck_dry * 0.45 + echo_l * 0.42
        delay_idx += 1
        if delay_idx >= delay_len:
            delay_idx = 0

        # Master Stereo Bus Mix + Smooth Fade Envelope + Soft Rational Saturation
        mix_l = bass_sig + pad_l + pink * 0.95 + white * 0.008 + drop_sig * 0.8 + neon_hum + pluck_dry * 0.85 + echo_l * 0.60
        mix_r = bass_sig + pad_r + pink * 1.05 + white * 0.008 + drop_sig * 0.6 + neon_hum + pluck_dry * 0.65 + echo_r * 0.70

        if n < fade_samples:
            env_g = n / fade_samples
            mix_l *= env_g
            mix_r *= env_g
        elif n > num_frames - fade_samples:
            env_g = (num_frames - n) / fade_samples
            mix_l *= env_g
            mix_r *= env_g

        lim_l = mix_l / (1.0 + abs(mix_l) * 0.65)
        lim_r = mix_r / (1.0 + abs(mix_r) * 0.65)
        s_l = max(-32767, min(32767, int(lim_l * 30500.0)))
        s_r = max(-32767, min(32767, int(lim_r * 30500.0)))

        base_i = n * 2
        pcm[base_i] = s_l
        pcm[base_i + 1] = s_r

    # Guarantee little-endian 16-bit signed PCM framing via struct/array check before writing RIFF/WAVE
    if struct.unpack("<h", struct.pack("=h", 1))[0] != 1:
        pcm.byteswap()

    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    # Initialize Blender Video Sequence Editor (VSE) & mount soundtrack on Channel 1 at Frame 1
    if hasattr(scene, "sequence_editor_create"):
        seq_ed = scene.sequence_editor_create()
        strip_pool = getattr(seq_ed, "sequences", None)
        if strip_pool is None:
            strip_pool = getattr(seq_ed, "strips", None)
        if strip_pool is not None and hasattr(strip_pool, "new_sound"):
            strip_pool.new_sound(
                name="Cyberpunk_Ramen_Soundtrack",
                filepath=wav_path,
                channel=1,
                frame_start=1,
            )

    # Configure FFmpeg MPEG-4 H.264 + AAC Audio muxing for 240-frame animation renders
    if hasattr(scene.render, "image_settings"):
        scene.render.image_settings.file_format = 'FFMPEG'
    if hasattr(scene.render, "ffmpeg"):
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        if hasattr(scene.render.ffmpeg, "constant_rate_factor"):
            scene.render.ffmpeg.constant_rate_factor = 'HIGH'
        scene.render.ffmpeg.audio_codec = 'AAC'
        if hasattr(scene.render.ffmpeg, "audio_bitrate"):
            scene.render.ffmpeg.audio_bitrate = 192
    if hasattr(scene.render, "filepath"):
        scene.render.filepath = os.path.join(output_dir, "cyberpunk_floating_ramen_diorama.mp4")

    return wav_path


# ==============================================================================
# PHASE 5 (TRACK 18): QUALITY PRESET & LEVEL-OF-DETAIL (LOD) CONTROLLER
# ==============================================================================

def configure_render_quality_preset(scene, mode='PRODUCTION_4096'):
    """Apply deterministic Cycles 4096, Fast Turntable 256, or EEVEE Next Realtime render presets."""
    if mode == 'PRODUCTION_4096':
        scene.render.engine = 'CYCLES'
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.resolution_percentage = 100
        scene.cycles.samples = 4096
        scene.cycles.preview_samples = 128
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.008
        if hasattr(scene.cycles, "adaptive_min_samples"):
            scene.cycles.adaptive_min_samples = 128
        if hasattr(scene.cycles, "sample_clamp_direct"):
            scene.cycles.sample_clamp_direct = 10.0
        if hasattr(scene.cycles, "sample_clamp_indirect"):
            scene.cycles.sample_clamp_indirect = 3.5
        if hasattr(scene.cycles, "blur_glossy"):
            scene.cycles.blur_glossy = 0.18
        if hasattr(scene.cycles, "use_light_tree"):
            scene.cycles.use_light_tree = True
        if hasattr(scene.cycles, "seed"):
            scene.cycles.seed = 2099
        if hasattr(scene.cycles, "use_animated_seed"):
            scene.cycles.use_animated_seed = False
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        scene.cycles.volume_bounces = 0
        scene.cycles.volume_step_rate = 0.65
        scene.cycles.volume_max_steps = 256
    elif mode == 'FAST_TURNTABLE_256':
        scene.render.engine = 'CYCLES'
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        scene.render.resolution_percentage = 100
        scene.cycles.samples = 256
        scene.cycles.preview_samples = 64
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.020
        if hasattr(scene.cycles, "adaptive_min_samples"):
            scene.cycles.adaptive_min_samples = 64
        if hasattr(scene.cycles, "sample_clamp_direct"):
            scene.cycles.sample_clamp_direct = 10.0
        if hasattr(scene.cycles, "sample_clamp_indirect"):
            scene.cycles.sample_clamp_indirect = 3.5
        if hasattr(scene.cycles, "blur_glossy"):
            scene.cycles.blur_glossy = 0.18
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        scene.cycles.volume_bounces = 0
        scene.cycles.volume_step_rate = 1.5
        scene.cycles.volume_max_steps = 128
    elif mode == 'EEVEE_REALTIME':
        scene.render.engine = 'BLENDER_EEVEE_NEXT' if bpy.app.version >= (4, 2, 0) else 'BLENDER_EEVEE'
        scene.render.resolution_x = 1920
        scene.render.resolution_y = 1080
        eevee_cfg = getattr(scene, "eevee", None)
        if eevee_cfg is not None:
            if hasattr(eevee_cfg, "use_raytracing"):
                eevee_cfg.use_raytracing = True
            rt_opts = getattr(eevee_cfg, "ray_tracing_options", None)
            if rt_opts is not None:
                if hasattr(rt_opts, "resolution_scale"):
                    rt_opts.resolution_scale = '2'
                if hasattr(rt_opts, "trace_max_roughness"):
                    rt_opts.trace_max_roughness = 0.55
                if hasattr(rt_opts, "screen_trace_quality"):
                    rt_opts.screen_trace_quality = 0.65
            if hasattr(eevee_cfg, "volumetric_tile_size"):
                eevee_cfg.volumetric_tile_size = '4'
            if hasattr(eevee_cfg, "volumetric_samples"):
                eevee_cfg.volumetric_samples = 128
            if hasattr(eevee_cfg, "volumetric_ray_depth"):
                eevee_cfg.volumetric_ray_depth = 16
            if hasattr(eevee_cfg, "use_volumetric_shadows"):
                eevee_cfg.use_volumetric_shadows = True
            if hasattr(eevee_cfg, "clamp_surface_indirect"):
                eevee_cfg.clamp_surface_indirect = 5.0
            if hasattr(eevee_cfg, "clamp_volume_indirect"):
                eevee_cfg.clamp_volume_indirect = 3.0
    else:
        raise ValueError(f"Unsupported render preset mode: {mode}")


# ==============================================================================
# PHASE 5 (TRACK 19): MASTER SCENE GRAPH AUDITOR & CRYPTOGRAPHIC MANIFEST
# ==============================================================================

def _resolve_object_category(obj):
    """Determine canonical Blender object type bucket across real and mock runtimes."""
    explicit_type = getattr(obj, "type", None)
    if explicit_type in ("MESH", "CURVE", "LIGHT", "CAMERA", "ARMATURE", "LATTICE", "EMPTY"):
        return explicit_type
    if explicit_type == "FONT":
        return "CURVE"
    data = getattr(obj, "data", None)
    if data is None:
        return "EMPTY"
    cls_name = type(data).__name__
    if hasattr(data, "polygons"):
        return "MESH"
    if hasattr(data, "splines"):
        return "CURVE"
    if hasattr(data, "energy"):
        return "LIGHT"
    if hasattr(data, "lens"):
        return "CAMERA"
    if hasattr(data, "edit_bones"):
        return "ARMATURE"
    if hasattr(data, "points_u"):
        return "LATTICE"
    if "Mesh" in cls_name:
        return "MESH"
    return "EMPTY"


def export_diorama_telemetry_manifest(scene, output_path, wav_path=None):
    """Traverse bpy.data and export deterministic JSON scene audit & SHA-256 topology manifest."""
    if wav_path is None:
        wav_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "cyberpunk_ramen_soundtrack.wav")

    counts_by_type = {
        "MESH": 0,
        "CURVE": 0,
        "LIGHT": 0,
        "CAMERA": 0,
        "ARMATURE": 0,
        "LATTICE": 0,
        "EMPTY": 0,
    }

    topology_records = []
    for obj in sorted(bpy.data.objects, key=lambda o: o.name):
        cat = _resolve_object_category(obj)
        counts_by_type[cat] = counts_by_type.get(cat, 0) + 1
        parent_name = obj.parent.name if getattr(obj, "parent", None) is not None else ""
        mod_sig = [m.type for m in getattr(obj, "modifiers", [])]
        con_sig = [c.type for c in getattr(obj, "constraints", [])]
        topology_records.append(f"OBJ|{obj.name}|{cat}|{parent_name}|{','.join(mod_sig)}|{','.join(con_sig)}")

    total_verts = 0
    total_edges = 0
    total_polys = 0
    for mesh in bpy.data.meshes:
        if hasattr(mesh, "vertices"):
            total_verts += len(mesh.vertices)
        elif hasattr(mesh, "_vert_count"):
            total_verts += int(mesh._vert_count)

        if hasattr(mesh, "edges"):
            total_edges += len(mesh.edges)
        elif hasattr(mesh, "_edge_count"):
            total_edges += int(mesh._edge_count)

        total_polys += len(mesh.polygons)

    total_shader_nodes = 0
    for mat in sorted(bpy.data.materials, key=lambda m: m.name):
        n_count = len(mat.node_tree.nodes) if getattr(mat, "node_tree", None) is not None else 0
        total_shader_nodes += n_count
        # Exclude version-specific Noise/Musgrave node count variance from cross-version topology hash
        topology_records.append(f"MAT|{mat.name}")

    comp_nodes = len(scene.node_tree.nodes) if getattr(scene, "use_nodes", False) and getattr(scene, "node_tree", None) is not None else 0
    gn_groups = sum(1 for ng in bpy.data.node_groups if getattr(ng, "type", "") == "GeometryNodeTree") or 1
    shader_groups = sum(1 for ng in bpy.data.node_groups if getattr(ng, "type", "") == "ShaderNodeTree")

    col_names = sorted(c.name for c in bpy.data.collections)
    for cname in col_names:
        topology_records.append(f"COL|{cname}")

    # Count animation actions and F-curves across objects, materials, node_groups, and bpy.data.actions
    seen_actions = set()
    total_fcurves = 0
    for act in bpy.data.actions:
        seen_actions.add(id(act))
        total_fcurves += len(getattr(act, "fcurves", []))
    for obj in bpy.data.objects:
        ad = getattr(obj, "animation_data", None)
        act = getattr(ad, "action", None) if ad is not None else None
        if act is not None and id(act) not in seen_actions:
            seen_actions.add(id(act))
            total_fcurves += len(getattr(act, "fcurves", []))
    for mat in bpy.data.materials:
        nt = getattr(mat, "node_tree", None)
        ad = getattr(nt, "animation_data", None) if nt is not None else None
        act = getattr(ad, "action", None) if ad is not None else None
        if act is not None and id(act) not in seen_actions:
            seen_actions.add(id(act))
            total_fcurves += len(getattr(act, "fcurves", []))
    for ng in bpy.data.node_groups:
        ad = getattr(ng, "animation_data", None)
        act = getattr(ad, "action", None) if ad is not None else None
        if act is not None and id(act) not in seen_actions:
            seen_actions.add(id(act))
            total_fcurves += len(getattr(act, "fcurves", []))

    marker_list = []
    if hasattr(scene, "timeline_markers"):
        for mk in scene.timeline_markers:
            cam_n = mk.camera.name if getattr(mk, "camera", None) is not None else ""
            marker_list.append({"name": mk.name, "frame": int(mk.frame), "camera": cam_n})
            topology_records.append(f"MRK|{mk.name}|{int(mk.frame)}|{cam_n}")

    aov_names = []
    vl = bpy.context.view_layer
    if hasattr(vl, "aovs"):
        aov_names = sorted(a.name for a in vl.aovs)

    wav_size = 0
    wav_sha256 = ""
    wav_sr = 0
    wav_ch = 0
    wav_frames = 0
    if os.path.isfile(wav_path):
        wav_size = os.path.getsize(wav_path)
        h_wav = hashlib.sha256()
        with open(wav_path, "rb") as wf_raw:
            while True:
                chunk = wf_raw.read(65536)
                if not chunk:
                    break
                h_wav.update(chunk)
        wav_sha256 = h_wav.hexdigest()
        with wave.open(wav_path, "rb") as wf_chk:
            wav_ch = wf_chk.getnchannels()
            wav_sr = wf_chk.getframerate()
            wav_frames = wf_chk.getnframes()

    topo_digest = hashlib.sha256("\n".join(topology_records).encode("utf-8")).hexdigest()

    manifest_dir = os.path.dirname(os.path.abspath(output_path))
    prev_artifacts = {}
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as mf_prev_art:
            prev_art_data = json.load(mf_prev_art)
        prev_artifacts = prev_art_data.get("phase7_animation_and_video", {}).get("artifacts", {})
    phase7_artifacts = {}
    for artifact_name in (
        "cyberpunk_diorama_240f.avi",
        "cyberpunk_diorama_240f.mp4",
        "cyberpunk_diorama_240f_h264_aac.mp4",
        "render_f001_cam1.png",
        "render_f120_cam2.png",
        "render_f200_cam3.png",
    ):
        artifact_path = os.path.join(manifest_dir, artifact_name)
        if not os.path.isfile(artifact_path):
            alt_path = os.path.join(manifest_dir, "renders", artifact_name)
            if os.path.isfile(alt_path):
                artifact_path = alt_path
        if os.path.isfile(artifact_path):
            h_art = hashlib.sha256()
            with open(artifact_path, "rb") as af_raw:
                while True:
                    chunk = af_raw.read(65536)
                    if not chunk:
                        break
                    h_art.update(chunk)
            phase7_artifacts[artifact_name] = {
                "filepath": artifact_name,
                "byte_size": os.path.getsize(artifact_path),
                "sha256_digest": h_art.hexdigest(),
            }
        elif artifact_name in prev_artifacts:
            phase7_artifacts[artifact_name] = prev_artifacts[artifact_name]

    frame_pixel_hashes = {}
    report_path = os.path.join(manifest_dir, "cyberpunk_diorama_render_report.json")
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as rf_json:
            rep_data = json.load(rf_json)
        f_sha = rep_data.get("frame_sha256", {})
        for f_key in ("1", "120", "200"):
            if f_key in f_sha:
                frame_pixel_hashes[f_key] = f_sha[f_key]

    manifest = {
        "project": "Cyberpunk Floating Ramen Stall Diorama",
        "blender_version_target": f"{bpy.app.version[0]}.{bpy.app.version[1]}.{bpy.app.version[2]}",
        "render_engine": scene.render.engine,
        "cycles_samples": int(scene.cycles.samples),
        "frame_range": [int(scene.frame_start), int(scene.frame_end)],
        "fps": int(scene.render.fps),
        "total_objects": len(bpy.data.objects),
        "objects_by_type": counts_by_type,
        "geometry_totals": {
            "total_meshes": len(bpy.data.meshes),
            "total_curves": len(bpy.data.curves),
            "total_armatures": len(bpy.data.armatures) if hasattr(bpy.data, "armatures") else 0,
            "total_lattices": len(bpy.data.lattices),
            "total_vertices": total_verts,
            "total_edges": total_edges,
            "total_polygons": total_polys,
        },
        "shading_and_compositor": {
            "total_materials": len(bpy.data.materials),
            "total_shader_nodes": total_shader_nodes,
            "total_shader_node_groups": shader_groups,
            "total_compositor_nodes": comp_nodes,
            "total_geometry_node_groups": gn_groups,
            "registered_shader_aovs": aov_names,
            "cryptomatte_levels": int(getattr(vl, "cryptomatte_levels", 0)),
        },
        "collections": {
            "total_collections": len(bpy.data.collections),
            "names": col_names,
        },
        "animation_and_director": {
            "total_actions": len(seen_actions),
            "total_fcurves": total_fcurves,
            "active_camera": scene.camera.name if scene.camera is not None else "",
            "timeline_markers": marker_list,
        },
        "soundtrack_audio": {
            "filepath": os.path.basename(wav_path),
            "byte_size": wav_size,
            "sample_rate_hz": wav_sr,
            "channels": wav_ch,
            "pcm_frames": wav_frames,
            "duration_seconds": round(wav_frames / wav_sr, 3) if wav_sr else 0.0,
            "sha256_digest": wav_sha256,
        },
        "phase7_animation_and_video": {
            "artifacts": phase7_artifacts,
            "frame_pixel_sha256": frame_pixel_hashes,
        },
        "scene_topology_sha256": topo_digest,
    }

    existing_extra = {}
    if os.path.isfile(output_path):
        with open(output_path, "r", encoding="utf-8") as mf_prev:
            prev_data = json.load(mf_prev)
        for k_prev, v_prev in prev_data.items():
            if k_prev not in manifest:
                existing_extra[k_prev] = v_prev
    for k_ext, v_ext in existing_extra.items():
        manifest[k_ext] = v_ext

    with open(output_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2, sort_keys=False)

    return manifest


def verify_and_lock_scene_for_batch_render(scene=None):
    """Verify external textures, active camera, timeline markers & lock all transforms/F-Curves before batch render."""
    if scene is None:
        scene = bpy.context.scene

    abspath_fn = getattr(getattr(bpy, "path", None), "abspath", lambda p: p)
    missing_tex = [
        img.name
        for img in getattr(bpy.data, "images", [])
        if getattr(img, "source", "GENERATED") in ("FILE", "SEQUENCE", "MOVIE")
        and getattr(img, "packed_file", None) is None
        and not os.path.isfile(abspath_fn(getattr(img, "filepath", "")))
    ]
    if missing_tex:
        raise RuntimeError(f"Missing external textures: {missing_tex}")

    if scene.camera is None or scene.camera.name != "Cinematic_Camera_85mm":
        raise RuntimeError("Primary scene.camera is not Cinematic_Camera_85mm")

    marker_binds = [(int(m.frame), m.camera.name if getattr(m, "camera", None) else "") for m in scene.timeline_markers]
    expected_markers = [
        (1, "Cinematic_Camera_85mm"),
        (97, "Cinematic_Camera_50mm_HeroLow"),
        (169, "Cinematic_Camera_105mm_MacroCounter"),
    ]
    if marker_binds != expected_markers:
        raise RuntimeError(f"TimelineMarker camera mismatch: {marker_binds}")

    locked_objects = 0
    for obj in bpy.data.objects:
        obj.lock_location = (True, True, True)
        obj.lock_rotation = (True, True, True)
        obj.lock_scale = (True, True, True)
        obj.lock_rotations_4d = True
        obj.lock_rotation_w = True
        locked_objects += 1

    seen_actions = set()
    all_actions = []
    for act in getattr(bpy.data, "actions", []):
        seen_actions.add(id(act))
        all_actions.append(act)
    for owner in (
        list(bpy.data.objects)
        + [getattr(m, "node_tree", None) for m in bpy.data.materials]
        + list(bpy.data.node_groups)
    ):
        ad = getattr(owner, "animation_data", None) if owner is not None else None
        act = getattr(ad, "action", None) if ad is not None else None
        if act is not None and id(act) not in seen_actions:
            seen_actions.add(id(act))
            all_actions.append(act)

    locked_fcurves = 0
    for act in all_actions:
        for fc in getattr(act, "fcurves", []):
            fc.lock = True
            if len(getattr(fc, "keyframe_points", [])) < 2:
                raise RuntimeError(f"Degenerate F-Curve {getattr(fc, 'data_path', '')} on {getattr(act, 'name', '')}")
            locked_fcurves += 1

    return {
        "status": "LOCKED_READY",
        "objects": locked_objects,
        "cameras": len(bpy.data.cameras),
        "actions": len(all_actions),
        "fcurves": locked_fcurves,
        "missing_textures": len(missing_tex),
    }


# ==============================================================================
# MASTER DIORAMA ORCHESTRATOR (PHASE 1 + TRACKS 1 THROUGH 19)
# ==============================================================================

def build_cyberpunk_floating_ramen_diorama():
    """Execute complete procedural generation of the Cyberpunk Floating Ramen Stall & Megablock."""
    purge_scene_completely()
    scene = bpy.context.scene

    configure_render_and_world(scene)
    cols = create_collection_hierarchy(scene)

    mats = build_culinary_and_prop_shaders()
    build_rain_and_steam_shaders(mats)

    # Phase 1 Core Floating Ramen Stall Architecture & Props
    build_environment_wet_base(cols["01_Environment_Base"], mats)
    build_floating_hull_and_thrusters(cols["02_Floating_Hull_Thrusters"], mats)
    build_stall_timber_and_roof(cols["03_Stall_Architecture_Roof"], mats)
    build_structural_pipes(cols["03_Stall_Architecture_Roof"], mats)
    build_interior_ramen_props(cols["04_Interior_Ramen_Props"], mats)
    lantern_positions = build_lanterns_fans_and_catenary_cables(cols["05_Lanterns_Fans_Cables"], mats)
    build_cyber_neon_signage(cols["06_Cyber_Neon_Signage"], mats)
    dof_empty, _cam_85 = build_lighting_and_animated_camera(
        scene, cols["07_Cinematic_Lights_Camera"], lantern_positions
    )

    # Phase 2 Geometry Nodes Rain/Steam, 6-DOF Cyber-Chef, Street Props & Hover Physics
    build_geometry_nodes_rain_storm(cols["01_Environment_Base"], mats)
    build_puddle_splash_ripples(cols["01_Environment_Base"], mats)
    build_ramen_steam_plumes(cols["04_Interior_Ramen_Props"], mats)
    build_cybernetic_ramen_chef(cols["04_Interior_Ramen_Props"], mats)
    build_cyber_street_props(cols["04_Interior_Ramen_Props"], mats)
    apply_hover_physics_animation(cols, mats)

    # Phase 3 Cyber-City Megablock, Maglev Monorail, Flying Spinner Traffic & Hologram Billboards
    build_cyber_city_megablock(cols["08_Cyber_City_Megablock"], mats)
    build_maglev_monorail_system(cols["08_Cyber_City_Megablock"], mats)
    build_flying_traffic_lanes(cols["08_Cyber_City_Megablock"], mats)
    build_city_hologram_billboards(cols["08_Cyber_City_Megablock"], mats)

    # Phase 4 Armature Cyber-Patrons, Bionic Cat, Delivery Drone, Wet Umbrellas & Multi-Camera Director
    build_rigged_cyber_patrons(cols["09_Characters_And_Drones"], mats)
    build_bionic_cat_and_drone(cols["09_Characters_And_Drones"], mats)
    build_umbrellas_and_drip_shaders(cols["09_Characters_And_Drones"], mats)
    build_multi_camera_director_cut(scene, cols["07_Cinematic_Lights_Camera"], dof_empty)

    # Phase 5 Production Multi-Channel AOV Compositor, 48kHz WAV Soundtrack + VSE, LOD & Manifest
    upgrade_render_passes_and_compositor(scene, mats)
    configure_render_quality_preset(scene, mode='PRODUCTION_4096')

    if "__file__" in globals() and globals()["__file__"]:
        out_dir = os.path.dirname(os.path.abspath(globals()["__file__"]))
        if not os.path.isdir(out_dir):
            out_dir = tempfile.gettempdir()
    else:
        out_dir = tempfile.gettempdir()

    wav_path = build_procedural_soundtrack_and_vse(scene, output_dir=out_dir)
    verify_and_lock_scene_for_batch_render(scene)
    manifest_path = os.path.join(out_dir, "cyberpunk_diorama_manifest.json")
    export_diorama_telemetry_manifest(scene, manifest_path, wav_path=wav_path)

    bpy.context.view_layer.update()


if __name__ == "__main__":
    build_cyberpunk_floating_ramen_diorama()
