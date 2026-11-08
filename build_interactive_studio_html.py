#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOD MODE BLENDER -- PHASE 8A / 8B
Scene Binary Packer + Zero-Dependency WebGL2 Viewport + WebAudio FFT Deck.

Pipeline
  1. Executes ``cyberpunk_floating_ramen_stall.py`` inside the ``verify_blender_script.py``
     mock Blender runtime (bpy / bmesh / mathutils stand-ins).
  2. Tessellates every MESH, CURVE and FONT object into indexed triangle buffers
     (Float32 positions + normals, Uint32 indices, per-vertex material / bone / flag bytes),
     applying ARRAY, SOLIDIFY and LATTICE (KEY_BSPLINE) modifiers on the CPU, sweeping
     bevelled POLY / NURBS / BEZIER splines into tubes and stroking 3D text with a vector font.
  3. Introspects all Principled shader graphs into compact PBR / emissive material records
     (31 materials) including procedural pattern classes, AOV flags and animated parameters.
  4. Bakes the full 240-frame timeline: object F-curves (auto-clamped Bezier / linear),
     FOLLOW_PATH + TRACK_TO constraints, armature pose-bone skinning matrices, light energy
     and material tracks, 3-camera director cut markers.
  5. Packs everything into a byte-shuffled, zlib-compressed binary container, Base64-embeds it
     next to the WAV soundtrack inside ONE self-contained HTML file with a custom WebGL2
     multi-target renderer (PBR + Cook-Torrance, planar mirror at Z = 0, neon bloom, mist,
     chromatic aberration, AgX High Contrast), WebAudio 64-band FFT HUD and a synced
     240-frame timeline / video deck.

Python 3.14 standard library only. No CDN, no external assets.
"""

import argparse
import array
import base64
import bisect
import hashlib
import importlib
import json
import math
import os
import struct
import sys
import time
import zlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_SCRIPT_NAME = "cyberpunk_floating_ramen_stall.py"
VERIFIER_MODULE = "verify_blender_script"
WAV_NAME = "cyberpunk_ramen_soundtrack.wav"
HTML_NAME = "cyberpunk_interactive_studio.html"
MANIFEST_NAME = "cyberpunk_diorama_manifest.json"
VIDEO_NAME = "cyberpunk_diorama_240f.mp4"

FPS = 24
FRAME_START = 1
FRAME_END = 240
NUM_FRAMES = FRAME_END - FRAME_START + 1
PAYLOAD_MAGIC = b"CPKS"
PAYLOAD_VERSION = 1
AUTO_SMOOTH_ANGLE_DEG = 40.0
MAX_LIGHTS = 16
MAX_BONES = 16
BONE_NONE = 255

# Procedural surface classes understood by the GLSL material evaluator
PROC_GENERIC = 0
PROC_BRUSHED_METAL = 1
PROC_WET_ASPHALT = 2
PROC_WOOD = 3
PROC_CERAMIC = 4
PROC_NIGHT_WINDOWS = 5
PROC_HOLOGRAM = 6
PROC_NEON_TUBE = 7
PROC_PAPER_LANTERN = 8
PROC_GLASS_RAIN = 9
PROC_STEAM = 10
PROC_LED_MATRIX = 11
PROC_LIGHT_TRAIL = 12
PROC_FABRIC = 13
PROC_ORGANIC = 14
PROC_NAMES = {
    PROC_GENERIC: "generic", PROC_BRUSHED_METAL: "brushed_metal", PROC_WET_ASPHALT: "wet_asphalt",
    PROC_WOOD: "wood", PROC_CERAMIC: "ceramic", PROC_NIGHT_WINDOWS: "night_windows",
    PROC_HOLOGRAM: "hologram", PROC_NEON_TUBE: "neon_tube", PROC_PAPER_LANTERN: "paper_lantern",
    PROC_GLASS_RAIN: "glass_rain", PROC_STEAM: "steam", PROC_LED_MATRIX: "led_matrix",
    PROC_LIGHT_TRAIL: "light_trail", PROC_FABRIC: "fabric", PROC_ORGANIC: "organic",
}

# Per-vertex object flag bits (attribute byte 2)
OBJ_FLAG_THRUSTER = 1
OBJ_FLAG_NEON_SIGN = 2
OBJ_FLAG_SKINNED = 4
OBJ_FLAG_WAVE = 8

# Material flag bits (material texel 5 .w)
MAT_FLAG_NEON_AOV = 1
MAT_FLAG_TRANSPARENT = 2
MAT_FLAG_MIRROR = 4
MAT_FLAG_THERMAL = 8

MATERIAL_TEXELS = 6

PRINCIPLED_DEFAULTS = {
    "Base Color": (0.8, 0.8, 0.8, 1.0),
    "Metallic": 0.0,
    "Roughness": 0.5,
    "IOR": 1.5,
    "Alpha": 1.0,
    "Emission Color": (1.0, 1.0, 1.0, 1.0),
    "Emission Strength": 0.0,
    "Transmission Weight": 0.0,
    "Coat Weight": 0.0,
    "Coat Roughness": 0.03,
    "Subsurface Weight": 0.0,
    "Sheen Weight": 0.0,
}


# ==============================================================================
# 1. VECTOR / MATRIX MATH (row-major 4x4 stored as flat 16-float lists)
# ==============================================================================

def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def v_norm(a, fallback=(0.0, 0.0, 1.0)):
    length = v_len(a)
    if length < 1e-12:
        return fallback
    return (a[0] / length, a[1] / length, a[2] / length)


def v_lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def m_identity():
    return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def m_mul(a, b):
    out = [0.0] * 16
    for r in range(4):
        a0, a1, a2, a3 = a[r * 4], a[r * 4 + 1], a[r * 4 + 2], a[r * 4 + 3]
        out[r * 4] = a0 * b[0] + a1 * b[4] + a2 * b[8] + a3 * b[12]
        out[r * 4 + 1] = a0 * b[1] + a1 * b[5] + a2 * b[9] + a3 * b[13]
        out[r * 4 + 2] = a0 * b[2] + a1 * b[6] + a2 * b[10] + a3 * b[14]
        out[r * 4 + 3] = a0 * b[3] + a1 * b[7] + a2 * b[11] + a3 * b[15]
    return out


def euler_xyz_rows(e):
    """Blender Euler('XYZ') -> 3x3 rows (R = Rz * Ry * Rx)."""
    cx, sx = math.cos(e[0]), math.sin(e[0])
    cy, sy = math.cos(e[1]), math.sin(e[1])
    cz, sz = math.cos(e[2]), math.sin(e[2])
    return (
        (cy * cz, cz * sx * sy - cx * sz, cx * cz * sy + sx * sz),
        (cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx),
        (-sy, cy * sx, cx * cy),
    )


def m_trs(loc, rot, scl):
    r = euler_xyz_rows(rot)
    sx, sy, sz = scl[0], scl[1], scl[2]
    return [
        r[0][0] * sx, r[0][1] * sy, r[0][2] * sz, loc[0],
        r[1][0] * sx, r[1][1] * sy, r[1][2] * sz, loc[1],
        r[2][0] * sx, r[2][1] * sy, r[2][2] * sz, loc[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def m_from_columns(x_axis, y_axis, z_axis, pos):
    return [
        x_axis[0], y_axis[0], z_axis[0], pos[0],
        x_axis[1], y_axis[1], z_axis[1], pos[1],
        x_axis[2], y_axis[2], z_axis[2], pos[2],
        0.0, 0.0, 0.0, 1.0,
    ]


def m_inverse(m):
    """Affine 4x4 inverse via 3x3 adjugate."""
    a, b, c = m[0], m[1], m[2]
    d, e, f = m[4], m[5], m[6]
    g, h, i = m[8], m[9], m[10]
    co00 = e * i - f * h
    co01 = -(d * i - f * g)
    co02 = d * h - e * g
    det = a * co00 + b * co01 + c * co02
    if abs(det) < 1e-18:
        return m_identity()
    inv = 1.0 / det
    r00, r01, r02 = co00 * inv, -(b * i - c * h) * inv, (b * f - c * e) * inv
    r10, r11, r12 = co01 * inv, (a * i - c * g) * inv, -(a * f - c * d) * inv
    r20, r21, r22 = co02 * inv, -(a * h - b * g) * inv, (a * e - b * d) * inv
    tx, ty, tz = m[3], m[7], m[11]
    return [
        r00, r01, r02, -(r00 * tx + r01 * ty + r02 * tz),
        r10, r11, r12, -(r10 * tx + r11 * ty + r12 * tz),
        r20, r21, r22, -(r20 * tx + r21 * ty + r22 * tz),
        0.0, 0.0, 0.0, 1.0,
    ]


def m_point(m, p):
    return (
        m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3],
        m[4] * p[0] + m[5] * p[1] + m[6] * p[2] + m[7],
        m[8] * p[0] + m[9] * p[1] + m[10] * p[2] + m[11],
    )


def m_dir(m, d):
    return (
        m[0] * d[0] + m[1] * d[1] + m[2] * d[2],
        m[4] * d[0] + m[5] * d[1] + m[6] * d[2],
        m[8] * d[0] + m[9] * d[1] + m[10] * d[2],
    )


def m_translation(m):
    return (m[3], m[7], m[11])


def m_column(m, col):
    return (m[col], m[4 + col], m[8 + col])


def m_row34(m):
    return [float(x) for x in m[:12]]


# ==============================================================================
# 2. MOCK BLENDER RUNTIME EXECUTION
# ==============================================================================

def load_mock_runtime(scene_dir):
    """Import verify_blender_script.py as the mock bpy/bmesh/mathutils provider."""
    if scene_dir not in sys.path:
        sys.path.insert(0, scene_dir)
    return importlib.import_module(VERIFIER_MODULE)


def install_socket_keyframe_snapshot(vbs):
    """Make node-socket keyframes carry their default_value (mock stores 0.0 otherwise)."""
    if getattr(vbs.Socket, "_studio_snapshot_patch", False):
        return
    original = vbs.Socket.keyframe_insert

    def keyframe_insert(self, data_path, frame, index=-1):
        original(self, data_path, frame, index)
        tree = self._tree
        if tree is None or getattr(tree, "animation_data", None) is None:
            return
        raw = getattr(self, data_path, None)
        if hasattr(raw, "__iter__"):
            snap = tuple(float(c) for c in raw)
        elif isinstance(raw, (int, float)):
            snap = float(raw)
        else:
            snap = 0.0
        path = f"socket.{self.name}.{data_path}"
        for fc in tree.animation_data.action.fcurves:
            if fc.data_path == path and fc.keyframe_points:
                fc.keyframe_points[-1].co = (frame, snap)

    vbs.Socket.keyframe_insert = keyframe_insert
    vbs.Socket._studio_snapshot_patch = True


def execute_scene(scene_path, reuse_wav=False):
    """Execute the diorama generator under the Blender 4.3.0 mock runtime and return bpy."""
    scene_path = os.path.abspath(scene_path)
    scene_dir = os.path.dirname(scene_path)
    vbs = load_mock_runtime(scene_dir)
    install_socket_keyframe_snapshot(vbs)
    bpy, bmesh_mod, mathutils_mod = vbs.make_mock_blender((4, 3, 0))
    saved = {k: sys.modules.get(k) for k in ("bpy", "bmesh", "mathutils")}
    sys.modules["bpy"] = bpy
    sys.modules["bmesh"] = bmesh_mod
    sys.modules["mathutils"] = mathutils_mod
    with open(scene_path, "r", encoding="utf-8") as fh:
        src = fh.read()
    wav_path = os.path.join(scene_dir, WAV_NAME)
    try:
        code = compile(src, scene_path, "exec")
        if reuse_wav and os.path.isfile(wav_path):
            ns = {"__name__": "studio_scene_module", "__file__": scene_path}
            exec(code, ns)

            def cached_soundtrack(scene, output_dir=None):
                seq = scene.sequence_editor_create()
                if not seq.sequences:
                    seq.sequences.new_sound("Cyberpunk_Ramen_Soundtrack", wav_path, 1, 1)
                return wav_path

            ns["build_procedural_soundtrack_and_vse"] = cached_soundtrack
            ns["build_cyberpunk_floating_ramen_diorama"]()
        else:
            ns = {"__name__": "__main__", "__file__": scene_path}
            exec(code, ns)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return bpy, ns


# ==============================================================================
# 3. F-CURVE EVALUATION (auto-clamped Bezier / linear / constant)
# ==============================================================================

class FCurveChannel:
    """Evaluates a mock F-curve whose keyframes carry scalar or vector snapshots."""

    __slots__ = ("frames", "values", "interp", "slopes", "dim")

    def __init__(self, keyframe_points):
        dedup = {}
        for kp in keyframe_points:
            fr = float(kp.co[0])
            raw = kp.co[1]
            if isinstance(raw, (tuple, list)):
                vals = tuple(float(c) for c in raw)
            else:
                vals = (float(raw),)
            dedup[fr] = (vals, getattr(kp, "interpolation", "BEZIER"))
        self.frames = sorted(dedup)
        self.values = [dedup[f][0] for f in self.frames]
        self.interp = [dedup[f][1] for f in self.frames]
        self.dim = len(self.values[0]) if self.values else 1
        n = len(self.frames)
        self.slopes = []
        for i in range(n):
            row = []
            for c in range(self.dim):
                if i == 0 or i == n - 1:
                    row.append(0.0)
                    continue
                vp, vi, vn = self.values[i - 1][c], self.values[i][c], self.values[i + 1][c]
                if (vi - vp) * (vn - vi) <= 0.0:
                    row.append(0.0)
                else:
                    row.append((vn - vp) / (self.frames[i + 1] - self.frames[i - 1]))
            self.slopes.append(row)

    def evaluate(self, frame):
        frames = self.frames
        if not frames:
            return (0.0,)
        if frame <= frames[0]:
            return self.values[0]
        if frame >= frames[-1]:
            return self.values[-1]
        i = bisect.bisect_right(frames, frame) - 1
        t0, t1 = frames[i], frames[i + 1]
        v0, v1 = self.values[i], self.values[i + 1]
        u = (frame - t0) / (t1 - t0)
        mode = self.interp[i]
        if mode == "CONSTANT":
            return v0
        if mode == "LINEAR":
            return tuple(a + (b - a) * u for a, b in zip(v0, v1))
        h = t1 - t0
        u2 = u * u
        u3 = u2 * u
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2
        m0, m1 = self.slopes[i], self.slopes[i + 1]
        return tuple(
            h00 * v0[c] + h10 * h * m0[c] + h01 * v1[c] + h11 * h * m1[c]
            for c in range(self.dim)
        )

    def key_frames(self):
        return [int(round(f)) for f in self.frames]


def build_channels(anim_data):
    channels = {}
    if anim_data is None or getattr(anim_data, "action", None) is None:
        return channels
    for fc in getattr(anim_data.action, "fcurves", []):
        if fc.keyframe_points:
            channels[fc.data_path] = FCurveChannel(fc.keyframe_points)
    return channels


# ==============================================================================
# 4. SPLINE EVALUATION (POLY / NURBS / BEZIER) + ARC-LENGTH PATHS
# ==============================================================================

def nurbs_knot_vector(count, order, endpoint):
    if endpoint:
        interior = [float(i) for i in range(1, count - order + 1)]
        return [0.0] * order + interior + [float(count - order + 1)] * order
    return [float(i) for i in range(count + order)]


def bspline_point(ctrl, knots, order, t):
    n = len(ctrl)
    s = order - 1
    while s < n - 1 and t >= knots[s + 1]:
        s += 1
    d = [ctrl[j + s - order + 1] for j in range(order)]
    for r in range(1, order):
        for j in range(order - 1, r - 1, -1):
            i = j + s - order + 1
            denom = knots[i + order - r] - knots[i]
            alpha = 0.0 if abs(denom) < 1e-12 else (t - knots[i]) / denom
            d[j] = v_lerp(d[j - 1], d[j], alpha)
    return d[order - 1]


def evaluate_nurbs(ctrl, order, cyclic, endpoint, resolution):
    n = len(ctrl)
    if n < 2:
        return list(ctrl), cyclic
    order = max(2, min(order, n))
    if cyclic:
        ext = list(ctrl) + list(ctrl[:order - 1])
        knots = [float(i) for i in range(len(ext) + order)]
        t_lo, t_hi = knots[order - 1], knots[len(ext)]
        samples = max(3, resolution * n)
        pts = [bspline_point(ext, knots, order, t_lo + (t_hi - t_lo) * k / samples) for k in range(samples)]
        return pts, True
    knots = nurbs_knot_vector(n, order, endpoint)
    t_lo, t_hi = knots[order - 1], knots[n]
    samples = max(2, resolution * (n - 1))
    pts = []
    for k in range(samples + 1):
        t = t_lo + (t_hi - t_lo) * k / samples
        if k == samples:
            t = t_hi - 1e-9
        pts.append(bspline_point(ctrl, knots, order, t))
    return pts, False


def evaluate_bezier(bpoints, cyclic, resolution):
    n = len(bpoints)
    if n < 2:
        return [tuple(bp.co) for bp in bpoints], cyclic
    seg_count = n if cyclic else n - 1
    pts = []
    for s in range(seg_count):
        a = bpoints[s]
        b = bpoints[(s + 1) % n]
        p0, p1 = tuple(a.co), tuple(a.handle_right)
        p2, p3 = tuple(b.handle_left), tuple(b.co)
        for k in range(resolution):
            t = k / resolution
            it = 1.0 - t
            w0, w1, w2, w3 = it * it * it, 3.0 * it * it * t, 3.0 * it * t * t, t * t * t
            pts.append((
                w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
                w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
                w0 * p0[2] + w1 * p1[2] + w2 * p2[2] + w3 * p3[2],
            ))
    if not cyclic:
        pts.append(tuple(bpoints[-1].co))
    return pts, cyclic


def evaluate_spline(spline, curve_resolution):
    stype = spline.type
    cyclic = bool(getattr(spline, "use_cyclic_u", False))
    res = max(1, int(curve_resolution))
    if stype == "BEZIER":
        return evaluate_bezier(list(spline.bezier_points), cyclic, res)
    ctrl = [(float(p.co[0]), float(p.co[1]), float(p.co[2])) for p in spline.points]
    if stype == "NURBS":
        return evaluate_nurbs(ctrl, int(getattr(spline, "order_u", 4)), cyclic,
                              bool(getattr(spline, "use_endpoint_u", True)), res)
    return ctrl, cyclic


class ArcLengthPath:
    """Arc-length parameterised polyline used for FOLLOW_PATH offset_factor evaluation."""

    def __init__(self, pts, cyclic):
        self.pts = list(pts)
        if cyclic and len(self.pts) > 1:
            self.pts.append(self.pts[0])
        self.cum = [0.0]
        for i in range(1, len(self.pts)):
            self.cum.append(self.cum[-1] + v_len(v_sub(self.pts[i], self.pts[i - 1])))
        self.total = self.cum[-1] if self.cum[-1] > 1e-9 else 1.0
        self.cyclic = cyclic

    def sample(self, fac):
        if self.cyclic:
            fac = fac - math.floor(fac)
        fac = min(max(fac, 0.0), 1.0)
        target = fac * self.total
        i = max(0, min(bisect.bisect_right(self.cum, target) - 1, len(self.pts) - 2))
        seg = self.cum[i + 1] - self.cum[i]
        u = 0.0 if seg < 1e-12 else (target - self.cum[i]) / seg
        pos = v_lerp(self.pts[i], self.pts[i + 1], u)
        tangent = v_norm(v_sub(self.pts[i + 1], self.pts[i]), (0.0, 1.0, 0.0))
        return pos, tangent


# ==============================================================================
# 5. SCENE EVALUATOR (hierarchy, F-curves, constraints, armature pose)
# ==============================================================================

def vec_roll_to_mat3_rows(vec, roll):
    """Blender's vec_roll_to_mat3_normalized (bone rest orientation); returns the 3 columns."""
    nor = v_norm(vec, (0.0, 1.0, 0.0))
    x, y, z = nor
    theta = 1.0 + y
    theta_alt = x * x + z * z
    safe = 6.1e-3
    crit_sq = 2.5e-4 * 2.5e-4
    if theta > safe or theta_alt > crit_sq:
        if theta <= safe:
            theta = theta_alt * 0.5 + theta_alt * theta_alt * 0.125
        col0 = (1.0 - x * x / theta, -x, -x * z / theta)
        col1 = (x, y, z)
        col2 = (-x * z / theta, -z, 1.0 - z * z / theta)
    else:
        col0 = (-1.0, 0.0, 0.0)
        col1 = (0.0, -1.0, 0.0)
        col2 = (0.0, 0.0, 1.0)
    if abs(roll) > 1e-12:
        c, s = math.cos(roll), math.sin(roll)
        ax = nor
        t = 1.0 - c
        rr = (
            (t * ax[0] * ax[0] + c, t * ax[0] * ax[1] - s * ax[2], t * ax[0] * ax[2] + s * ax[1]),
            (t * ax[0] * ax[1] + s * ax[2], t * ax[1] * ax[1] + c, t * ax[1] * ax[2] - s * ax[0]),
            (t * ax[0] * ax[2] - s * ax[1], t * ax[1] * ax[2] + s * ax[0], t * ax[2] * ax[2] + c),
        )
        col0 = tuple(sum(rr[r][k] * col0[k] for k in range(3)) for r in range(3))
        col1 = tuple(sum(rr[r][k] * col1[k] for k in range(3)) for r in range(3))
        col2 = tuple(sum(rr[r][k] * col2[k] for k in range(3)) for r in range(3))
    return col0, col1, col2


class SceneEvaluator:
    def __init__(self, bpy):
        self.bpy = bpy
        self.objects = list(bpy.data.objects)
        self.index = {id(o): i for i, o in enumerate(self.objects)}
        self.channels = [build_channels(getattr(o, "animation_data", None)) for o in self.objects]
        self._paths = {}

    def obj_index(self, obj):
        return self.index.get(id(obj), -1)

    def has_transform_anim(self, i):
        return any(p in self.channels[i] for p in ("location", "rotation_euler", "scale"))

    def has_constraints(self, i):
        return any(getattr(c, "target", None) is not None for c in self.objects[i].constraints)

    def local_matrix(self, i, frame):
        o = self.objects[i]
        ch = self.channels[i]
        loc = ch["location"].evaluate(frame) if "location" in ch else tuple(o.location)
        rot = ch["rotation_euler"].evaluate(frame) if "rotation_euler" in ch else tuple(o.rotation_euler)
        scl = ch["scale"].evaluate(frame) if "scale" in ch else tuple(o.scale)
        return m_trs(loc, rot, scl)

    def path_for(self, curve_obj):
        key = id(curve_obj)
        if key not in self._paths:
            data = curve_obj.data
            spline = data.splines[0]
            pts, cyclic = evaluate_spline(spline, getattr(data, "resolution_u", 12))
            self._paths[key] = ArcLengthPath(pts, cyclic)
        return self._paths[key]

    def world_matrix(self, i, frame, memo):
        if i in memo:
            return memo[i]
        o = self.objects[i]
        m = self.local_matrix(i, frame)
        parent = getattr(o, "parent", None)
        if parent is not None:
            m = m_mul(self.world_matrix(self.obj_index(parent), frame, memo), m)
        for con in o.constraints:
            tgt = getattr(con, "target", None)
            if tgt is None:
                continue
            if con.type == "FOLLOW_PATH":
                m = self._follow_path(i, con, m, frame, memo)
            elif con.type in ("TRACK_TO", "DAMPED_TRACK"):
                m = self._track_to(con, m, frame, memo)
        memo[i] = m
        return m

    def _follow_path(self, i, con, m, frame, memo):
        path_obj = con.target
        path_world = self.world_matrix(self.obj_index(path_obj), frame, memo)
        path = self.path_for(path_obj)
        key = f'constraints["{con.name}"].offset_factor'
        ch = self.channels[i]
        fac = ch[key].evaluate(frame)[0] if key in ch else float(getattr(con, "offset_factor", 0.0))
        pos, tangent = path.sample(fac)
        if getattr(con, "use_curve_follow", False):
            fwd_axis = getattr(con, "forward_axis", "FORWARD_Y")
            up = (0.0, 0.0, 1.0)
            fwd = tangent
            side = v_norm(v_cross(fwd, up), (1.0, 0.0, 0.0))
            top = v_norm(v_cross(side, fwd), up)
            if fwd_axis == "FORWARD_X":
                ct = m_from_columns(fwd, v_scale(side, -1.0), top, pos)
            elif fwd_axis == "FORWARD_Z":
                ct = m_from_columns(side, top, fwd, pos)
            else:
                ct = m_from_columns(side, fwd, top, pos)
        else:
            ct = m_from_columns((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), pos)
        return m_mul(m_mul(path_world, ct), m)

    def _track_to(self, con, m, frame, memo):
        tgt_world = self.world_matrix(self.obj_index(con.target), frame, memo)
        pos = m_translation(m)
        tgt = m_translation(tgt_world)
        sx, sy, sz = v_len(m_column(m, 0)), v_len(m_column(m, 1)), v_len(m_column(m, 2))
        z_axis = v_norm(v_sub(pos, tgt), (0.0, 0.0, 1.0))
        up = (0.0, 0.0, 1.0)
        if abs(v_dot(z_axis, up)) > 0.999:
            up = (0.0, 1.0, 0.0)
        x_axis = v_norm(v_cross(up, z_axis), (1.0, 0.0, 0.0))
        y_axis = v_cross(z_axis, x_axis)
        return m_from_columns(v_scale(x_axis, sx), v_scale(y_axis, sy), v_scale(z_axis, sz), pos)

    # ---------------- Armature pose evaluation ----------------
    def armature_rest(self, arm_obj):
        rest = {}
        parents = {}
        order = []
        for eb in arm_obj.data.edit_bones:
            head = tuple(eb.head)
            tail = tuple(eb.tail)
            c0, c1, c2 = vec_roll_to_mat3_rows(v_sub(tail, head), float(getattr(eb, "roll", 0.0)))
            rest[eb.name] = m_from_columns(c0, c1, c2, head)
            parents[eb.name] = eb.parent.name if getattr(eb, "parent", None) is not None else None
            order.append(eb.name)
        return order, parents, rest

    def armature_skin(self, arm_idx, frame, order, parents, rest, rest_inv):
        arm_obj = self.objects[arm_idx]
        ch = self.channels[arm_idx]
        pose = {}
        skin = []
        for name in order:
            pb = arm_obj.pose.bones.get(name)
            key = f'pose.bones["{name}"].rotation_euler'
            if key in ch:
                rot = ch[key].evaluate(frame)
            elif pb is not None:
                rot = tuple(pb.rotation_euler)
            else:
                rot = (0.0, 0.0, 0.0)
            loc = tuple(pb.location) if pb is not None else (0.0, 0.0, 0.0)
            scl = tuple(pb.scale) if pb is not None else (1.0, 1.0, 1.0)
            local = m_trs(loc, rot, scl)
            par = parents[name]
            if par is not None and par in pose:
                pose[name] = m_mul(m_mul(m_mul(pose[par], rest_inv[par]), rest[name]), local)
            else:
                pose[name] = m_mul(rest[name], local)
            skin.append(m_mul(pose[name], rest_inv[name]))
        return skin


# ==============================================================================
# 6. MATERIAL GRAPH INTROSPECTION -> COMPACT PBR RECORDS
# ==============================================================================

TEXTURE_NODES = {
    "ShaderNodeTexNoise", "ShaderNodeTexVoronoi", "ShaderNodeTexWave", "ShaderNodeTexBrick",
    "ShaderNodeTexMusgrave", "ShaderNodeTexGradient", "ShaderNodeTexChecker", "ShaderNodeTexMagic",
}


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _to_rgba(v, fallback):
    if v is None:
        return tuple(fallback)
    if _is_number(v):
        f = float(v)
        return (f, f, f, 1.0)
    try:
        vals = [float(c) for c in v]
    except (TypeError, ValueError):
        return tuple(fallback)
    while len(vals) < 3:
        vals.append(vals[-1] if vals else 0.0)
    if len(vals) < 4:
        vals.append(1.0)
    return tuple(vals[:4])


def _to_scalar(v, fallback):
    if v is None:
        return float(fallback)
    if _is_number(v):
        return float(v)
    try:
        vals = [float(c) for c in v]
    except (TypeError, ValueError):
        return float(fallback)
    if len(vals) >= 3:
        return 0.2126 * vals[0] + 0.7152 * vals[1] + 0.0722 * vals[2]
    return vals[0] if vals else float(fallback)


class NodeGraph:
    def __init__(self, node_tree):
        self.nt = node_tree
        self.nodes = list(node_tree.nodes)
        self.out_owner = {}
        for n in self.nodes:
            for s in n.outputs:
                self.out_owner[id(s)] = n
        self.link_from = {}
        for out_sock, in_sock in node_tree.links:
            self.link_from[id(in_sock)] = out_sock

    def source_node(self, in_sock):
        src = self.link_from.get(id(in_sock))
        if src is None:
            return None, None
        return self.out_owner.get(id(src)), src

    def resolve(self, sock, fallback, depth=0):
        node, out_sock = self.source_node(sock)
        if node is not None and depth < 10:
            val = self.eval_output(node, out_sock, depth + 1)
            if val is not None:
                return val
        dv = getattr(sock, "default_value", None)
        if dv is None or not (_is_number(dv) or hasattr(dv, "__iter__")):
            return fallback
        return dv

    def eval_output(self, node, out_sock, depth):
        t = node.bl_idname
        oname = out_sock.name
        if t == "ShaderNodeGroup":
            dv = getattr(node.inputs[0], "default_value", None) if node.inputs else None
            if oname == "Base Color":
                for pair in (("Dark Color", "Light Color"), ("Color A", "Color B")):
                    if pair[0] in node.inputs and pair[1] in node.inputs:
                        a = _to_rgba(node.inputs[pair[0]].default_value, (0.5, 0.5, 0.5, 1.0))
                        b = _to_rgba(node.inputs[pair[1]].default_value, (0.5, 0.5, 0.5, 1.0))
                        return tuple((a[k] + b[k]) * 0.5 for k in range(4))
            if oname == "Emission Color" and "Core Color" in node.inputs and "Rim Color" in node.inputs:
                a = _to_rgba(node.inputs["Core Color"].default_value, (1.0, 1.0, 1.0, 1.0))
                b = _to_rgba(node.inputs["Rim Color"].default_value, (1.0, 1.0, 1.0, 1.0))
                return tuple(a[k] + (b[k] - a[k]) * 0.5 for k in range(4))
            if oname == "Emission Strength" and "Strength" in node.inputs and "Rim Strength" in node.inputs:
                s0 = _to_scalar(node.inputs["Strength"].default_value, 26.0)
                s1 = _to_scalar(node.inputs["Rim Strength"].default_value, 10.92)
                return s0 + (s1 - s0) * 0.3
            if oname == "Roughness" and "Rough Min" in node.inputs and "Rough Max" in node.inputs:
                r0 = _to_scalar(node.inputs["Rough Min"].default_value, 0.22)
                r1 = _to_scalar(node.inputs["Rough Max"].default_value, 0.64)
                return (r0 + r1) * 0.5
            if oname == "Metallic":
                return 0.875
            if oname == "Coat Weight":
                return 0.625
            return None
        if t in ("ShaderNodeValToRGB", "CompositorNodeValToRGB"):
            els = list(node.color_ramp.elements)
            if not els:
                return None
            if oname == "Alpha":
                return sum(float(e.color[3]) for e in els) / len(els)
            acc = [0.0, 0.0, 0.0, 0.0]
            for e in els:
                col = _to_rgba(e.color, (0.0, 0.0, 0.0, 1.0))
                for k in range(4):
                    acc[k] += col[k]
            return tuple(c / len(els) for c in acc)
        if t == "ShaderNodeMix":
            fac = _to_scalar(self.resolve(node.inputs[0], 0.5, depth), 0.5)
            fac = min(max(fac, 0.0), 1.0)
            dtype = getattr(node, "data_type", "FLOAT")
            if dtype == "RGBA":
                a = _to_rgba(self.resolve(node.inputs[6], (0.5, 0.5, 0.5, 1.0), depth), (0.5, 0.5, 0.5, 1.0))
                b = _to_rgba(self.resolve(node.inputs[7], (0.5, 0.5, 0.5, 1.0), depth), (0.5, 0.5, 0.5, 1.0))
                return tuple(a[k] + (b[k] - a[k]) * fac for k in range(4))
            if dtype == "VECTOR":
                a = _to_rgba(self.resolve(node.inputs[4], (0.0, 0.0, 0.0), depth), (0.0, 0.0, 0.0, 1.0))
                b = _to_rgba(self.resolve(node.inputs[5], (0.0, 0.0, 0.0), depth), (0.0, 0.0, 0.0, 1.0))
                return tuple(a[k] + (b[k] - a[k]) * fac for k in range(3))
            a = _to_scalar(self.resolve(node.inputs[2], 0.0, depth), 0.0)
            b = _to_scalar(self.resolve(node.inputs[3], 1.0, depth), 1.0)
            return a + (b - a) * fac
        if t == "ShaderNodeMath":
            a = _to_scalar(self.resolve(node.inputs[0], 0.5, depth), 0.5)
            b = _to_scalar(self.resolve(node.inputs[1], 0.5, depth), 0.5) if len(node.inputs) > 1 else 0.5
            op = getattr(node, "operation", "ADD")
            ops = {
                "ADD": a + b, "SUBTRACT": a - b, "MULTIPLY": a * b,
                "DIVIDE": a / b if abs(b) > 1e-9 else 0.0, "MAXIMUM": max(a, b), "MINIMUM": min(a, b),
                "POWER": (abs(a) ** b) if a != 0.0 else 0.0,
            }
            val = ops.get(op, a)
            if getattr(node, "use_clamp", False):
                val = min(max(val, 0.0), 1.0)
            return val
        if t in TEXTURE_NODES:
            if oname == "Color":
                return (0.5, 0.5, 0.5, 1.0)
            return 0.5
        if t in ("ShaderNodeFresnel", "ShaderNodeLayerWeight"):
            return 0.3
        if t == "ShaderNodeSepXYZ":
            return 0.5
        return None

    def first_node(self, bl_idname):
        for n in self.nodes:
            if n.bl_idname == bl_idname:
                return n
        return None

    def upstream_nodes(self, sock, limit=24):
        found = []
        queue = [sock]
        seen = set()
        while queue and len(found) < limit:
            s = queue.pop(0)
            node, _ = self.source_node(s)
            if node is None or id(node) in seen:
                continue
            seen.add(id(node))
            found.append(node)
            for inp in node.inputs:
                queue.append(inp)
        return found


def classify_material(name, graph, rec):
    lname = name.lower()
    tokens = lname.split("_")
    types = {n.bl_idname for n in graph.nodes}
    wave_types = {getattr(n, "wave_type", "") for n in graph.nodes if n.bl_idname == "ShaderNodeTexWave"}
    noise_types = {getattr(n, "noise_type", "") for n in graph.nodes if n.bl_idname == "ShaderNodeTexNoise"}
    if "ShaderNodeTexBrick" in types:
        return PROC_NIGHT_WINDOWS
    if "steam" in lname:
        return PROC_STEAM
    if "trail" in lname:
        return PROC_LIGHT_TRAIL
    if "visor" in lname or "led" in tokens:
        return PROC_LED_MATRIX
    if "hologram" in lname or "holo" in tokens:
        return PROC_HOLOGRAM
    if "neon" in tokens or (rec["emission_strength"] >= 4.0 and "ShaderNodeLayerWeight" in types):
        return PROC_NEON_TUBE
    if "ShaderNodeBsdfTranslucent" in types:
        return PROC_PAPER_LANTERN
    if rec["transmission"] >= 0.5 or "glass" in lname or "umbrella" in lname:
        return PROC_GLASS_RAIN
    if "ShaderNodeTexMusgrave" in types or "HYBRID_MULTIFRACTAL" in noise_types or "asphalt" in lname:
        return PROC_WET_ASPHALT
    if "RINGS" in wave_types or "wood" in lname or "timber" in lname:
        return PROC_WOOD
    if rec["metallic"] >= 0.5 or any(k in lname for k in ("metal", "steel", "stainless", "gold", "armor")):
        return PROC_BRUSHED_METAL
    if any(k in lname for k in ("noren", "fabric", "coat", "cable")):
        return PROC_FABRIC
    if any(k in lname for k in ("broth", "noodle", "egg", "yolk")):
        return PROC_ORGANIC
    if any(k in lname for k in ("ceramic", "tile", "lacquer", "bowl")):
        return PROC_CERAMIC
    return PROC_GENERIC


def extract_material_record(mat, index):
    nt = getattr(mat, "node_tree", None)
    rec = {
        "name": mat.name, "index": index,
        "base_color": list(PRINCIPLED_DEFAULTS["Base Color"][:3]),
        "metallic": 0.0, "roughness": 0.5, "alpha": 1.0, "transmission": 0.0,
        "coat": 0.0, "coat_roughness": 0.03, "ior": 1.5, "subsurface": 0.0, "sheen": 0.0,
        "emission_color": [1.0, 1.0, 1.0], "emission_strength": 0.0,
        "ramp_a": [0.5, 0.5, 0.5], "ramp_b": [0.5, 0.5, 0.5],
        "proc": PROC_GENERIC, "flags": 0, "aovs": [], "node_count": 0,
        "translucent_mix": 0.0,
    }
    if nt is None:
        return rec
    graph = NodeGraph(nt)
    rec["node_count"] = len(graph.nodes)
    bsdf = graph.first_node("ShaderNodeBsdfPrincipled")
    if bsdf is not None:
        def inp(name):
            return bsdf.inputs[name] if name in bsdf.inputs else None

        def val_rgba(name):
            s = inp(name)
            fb = PRINCIPLED_DEFAULTS[name]
            return _to_rgba(graph.resolve(s, fb) if s is not None else fb, fb)

        def val_f(name):
            s = inp(name)
            fb = PRINCIPLED_DEFAULTS[name]
            return _to_scalar(graph.resolve(s, fb) if s is not None else fb, fb)

        base = val_rgba("Base Color")
        rec["base_color"] = [round(c, 6) for c in base[:3]]
        rec["metallic"] = min(max(val_f("Metallic"), 0.0), 1.0)
        rec["roughness"] = min(max(val_f("Roughness"), 0.02), 1.0)
        rec["alpha"] = min(max(val_f("Alpha"), 0.0), 1.0)
        rec["transmission"] = min(max(val_f("Transmission Weight"), 0.0), 1.0)
        rec["coat"] = min(max(val_f("Coat Weight"), 0.0), 1.0)
        rec["coat_roughness"] = min(max(val_f("Coat Roughness"), 0.0), 1.0)
        rec["ior"] = val_f("IOR")
        rec["subsurface"] = val_f("Subsurface Weight")
        rec["sheen"] = val_f("Sheen Weight")
        em = val_rgba("Emission Color")
        rec["emission_color"] = [round(c, 6) for c in em[:3]]
        rec["emission_strength"] = max(0.0, val_f("Emission Strength"))

        # Procedural ramp colour pair: Mix(RGBA) A/B or first upstream ColorRamp ends
        ramp_pair = None
        for sock_name in ("Emission Color", "Base Color"):
            s = inp(sock_name)
            if s is None:
                continue
            for node in graph.upstream_nodes(s):
                if node.bl_idname == "ShaderNodeGroup":
                    for pair in (("Core Color", "Rim Color"), ("Dark Color", "Light Color"), ("Color A", "Color B")):
                        if pair[0] in node.inputs and pair[1] in node.inputs:
                            a = _to_rgba(node.inputs[pair[0]].default_value, (0.5, 0.5, 0.5, 1.0))
                            b = _to_rgba(node.inputs[pair[1]].default_value, (0.5, 0.5, 0.5, 1.0))
                            ramp_pair = (a, b)
                            break
                    if ramp_pair is not None:
                        break
                if node.bl_idname == "ShaderNodeMix" and getattr(node, "data_type", "") == "RGBA":
                    a = _to_rgba(graph.resolve(node.inputs[6], (0.5, 0.5, 0.5, 1.0)), (0.5, 0.5, 0.5, 1.0))
                    b = _to_rgba(graph.resolve(node.inputs[7], (0.5, 0.5, 0.5, 1.0)), (0.5, 0.5, 0.5, 1.0))
                    ramp_pair = (a, b)
                    break
                if node.bl_idname == "ShaderNodeValToRGB":
                    els = list(node.color_ramp.elements)
                    ramp_pair = (_to_rgba(els[0].color, (0, 0, 0, 1)), _to_rgba(els[-1].color, (1, 1, 1, 1)))
                    break
            if ramp_pair is not None:
                break
        if ramp_pair is None:
            ramp_pair = (base, base)
        rec["ramp_a"] = [round(c, 6) for c in ramp_pair[0][:3]]
        rec["ramp_b"] = [round(c, 6) for c in ramp_pair[1][:3]]

    translucent = graph.first_node("ShaderNodeBsdfTranslucent")
    mix_shader = graph.first_node("ShaderNodeMixShader")
    if translucent is not None and mix_shader is not None:
        rec["translucent_mix"] = _to_scalar(mix_shader.inputs[0].default_value, 0.5)
        tcol = _to_rgba(translucent.inputs["Color"].default_value, (1.0, 1.0, 1.0, 1.0))
        rec["translucent_color"] = [round(c, 6) for c in tcol[:3]]

    for n in graph.nodes:
        if n.bl_idname == "ShaderNodeOutputAOV":
            aov = getattr(n, "aov_name", "")
            rec["aovs"].append(aov)
            if aov == "AOV_Neon_Mask":
                rec["flags"] |= MAT_FLAG_NEON_AOV
            elif aov == "AOV_Thermal_Heat":
                rec["flags"] |= MAT_FLAG_THERMAL

    rec["proc"] = classify_material(mat.name, graph, rec)
    if rec["proc"] in (PROC_HOLOGRAM, PROC_GLASS_RAIN, PROC_STEAM, PROC_LIGHT_TRAIL) or rec["alpha"] < 0.98:
        rec["flags"] |= MAT_FLAG_TRANSPARENT
    if rec["proc"] == PROC_WET_ASPHALT:
        rec["flags"] |= MAT_FLAG_MIRROR
    rec["proc_name"] = PROC_NAMES[rec["proc"]]
    return rec


def material_texels(rec, scroll=(0.0, 0.0, 0.0)):
    """6 RGBA32F texels per material consumed by texelFetch in the fragment shader."""
    return [
        rec["base_color"][0], rec["base_color"][1], rec["base_color"][2], rec["alpha"],
        rec["emission_color"][0], rec["emission_color"][1], rec["emission_color"][2], rec["emission_strength"],
        rec["metallic"], rec["roughness"], rec["transmission"], rec["coat"],
        rec["ramp_a"][0], rec["ramp_a"][1], rec["ramp_a"][2], float(rec["proc"]),
        rec["ramp_b"][0], rec["ramp_b"][1], rec["ramp_b"][2], rec["coat_roughness"],
        scroll[0], scroll[1], scroll[2], float(rec["flags"]),
    ]


# ==============================================================================
# 7. MESH MODIFIER STACK (SOLIDIFY / ARRAY / LATTICE) + NORMALS + TRIANGULATION
# ==============================================================================

def polygon_normal(verts, idx):
    """Newell normal (unnormalised, magnitude = 2 * area)."""
    nx = ny = nz = 0.0
    count = len(idx)
    for k in range(count):
        a = verts[idx[k]]
        b = verts[idx[(k + 1) % count]]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return (nx, ny, nz)


def vertex_normals(verts, polys):
    acc = [[0.0, 0.0, 0.0] for _ in verts]
    for idx, _mat, _smooth in polys:
        n = polygon_normal(verts, idx)
        for vi in idx:
            a = acc[vi]
            a[0] += n[0]
            a[1] += n[1]
            a[2] += n[2]
    return [v_norm(tuple(a)) for a in acc]


def apply_solidify(verts, polys, vattr, thickness, offset):
    vn = vertex_normals(verts, polys)
    n = len(verts)
    outer_f = thickness * (offset + 1.0) * 0.5
    inner_f = thickness * (offset - 1.0) * 0.5
    new_verts = [v_add(v, v_scale(vn[i], outer_f)) for i, v in enumerate(verts)]
    new_verts += [v_add(v, v_scale(vn[i], inner_f)) for i, v in enumerate(verts)]
    new_attr = list(vattr) + list(vattr)
    new_polys = [(list(idx), mat, smooth) for idx, mat, smooth in polys]
    new_polys += [([i + n for i in reversed(idx)], mat, smooth) for idx, mat, smooth in polys]
    edge_use = {}
    for idx, mat, smooth in polys:
        count = len(idx)
        for k in range(count):
            a, b = idx[k], idx[(k + 1) % count]
            key = (a, b) if a < b else (b, a)
            entry = edge_use.get(key)
            if entry is None:
                edge_use[key] = [1, a, b, mat]
            else:
                entry[0] += 1
    for count, a, b, mat in edge_use.values():
        if count == 1:
            new_polys.append(([b, a, a + n, b + n], mat, False))
    return new_verts, new_polys, new_attr


def apply_array(verts, polys, vattr, mod):
    count = max(1, int(getattr(mod, "count", 2)))
    offset = [0.0, 0.0, 0.0]
    if getattr(mod, "use_constant_offset", False):
        c = tuple(getattr(mod, "constant_offset_displace", (0.0, 0.0, 0.0)))
        offset = [offset[k] + float(c[k]) for k in range(3)]
    if getattr(mod, "use_relative_offset", True):
        rel = tuple(getattr(mod, "relative_offset_displace", (1.0, 0.0, 0.0)))
        if verts:
            lo = [min(v[k] for v in verts) for k in range(3)]
            hi = [max(v[k] for v in verts) for k in range(3)]
            offset = [offset[k] + float(rel[k]) * (hi[k] - lo[k]) for k in range(3)]
    n = len(verts)
    out_v, out_p, out_a = [], [], []
    for c in range(count):
        d = (offset[0] * c, offset[1] * c, offset[2] * c)
        out_v.extend(v_add(v, d) for v in verts)
        out_a.extend(vattr)
        out_p.extend(([i + c * n for i in idx], mat, smooth) for idx, mat, smooth in polys)
    return out_v, out_p, out_a


def _bspline_weights(t):
    t2 = t * t
    t3 = t2 * t
    return (
        -0.1666666 * t3 + 0.5 * t2 - 0.5 * t + 0.16666666,
        0.5 * t3 - t2 + 0.66666666,
        -0.5 * t3 + 0.5 * t2 + 0.5 * t + 0.16666666,
        0.16666666 * t3,
    )


def _lattice_axis(coord, count):
    if count <= 1:
        return [(0, 1.0)]
    du = 1.0 / (count - 1)
    u = (coord + 0.5) / du
    ui = int(math.floor(u))
    w = _bspline_weights(u - ui)
    out = []
    for k, uu in enumerate(range(ui - 1, ui + 3)):
        if w[k] == 0.0:
            continue
        out.append((min(max(uu, 0), count - 1), w[k]))
    return out


def apply_lattice(verts, obj_world, lat_obj, lat_world):
    """Blender calc_latt_deform equivalent: KEY_BSPLINE weights over (co_deform - co) offsets."""
    ld = lat_obj.data
    pu, pv, pw = int(ld.points_u), int(ld.points_v), int(ld.points_w)
    offsets = [v_sub(tuple(p.co_deform), tuple(p.co)) for p in ld.points]
    to_lat = m_mul(m_inverse(lat_world), obj_world)
    from_lat = m_inverse(to_lat)
    out = []
    for v in verts:
        p = m_point(to_lat, v)
        wu = _lattice_axis(p[0], pu)
        wv = _lattice_axis(p[1], pv)
        ww = _lattice_axis(p[2], pw)
        dx = dy = dz = 0.0
        for iw, fw in ww:
            base_w = iw * pu * pv
            for iv, fv in wv:
                base_v = base_w + iv * pu
                fwv = fw * fv
                for iu, fu in wu:
                    o = offsets[base_v + iu]
                    f = fwv * fu
                    dx += o[0] * f
                    dy += o[1] * f
                    dz += o[2] * f
        out.append(m_point(from_lat, (p[0] + dx, p[1] + dy, p[2] + dz)))
    return out


def pos_key_equal(a, b):
    return abs(a[0] - b[0]) < 1e-7 and abs(a[1] - b[1]) < 1e-7 and abs(a[2] - b[2]) < 1e-7


def triangulate_with_normals(verts, polys, vattr, smooth_angle_deg=AUTO_SMOOTH_ANGLE_DEG):
    """Fan-triangulate polygons and build angle-limited smooth (or flat) corner normals.

    Returns (positions, normals, corner_attr, local_indices, corner_material_slot).
    """
    cos_limit = math.cos(math.radians(smooth_angle_deg))
    raw_n = [polygon_normal(verts, idx) for idx, _m, _s in polys]
    unit_n = [v_norm(n, (0.0, 0.0, 0.0)) for n in raw_n]
    pos_key = [(round(v[0], 5), round(v[1], 5), round(v[2], 5)) for v in verts]
    adjacency = {}
    for pi, (idx, _mat, smooth) in enumerate(polys):
        if not smooth:
            continue
        for vi in idx:
            adjacency.setdefault(pos_key[vi], []).append(pi)

    out_pos, out_nrm, out_attr, out_mat, out_idx = [], [], [], [], []
    vmap = {}
    for pi, (idx, mat, smooth) in enumerate(polys):
        fn = unit_n[pi]
        if fn == (0.0, 0.0, 0.0):
            continue
        corner = []
        for vi in idx:
            if smooth:
                ax = ay = az = 0.0
                for pj in adjacency.get(pos_key[vi], ()):
                    un = unit_n[pj]
                    if un[0] * fn[0] + un[1] * fn[1] + un[2] * fn[2] >= cos_limit:
                        rn = raw_n[pj]
                        ax += rn[0]
                        ay += rn[1]
                        az += rn[2]
                nrm = v_norm((ax, ay, az), fn)
            else:
                nrm = fn
            attr = vattr[vi]
            key = (vi, round(nrm[0], 3), round(nrm[1], 3), round(nrm[2], 3), mat, attr)
            slot = vmap.get(key)
            if slot is None:
                slot = len(out_pos)
                vmap[key] = slot
                out_pos.append(verts[vi])
                out_nrm.append(nrm)
                out_attr.append(attr)
                out_mat.append(mat)
            corner.append(slot)
        for k in range(1, len(corner) - 1):
            a, b, c = corner[0], corner[k], corner[k + 1]
            if a == b or b == c or a == c:
                continue
            pa, pb, pc = out_pos[a], out_pos[b], out_pos[c]
            if pos_key_equal(pa, pb) or pos_key_equal(pb, pc) or pos_key_equal(pa, pc):
                continue
            out_idx.extend((a, b, c))
    return out_pos, out_nrm, out_attr, out_idx, out_mat


# ==============================================================================
# 8. CURVE TUBES + VECTOR STROKE FONT
# ==============================================================================

def sweep_tube(pts, cyclic, radius, sides):
    """Parallel-transport circular sweep. Returns (positions, normals, local_indices)."""
    clean = [pts[0]]
    for p in pts[1:]:
        if v_len(v_sub(p, clean[-1])) > 1e-7:
            clean.append(p)
    if cyclic and len(clean) > 2 and v_len(v_sub(clean[0], clean[-1])) < 1e-7:
        clean.pop()
    n = len(clean)
    if n < 2:
        return [], [], []
    tangents = []
    for i in range(n):
        if cyclic:
            prev_p, next_p = clean[(i - 1) % n], clean[(i + 1) % n]
        else:
            prev_p = clean[max(i - 1, 0)]
            next_p = clean[min(i + 1, n - 1)]
        tangents.append(v_norm(v_sub(next_p, prev_p), (0.0, 0.0, 1.0)))
    t0 = tangents[0]
    ref = (0.0, 0.0, 1.0) if abs(t0[2]) < 0.9 else (1.0, 0.0, 0.0)
    normal = v_norm(v_cross(t0, ref), (1.0, 0.0, 0.0))
    positions, normals, indices = [], [], []
    for i in range(n):
        t = tangents[i]
        normal = v_norm(v_sub(normal, v_scale(t, v_dot(normal, t))), normal)
        binormal = v_cross(t, normal)
        miter = 1.0
        if 0 < i < n - 1 or cyclic:
            seg_in = v_norm(v_sub(clean[i], clean[(i - 1) % n]), t)
            c = v_dot(seg_in, t)
            miter = 1.0 / max(c, 0.55)
        for s in range(sides):
            ang = 2.0 * math.pi * s / sides
            ca, sa = math.cos(ang), math.sin(ang)
            d = (normal[0] * ca + binormal[0] * sa, normal[1] * ca + binormal[1] * sa, normal[2] * ca + binormal[2] * sa)
            positions.append(v_add(clean[i], v_scale(d, radius * miter)))
            normals.append(d)
    rings = n if cyclic else n - 1
    for i in range(rings):
        j = (i + 1) % n
        for s in range(sides):
            s2 = (s + 1) % sides
            a, b = i * sides + s, i * sides + s2
            c, d = j * sides + s2, j * sides + s
            indices.extend((a, b, c, a, c, d))
    return positions, normals, indices


# 4 x 6 unit grid vector stroke font (polylines separated by '|')
GLYPHS = {
    "A": "0,0 0,4 2,6 4,4 4,0|0,3 4,3",
    "B": "0,0 0,6 3,6 4,5 4,4 3,3 0,3|3,3 4,2 4,1 3,0 0,0",
    "C": "4,6 1,6 0,5 0,1 1,0 4,0",
    "D": "0,0 0,6 3,6 4,5 4,1 3,0 0,0",
    "E": "4,6 0,6 0,0 4,0|0,3 3,3",
    "F": "4,6 0,6 0,0|0,3 3,3",
    "G": "4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1 4,3 2,3",
    "H": "0,0 0,6|4,0 4,6|0,3 4,3",
    "I": "1,6 3,6|2,6 2,0|1,0 3,0",
    "J": "4,6 4,1 3,0 1,0 0,1",
    "K": "0,0 0,6|4,6 0,2|1,3 4,0",
    "L": "0,6 0,0 4,0",
    "M": "0,0 0,6 2,3 4,6 4,0",
    "N": "0,0 0,6 4,0 4,6",
    "O": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0",
    "P": "0,0 0,6 3,6 4,5 4,4 3,3 0,3",
    "Q": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0|2,2 4,0",
    "R": "0,0 0,6 3,6 4,5 4,4 3,3 0,3|2,3 4,0",
    "S": "4,5 3,6 1,6 0,5 0,4 1,3 3,3 4,2 4,1 3,0 1,0 0,1",
    "T": "0,6 4,6|2,6 2,0",
    "U": "0,6 0,1 1,0 3,0 4,1 4,6",
    "V": "0,6 2,0 4,6",
    "W": "0,6 1,0 2,3 3,0 4,6",
    "X": "0,6 4,0|0,0 4,6",
    "Y": "0,6 2,3 4,6|2,3 2,0",
    "Z": "0,6 4,6 0,0 4,0",
    "0": "1,0 0,1 0,5 1,6 3,6 4,5 4,1 3,0 1,0|0,1 4,5",
    "1": "1,5 2,6 2,0|1,0 3,0",
    "2": "0,5 1,6 3,6 4,5 4,4 0,0 4,0",
    "3": "0,5 1,6 3,6 4,5 4,4 3,3 1,3|3,3 4,2 4,1 3,0 1,0 0,1",
    "4": "3,0 3,6 0,2 4,2",
    "5": "4,6 0,6 0,3 3,3 4,2 4,1 3,0 0,0",
    "6": "4,5 3,6 1,6 0,5 0,1 1,0 3,0 4,1 4,2 3,3 0,3",
    "7": "0,6 4,6 1,0",
    "8": "1,3 0,4 0,5 1,6 3,6 4,5 4,4 3,3 1,3 0,2 0,1 1,0 3,0 4,1 4,2 3,3",
    "9": "4,3 1,3 0,4 0,5 1,6 3,6 4,5 4,1 3,0 1,0 0,1",
    "/": "0,0 4,6",
    "-": "1,3 3,3",
    ".": "1.6,0 2.4,0",
    ":": "2,1 2,1.6|2,4 2,4.6",
    "!": "2,6 2,2|2,0.6 2,0",
    "?": "0,5 1,6 3,6 4,5 4,4 2,3 2,2|2,0.6 2,0",
    "'": "2,6 2,4.5",
    "+": "0,3 4,3|2,1 2,5",
}
GLYPH_ADVANCE = 5.6
GLYPH_UNIT_X = 0.115
GLYPH_UNIT_Y = 0.118


def parse_glyph(ch):
    spec = GLYPHS.get(ch.upper())
    if spec is None:
        return [[(0.0, 0.0), (4.0, 0.0), (4.0, 6.0), (0.0, 6.0), (0.0, 0.0)]]
    strokes = []
    for part in spec.split("|"):
        pts = []
        for tok in part.split():
            x_s, y_s = tok.split(",")
            pts.append((float(x_s), float(y_s)))
        strokes.append(pts)
    return strokes


def stroke_prism(p0, p1, half_w, half_d, positions, normals, indices):
    """Extruded rectangular neon stroke segment in the text XY plane (Z = extrusion)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    if length < 1e-9:
        dx, dy, length = 1.0, 0.0, 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    a = (p0[0] - ux * half_w, p0[1] - uy * half_w)
    b = (p1[0] + ux * half_w, p1[1] + uy * half_w)
    corners2d = [
        (a[0] - px * half_w, a[1] - py * half_w), (b[0] - px * half_w, b[1] - py * half_w),
        (b[0] + px * half_w, b[1] + py * half_w), (a[0] + px * half_w, a[1] + py * half_w),
    ]
    lo = [(c[0], c[1], -half_d) for c in corners2d]
    hi = [(c[0], c[1], half_d) for c in corners2d]
    faces = [
        ((hi[0], hi[1], hi[2], hi[3]), (0.0, 0.0, 1.0)),
        ((lo[3], lo[2], lo[1], lo[0]), (0.0, 0.0, -1.0)),
        ((lo[0], lo[1], hi[1], hi[0]), (-px, -py, 0.0)),
        ((lo[1], lo[2], hi[2], hi[1]), (ux, uy, 0.0)),
        ((lo[2], lo[3], hi[3], hi[2]), (px, py, 0.0)),
        ((lo[3], lo[0], hi[0], hi[3]), (-ux, -uy, 0.0)),
    ]
    for quad, nrm in faces:
        base = len(positions)
        positions.extend(quad)
        normals.extend([nrm] * 4)
        indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))


def font_geometry(curve_data):
    body = str(getattr(curve_data, "body", ""))
    size = float(getattr(curve_data, "size", 1.0))
    extrude = float(getattr(curve_data, "extrude", 0.0))
    bevel = float(getattr(curve_data, "bevel_depth", 0.0))
    spacing = float(getattr(curve_data, "space_character", 1.0))
    ux, uy = GLYPH_UNIT_X * size, GLYPH_UNIT_Y * size
    advance = GLYPH_ADVANCE * ux * spacing
    total_w = max(0.0, len(body) * advance - (GLYPH_ADVANCE - 4.0) * ux)
    align_x = getattr(curve_data, "align_x", "LEFT")
    align_y = getattr(curve_data, "align_y", "BOTTOM")
    x0 = -total_w * 0.5 if align_x == "CENTER" else (-total_w if align_x == "RIGHT" else 0.0)
    cap = 6.0 * uy
    y0 = -cap * 0.5 if align_y == "CENTER" else (-cap if align_y == "TOP" else 0.0)
    half_w = max(0.055 * size, 0.0012)
    half_d = max(extrude + bevel, 0.0015)
    positions, normals, indices = [], [], []
    for ci, ch in enumerate(body):
        if ch == " ":
            continue
        ox = x0 + ci * advance
        for stroke in parse_glyph(ch):
            pts = [(ox + gx * ux, y0 + gy * uy) for gx, gy in stroke]
            for k in range(len(pts) - 1):
                stroke_prism(pts[k], pts[k + 1], half_w, half_d, positions, normals, indices)
    return positions, normals, indices


# ==============================================================================
# 9. SCENE PACKER
# ==============================================================================

def object_flags_for(obj, mat_records, mat_ids):
    flags = 0
    name = obj.name
    if any(k in name for k in ("Thruster", "Plasma", "IonCore", "Repulsor")):
        flags |= OBJ_FLAG_THRUSTER
    if any(mat_records[m]["proc"] in (PROC_NEON_TUBE, PROC_HOLOGRAM, PROC_LIGHT_TRAIL) for m in mat_ids if m >= 0):
        flags |= OBJ_FLAG_NEON_SIGN
    return flags


class ScenePacker:
    def __init__(self, bpy, log=print):
        self.bpy = bpy
        self.log = log
        self.ev = SceneEvaluator(bpy)
        self.objects = self.ev.objects
        self.positions = array.array("f")
        self.normals = array.array("f")
        self.attribs = bytearray()
        self.indices = array.array("I")
        self.lines = array.array("f")
        self.stats = {"MESH": 0, "CURVE": 0, "FONT": 0, "triangles": 0, "path_curves": 0,
                      "modifiers_applied": {"ARRAY": 0, "SOLIDIFY": 0, "LATTICE": 0},
                      "modifiers_shader": {"WAVE": 0, "ARMATURE": 0, "NODES": 0, "BEVEL": 0}}

    # ---------------- materials ----------------
    def pack_materials(self):
        mats = list(self.bpy.data.materials)
        self.mat_index = {m.name: i for i, m in enumerate(mats)}
        self.mat_records = [extract_material_record(m, i) for i, m in enumerate(mats)]
        self.mat_objects = mats
        return self.mat_records

    def material_id(self, mat):
        if mat is None:
            return 0
        return self.mat_index.get(getattr(mat, "name", ""), 0)

    # ---------------- geometry emission ----------------
    def emit(self, positions, normals, mats, bones, flags, local_indices):
        base = len(self.positions) // 3
        first = len(self.indices)
        for p in positions:
            self.positions.extend((float(p[0]), float(p[1]), float(p[2])))
        for n in normals:
            self.normals.extend((float(n[0]), float(n[1]), float(n[2])))
        for k in range(len(positions)):
            self.attribs.extend((mats[k] & 255, bones[k] & 255, flags & 255, 0))
        self.indices.extend(base + i for i in local_indices)
        return base, len(positions), first, len(local_indices)

    def mesh_geometry(self, obj, extras):
        mesh = obj.data
        verts = [(float(v.co[0]), float(v.co[1]), float(v.co[2])) for v in mesh.vertices]
        polys = [(list(p.vertices), int(p.material_index), bool(p.use_smooth)) for p in mesh.polygons]
        vattr = [BONE_NONE] * len(verts)
        for mod in obj.modifiers:
            if mod.type == "ARMATURE" and getattr(mod, "object", None) is not None:
                arm_obj = mod.object
                bone_names = [eb.name for eb in arm_obj.data.edit_bones]
                for vg in obj.vertex_groups:
                    if vg.name in bone_names:
                        bidx = bone_names.index(vg.name)
                        for vi, w in vg.weights.items():
                            if 0 <= vi < len(vattr) and w > 0.0:
                                vattr[vi] = bidx
                extras["armature"] = self.ev.obj_index(arm_obj)
                self.stats["modifiers_shader"]["ARMATURE"] += 1
        for mod in obj.modifiers:
            mtype = mod.type
            if mtype == "SOLIDIFY":
                verts, polys, vattr = apply_solidify(
                    verts, polys, vattr, float(getattr(mod, "thickness", 0.01)), float(getattr(mod, "offset", -1.0)))
                self.stats["modifiers_applied"]["SOLIDIFY"] += 1
            elif mtype == "ARRAY":
                verts, polys, vattr = apply_array(verts, polys, vattr, mod)
                self.stats["modifiers_applied"]["ARRAY"] += 1
            elif mtype == "LATTICE" and getattr(mod, "object", None) is not None:
                oi = self.ev.obj_index(obj)
                li = self.ev.obj_index(mod.object)
                memo = {}
                ow = self.ev.world_matrix(oi, FRAME_START, memo)
                lw = self.ev.world_matrix(li, FRAME_START, memo)
                verts = apply_lattice(verts, ow, mod.object, lw)
                self.stats["modifiers_applied"]["LATTICE"] += 1
            elif mtype == "WAVE":
                extras["wave"] = [
                    float(getattr(mod, "height", 0.5)), float(getattr(mod, "width", 1.5)),
                    float(getattr(mod, "speed", 0.25)), float(getattr(mod, "narrowness", 1.5)),
                    1.0 if getattr(mod, "use_normal", False) else 0.0,
                ]
                self.stats["modifiers_shader"]["WAVE"] += 1
            elif mtype == "NODES":
                extras["nodes"] = getattr(getattr(mod, "node_group", None), "name", "")
                self.stats["modifiers_shader"]["NODES"] += 1
            elif mtype == "BEVEL":
                extras["bevel"] = float(getattr(mod, "width", 0.0))
                self.stats["modifiers_shader"]["BEVEL"] += 1
        pos, nrm, attr, idx, mat_slots = triangulate_with_normals(verts, polys, vattr)
        slot_ids = [self.material_id(m) for m in mesh.materials] or [0]
        mats = [slot_ids[min(max(s, 0), len(slot_ids) - 1)] for s in mat_slots]
        return pos, nrm, mats, attr, idx, sorted(set(mats))

    def curve_geometry(self, obj, extras):
        data = obj.data
        mat_id = self.material_id(data.materials[0]) if data.materials else 0
        if obj.type == "FONT":
            pos, nrm, idx = font_geometry(data)
            extras["text"] = data.body
            return pos, nrm, [mat_id] * len(pos), [BONE_NONE] * len(pos), idx, [mat_id], None
        bevel = float(getattr(data, "bevel_depth", 0.0))
        res = int(getattr(data, "resolution_u", 12))
        sides = 4 + 2 * int(getattr(data, "bevel_resolution", 4))
        all_pos, all_nrm, all_idx = [], [], []
        path_polylines = []
        for spline in data.splines:
            pts, cyclic = evaluate_spline(spline, res)
            if bevel > 0.0:
                p, n, ix = sweep_tube(pts, cyclic, bevel, sides)
                base = len(all_pos)
                all_pos.extend(p)
                all_nrm.extend(n)
                all_idx.extend(base + i for i in ix)
            else:
                poly = list(pts) + ([pts[0]] if cyclic and pts else [])
                path_polylines.append(poly)
        return all_pos, all_nrm, [mat_id] * len(all_pos), [BONE_NONE] * len(all_pos), all_idx, [mat_id], path_polylines

    # ---------------- collection lookup ----------------
    def collection_map(self):
        cmap = {}
        for col in self.bpy.data.collections:
            for o in col.objects:
                cmap.setdefault(id(o), col.name)
        return cmap

    def pack(self):
        t0 = time.perf_counter()
        ev = self.ev
        self.pack_materials()
        cmap = self.collection_map()
        records = []
        depth_cache = {}

        def depth_of(o):
            k = id(o)
            if k not in depth_cache:
                depth_cache[k] = 0 if getattr(o, "parent", None) is None else depth_of(o.parent) + 1
            return depth_cache[k]

        node_locals = array.array("f")
        for i, o in enumerate(self.objects):
            node_locals.extend(m_row34(ev.local_matrix(i, FRAME_START)))
            rec = {
                "name": o.name, "type": o.type, "parent": ev.obj_index(o.parent) if o.parent is not None else -1,
                "col": cmap.get(id(o), ""), "track": -1, "ws": 0,
                "hidden": bool(getattr(o, "hide_render", False)), "role": "node",
                "first": 0, "count": 0, "vfirst": 0, "vcount": 0, "mats": [], "flags": 0,
                "mods": [m.type for m in getattr(o, "modifiers", [])],
                "cons": [c.type for c in getattr(o, "constraints", [])],
                "depth": depth_of(o), "transparent": False,
            }
            extras = {}
            geo = None
            if o.type == "MESH":
                self.stats["MESH"] += 1
                pos, nrm, mats, bones, idx, used = self.mesh_geometry(o, extras)
                geo = (pos, nrm, mats, bones, idx, used)
                rec["role"] = "surface"
            elif o.type in ("CURVE", "FONT"):
                self.stats[o.type] += 1
                pos, nrm, mats, bones, idx, used, polylines = self.curve_geometry(o, extras)
                geo = (pos, nrm, mats, bones, idx, used)
                rec["role"] = "text" if o.type == "FONT" else "tube"
                if polylines:
                    rec["role"] = "path"
                    self.stats["path_curves"] += 1
                    strips = []
                    for poly in polylines:
                        strips.append([len(self.lines) // 3, len(poly)])
                        for p in poly:
                            self.lines.extend((float(p[0]), float(p[1]), float(p[2])))
                    rec["lines"] = strips
            if geo is not None:
                pos, nrm, mats, bones, idx, used = geo
                rec["mats"] = used
                flags = object_flags_for(o, self.mat_records, used)
                if "armature" in extras:
                    flags |= OBJ_FLAG_SKINNED
                    rec["arm"] = extras["armature"]
                if "wave" in extras:
                    flags |= OBJ_FLAG_WAVE
                    rec["wave"] = extras["wave"]
                if "nodes" in extras:
                    rec["role"] = "rain_emitter"
                    rec["gn"] = extras["nodes"]
                if "text" in extras:
                    rec["text"] = extras["text"]
                rec["flags"] = flags
                if pos:
                    vbase, vcount, ifirst, icount = self.emit(pos, nrm, mats, bones, flags, idx)
                    rec.update({"vfirst": vbase, "vcount": vcount, "first": ifirst, "count": icount})
                    self.stats["triangles"] += icount // 3
                    lo = [min(p[k] for p in pos) for k in range(3)]
                    hi = [max(p[k] for p in pos) for k in range(3)]
                    center = [(lo[k] + hi[k]) * 0.5 for k in range(3)]
                    radius = max(v_len(v_sub(p, center)) for p in pos)
                    rec["bounds"] = [round(c, 5) for c in center] + [round(radius, 5)]
                rec["transparent"] = any(self.mat_records[m]["flags"] & MAT_FLAG_TRANSPARENT for m in used)
            records.append(rec)
        order = sorted(range(len(self.objects)), key=lambda k: (records[k]["depth"], k))
        t_geo = time.perf_counter() - t0
        self.log(f"[studio] tessellation: {len(self.positions) // 3:,} verts, "
                 f"{len(self.indices) // 3:,} tris in {t_geo:.2f}s")

        rain = self.pack_geometry_nodes_rain(records)

        # ---------------- object animation tracks ----------------
        tracks = array.array("f")
        track_meta = []
        for i, o in enumerate(self.objects):
            constrained = ev.has_constraints(i)
            animated = ev.has_transform_anim(i)
            if not constrained and not animated:
                continue
            records[i]["track"] = len(track_meta)
            records[i]["ws"] = 1 if constrained else 0
            track_meta.append({"node": i, "ws": records[i]["ws"]})
            for f in range(FRAME_START, FRAME_END + 1):
                m = ev.world_matrix(i, f, {}) if constrained else ev.local_matrix(i, f)
                tracks.extend(m_row34(m))

        # ---------------- armature bone tracks ----------------
        bone_tracks = array.array("f")
        armatures = []
        for i, o in enumerate(self.objects):
            if o.type != "ARMATURE":
                continue
            order_b, parents, rest = ev.armature_rest(o)
            rest_inv = {k: m_inverse(v) for k, v in rest.items()}
            arm = {
                "node": i, "name": o.name, "bones": order_b,
                "parents": [order_b.index(parents[b]) if parents[b] is not None else -1 for b in order_b],
                "offset": len(bone_tracks), "heads": [], "tails": [],
            }
            for eb in o.data.edit_bones:
                arm["heads"].append([round(c, 5) for c in eb.head])
                arm["tails"].append([round(c, 5) for c in eb.tail])
            for f in range(FRAME_START, FRAME_END + 1):
                for m in ev.armature_skin(i, f, order_b, parents, rest, rest_inv):
                    bone_tracks.extend(m_row34(m))
            armatures.append(arm)
        arm_lookup = {a["node"]: k for k, a in enumerate(armatures)}
        for rec in records:
            if "arm" in rec:
                rec["arm"] = arm_lookup.get(rec["arm"], -1)

        # ---------------- lights ----------------
        light_tracks = array.array("f")
        lights = []
        for i, o in enumerate(self.objects):
            if o.type != "LIGHT":
                continue
            ld = o.data
            ch = build_channels(getattr(ld, "animation_data", None))
            entry = {
                "node": i, "name": o.name, "type": ld.type, "energy": float(ld.energy),
                "color": [float(c) for c in ld.color], "size": float(getattr(ld, "size", 0.25)),
                "size_y": float(getattr(ld, "size_y", 0.25)), "soft": float(getattr(ld, "shadow_soft_size", 0.1)),
                "spot_size": float(getattr(ld, "spot_size", math.radians(45.0))),
                "spot_blend": float(getattr(ld, "spot_blend", 0.15)), "track": -1,
            }
            if "energy" in ch:
                entry["track"] = len(light_tracks)
                for f in range(FRAME_START, FRAME_END + 1):
                    light_tracks.append(float(ch["energy"].evaluate(f)[0]))
            lights.append(entry)
        lights = lights[:MAX_LIGHTS]

        # ---------------- material tracks ----------------
        mat_tracks = array.array("f")
        mat_track_meta = []
        for mi, mat in enumerate(self.mat_objects):
            ch = dict(build_channels(getattr(mat.node_tree, "animation_data", None)))
            if getattr(mat, "node_tree", None) is not None:
                for gn in mat.node_tree.nodes:
                    if gn.bl_idname == "ShaderNodeGroup" and getattr(gn, "node_tree", None) is not None:
                        for gp, gch in build_channels(getattr(gn.node_tree, "animation_data", None)).items():
                            ch.setdefault(gp, gch)
            for path, chan in ch.items():
                if path.endswith("Emission Strength.default_value"):
                    peak = max(max(abs(v[0]) for v in chan.values), 1e-6)
                    mat_track_meta.append({"mat": mi, "param": "emission_scale", "offset": len(mat_tracks), "comps": 1})
                    for f in range(FRAME_START, FRAME_END + 1):
                        mat_tracks.append(float(chan.evaluate(f)[0]) / peak)
                elif path.endswith("Location.default_value"):
                    mat_track_meta.append({"mat": mi, "param": "scroll", "offset": len(mat_tracks), "comps": 3})
                    for f in range(FRAME_START, FRAME_END + 1):
                        v = tuple(chan.evaluate(f)) + (0.0, 0.0, 0.0)
                        mat_tracks.extend((float(v[0]), float(v[1]), float(v[2])))

        # ---------------- cameras + director markers ----------------
        scene = self.bpy.context.scene
        cameras = []
        cam_lookup = {}
        for i, o in enumerate(self.objects):
            if o.type != "CAMERA":
                continue
            cd = o.data
            focus = getattr(cd.dof, "focus_object", None)
            cam_lookup[id(o)] = len(cameras)
            cameras.append({
                "node": i, "name": o.name, "lens": float(cd.lens), "sensor": float(cd.sensor_width),
                "clip_start": float(cd.clip_start), "clip_end": float(cd.clip_end),
                "fstop": float(cd.dof.aperture_fstop), "focus_node": ev.obj_index(focus) if focus is not None else -1,
            })
        markers = []
        for mk in scene.timeline_markers:
            cam = getattr(mk, "camera", None)
            markers.append({"name": mk.name, "frame": int(mk.frame),
                            "camera": cam_lookup.get(id(cam), -1) if cam is not None else -1})
        markers.sort(key=lambda m: m["frame"])

        keyframes = set()
        for ch in ev.channels:
            for chan in ch.values():
                keyframes.update(chan.key_frames())
        for mat in self.mat_objects:
            for chan in build_channels(getattr(mat.node_tree, "animation_data", None)).values():
                keyframes.update(chan.key_frames())
        keyframes = sorted(k for k in keyframes if FRAME_START <= k <= FRAME_END)

        world = self.pack_world(scene)
        compositor = self.pack_compositor(scene)

        mat_tex = array.array("f")
        for rec in self.mat_records:
            mat_tex.extend(material_texels(rec))

        header = {
            "format": "CPKS", "version": PAYLOAD_VERSION, "generator": "build_interactive_studio_html.py",
            "fps": FPS, "frame_start": FRAME_START, "frame_end": FRAME_END, "frames": NUM_FRAMES,
            "counts": {
                "objects": len(self.objects), "meshes": self.stats["MESH"],
                "curves": self.stats["CURVE"] + self.stats["FONT"],
                "curve_tubes": self.stats["CURVE"] - self.stats["path_curves"],
                "path_curves": self.stats["path_curves"], "fonts": self.stats["FONT"],
                "materials": len(self.mat_records), "lights": len(lights), "cameras": len(cameras),
                "armatures": len(armatures), "tracks": len(track_meta),
                "vertices": len(self.positions) // 3, "triangles": len(self.indices) // 3,
                "line_vertices": len(self.lines) // 3,
            },
            "modifiers": {"applied": self.stats["modifiers_applied"], "shader": self.stats["modifiers_shader"]},
            "order": order, "objects": records, "materials": self.mat_records,
            "material_texels": MATERIAL_TEXELS, "material_tracks": mat_track_meta,
            "tracks": track_meta, "armatures": armatures, "max_bones": MAX_BONES,
            "lights": lights, "cameras": cameras, "markers": markers, "keyframes": keyframes,
            "rain": rain, "world": world, "compositor": compositor,
            "proc_names": {str(k): v for k, v in PROC_NAMES.items()},
        }
        sections = [
            ("positions", self.positions.tobytes(), "f32", 4),
            ("normals", self.normals.tobytes(), "f32", 4),
            ("attribs", bytes(self.attribs), "u8", 0),
            ("indices", self.indices.tobytes(), "u32", 4),
            ("lines", self.lines.tobytes(), "f32", 4),
            ("node_locals", node_locals.tobytes(), "f32", 4),
            ("tracks", tracks.tobytes(), "f32", 4),
            ("bone_tracks", bone_tracks.tobytes(), "f32", 4),
            ("light_tracks", light_tracks.tobytes(), "f32", 4),
            ("material_tracks", mat_tracks.tobytes(), "f32", 4),
            ("material_table", mat_tex.tobytes(), "f32", 4),
        ]
        self.log(f"[studio] baked {len(track_meta)} object tracks, {len(armatures)} armatures, "
                 f"{len(lights)} lights, {len(mat_track_meta)} material tracks in {time.perf_counter() - t0:.2f}s")
        return header, sections

    def pack_geometry_nodes_rain(self, records):
        gn = self.bpy.data.node_groups.get("GN_Cyber_Rain_Field")
        info = {"enabled": False}
        if gn is None:
            return info
        density, seed = 7.5, 0
        smin, smax = (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)
        proto, mat = None, None
        for n in gn.nodes:
            if n.bl_idname == "GeometryNodeDistributePointsOnFaces":
                density = float(n.inputs["Density"].default_value or density)
                seed = int(n.inputs["Seed"].default_value or 0)
            elif n.bl_idname == "FunctionNodeRandomValue":
                smin = tuple(float(c) for c in (n.inputs["Min"].default_value or smin))
                smax = tuple(float(c) for c in (n.inputs["Max"].default_value or smax))
            elif n.bl_idname == "GeometryNodeObjectInfo":
                proto = n.inputs["Object"].default_value
            elif n.bl_idname == "GeometryNodeSetMaterial":
                mat = n.inputs["Material"].default_value
        if mat is None and proto is not None and getattr(proto, "data", None) is not None:
            pmats = getattr(proto.data, "materials", [])
            if pmats:
                mat = pmats[0]
        emitters = [i for i, r in enumerate(records) if r.get("role") == "rain_emitter"]
        proto_idx = self.ev.obj_index(proto) if proto is not None else -1
        if proto_idx >= 0:
            records[proto_idx]["role"] = "rain_prototype"
        faces = []
        area = 0.0
        if emitters:
            mesh = self.objects[emitters[0]].data
            for p in mesh.polygons:
                vs = [tuple(mesh.vertices[k].co) for k in p.vertices]
                faces.append([[round(c, 5) for c in v] for v in vs])
                area += v_len(polygon_normal(vs, list(range(len(vs))))) * 0.5
        info.update({
            "enabled": bool(emitters) and proto_idx >= 0, "tree": gn.name, "emitters": emitters,
            "prototype": proto_idx, "density": density, "seed": seed,
            "scale_min": list(smin), "scale_max": list(smax),
            "material": self.material_id(mat), "faces": faces, "face_area": round(area, 4),
            "points_per_emitter": int(round(density * area)),
        })
        return info

    def pack_world(self, scene):
        world = scene.world
        out = {"sky_bottom": [0.005, 0.009, 0.022], "sky_top": [0.016, 0.007, 0.030], "strength": 0.32,
               "fog_color": [0.32, 0.45, 0.62], "fog_anisotropy": 0.66,
               "mist_start": 8.0, "mist_depth": 28.0, "mist_falloff": "QUADRATIC",
               "view_transform": scene.view_settings.view_transform, "look": scene.view_settings.look,
               "exposure": float(scene.view_settings.exposure)}
        nt = getattr(world, "node_tree", None)
        if nt is not None:
            for n in nt.nodes:
                if n.bl_idname == "ShaderNodeValToRGB":
                    els = list(n.color_ramp.elements)
                    out["sky_bottom"] = [float(c) for c in els[0].color[:3]]
                    out["sky_top"] = [float(c) for c in els[-1].color[:3]]
                elif n.bl_idname == "ShaderNodeBackground":
                    out["strength"] = float(n.inputs["Strength"].default_value or 1.0)
                elif n.bl_idname == "ShaderNodeVolumePrincipled":
                    col = n.inputs["Color"].default_value
                    if col is not None:
                        out["fog_color"] = [float(c) for c in col[:3]]
                    ani = n.inputs["Anisotropy"].default_value
                    if ani is not None:
                        out["fog_anisotropy"] = float(ani)
        ms = getattr(world, "mist_settings", None)
        if ms is not None:
            out["mist_start"] = float(ms.start)
            out["mist_depth"] = float(ms.depth)
            out["mist_falloff"] = ms.falloff
        return out

    def pack_compositor(self, scene):
        out = {"glare": [], "dispersion": 0.012, "distortion": -0.015, "exposure": 0.06,
               "lift": [1.0, 1.0, 1.0], "gamma": [1.0, 1.0, 1.0], "gain": [1.0, 1.0, 1.0],
               "mist_tint": [0.038, 0.065, 0.115], "mist_mix": 0.42, "emit_feed": 0.22}
        nt = getattr(scene, "node_tree", None)
        if nt is None:
            return out
        for n in nt.nodes:
            t = n.bl_idname
            if t == "CompositorNodeGlare":
                out["glare"].append({
                    "type": n.glare_type, "threshold": float(n.threshold), "size": int(n.size),
                    "mix": float(n.mix), "streaks": int(getattr(n, "streaks", 4)),
                    "angle_offset": float(getattr(n, "angle_offset", 0.0)), "fade": float(getattr(n, "fade", 0.9)),
                })
            elif t == "CompositorNodeLensdist":
                if "Dispersion" in n.inputs and n.inputs["Dispersion"].default_value is not None:
                    out["dispersion"] = float(n.inputs["Dispersion"].default_value)
                for key in ("Distortion", "Distort"):
                    if key in n.inputs and n.inputs[key].default_value is not None:
                        out["distortion"] = float(n.inputs[key].default_value)
            elif t == "CompositorNodeExposure":
                if n.inputs["Exposure"].default_value is not None:
                    out["exposure"] = float(n.inputs["Exposure"].default_value)
            elif t == "CompositorNodeColorBalance":
                out["lift"] = [float(c) for c in n.lift]
                out["gamma"] = [float(c) for c in n.gamma]
                out["gain"] = [float(c) for c in n.gain]
            elif t == "CompositorNodeValToRGB":
                els = list(n.color_ramp.elements)
                out["mist_tint"] = [float(c) for c in els[-1].color[:3]]
        return out


# ==============================================================================
# 10. BINARY CONTAINER (byte-shuffle + zlib + Base64)
# ==============================================================================

def byte_shuffle(data, width):
    if width <= 1 or len(data) % width != 0:
        return data
    return b"".join(data[k::width] for k in range(width))


def byte_unshuffle(data, width):
    if width <= 1 or len(data) % width != 0:
        return data
    n = len(data) // width
    out = bytearray(len(data))
    for k in range(width):
        out[k::width] = data[k * n:(k + 1) * n]
    return bytes(out)


def assemble_container(header, sections):
    blobs = []
    table = {}
    offset = 0
    for name, raw, dtype, shuffle in sections:
        data = byte_shuffle(raw, shuffle)
        pad = (-offset) % 4
        if pad:
            blobs.append(b"\0" * pad)
            offset += pad
        table[name] = {"offset": offset, "length": len(raw), "type": dtype, "shuffle": shuffle,
                       "sha256": hashlib.sha256(raw).hexdigest()}
        blobs.append(data)
        offset += len(data)
    header = dict(header)
    header["sections"] = table
    header_json = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    head = PAYLOAD_MAGIC + struct.pack("<II", PAYLOAD_VERSION, len(header_json)) + header_json
    head += b"\0" * ((-len(head)) % 4)
    return head + b"".join(blobs), header


def parse_container(blob):
    """Inverse of assemble_container (used by the verifier for integrity checks)."""
    if blob[:4] != PAYLOAD_MAGIC:
        raise ValueError("bad payload magic")
    version, jlen = struct.unpack_from("<II", blob, 4)
    header = json.loads(blob[12:12 + jlen].decode("ascii"))
    start = (12 + jlen + 3) & ~3
    sections = {}
    for name, meta in header["sections"].items():
        raw = blob[start + meta["offset"]:start + meta["offset"] + meta["length"]]
        sections[name] = byte_unshuffle(raw, meta["shuffle"])
    return version, header, sections


# ==============================================================================
# 11. HTML ASSEMBLY + MANIFEST
# ==============================================================================

def render_html(template, replacements):
    html = template
    for key, value in replacements.items():
        token = "@@" + key + "@@"
        if token not in html:
            raise KeyError(f"template token missing: {token}")
        html = html.replace(token, value)
    return html


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def update_manifest(manifest_path, studio_info):
    manifest = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    manifest["interactive_studio"] = studio_info
    manifest["interactive_studio_html_sha256"] = studio_info["sha256_digest"]
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
    return manifest


def build_studio(scene_path, output_path=None, reuse_wav=False, write_manifest=True, log=print, bpy=None):
    t_start = time.perf_counter()
    scene_path = os.path.abspath(scene_path)
    scene_dir = os.path.dirname(scene_path)
    output_path = os.path.abspath(output_path or os.path.join(scene_dir, HTML_NAME))
    if bpy is None:
        log(f"[studio] executing {os.path.basename(scene_path)} in mock Blender 4.3.0 runtime")
        bpy, _ns = execute_scene(scene_path, reuse_wav=reuse_wav)
    log(f"[studio] scene ready: {len(bpy.data.objects)} objects, {len(bpy.data.meshes)} meshes, "
        f"{len(bpy.data.curves)} curves, {len(bpy.data.materials)} materials "
        f"({time.perf_counter() - t_start:.1f}s)")

    packer = ScenePacker(bpy, log=log)
    header, sections = packer.pack()
    blob, header = assemble_container(header, sections)
    compressed = zlib.compress(blob, 9)
    scene_b64 = base64.b64encode(compressed).decode("ascii")
    payload_sha = hashlib.sha256(blob).hexdigest()

    wav_path = os.path.join(scene_dir, WAV_NAME)
    with open(wav_path, "rb") as fh:
        wav_bytes = fh.read()
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    audio_sha = hashlib.sha256(wav_bytes).hexdigest()

    build_info = {
        "generator": "build_interactive_studio_html.py",
        "scene_script": os.path.basename(scene_path),
        "payload_format": "CPKS v1 / byte-shuffle(4) / zlib(9) / base64",
        "payload_raw_bytes": len(blob), "payload_zlib_bytes": len(compressed), "payload_b64_chars": len(scene_b64),
        "payload_sha256": payload_sha, "audio_file": WAV_NAME, "audio_bytes": len(wav_bytes), "audio_sha256": audio_sha,
        "counts": header["counts"], "video_file": VIDEO_NAME,
    }
    html = render_html(HTML_TEMPLATE, {
        "SCENE_B64": scene_b64,
        "SCENE_SHA256": payload_sha,
        "SCENE_RAW_BYTES": str(len(blob)),
        "SCENE_ZLIB_BYTES": str(len(compressed)),
        "AUDIO_B64": audio_b64,
        "AUDIO_SHA256": audio_sha,
        "AUDIO_BYTES": str(len(wav_bytes)),
        "BUILD_INFO": json.dumps(build_info, separators=(",", ":")),
        "VIDEO_NAME": VIDEO_NAME,
    })
    with open(output_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    html_sha = sha256_file(output_path)
    html_size = os.path.getsize(output_path)

    studio_info = {
        "html_file": os.path.basename(output_path),
        "byte_size": html_size,
        "sha256_digest": html_sha,
        "builder": "build_interactive_studio_html.py",
        "scene_payload": {
            "format": "CPKS", "version": PAYLOAD_VERSION, "raw_bytes": len(blob),
            "zlib_bytes": len(compressed), "base64_chars": len(scene_b64), "sha256": payload_sha,
            "sections": {k: {"bytes": v["length"], "type": v["type"]} for k, v in header["sections"].items()},
        },
        "geometry": header["counts"],
        "modifiers": header["modifiers"],
        "audio_payload": {"file": WAV_NAME, "bytes": len(wav_bytes), "sha256": audio_sha},
        "renderer": "WebGL2 / GLSL ES 3.00 multi-target forward pipeline",
        "verified": True,
    }
    if write_manifest:
        update_manifest(os.path.join(scene_dir, MANIFEST_NAME), studio_info)

    c = header["counts"]
    rows = [
        ("Output", f"{os.path.basename(output_path)} ({html_size:,} bytes)"),
        ("HTML SHA-256", html_sha),
        ("Meshes / Curves", f"{c['meshes']} meshes | {c['curves']} curves ({c['curve_tubes']} tubes, "
                            f"{c['fonts']} fonts, {c['path_curves']} paths)"),
        ("Triangle Buffers", f"{c['vertices']:,} verts | {c['triangles']:,} tris | Uint32 indices"),
        ("Modifiers Applied", ", ".join(f"{k} x{v}" for k, v in header["modifiers"]["applied"].items())),
        ("Materials", f"{c['materials']} PBR/emissive records x {MATERIAL_TEXELS} RGBA32F texels"),
        ("Animation", f"{NUM_FRAMES} frames | {c['tracks']} object tracks | {c['armatures']} armatures | "
                      f"{c['lights']} lights"),
        ("Scene Payload", f"{len(blob):,} raw -> {len(compressed):,} zlib -> {len(scene_b64):,} b64"),
        ("Payload SHA-256", payload_sha),
        ("Audio Payload", f"{WAV_NAME} {len(wav_bytes):,} bytes ({audio_sha[:16]})"),
        ("Build Time", f"{time.perf_counter() - t_start:.1f}s"),
    ]
    log("=" * 98)
    log(" CYBERPUNK INTERACTIVE STUDIO -- PHASE 8A/8B BUILD SUMMARY")
    log("=" * 98)
    for k, v in rows:
        log(f" {k:<20} | {v}")
    log("=" * 98)
    return {"html_path": output_path, "sha256": html_sha, "size": html_size, "header": header,
            "payload_sha256": payload_sha, "audio_sha256": audio_sha, "studio_info": studio_info}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the self-contained WebGL2 cyberpunk interactive studio HTML.")
    ap.add_argument("--scene", default=os.path.join(SCRIPT_DIR, SCENE_SCRIPT_NAME), help="diorama generator script")
    ap.add_argument("--output", default=None, help="output HTML path (default: next to the scene script)")
    ap.add_argument("--reuse-wav", action="store_true", help="reuse an existing soundtrack WAV instead of re-synthesising")
    ap.add_argument("--no-manifest", action="store_true", help="do not record the HTML digest in the manifest")
    args = ap.parse_args(argv)
    build_studio(args.scene, args.output, reuse_wav=args.reuse_wav, write_manifest=not args.no_manifest)
    return 0


# ==============================================================================
# 12. SELF-CONTAINED HTML TEMPLATE (markup + CSS + GLSL ES 3.00 + JavaScript)
# ==============================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="generator" content="build_interactive_studio_html.py (Phase 8A/8B)">
<meta name="description" content="Self-contained WebGL2 + WebAudio interactive studio for the Cyberpunk Floating Ramen Stall diorama. Zero external dependencies.">
<title>CYBER RAMEN // Interactive Studio - WebGL2 + WebAudio</title>
<style>
:root{
  --cyan:#00eaff;--mag:#ff2bd6;--amber:#ffb340;--lime:#7dff9a;
  --bg:rgba(5,9,18,.74);--bg2:rgba(10,16,30,.86);--line:rgba(0,234,255,.26);--line2:rgba(255,43,214,.35);
  --txt:#d4f3ff;--dim:#7895ad;--warn:#ff5577;
  --mono:"Cascadia Mono","Cascadia Code","JetBrains Mono",Consolas,"SFMono-Regular",Menlo,"DejaVu Sans Mono",monospace;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#010207;color:var(--txt);overflow:hidden;font:12px/1.35 var(--mono);-webkit-font-smoothing:antialiased}
#gl{position:fixed;inset:0;width:100vw;height:100vh;display:block;touch-action:none;cursor:grab;outline:none}
#gl.dragging{cursor:grabbing}
.panel{position:fixed;z-index:5;background:var(--bg);border:1px solid var(--line);border-radius:7px;
  box-shadow:0 0 22px rgba(0,234,255,.07),inset 0 0 26px rgba(255,43,214,.035);-webkit-backdrop-filter:blur(7px);backdrop-filter:blur(7px)}
.panel::before{content:"";position:absolute;inset:-1px;border-radius:7px;pointer-events:none;
  background:linear-gradient(90deg,var(--cyan),transparent 22%,transparent 78%,var(--mag)) border-box;
  -webkit-mask:linear-gradient(#000 0 0) padding-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;border:1px solid transparent;opacity:.55}
.lbl{font-size:10px;letter-spacing:.14em;color:var(--dim);text-transform:uppercase;margin:0 0 6px}
.lbl b{color:var(--cyan);font-weight:600}
button,select,label.btn{font:inherit;font-size:11px;color:var(--txt);background:rgba(0,234,255,.06);border:1px solid rgba(0,234,255,.28);
  border-radius:4px;padding:5px 8px;cursor:pointer;letter-spacing:.05em;transition:background .15s,border-color .15s,box-shadow .15s,color .15s}
button:hover,label.btn:hover{background:rgba(0,234,255,.16);border-color:var(--cyan)}
button.on{background:rgba(0,234,255,.22);border-color:var(--cyan);color:#fff;box-shadow:0 0 10px rgba(0,234,255,.35),inset 0 0 8px rgba(0,234,255,.25)}
button.mag.on{background:rgba(255,43,214,.22);border-color:var(--mag);box-shadow:0 0 10px rgba(255,43,214,.4),inset 0 0 8px rgba(255,43,214,.25)}
button.rec.on{background:rgba(255,40,80,.35);border-color:#ff3b6b;box-shadow:0 0 14px rgba(255,40,80,.6);animation:blink 1s steps(2) infinite}
button:focus-visible,select:focus-visible{outline:1px solid var(--mag);outline-offset:1px}
select{padding:4px 6px;background:var(--bg2)}
@keyframes blink{50%{opacity:.55}}
#p-title{top:12px;left:12px;width:330px;padding:11px 13px 10px}
.brand{font-size:15px;letter-spacing:.12em;font-weight:700;white-space:nowrap}
.brand .g{color:var(--cyan);text-shadow:0 0 8px rgba(0,234,255,.8),2px 0 0 rgba(255,43,214,.55)}
.brand b{color:var(--mag);font-weight:600;font-size:11px;letter-spacing:.18em;text-shadow:0 0 8px rgba(255,43,214,.6)}
.sub{color:var(--dim);font-size:10px;letter-spacing:.08em;margin:3px 0 9px}
#stats{display:grid;grid-template-columns:auto 1fr auto 1fr;gap:2px 10px;margin:0;font-size:11px}
#stats dt{color:var(--dim)}
#stats dd{margin:0;color:var(--txt);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#stats dd.ok{color:var(--lime)}#stats dd.bad{color:var(--warn)}
#p-cams{top:12px;right:12px;width:392px;padding:10px 12px}
.grp{margin-bottom:9px}.grp:last-child{margin-bottom:0}
.btns{display:flex;flex-wrap:wrap;gap:5px}
.toggles{display:grid;grid-template-columns:repeat(4,1fr);gap:4px 8px;font-size:11px}
.toggles label{display:flex;align-items:center;gap:5px;cursor:pointer;color:var(--txt);white-space:nowrap}
.toggles input{accent-color:var(--cyan);margin:0}
.sliders{display:grid;grid-template-columns:auto 1fr 42px;gap:4px 8px;align-items:center;font-size:11px;margin-top:7px}
.sliders input[type=range]{width:100%;accent-color:var(--mag)}
.sliders output{color:var(--cyan);text-align:right}
#p-audio{right:12px;top:268px;width:392px;padding:10px 12px}
#hud-audio{width:100%;height:208px;display:block;border-radius:4px;background:rgba(0,0,0,.35)}
.bands{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:7px;font-size:10px;color:var(--dim)}
.band .bar{height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;margin-top:3px}
.band .bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--mag));box-shadow:0 0 8px var(--cyan)}
.band span{float:right;color:var(--txt)}
#p-pip{right:12px;bottom:118px;width:392px;padding:9px 12px 10px}
#pip-view{position:relative;width:100%;aspect-ratio:16/9;border:1px solid var(--line2);border-radius:4px;overflow:hidden;background:transparent}
#pip-view video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;display:none;background:#000}
#pip-view.mp4 video{display:block}
#pip-shot{position:absolute;left:6px;top:5px;font-size:10px;letter-spacing:.1em;color:#fff;text-shadow:0 0 6px #000,0 0 2px #000;pointer-events:none}
#pip-rec{position:absolute;right:6px;top:5px;font-size:10px;color:#ff3b6b;display:none;text-shadow:0 0 6px #000}
#pip-rec.on{display:block;animation:blink 1s steps(2) infinite}
.row{display:flex;flex-wrap:wrap;gap:5px;margin-top:7px;align-items:center}
#p-deck{left:12px;right:416px;bottom:12px;padding:9px 12px 10px}
.transport{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.transport .tc{margin-left:auto;font-size:13px;letter-spacing:.08em;color:var(--cyan);text-shadow:0 0 8px rgba(0,234,255,.6)}
.transport .tc small{color:var(--dim);font-size:10px;margin-left:6px}
#btn-play{min-width:74px;font-weight:700}
#scrubber{width:100%;height:58px;display:block;margin-top:8px;cursor:ew-resize;border-radius:4px;background:rgba(0,0,0,.35);touch-action:none}
#loader{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center;background:radial-gradient(ellipse at center,#0a1426 0%,#010207 70%);transition:opacity .5s}
#loader.done{opacity:0;pointer-events:none}
.ld-box{width:min(520px,90vw);padding:22px 24px;border:1px solid var(--line);border-radius:8px;background:var(--bg2);box-shadow:0 0 40px rgba(0,234,255,.12)}
.ld-title{font-size:18px;letter-spacing:.2em;color:var(--cyan);text-shadow:0 0 12px rgba(0,234,255,.7);margin-bottom:12px}
#ld-steps{font-size:11px;color:var(--dim);min-height:110px;white-space:pre-wrap}
#ld-steps .ok{color:var(--lime)}#ld-steps .err{color:var(--warn)}
.ld-bar{height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;margin-top:10px}
.ld-bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--cyan),var(--mag));transition:width .25s}
#toast{position:fixed;left:50%;top:14px;transform:translate(-50%,-30px);opacity:0;z-index:60;padding:7px 14px;border-radius:5px;background:var(--bg2);border:1px solid var(--mag);color:#fff;font-size:11px;letter-spacing:.06em;transition:all .25s;pointer-events:none;max-width:70vw}
#toast.show{opacity:1;transform:translate(-50%,0)}
#help{left:50%;top:50%;transform:translate(-50%,-50%);width:min(560px,92vw);padding:16px 18px;z-index:40;display:none}
#help.show{display:block}
#help table{width:100%;border-collapse:collapse;font-size:11px}
#help td{padding:3px 6px;border-bottom:1px solid rgba(0,234,255,.08)}
#help td:first-child{color:var(--cyan);white-space:nowrap;width:38%}
#hint{position:fixed;left:50%;bottom:92px;transform:translateX(-50%);z-index:4;font-size:10px;color:var(--dim);letter-spacing:.12em;pointer-events:none;text-shadow:0 0 6px #000}
.hidden-ui .panel:not(#help){display:none}
#fatal{position:fixed;inset:0;z-index:70;display:none;align-items:center;justify-content:center;background:#010207;color:var(--warn);font-size:13px;text-align:center;padding:30px}
#fatal.show{display:flex}
@media (max-width:1180px){#p-audio{display:none}#p-pip{bottom:118px}#p-deck{right:12px;bottom:12px}#p-pip{display:none}}
@media (max-width:760px){#p-cams{left:12px;right:12px;width:auto;top:auto;bottom:122px}#p-title{width:auto;right:12px}#stats{grid-template-columns:auto 1fr auto 1fr}}
</style>
</head>
<body>
<canvas id="gl" tabindex="0" aria-label="WebGL2 cyberpunk diorama viewport"></canvas>

<div id="loader" role="status" aria-live="polite"><div class="ld-box">
  <div class="ld-title">CYBER RAMEN // STUDIO</div>
  <div id="ld-steps"></div>
  <div class="ld-bar"><i id="ld-fill"></i></div>
</div></div>

<div id="fatal"></div>

<header id="p-title" class="panel">
  <div class="brand"><span class="g">CYBER RAMEN</span> <b>// INTERACTIVE STUDIO</b></div>
  <div class="sub">PHASE 8 &middot; WEBGL2 GLSL ES 3.00 &middot; WEBAUDIO FFT &middot; 240F @ 24FPS</div>
  <dl id="stats">
    <dt>FPS</dt><dd id="st-fps">--</dd>
    <dt>FRAME</dt><dd id="st-frame">1</dd>
    <dt>TRIS</dt><dd id="st-tris">--</dd>
    <dt>DRAWS</dt><dd id="st-draws">--</dd>
    <dt>CAMERA</dt><dd id="st-cam">FREE</dd>
    <dt>SHOT</dt><dd id="st-shot">--</dd>
    <dt>PAYLOAD</dt><dd id="st-payload">--</dd>
    <dt>SHA-256</dt><dd id="st-integrity">--</dd>
    <dt>MESH/CRV</dt><dd id="st-geo">--</dd>
    <dt>INFLATE</dt><dd id="st-inflate">--</dd>
    <dt>GPU</dt><dd id="st-gpu" style="grid-column:span 3">--</dd>
  </dl>
</header>

<nav id="p-cams" class="panel" aria-label="Camera and AOV render controls">
  <div class="grp"><div class="lbl">CAMERA <b>DIRECTOR SWITCHER</b></div>
    <div class="btns" id="cam-btns">
      <button data-cam="free" class="on" title="Free orbit / pan / WASD fly (F)">FREE ORBIT</button>
      <button data-cam="director" class="mag" title="Follow TimelineMarker camera binding (C)">DIRECTOR CUT</button>
      <button data-cam="0" title="Shot 01 f1-96 85mm orbit (1)">85MM ORBIT</button>
      <button data-cam="1" title="Shot 02 f97-168 50mm low puddle hero (2)">50MM HERO</button>
      <button data-cam="2" title="Shot 03 f169-240 105mm counter macro (3)">105MM MACRO</button>
    </div>
  </div>
  <div class="grp"><div class="lbl">AOV <b>LIVE RENDER PASSES</b></div>
    <div class="btns" id="aov-btns">
      <button data-aov="0" class="on" title="AgX High Contrast final composite (Z)">FINAL AGX</button>
      <button data-aov="1" title="AOV_Neon_Mask emission + bloom (X)">NEON EMISSION</button>
      <button data-aov="2" title="Linear depth + volumetric mist (V)">Z-DEPTH MIST</button>
      <button data-aov="3" title="World-space normals (N)">NORMALS</button>
      <button data-aov="4" title="Hidden-line wireframe (B)">WIREFRAME</button>
    </div>
  </div>
  <div class="grp">
    <div class="toggles">
      <label><input type="checkbox" id="t-bloom" checked>BLOOM</label>
      <label><input type="checkbox" id="t-mist" checked>MIST</label>
      <label><input type="checkbox" id="t-chroma" checked>CHROMA</label>
      <label><input type="checkbox" id="t-mirror" checked>MIRROR</label>
      <label><input type="checkbox" id="t-rain" checked>RAIN</label>
      <label><input type="checkbox" id="t-react" checked>AUDIO&#8596;GL</label>
      <label><input type="checkbox" id="t-paths">PATHS</label>
      <label><input type="checkbox" id="t-streaks" checked>STREAKS</label>
    </div>
    <div class="sliders">
      <span>EXPOSURE</span><input type="range" id="s-exposure" min="-3" max="4" step="0.05" value="0"><output id="o-exposure">0</output>
      <span>BLOOM</span><input type="range" id="s-bloom" min="0" max="3" step="0.05" value="1"><output id="o-bloom">1</output>
      <span>RAIN DENSITY</span><input type="range" id="s-rain" min="0" max="1" step="0.05" value="0.6"><output id="o-rain">60%</output>
    </div>
  </div>
</nav>

<aside id="p-audio" class="panel" aria-label="WebAudio spectrum analyzer">
  <div class="lbl">WEBAUDIO <b>FFT 2048 &middot; 64-BAND LOG &middot; LISSAJOUS &middot; SCOPE</b></div>
  <canvas id="hud-audio"></canvas>
  <div class="bands">
    <div class="band">SUB 20-120HZ <span id="b-sub">0.00</span><div class="bar"><i id="bb-sub"></i></div></div>
    <div class="band">MID 250-2KHZ <span id="b-mid">0.00</span><div class="bar"><i id="bb-mid"></i></div></div>
    <div class="band">HIGH 4-16KHZ <span id="b-high">0.00</span><div class="bar"><i id="bb-high"></i></div></div>
  </div>
</aside>

<section id="p-pip" class="panel" aria-label="Video deck">
  <div class="lbl">VIDEO DECK <b id="pip-src">LIVE WEBGL2 DIRECTOR FEED</b></div>
  <div id="pip-view"><video id="deck-video" muted playsinline preload="auto"></video><div id="pip-shot">--</div><div id="pip-rec">&#9679; REC</div></div>
  <div class="row">
    <button id="btn-pip" class="on" title="Picture-in-picture director monitor">PIP</button>
    <button id="btn-split" title="Split-screen free / director">SPLIT</button>
    <button id="btn-pipoff" title="Hide secondary view">OFF</button>
    <label class="btn" title="Load a rendered MP4/WebM for synced playback">LOAD MP4<input type="file" id="file-video" accept="video/*" hidden></label>
    <button id="btn-rec" class="rec" title="Record the 240-frame cut with soundtrack to .webm">REC .WEBM</button>
    <button id="btn-png" title="Export the current frame as .png (P)">PNG</button>
  </div>
</section>

<footer id="p-deck" class="panel" aria-label="Timeline control deck">
  <div class="transport">
    <button id="btn-start" title="Jump to f1 (Home)">&#9198;</button>
    <button id="btn-back" title="Step -1 frame (,)">-1</button>
    <button id="btn-play" class="on" title="Play / Pause (Space)">&#9654; PLAY</button>
    <button id="btn-fwd" title="Step +1 frame (.)">+1</button>
    <button id="btn-end" title="Jump to f240 (End)">&#9197;</button>
    <button id="btn-loop" class="on mag" title="Loop (L)">LOOP</button>
    <select id="sel-speed" title="Playback speed">
      <option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="0.75">0.75x</option>
      <option value="1" selected>1.0x</option><option value="1.25">1.25x</option><option value="1.5">1.5x</option><option value="2">2.0x</option>
    </select>
    <button id="btn-mute" title="Mute (M)">AUDIO ON</button>
    <button id="btn-help" title="Shortcuts (H)">?</button>
    <span class="tc" id="tc">F001 / 240<small id="tc-time">00:00.00</small></span>
  </div>
  <canvas id="scrubber" aria-label="240-frame keyframe scrubber"></canvas>
</footer>

<div id="hint">DRAG ORBIT &middot; RIGHT/SHIFT-DRAG PAN &middot; WHEEL ZOOM &middot; WASD+QE FLY &middot; H HELP</div>
<div id="toast"></div>

<div id="help" class="panel" role="dialog" aria-label="Keyboard shortcuts">
  <div class="lbl">SHORTCUTS <b>CYBER RAMEN STUDIO</b></div>
  <table>
    <tr><td>Space</td><td>Play / Pause (audio + scene + video in sync)</td></tr>
    <tr><td>, / .</td><td>Frame step -1 / +1</td></tr>
    <tr><td>Home / End</td><td>Jump to f1 / f240</td></tr>
    <tr><td>L</td><td>Toggle loop</td></tr>
    <tr><td>F / C / 1 / 2 / 3</td><td>Free orbit / Director cut / 85mm / 50mm / 105mm</td></tr>
    <tr><td>Z X V N B</td><td>AOV: Final AgX / Neon / Z-Depth Mist / Normals / Wireframe</td></tr>
    <tr><td>Drag / Right-drag / Wheel</td><td>Orbit / Pan / Dolly (touch: 1 finger orbit, 2 finger pinch + pan)</td></tr>
    <tr><td>W A S D Q E (+Shift)</td><td>Fly the free camera (fast)</td></tr>
    <tr><td>R</td><td>Reset free camera</td></tr>
    <tr><td>P</td><td>PNG snapshot of the current frame</td></tr>
    <tr><td>M</td><td>Mute audio</td></tr>
    <tr><td>U</td><td>Hide / show HUD</td></tr>
    <tr><td>H / ?</td><td>Toggle this help</td></tr>
  </table>
</div>

"""

HTML_TEMPLATE += r"""
<!-- ===================== GLSL ES 3.00 SHADER LIBRARY ===================== -->
<script type="x-shader/x-common" id="glsl-common">
const float PI = 3.14159265359;
float hash11(float p){ p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
float hash31(vec3 p3){ p3 = fract(p3 * 0.1031); p3 += dot(p3, p3.zyx + 31.32); return fract((p3.x + p3.y) * p3.z); }
float vnoise(vec3 p){
  vec3 i = floor(p); vec3 f = fract(p); f = f * f * (3.0 - 2.0 * f);
  float n000 = hash31(i), n100 = hash31(i + vec3(1.0, 0.0, 0.0)), n010 = hash31(i + vec3(0.0, 1.0, 0.0)), n110 = hash31(i + vec3(1.0, 1.0, 0.0));
  float n001 = hash31(i + vec3(0.0, 0.0, 1.0)), n101 = hash31(i + vec3(1.0, 0.0, 1.0)), n011 = hash31(i + vec3(0.0, 1.0, 1.0)), n111 = hash31(i + vec3(1.0, 1.0, 1.0));
  return mix(mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y), mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y), f.z);
}
float fbm(vec3 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + vec3(17.1, 9.7, 3.3); a *= 0.5; } return s; }
</script>

<script type="x-shader/x-vertex" id="vs-scene">#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNor;
layout(location = 2) in uvec4 aAttr;
uniform mat4 uViewProj;
uniform mat4 uModel;
uniform mat3 uNormalMat;
uniform mat4 uBones[16];
uniform int uSkinned;
uniform vec4 uWave;
uniform float uWaveNormal;
uniform float uFrameF;
out vec3 vWorld;
out vec3 vNor;
out vec3 vLocal;
out vec3 vLocalNor;
flat out uint vMat;
flat out uint vFlags;
void main() {
  vec4 p = vec4(aPos, 1.0);
  vec3 n = aNor;
  if (uSkinned == 1 && aAttr.y < 255u) {
    mat4 B = uBones[int(aAttr.y)];
    p = B * p;
    n = mat3(B) * n;
  }
  if (uWave.x > 0.0) {
    float span = max(uWave.y, 1e-3);
    float d = length(p.xy);
    float a = mod(d - uFrameF * uWave.z, span * 2.0) - span;
    float g = a * uWave.w;
    float amp = exp(-g * g) * uWave.x;
    vec3 dir = uWaveNormal > 0.5 ? normalize(n) : vec3(0.0, 0.0, 1.0);
    p.xyz += dir * amp;
  }
  vLocal = p.xyz;
  vLocalNor = n;
  vec4 wp = uModel * p;
  vWorld = wp.xyz;
  vNor = uNormalMat * n;
  vMat = aAttr.x;
  vFlags = aAttr.z;
  gl_Position = uViewProj * wp;
}
</script>

<script type="x-shader/x-fragment" id="fs-scene">#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec3 vWorld;
in vec3 vNor;
in vec3 vLocal;
in vec3 vLocalNor;
flat in uint vMat;
flat in uint vFlags;
uniform sampler2D uMatTex;
uniform sampler2D uReflTex;
uniform int uUseRefl;
uniform int uMirrorPass;
uniform int uFlipFacing;
uniform vec4 uViewport;
uniform vec3 uCamPos;
uniform float uTime;
uniform float uFar;
uniform int uLightCount;
uniform vec4 uLightPos[16];
uniform vec4 uLightCol[16];
uniform vec4 uLightDir[16];
uniform vec4 uLightAux[16];
uniform vec3 uSkyTop;
uniform vec3 uSkyBottom;
uniform float uAmbient;
uniform vec3 uAudio;
uniform float uAudioReactive;
uniform float uRainWet;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oNeon;
layout(location = 2) out vec4 oAux;
//#include common

vec3 fresnelSchlick(vec3 F0, float c) { return F0 + (1.0 - F0) * pow(1.0 - c, 5.0); }
float distributionGGX(float NoH, float a) { float a2 = a * a; float d = NoH * NoH * (a2 - 1.0) + 1.0; return a2 / (PI * d * d + 1e-7); }
float visibilitySmith(float NoV, float NoL, float a) {
  float k = a * 0.5;
  float gv = NoV / (NoV * (1.0 - k) + k);
  float gl = NoL / (NoL * (1.0 - k) + k);
  return gv * gl / max(4.0 * NoV * NoL, 1e-4);
}

void main() {
  if (uMirrorPass == 1 && vWorld.z < -0.003) discard;
  int mid = int(vMat);
  vec4 m0 = texelFetch(uMatTex, ivec2(0, mid), 0);
  vec4 m1 = texelFetch(uMatTex, ivec2(1, mid), 0);
  vec4 m2 = texelFetch(uMatTex, ivec2(2, mid), 0);
  vec4 m3 = texelFetch(uMatTex, ivec2(3, mid), 0);
  vec4 m4 = texelFetch(uMatTex, ivec2(4, mid), 0);
  vec4 m5 = texelFetch(uMatTex, ivec2(5, mid), 0);
  vec3 base = m0.rgb;
  float alpha = m0.a;
  vec3 emis = m1.rgb * m1.a;
  float metal = m2.x;
  float rough = m2.y;
  float trans = m2.z;
  float coat = m2.w;
  vec3 rA = m3.rgb;
  vec3 rB = m4.rgb;
  float coatRough = m4.a;
  int proc = int(m3.w + 0.5);
  int mflags = int(m5.w + 0.5);
  vec3 P = vLocal + m5.xyz;
  vec3 Ln = normalize(vLocalNor);
  vec3 N = normalize(vNor);
  bool front = gl_FrontFacing;
  if (uFlipFacing == 1) front = !front;
  if (!front) N = -N;
  vec3 V = normalize(uCamPos - vWorld);
  float NoV = clamp(dot(N, V), 0.0, 1.0);
  float fres = pow(1.0 - NoV, 5.0);
  float mirror = 0.0;
  float neonMask = (mflags & 1) != 0 ? 1.0 : 0.0;
  bool isTransparent = (mflags & 2) != 0;

  if (proc == 1) {
    float n = fbm(P * vec3(14.0, 1.1, 14.0));
    float g = vnoise(P * vec3(160.0, 6.0, 160.0));
    base = mix(rA, rB, smoothstep(0.2, 0.8, n));
    rough = clamp(rough * (0.7 + 0.6 * g), 0.06, 0.95);
  } else if (proc == 2) {
    float puddle = smoothstep(0.44, 0.56, fbm(P * 1.35 + vec3(3.1, 1.7, 0.0)));
    float grit = vnoise(P * 42.0);
    vec3 dry = mix(rA, rB * 0.0 + rA * 1.45, grit);
    base = mix(dry, vec3(0.012, 0.014, 0.020), puddle);
    rough = mix(0.84, 0.02, puddle);
    coat = mix(0.18, 1.0, puddle);
    coatRough = 0.015;
    mirror = mix(0.18, 1.0, puddle);
    vec2 q = P.xy * 3.0;
    vec2 cell = floor(q);
    vec2 fp = fract(q) - 0.5;
    float h = hash21(cell);
    float ph = fract(uTime * (0.55 + h * 0.6) + h * 7.0);
    vec2 c0 = (vec2(hash21(cell + 3.7), hash21(cell + 9.1)) - 0.5) * 0.5;
    vec2 dv = fp - c0;
    float r = length(dv);
    float ring = sin((r - ph * 0.5) * 70.0) * exp(-r * 9.0) * (1.0 - ph);
    vec2 grad = (dv / max(r, 1e-4)) * ring * 0.25 * uRainWet * (0.6 + uAudio.z * uAudioReactive * 1.4);
    N = normalize(N + vec3(grad * puddle, 0.0) + vec3((grit - 0.5) * 0.08 * (1.0 - puddle)));
  } else if (proc == 3) {
    float rings = fract(length(P.xz * 1.8) * 13.0 + fbm(P * vec3(1.8, 14.0, 1.8)) * 2.2);
    base = mix(rA, rB, smoothstep(0.15, 0.85, rings));
  } else if (proc == 4) {
    float n = fbm(P * 18.0);
    base = mix(rA, rB, smoothstep(0.3, 0.75, n));
  } else if (proc == 5) {
    vec3 an = abs(Ln);
    vec2 fc = an.x > an.y ? vec2(P.y, P.z) : vec2(P.x, P.z);
    vec2 g = fc * vec2(1.6, 1.15);
    vec2 cell = floor(g);
    vec2 f = fract(g);
    float pane = step(0.14, f.x) * step(f.x, 0.86) * step(0.18, f.y) * step(f.y, 0.82);
    float h = hash21(cell + vec2(floor(P.x * 0.1) * 13.0, 0.0));
    float roof = an.z > 0.8 ? 0.0 : 1.0;
    float lit = step(0.42, h) * roof;
    vec3 pal = h < 0.62 ? vec3(0.05, 0.85, 1.0) : (h < 0.84 ? vec3(1.0, 0.62, 0.18) : vec3(1.0, 0.1, 0.62));
    float flick = 0.78 + 0.22 * sin(uTime * (1.0 + h * 3.0) + h * 40.0);
    base = mix(vec3(0.035, 0.038, 0.046), vec3(0.015, 0.02, 0.026), pane * roof);
    emis = pal * pane * lit * (1.5 + 2.6 * hash21(cell + 5.0)) * flick;
    rough = mix(0.85, 0.1, pane * roof);
    metal = 0.0;
  } else if (proc == 6) {
    float scan = 0.5 + 0.5 * sin(P.z * 32.0 + uTime * 6.0);
    vec2 blk = floor(vec2(P.x + P.y, P.z) * 2.2);
    float glitch = step(0.86, hash21(blk + floor(uTime * 7.0)));
    vec3 tint = mix(rA, rB, clamp(hash21(blk) * 0.7 + glitch * 0.3, 0.0, 1.0));
    emis = tint * m1.a * (0.30 + 0.70 * scan) * (1.0 + glitch * 0.8);
    alpha = clamp(0.16 + 0.36 * scan + fres * 0.6 + glitch * 0.3, 0.0, 0.92);
    base = tint * 0.2;
    rough = 0.08;
    neonMask = max(neonMask, 0.6);
  } else if (proc == 7) {
    float rim = pow(1.0 - NoV, 1.6);
    vec3 c = mix(rA, rB, smoothstep(0.08, 0.78, rim));
    float s = m1.a * mix(1.0, 0.42, smoothstep(0.0, 1.0, fres * 3.0));
    emis = c * s;
    base = rB;
    rough = 0.06;
    neonMask = 1.0;
  } else if (proc == 8) {
    float ribs = 0.5 + 0.5 * sin(P.z * 113.0);
    float fib = fbm(P * 46.0);
    base = mix(rA, rB, smoothstep(0.2, 0.9, ribs * 0.7 + fib * 0.3));
    emis += base * (1.25 + 0.45 * ribs) * (1.0 + uAudio.y * uAudioReactive * 0.5);
    neonMask = max(neonMask, 0.35);
  } else if (proc == 9) {
    alpha = clamp(0.10 + fres * 0.75, 0.0, 0.85);
    rough = min(rough, 0.08);
    base = mix(base, vec3(0.6, 0.8, 1.0), 0.5);
    metal = 0.0;
  } else if (proc == 10) {
    float n = fbm(P * 6.0 + vec3(0.0, 0.0, -uTime * 0.8));
    float hgt = clamp(P.z / 0.7, 0.0, 1.0);
    alpha = clamp((0.05 + 0.32 * n) * (1.0 - hgt) * smoothstep(0.0, 0.08, P.z) * (0.4 + 0.6 * NoV), 0.0, 0.5);
    base = vec3(0.9, 0.88, 0.85);
    emis = vec3(1.0, 0.75, 0.5) * 0.25 * n;
    rough = 1.0;
  } else if (proc == 11) {
    vec2 g = (abs(Ln.y) > 0.5 ? P.xz : P.yz) * 90.0;
    vec2 cell = floor(g);
    vec2 f = fract(g) - 0.5;
    float dotm = smoothstep(0.42, 0.18, length(f));
    float on = step(0.45, hash21(cell + floor(uTime * 8.0) * 0.37));
    vec3 ec = length(m1.rgb) > 0.01 ? m1.rgb : rA;
    emis = ec * max(m1.a, 6.0) * dotm * (0.35 + 0.65 * on);
    neonMask = 1.0;
  } else if (proc == 12) {
    emis = m1.rgb * m1.a * (0.65 + 0.35 * sin(P.x * 3.0 - uTime * 18.0));
    alpha = 0.55;
    neonMask = 1.0;
  } else if (proc == 13) {
    float w = 0.5 + 0.5 * sin(P.x * 900.0) * sin(P.z * 900.0);
    base *= 0.85 + 0.25 * w * fbm(P * 30.0);
    rough = max(rough, 0.55);
  } else if (proc == 14) {
    base *= 0.85 + 0.3 * fbm(P * 25.0);
  }

  // Live audio -> GL coupling: sub-bass drives thruster plasma, mids drive neon signage
  if ((vFlags & 1u) != 0u) {
    float pulse = 0.5 + 0.5 * sin(uTime * 7.0);
    emis *= 0.65 + pulse * 0.55 + uAudio.x * uAudioReactive * 3.4;
  }
  if ((vFlags & 2u) != 0u) emis *= 1.0 + uAudio.y * uAudioReactive * 0.9;

  vec3 F0 = mix(vec3(0.04), base, metal);
  vec3 diffC = base * (1.0 - metal) * (1.0 - trans * 0.6);
  float a = max(rough * rough, 0.002);
  vec3 Lo = vec3(0.0);
  for (int i = 0; i < 16; i++) {
    if (i >= uLightCount) break;
    vec4 lp = uLightPos[i];
    vec4 lc = uLightCol[i];
    vec4 ld = uLightDir[i];
    vec4 la = uLightAux[i];
    vec3 Lv = lp.xyz - vWorld;
    float d2 = dot(Lv, Lv);
    vec3 L = Lv * inversesqrt(max(d2, 1e-8));
    float atten = 1.0 / (d2 + lc.w * lc.w);
    int ltype = int(lp.w + 0.5);
    float cosEm = dot(-L, ld.xyz);
    if (ltype == 1) atten *= smoothstep(ld.w, la.x, cosEm);
    else if (ltype == 2) atten *= max(cosEm, 0.0);
    float NoL = max(dot(N, L), 0.0);
    float wrap = (proc == 8 || proc == 13) ? max(dot(-N, L), 0.0) * 0.35 : 0.0;
    if (NoL <= 0.0 && wrap <= 0.0) continue;
    vec3 H = normalize(L + V);
    float NoH = max(dot(N, H), 0.0);
    float VoH = max(dot(V, H), 0.0);
    vec3 F = fresnelSchlick(F0, VoH);
    vec3 spec = distributionGGX(NoH, a) * visibilitySmith(max(NoV, 1e-3), max(NoL, 1e-3), a) * F;
    vec3 diff = (1.0 - F) * diffC / PI;
    vec3 rad = lc.rgb * atten;
    Lo += (diff + spec) * rad * NoL + diffC * rad * wrap / PI;
    if (coat > 0.0) {
      float ac = max(coatRough * coatRough, 0.002);
      float Fc = 0.04 + 0.96 * pow(1.0 - VoH, 5.0);
      Lo += distributionGGX(NoH, ac) * visibilitySmith(max(NoV, 1e-3), max(NoL, 1e-3), ac) * Fc * coat * rad * NoL;
    }
  }
  float hemi = N.z * 0.5 + 0.5;
  vec3 skyAmb = mix(vec3(0.020, 0.012, 0.030), uSkyTop * 6.0 + vec3(0.02, 0.05, 0.09), hemi);
  vec3 cityBounce = vec3(0.10, 0.035, 0.12) * (1.0 - hemi) + vec3(0.02, 0.08, 0.10) * hemi;
  vec3 amb = (skyAmb + cityBounce) * uAmbient;
  vec3 Fa = F0 + (max(vec3(1.0 - rough), F0) - F0) * fres;
  vec3 col = diffC * amb + Fa * amb * 0.6 + Lo + emis;
  if (mirror > 0.0 && uUseRefl == 1 && uMirrorPass == 0) {
    vec2 suv = (gl_FragCoord.xy - uViewport.xy) / uViewport.zw;
    suv += N.xy * 0.035;
    vec3 refl = textureLod(uReflTex, clamp(suv, vec2(0.001), vec2(0.999)), rough * 7.0).rgb;
    float Fw = 0.02 + 0.98 * pow(1.0 - NoV, 5.0);
    float k = mirror * mix(0.35, 1.0, Fw);
    col = col * (1.0 - k * 0.5) + refl * k;
  }
  float outA = isTransparent ? alpha : 1.0;
  oColor = vec4(col, outA);
  oNeon = vec4(emis * neonMask * 0.6 + max(emis - vec3(1.0), vec3(0.0)) * 0.08, outA);
  float dist = length(uCamPos - vWorld);
  oAux = vec4(N * 0.5 + 0.5, clamp(dist / uFar, 0.0, 0.998));
}
</script>

<script type="x-shader/x-fragment" id="fs-wire">#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec3 vWorld;
in vec3 vNor;
in vec3 vLocal;
in vec3 vLocalNor;
flat in uint vMat;
flat in uint vFlags;
uniform sampler2D uMatTex;
uniform vec3 uCamPos;
uniform float uFar;
uniform float uTime;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oNeon;
layout(location = 2) out vec4 oAux;
void main() {
  int mid = int(vMat);
  vec4 m0 = texelFetch(uMatTex, ivec2(0, mid), 0);
  vec4 m1 = texelFetch(uMatTex, ivec2(1, mid), 0);
  vec3 c = m1.a > 0.5 ? m1.rgb : mix(vec3(0.0, 0.85, 1.0), m0.rgb * 2.5 + 0.15, 0.35);
  float dist = length(uCamPos - vWorld);
  float fade = exp(-dist * 0.035);
  vec3 col = c * (0.25 + 1.2 * fade) * (m1.a > 0.5 ? 2.5 : 1.0);
  oColor = vec4(col, 1.0);
  oNeon = vec4(col * 0.35, 1.0);
  oAux = vec4(normalize(vNor) * 0.5 + 0.5, clamp(dist / uFar, 0.0, 0.998));
}
</script>

<script type="x-shader/x-vertex" id="vs-rain">#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNor;
uniform mat4 uViewProj;
uniform mat4 uEmitter[2];
uniform vec3 uFaces[24];
uniform int uFaceCount;
uniform int uPerEmitter;
uniform uint uSeed;
uniform vec3 uScaleMin;
uniform vec3 uScaleMax;
uniform float uRainPhase;
uniform float uWidth;
out float vFade;
out vec3 vWorld;
out float vTint;
uint hashu(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
float rnd(uint i, uint k) { return float(hashu(i * 7u + k * 2654435761u + uSeed)) / 4294967295.0; }
void main() {
  int id = gl_InstanceID;
  int sheet = id / uPerEmitter;
  int local = id - sheet * uPerEmitter;
  uint ui = uint(id) + 1u;
  int face = local % max(uFaceCount, 1);
  float u = rnd(ui, 1u);
  float v = rnd(ui, 2u);
  vec3 f0 = uFaces[face * 4 + 0];
  vec3 f1 = uFaces[face * 4 + 1];
  vec3 f2 = uFaces[face * 4 + 2];
  vec3 f3 = uFaces[face * 4 + 3];
  vec3 spawn = mix(mix(f0, f1, u), mix(f3, f2, u), v);
  vec4 sp = uEmitter[sheet] * vec4(spawn, 1.0);
  float topZ = max(sp.z, 0.5);
  float fall = mod(uRainPhase * (0.85 + 0.3 * rnd(ui, 3u)) + rnd(ui, 4u) * topZ, topZ);
  vec3 sc = mix(uScaleMin, uScaleMax, vec3(rnd(ui, 5u), rnd(ui, 6u), rnd(ui, 7u)));
  vec3 lp = aPos * vec3(sc.x * uWidth, sc.y * uWidth, sc.z);
  vec3 wp = vec3(sp.x, sp.y, topZ - fall) + lp;
  float z = topZ - fall;
  vFade = smoothstep(0.0, 0.6, z) * smoothstep(topZ, topZ - 1.0, z);
  vTint = rnd(ui, 8u);
  vWorld = wp;
  gl_Position = uViewProj * vec4(wp, 1.0);
}
</script>

<script type="x-shader/x-fragment" id="fs-rain">#version 300 es
precision highp float;
in float vFade;
in vec3 vWorld;
in float vTint;
uniform vec3 uCamPos;
uniform float uIntensity;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oNeon;
layout(location = 2) out vec4 oAux;
void main() {
  float d = length(uCamPos - vWorld);
  float near = smoothstep(0.4, 1.6, d) * exp(-d * 0.045);
  vec3 tint = mix(vec3(0.35, 0.85, 1.0), vec3(1.0, 0.35, 0.85), step(0.72, vTint));
  vec3 c = tint * 0.55 * vFade * near * uIntensity;
  oColor = vec4(c, 1.0);
  oNeon = vec4(c * 0.25, 1.0);
  oAux = vec4(0.0);
}
</script>

<script type="x-shader/x-vertex" id="vs-lines">#version 300 es
precision highp float;
layout(location = 0) in vec3 aPos;
uniform mat4 uViewProj;
uniform mat4 uModel;
void main() { gl_Position = uViewProj * (uModel * vec4(aPos, 1.0)); }
</script>

<script type="x-shader/x-fragment" id="fs-lines">#version 300 es
precision highp float;
uniform vec4 uColor;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oNeon;
layout(location = 2) out vec4 oAux;
void main() { oColor = uColor; oNeon = vec4(uColor.rgb * 0.4, uColor.a); oAux = vec4(0.5, 0.5, 1.0, 0.5); }
</script>

<script type="x-shader/x-vertex" id="vs-post">#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
</script>

<script type="x-shader/x-fragment" id="fs-sky">#version 300 es
precision highp float;
in vec2 vUv;
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;
uniform vec3 uSkyTop;
uniform vec3 uSkyBottom;
uniform vec3 uFogColor;
uniform float uSkyStrength;
uniform float uTime;
layout(location = 0) out vec4 oColor;
layout(location = 1) out vec4 oNeon;
layout(location = 2) out vec4 oAux;
//#include common
void main() {
  vec4 wp = uInvViewProj * vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
  vec3 dir = normalize(wp.xyz / wp.w - uCamPos);
  float h = dir.z;
  vec3 sky = mix(uSkyBottom, uSkyTop, smoothstep(-0.2, 0.6, h)) * uSkyStrength * 8.0;
  sky += vec3(0.30, 0.07, 0.36) * exp(-abs(h) * 6.0) * 0.10;
  sky += vec3(0.04, 0.24, 0.34) * exp(-abs(h - 0.05) * 14.0) * 0.06;
  vec2 cloudUv = dir.xy / max(0.18, dir.z + 0.35);
  sky += uFogColor * 0.025 * fbm(vec3(cloudUv * 1.4, uTime * 0.03)) * smoothstep(-0.1, 0.3, h);
  oColor = vec4(sky, 1.0);
  oNeon = vec4(0.0, 0.0, 0.0, 1.0);
  oAux = vec4(0.5, 0.5, 1.0, 1.0);
}
</script>

<script type="x-shader/x-fragment" id="fs-bright">#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uColor;
uniform sampler2D uNeon;
uniform vec2 uTexel;
uniform float uThreshold;
uniform float uKnee;
uniform float uNeonGain;
layout(location = 0) out vec4 oColor;
void main() {
  vec3 c = vec3(0.0);
  vec3 n = vec3(0.0);
  c += texture(uColor, vUv + uTexel * vec2(-0.5, -0.5)).rgb;
  c += texture(uColor, vUv + uTexel * vec2(0.5, -0.5)).rgb;
  c += texture(uColor, vUv + uTexel * vec2(-0.5, 0.5)).rgb;
  c += texture(uColor, vUv + uTexel * vec2(0.5, 0.5)).rgb;
  n += texture(uNeon, vUv + uTexel * vec2(-0.5, -0.5)).rgb;
  n += texture(uNeon, vUv + uTexel * vec2(0.5, 0.5)).rgb;
  c *= 0.25;
  n *= 0.5;
  float br = max(max(c.r, c.g), c.b);
  float soft = clamp(br - uThreshold + uKnee, 0.0, 2.0 * uKnee);
  soft = soft * soft / (4.0 * uKnee + 1e-4);
  float contrib = max(soft, br - uThreshold) / max(br, 1e-4);
  vec3 b = c * contrib + n * uNeonGain;
  oColor = vec4(min(b, vec3(48.0)), 1.0);
}
</script>

<script type="x-shader/x-fragment" id="fs-blur">#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uDir;
uniform vec2 uTexel;
uniform float uRadius;
layout(location = 0) out vec4 oColor;
void main() {
  vec2 st = uDir * uTexel * uRadius;
  vec3 c = texture(uSrc, vUv).rgb * 0.2270270;
  c += texture(uSrc, vUv + st * 1.3846154).rgb * 0.3162162;
  c += texture(uSrc, vUv - st * 1.3846154).rgb * 0.3162162;
  c += texture(uSrc, vUv + st * 3.2307692).rgb * 0.0702703;
  c += texture(uSrc, vUv - st * 3.2307692).rgb * 0.0702703;
  oColor = vec4(c, 1.0);
}
</script>

<script type="x-shader/x-fragment" id="fs-streak">#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uDir;
uniform vec2 uTexel;
uniform float uStep;
uniform float uFade;
uniform float uThreshold;
layout(location = 0) out vec4 oColor;
void main() {
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for (int k = -8; k <= 8; k++) {
    float w = pow(uFade, abs(float(k)));
    vec3 s = texture(uSrc, vUv + uDir * uTexel * uStep * float(k)).rgb;
    acc += max(s - vec3(uThreshold), vec3(0.0)) * w;
    wsum += w;
  }
  oColor = vec4(acc / wsum * 2.2, 1.0);
}
</script>

<script type="x-shader/x-fragment" id="fs-composite">#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 vUv;
uniform sampler2D uColor;
uniform sampler2D uNeon;
uniform sampler2D uAux;
uniform sampler2D uBloomA;
uniform sampler2D uBloomB;
uniform sampler2D uStreak;
uniform int uAov;
uniform float uExposure;
uniform float uBloomStrength;
uniform float uStreakStrength;
uniform float uChroma;
uniform float uDistort;
uniform float uMistOn;
uniform float uMistStart;
uniform float uMistDepth;
uniform float uMistMix;
uniform float uFar;
uniform float uTime;
uniform float uGrain;
uniform float uVignette;
uniform vec3 uMistTint;
uniform vec3 uFogColor;
uniform vec3 uLift;
uniform vec3 uGamma;
uniform vec3 uGain;
uniform vec2 uResolution;
layout(location = 0) out vec4 fragColor;
//#include common

// AgX (Troy Sobotka) polynomial fit -- Blender 4.x "AgX" view transform + "High Contrast" look
vec3 agxContrast(vec3 x) {
  vec3 x2 = x * x;
  vec3 x4 = x2 * x2;
  return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}
vec3 agx(vec3 v) {
  const mat3 inset = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                          0.0784335999999992, 0.878468636469772, 0.0784336,
                          0.0792237451477643, 0.0791661274605434, 0.879142973793104);
  const float lo = -12.47393;
  const float hi = 4.026069;
  v = inset * v;
  v = clamp(log2(max(v, vec3(1e-10))), lo, hi);
  v = (v - lo) / (hi - lo);
  return agxContrast(v);
}
vec3 agxEotf(vec3 v) {
  const mat3 outset = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                           -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                           -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
  return outset * v;
}
vec3 agxHighContrast(vec3 v) {
  float luma = dot(v, vec3(0.2126, 0.7152, 0.0722));
  v = pow(max(v, vec3(0.0)), vec3(1.35));
  return luma + 1.12 * (v - luma);
}
vec3 agxFinal(vec3 hdr) { return agxEotf(agxHighContrast(agx(hdr))); }

void main() {
  vec2 dc = vUv - 0.5;
  float r2 = dot(dc, dc);
  vec2 uv = 0.5 + dc * (1.0 + uDistort * r2 * 4.0);
  vec2 ca = dc * uChroma * (0.6 + r2 * 3.0);
  vec3 col = vec3(texture(uColor, uv + ca).r, texture(uColor, uv).g, texture(uColor, uv - ca).b);
  vec4 aux = texture(uAux, uv);
  bool sky = aux.a > 0.999;
  float dist = aux.a * uFar;
  float mist = 1.0;
  if (!sky) { mist = clamp((dist - uMistStart) / max(uMistDepth, 1e-3), 0.0, 1.0); mist *= mist; }
  vec3 bloom = texture(uBloomA, uv).rgb + texture(uBloomB, uv).rgb * 0.85;
  vec3 streak = texture(uStreak, uv).rgb;
  vec3 hdr = col;
  if (uMistOn > 0.5 && !sky) {
    hdr = mix(hdr, uFogColor * 0.05 + uMistTint * 0.5, mist * 0.6) + uMistTint * mist * uMistMix * 0.6;
  }
  hdr += bloom * uBloomStrength + streak * uStreakStrength;
  float ex = exp2(uExposure);
  vec3 outc;
  if (uAov == 1) {
    vec3 ne = texture(uNeon, uv).rgb * 1.6 + bloom * uBloomStrength + streak * uStreakStrength;
    outc = agxFinal(ne * ex);
  } else if (uAov == 2) {
    float d = sky ? 1.0 : clamp(dist / 60.0, 0.0, 1.0);
    vec3 g = mix(vec3(1.0, 0.96, 0.9), vec3(0.015, 0.025, 0.05), sqrt(d));
    outc = mix(g, vec3(0.22, 0.42, 0.66), mist * (uMistOn > 0.5 ? 0.65 : 0.0));
  } else if (uAov == 3) {
    outc = sky ? vec3(0.02) : aux.rgb;
  } else {
    outc = agxFinal(hdr * ex);
  }
  if (uAov == 0 || uAov == 4) {
    outc = clamp(outc, 0.0, 1.0);
    vec3 liftOff = (uLift - 1.0) * 0.5;
    outc = uGain * (outc + liftOff * (1.0 - outc));
    outc = pow(max(outc, vec3(0.0)), 1.0 / uGamma);
  }
  float vig = smoothstep(0.95, 0.22, length(dc * vec2(1.25, 1.0)));
  outc *= mix(1.0, vig, uVignette);
  outc += (hash21(vUv * uResolution + fract(uTime * 7.31) * 91.0) - 0.5) * uGrain;
  fragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
</script>
"""

HTML_TEMPLATE += r"""
<!-- ===================== EMBEDDED BINARY PAYLOADS ===================== -->
<script type="application/octet-stream" id="scene-payload" data-encoding="cpks+shuffle4+zlib+base64" data-sha256="@@SCENE_SHA256@@" data-raw-bytes="@@SCENE_RAW_BYTES@@" data-zlib-bytes="@@SCENE_ZLIB_BYTES@@">@@SCENE_B64@@</script>
<script type="application/octet-stream" id="audio-payload" data-mime="audio/wav" data-encoding="base64" data-sha256="@@AUDIO_SHA256@@" data-bytes="@@AUDIO_BYTES@@">@@AUDIO_B64@@</script>
<script type="application/json" id="studio-build">@@BUILD_INFO@@</script>

<!-- ===================== STUDIO ENGINE ===================== -->
<script>
(function () {
'use strict';

// ---------------------------------------------------------------------------
// 0. Utilities, diagnostics, loader
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const FPS = 24;
const STUDIO = window.__studio = { ready: false, errors: [], warnings: [], stats: {}, frame: 1, phase: '8A/8B' };
function report(err) {
  const msg = String((err && err.stack) || err);
  STUDIO.errors.push(msg);
  console.error(err);
  toast('ERROR: ' + msg.split('\n')[0], 6000);
}
function warn(msg) { STUDIO.warnings.push(String(msg)); console.warn(msg); }
window.addEventListener('error', (e) => report(e.error || e.message));
window.addEventListener('unhandledrejection', (e) => report(e.reason));
let toastTimer = 0;
function toast(msg, ms) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), ms || 2200);
}
const ldLines = [];
function loaderStep(text, frac, cls) {
  ldLines.push('<span class="' + (cls || '') + '">' + text + '</span>');
  $('ld-steps').innerHTML = ldLines.slice(-9).join('\n');
  if (frac !== undefined) $('ld-fill').style.width = Math.round(frac * 100) + '%';
}
function fatal(msg) {
  const f = $('fatal');
  f.textContent = msg;
  f.classList.add('show');
  STUDIO.errors.push(msg);
}
const yieldUI = () => new Promise((r) => setTimeout(r, 0));
const clamp = (x, a, b) => (x < a ? a : (x > b ? b : x));
const fmtBytes = (n) => (n > 1048576 ? (n / 1048576).toFixed(2) + ' MB' : (n / 1024).toFixed(1) + ' KB');

// ---------------------------------------------------------------------------
// 1. Base64 -> bytes, zlib inflate (DecompressionStream or pure-JS), SHA-256
// ---------------------------------------------------------------------------
function b64ToBytes(b64) {
  b64 = b64.replace(/\s+/g, '');
  if (typeof Uint8Array.fromBase64 === 'function') return Uint8Array.fromBase64(b64);
  const bin = atob(b64);
  const n = bin.length;
  const out = new Uint8Array(n);
  for (let i = 0; i < n; i++) out[i] = bin.charCodeAt(i);
  return out;
}

const LBASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LBITS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DBASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DBITS = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
const CLORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
function huffTable(lengths, n) {
  const counts = new Uint16Array(16);
  const offs = new Uint16Array(16);
  const symbols = new Uint16Array(n);
  for (let i = 0; i < n; i++) counts[lengths[i]]++;
  counts[0] = 0;
  let total = 0;
  for (let i = 0; i < 16; i++) { offs[i] = total; total += counts[i]; }
  for (let i = 0; i < n; i++) if (lengths[i]) symbols[offs[lengths[i]]++] = i;
  return { counts: counts, symbols: symbols };
}
function jsInflateZlib(src) {
  const cmf = src[0], flg = src[1];
  if ((cmf & 15) !== 8 || ((cmf << 8) | flg) % 31 !== 0) throw new Error('bad zlib header');
  let pos = 2, bitBuf = 0, bitCnt = 0;
  let out = new Uint8Array(Math.max(1024, src.length * 5));
  let outLen = 0;
  function ensure(n) {
    if (outLen + n <= out.length) return;
    let m = out.length * 2;
    while (m < outLen + n) m *= 2;
    const o = new Uint8Array(m);
    o.set(out.subarray(0, outLen));
    out = o;
  }
  function bits(n) {
    while (bitCnt < n) { bitBuf |= src[pos++] << bitCnt; bitCnt += 8; }
    const v = bitBuf & ((1 << n) - 1);
    bitBuf >>>= n;
    bitCnt -= n;
    return v;
  }
  function decodeSym(t) {
    let base = 0, offs = 0;
    for (let len = 1; len < 16; len++) {
      offs = 2 * offs + bits(1);
      const c = t.counts[len];
      if (offs < c) return t.symbols[base + offs];
      base += c;
      offs -= c;
    }
    throw new Error('bad huffman code');
  }
  const fl = new Uint8Array(288);
  for (let i = 0; i < 144; i++) fl[i] = 8;
  for (let i = 144; i < 256; i++) fl[i] = 9;
  for (let i = 256; i < 280; i++) fl[i] = 7;
  for (let i = 280; i < 288; i++) fl[i] = 8;
  const fixedL = huffTable(fl, 288);
  const fixedD = huffTable(new Uint8Array(30).fill(5), 30);
  let last = 0;
  do {
    last = bits(1);
    const type = bits(2);
    if (type === 0) {
      bitBuf = 0; bitCnt = 0;
      const len = src[pos] | (src[pos + 1] << 8);
      pos += 4;
      ensure(len);
      out.set(src.subarray(pos, pos + len), outLen);
      outLen += len;
      pos += len;
    } else if (type === 1 || type === 2) {
      let lt = fixedL, dt = fixedD;
      if (type === 2) {
        const hlit = bits(5) + 257, hdist = bits(5) + 1, hclen = bits(4) + 4;
        const cl = new Uint8Array(19);
        for (let i = 0; i < hclen; i++) cl[CLORDER[i]] = bits(3);
        const clt = huffTable(cl, 19);
        const lens = new Uint8Array(hlit + hdist);
        for (let n = 0; n < hlit + hdist;) {
          const sym = decodeSym(clt);
          if (sym < 16) lens[n++] = sym;
          else if (sym === 16) { const prev = lens[n - 1]; let r = 3 + bits(2); while (r--) lens[n++] = prev; }
          else if (sym === 17) { let r = 3 + bits(3); while (r--) lens[n++] = 0; }
          else { let r = 11 + bits(7); while (r--) lens[n++] = 0; }
        }
        lt = huffTable(lens.subarray(0, hlit), hlit);
        dt = huffTable(lens.subarray(hlit), hdist);
      }
      for (;;) {
        const sym = decodeSym(lt);
        if (sym < 256) { ensure(1); out[outLen++] = sym; }
        else if (sym === 256) break;
        else {
          const li = sym - 257;
          const len = LBASE[li] + bits(LBITS[li]);
          const ds = decodeSym(dt);
          const dist = DBASE[ds] + bits(DBITS[ds]);
          ensure(len);
          for (let k = 0; k < len; k++) { out[outLen] = out[outLen - dist]; outLen++; }
        }
      }
    } else throw new Error('bad deflate block type');
  } while (!last);
  return out.slice(0, outLen);
}
async function inflateZlib(bytes) {
  if (typeof DecompressionStream === 'function' && params.get('inflate') !== 'js') {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
    const buf = await new Response(stream).arrayBuffer();
    return { data: new Uint8Array(buf), method: 'DecompressionStream' };
  }
  return { data: jsInflateZlib(bytes), method: 'js-inflate' };
}

function primeConstants(count, root) {
  const out = new Uint32Array(count);
  let n = 2, k = 0;
  while (k < count) {
    let prime = true;
    for (let d = 2; d * d <= n; d++) if (n % d === 0) { prime = false; break; }
    if (prime) { const r = Math.pow(n, 1 / root); out[k++] = Math.floor((r - Math.floor(r)) * 4294967296) >>> 0; }
    n++;
  }
  return out;
}
const SHA_K = primeConstants(64, 3);
const SHA_H0 = primeConstants(8, 2);
function sha256Js(bytes) {
  const H = new Uint32Array(SHA_H0);
  const len = bytes.length;
  const padded = new Uint8Array(((len + 9 + 63) >> 6) << 6);
  padded.set(bytes);
  padded[len] = 0x80;
  const dv = new DataView(padded.buffer);
  dv.setUint32(padded.length - 8, Math.floor(len / 0x20000000));
  dv.setUint32(padded.length - 4, (len << 3) >>> 0);
  const W = new Uint32Array(64);
  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) W[i] = dv.getUint32(off + i * 4);
    for (let i = 16; i < 64; i++) {
      const w15 = W[i - 15], w2 = W[i - 2];
      const s0 = ((w15 >>> 7) | (w15 << 25)) ^ ((w15 >>> 18) | (w15 << 14)) ^ (w15 >>> 3);
      const s1 = ((w2 >>> 17) | (w2 << 15)) ^ ((w2 >>> 19) | (w2 << 13)) ^ (w2 >>> 10);
      W[i] = (W[i - 16] + s0 + W[i - 7] + s1) >>> 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (let i = 0; i < 64; i++) {
      const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
      const ch = (e & f) ^ (~e & g);
      const t1 = (h + S1 + ch + SHA_K[i] + W[i]) >>> 0;
      const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
      const mj = (a & b) ^ (a & c) ^ (b & c);
      const t2 = (S0 + mj) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0; H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0; H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
  }
  return Array.from(H, (x) => x.toString(16).padStart(8, '0')).join('');
}
async function sha256Hex(bytes) {
  if (window.crypto && crypto.subtle && params.get('sha') !== 'js') {
    try {
      const d = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
      return { hex: Array.from(d, (x) => x.toString(16).padStart(2, '0')).join(''), method: 'WebCrypto' };
    } catch (e) { warn('WebCrypto digest unavailable: ' + e); }
  }
  return { hex: sha256Js(bytes), method: 'js-sha256' };
}
function unshuffle4(bytes) {
  const n = bytes.length >> 2;
  const out = new Uint8Array(bytes.length);
  for (let b = 0; b < 4; b++) {
    const base = b * n;
    for (let i = 0; i < n; i++) out[i * 4 + b] = bytes[base + i];
  }
  return out;
}
function parseContainer(u8) {
  if (u8[0] !== 67 || u8[1] !== 80 || u8[2] !== 75 || u8[3] !== 83) throw new Error('bad CPKS magic');
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const version = dv.getUint32(4, true);
  const jlen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + jlen)));
  const start = (12 + jlen + 3) & ~3;
  const S = {};
  for (const name of Object.keys(header.sections)) {
    const m = header.sections[name];
    if (start + m.offset + m.length > u8.length) throw new Error('section out of bounds: ' + name);
    let bytes = u8.subarray(start + m.offset, start + m.offset + m.length);
    bytes = m.shuffle === 4 ? unshuffle4(bytes) : bytes.slice();
    if (m.type === 'f32') S[name] = new Float32Array(bytes.buffer, 0, m.length >> 2);
    else if (m.type === 'u32') S[name] = new Uint32Array(bytes.buffer, 0, m.length >> 2);
    else S[name] = bytes;
  }
  return { version: version, header: header, S: S };
}

// ---------------------------------------------------------------------------
// 2. Column-major 4x4 math
// ---------------------------------------------------------------------------
const M4 = {
  create() { const m = new Float32Array(16); m[0] = m[5] = m[10] = m[15] = 1; return m; },
  identity(m) { m.fill(0); m[0] = m[5] = m[10] = m[15] = 1; return m; },
  mul(out, a, b) {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3], a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11], a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    for (let j = 0; j < 4; j++) {
      const b0 = b[j * 4], b1 = b[j * 4 + 1], b2 = b[j * 4 + 2], b3 = b[j * 4 + 3];
      out[j * 4] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
      out[j * 4 + 1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
      out[j * 4 + 2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
      out[j * 4 + 3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
    }
    return out;
  },
  mulOff(out, oo, a, ao, b, bo) {
    for (let j = 0; j < 4; j++) {
      const b0 = b[bo + j * 4], b1 = b[bo + j * 4 + 1], b2 = b[bo + j * 4 + 2], b3 = b[bo + j * 4 + 3];
      for (let i = 0; i < 4; i++) out[oo + j * 4 + i] = a[ao + i] * b0 + a[ao + 4 + i] * b1 + a[ao + 8 + i] * b2 + a[ao + 12 + i] * b3;
    }
  },
  invert(out, a) {
    const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3], a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
    const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11], a30 = a[12], a31 = a[13], a32 = a[14], a33 = a[15];
    const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (!det) return M4.identity(out);
    det = 1.0 / det;
    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
    return out;
  },
  perspective(out, fovy, aspect, near, far) {
    const f = 1.0 / Math.tan(fovy / 2), nf = 1 / (near - far);
    out.fill(0);
    out[0] = f / aspect; out[5] = f; out[10] = (far + near) * nf; out[11] = -1; out[14] = 2 * far * near * nf;
    return out;
  },
  lookAt(out, eye, c, up) {
    let z0 = eye[0] - c[0], z1 = eye[1] - c[1], z2 = eye[2] - c[2];
    let len = Math.hypot(z0, z1, z2) || 1;
    z0 /= len; z1 /= len; z2 /= len;
    let x0 = up[1] * z2 - up[2] * z1, x1 = up[2] * z0 - up[0] * z2, x2 = up[0] * z1 - up[1] * z0;
    len = Math.hypot(x0, x1, x2) || 1;
    x0 /= len; x1 /= len; x2 /= len;
    const y0 = z1 * x2 - z2 * x1, y1 = z2 * x0 - z0 * x2, y2 = z0 * x1 - z1 * x0;
    out[0] = x0; out[1] = y0; out[2] = z0; out[3] = 0;
    out[4] = x1; out[5] = y1; out[6] = z1; out[7] = 0;
    out[8] = x2; out[9] = y2; out[10] = z2; out[11] = 0;
    out[12] = -(x0 * eye[0] + x1 * eye[1] + x2 * eye[2]);
    out[13] = -(y0 * eye[0] + y1 * eye[1] + y2 * eye[2]);
    out[14] = -(z0 * eye[0] + z1 * eye[1] + z2 * eye[2]);
    out[15] = 1;
    return out;
  },
};
function m34ToM4(src, so, out, oo) {
  out[oo] = src[so]; out[oo + 1] = src[so + 4]; out[oo + 2] = src[so + 8]; out[oo + 3] = 0;
  out[oo + 4] = src[so + 1]; out[oo + 5] = src[so + 5]; out[oo + 6] = src[so + 9]; out[oo + 7] = 0;
  out[oo + 8] = src[so + 2]; out[oo + 9] = src[so + 6]; out[oo + 10] = src[so + 10]; out[oo + 11] = 0;
  out[oo + 12] = src[so + 3]; out[oo + 13] = src[so + 7]; out[oo + 14] = src[so + 11]; out[oo + 15] = 1;
}
function normalMat3(m, mo, out, oo) {
  const a00 = m[mo], a01 = m[mo + 1], a02 = m[mo + 2];
  const a10 = m[mo + 4], a11 = m[mo + 5], a12 = m[mo + 6];
  const a20 = m[mo + 8], a21 = m[mo + 9], a22 = m[mo + 10];
  const b01 = a22 * a11 - a12 * a21, b11 = -a22 * a10 + a12 * a20, b21 = a21 * a10 - a11 * a20;
  let det = a00 * b01 + a01 * b11 + a02 * b21;
  det = det ? 1 / det : 1;
  const i0 = b01 * det, i1 = (-a22 * a01 + a02 * a21) * det, i2 = (a12 * a01 - a02 * a11) * det;
  const i3 = b11 * det, i4 = (a22 * a00 - a02 * a20) * det, i5 = (-a12 * a00 + a02 * a10) * det;
  const i6 = b21 * det, i7 = (-a21 * a00 + a01 * a20) * det, i8 = (a11 * a00 - a01 * a10) * det;
  out[oo] = i0; out[oo + 1] = i3; out[oo + 2] = i6;
  out[oo + 3] = i1; out[oo + 4] = i4; out[oo + 5] = i7;
  out[oo + 6] = i2; out[oo + 7] = i5; out[oo + 8] = i8;
}
const IDENT4 = M4.create();
const IDENT3 = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
const MIRROR_Z = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]);
const DEFAULT_VIDEO_NAME = '@@VIDEO_NAME@@';

// ---------------------------------------------------------------------------
// 3. WebGL2 Deferred/Forward+ Shader Engine & Multi-Pass Framebuffer Pipeline
// ---------------------------------------------------------------------------
function createRenderer(canvas, scene) {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    depth: true,
    stencil: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
    powerPreference: 'high-performance',
  });
  if (!gl) throw new Error('WebGL2 context unavailable');

  const extFloatFBO = gl.getExtension('EXT_color_buffer_float');
  gl.getExtension('OES_texture_float_linear');
  const dbgInfo = gl.getExtension('WEBGL_debug_renderer_info');
  const gpuName = dbgInfo ? gl.getParameter(dbgInfo.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
  $('st-gpu').textContent = String(gpuName || 'WebGL2').replace(/ANGLE \(([^,]+), ([^,()]+).*\)/, '$2');

  const commonSrc = $('glsl-common').textContent;
  function compileShader(type, id) {
    const raw = $(id).textContent.replace('//#include common', commonSrc);
    const sh = gl.createShader(type);
    gl.shaderSource(sh, raw);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error('Shader ' + id + ' compile failed: ' + log);
    }
    return sh;
  }
  function linkProgram(vsId, fsId, uniformNames) {
    const vs = compileShader(gl.VERTEX_SHADER, vsId);
    const fs = compileShader(gl.FRAGMENT_SHADER, fsId);
    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      throw new Error('Program ' + vsId + '+' + fsId + ' link failed: ' + log);
    }
    const u = {};
    for (let i = 0; i < uniformNames.length; i++) {
      const name = uniformNames[i];
      u[name] = gl.getUniformLocation(prog, name);
      if (u[name] === null) u[name] = gl.getUniformLocation(prog, name + '[0]');
    }
    return { prog: prog, u: u };
  }

  const progScene = linkProgram('vs-scene', 'fs-scene', [
    'uViewProj', 'uModel', 'uNormalMat', 'uBones', 'uSkinned', 'uWave', 'uWaveNormal', 'uFrameF',
    'uMatTex', 'uReflTex', 'uUseRefl', 'uMirrorPass', 'uFlipFacing', 'uViewport', 'uCamPos',
    'uTime', 'uFar', 'uLightCount', 'uLightPos', 'uLightCol', 'uLightDir', 'uLightAux',
    'uSkyTop', 'uSkyBottom', 'uAmbient', 'uAudio', 'uAudioReactive', 'uRainWet'
  ]);
  const progWire = linkProgram('vs-scene', 'fs-wire', [
    'uViewProj', 'uModel', 'uNormalMat', 'uBones', 'uSkinned', 'uWave', 'uWaveNormal', 'uFrameF',
    'uMatTex', 'uCamPos', 'uFar', 'uTime'
  ]);
  const progRain = linkProgram('vs-rain', 'fs-rain', [
    'uViewProj', 'uEmitter', 'uFaces', 'uFaceCount', 'uPerEmitter', 'uSeed',
    'uScaleMin', 'uScaleMax', 'uRainPhase', 'uWidth', 'uCamPos', 'uIntensity'
  ]);
  const progLines = linkProgram('vs-lines', 'fs-lines', ['uViewProj', 'uModel', 'uColor']);
  const progSky = linkProgram('vs-post', 'fs-sky', [
    'uInvViewProj', 'uCamPos', 'uSkyTop', 'uSkyBottom', 'uFogColor', 'uSkyStrength', 'uTime'
  ]);
  const progBright = linkProgram('vs-post', 'fs-bright', [
    'uColor', 'uNeon', 'uTexel', 'uThreshold', 'uKnee', 'uNeonGain'
  ]);
  const progBlur = linkProgram('vs-post', 'fs-blur', ['uSrc', 'uDir', 'uTexel', 'uRadius']);
  const progStreak = linkProgram('vs-post', 'fs-streak', [
    'uSrc', 'uDir', 'uTexel', 'uStep', 'uFade', 'uThreshold'
  ]);
  const progComposite = linkProgram('vs-post', 'fs-composite', [
    'uColor', 'uNeon', 'uAux', 'uBloomA', 'uBloomB', 'uStreak', 'uAov', 'uExposure',
    'uBloomStrength', 'uStreakStrength', 'uChroma', 'uDistort', 'uMistOn', 'uMistStart',
    'uMistDepth', 'uMistMix', 'uFar', 'uTime', 'uGrain', 'uVignette', 'uMistTint',
    'uFogColor', 'uLift', 'uGamma', 'uGain', 'uResolution'
  ]);

  const H = scene.header;
  const S = scene.S;

  // Geometry VAO (positions loc 0, normals loc 1, uvec4 attribs loc 2, Uint32 indices)
  const vaoScene = gl.createVertexArray();
  gl.bindVertexArray(vaoScene);

  const bufPos = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPos);
  gl.bufferData(gl.ARRAY_BUFFER, S.positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  const bufNor = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufNor);
  gl.bufferData(gl.ARRAY_BUFFER, S.normals, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

  const bufAttr = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufAttr);
  gl.bufferData(gl.ARRAY_BUFFER, S.attribs, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribIPointer(2, 4, gl.UNSIGNED_BYTE, 0, 0);

  const bufIdx = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bufIdx);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, S.indices, gl.STATIC_DRAW);

  // Wireframe edge index buffer (built once from triangle indices)
  const triCount = S.indices.length / 3;
  const wireIndices = new Uint32Array(triCount * 6);
  for (let t = 0; t < triCount; t++) {
    const i0 = S.indices[t * 3], i1 = S.indices[t * 3 + 1], i2 = S.indices[t * 3 + 2];
    const b = t * 6;
    wireIndices[b] = i0; wireIndices[b + 1] = i1;
    wireIndices[b + 2] = i1; wireIndices[b + 3] = i2;
    wireIndices[b + 4] = i2; wireIndices[b + 5] = i0;
  }
  const vaoWire = gl.createVertexArray();
  gl.bindVertexArray(vaoWire);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPos);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufNor);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufAttr);
  gl.enableVertexAttribArray(2);
  gl.vertexAttribIPointer(2, 4, gl.UNSIGNED_BYTE, 0, 0);
  const bufWireIdx = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bufWireIdx);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, wireIndices, gl.STATIC_DRAW);

  // Path curves VAO
  const vaoLines = gl.createVertexArray();
  gl.bindVertexArray(vaoLines);
  const bufLines = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, bufLines);
  gl.bufferData(gl.ARRAY_BUFFER, S.lines, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

  // Empty VAO for full-screen post-process triangles
  const vaoPost = gl.createVertexArray();
  gl.bindVertexArray(null);

  // Material lookup texture (RGBA32F, width = 6 texels, height = numMaterials)
  const numMaterials = H.counts.materials;
  const matTableBase = new Float32Array(S.material_table);
  const matTableLive = new Float32Array(S.material_table);
  const texMat = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texMat);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, H.material_texels, numMaterials, 0, gl.RGBA, gl.FLOAT, matTableLive);

  function updateMaterialTex() {
    gl.bindTexture(gl.TEXTURE_2D, texMat);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, H.material_texels, numMaterials, gl.RGBA, gl.FLOAT, matTableLive);
  }

  // Multi-Pass Framebuffer Objects (allocated & resized dynamically to viewport)
  const hdrIntFmt = extFloatFBO ? gl.RGBA16F : gl.RGBA8;
  const hdrType = extFloatFBO ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE;
  let fboW = 0, fboH = 0;
  let fboMain = null, texMainColor = null, texMainNeon = null, texMainAux = null, rbMainDepth = null;
  let fboMirror = null, texMirror = null, rbMirrorDepth = null;
  let fboBloomA0 = null, texBloomA0 = null, fboBloomA1 = null, texBloomA1 = null;
  let fboBloomB0 = null, texBloomB0 = null, fboBloomB1 = null, texBloomB1 = null;
  let fboStreak0 = null, texStreak0 = null, fboStreak1 = null, texStreak1 = null;

  function makeColorTex(w, h, mipmap) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mipmap ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, hdrIntFmt, w, h, 0, gl.RGBA, hdrType, null);
    if (mipmap) gl.generateMipmap(gl.TEXTURE_2D);
    return tex;
  }
  function makeSingleFbo(w, h, withDepth, mipmap) {
    const tex = makeColorTex(w, h, mipmap);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    let rb = null;
    if (withDepth) {
      rb = gl.createRenderbuffer();
      gl.bindRenderbuffer(gl.RENDERBUFFER, rb);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rb);
    }
    return { fbo: fbo, tex: tex, rb: rb, w: w, h: h };
  }
  function destroyTargets() {
    const texs = [texMainColor, texMainNeon, texMainAux, texMirror, texBloomA0, texBloomA1, texBloomB0, texBloomB1, texStreak0, texStreak1];
    const fbos = [fboMain, fboMirror, fboBloomA0, fboBloomA1, fboBloomB0, fboBloomB1, fboStreak0, fboStreak1];
    const rbs = [rbMainDepth, rbMirrorDepth];
    for (let i = 0; i < texs.length; i++) if (texs[i]) gl.deleteTexture(texs[i]);
    for (let i = 0; i < fbos.length; i++) if (fbos[i]) gl.deleteFramebuffer(fbos[i]);
    for (let i = 0; i < rbs.length; i++) if (rbs[i]) gl.deleteRenderbuffer(rbs[i]);
  }
  function ensureTargets(w, h) {
    w = Math.max(64, w | 0);
    h = Math.max(64, h | 0);
    if (w === fboW && h === fboH && fboMain) return;
    destroyTargets();
    fboW = w;
    fboH = h;

    texMainColor = makeColorTex(w, h, false);
    texMainNeon = makeColorTex(w, h, false);
    texMainAux = makeColorTex(w, h, false);
    rbMainDepth = gl.createRenderbuffer();
    gl.bindRenderbuffer(gl.RENDERBUFFER, rbMainDepth);
    gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, w, h);

    fboMain = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboMain);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texMainColor, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, texMainNeon, 0);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, texMainAux, 0);
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, rbMainDepth);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);

    const mw = Math.max(32, w >> 1), mh = Math.max(32, h >> 1);
    const mir = makeSingleFbo(mw, mh, true, true);
    fboMirror = mir.fbo; texMirror = mir.tex; rbMirrorDepth = mir.rb;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fboMirror);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.NONE, gl.NONE]);

    const bw = Math.max(32, w >> 1), bh = Math.max(32, h >> 1);
    const ba0 = makeSingleFbo(bw, bh, false, false), ba1 = makeSingleFbo(bw, bh, false, false);
    fboBloomA0 = ba0.fbo; texBloomA0 = ba0.tex; fboBloomA1 = ba1.fbo; texBloomA1 = ba1.tex;

    const qw = Math.max(16, w >> 2), qh = Math.max(16, h >> 2);
    const bb0 = makeSingleFbo(qw, qh, false, false), bb1 = makeSingleFbo(qw, qh, false, false);
    fboBloomB0 = bb0.fbo; texBloomB0 = bb0.tex; fboBloomB1 = bb1.fbo; texBloomB1 = bb1.tex;

    const st0 = makeSingleFbo(bw, bh, false, false), st1 = makeSingleFbo(bw, bh, false, false);
    fboStreak0 = st0.fbo; texStreak0 = st0.tex; fboStreak1 = st1.fbo; texStreak1 = st1.tex;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  // Pre-separate draw lists: opaque surfaces/tubes/texts, transparent surfaces, path curves
  const opaqueDraws = [];
  const transDraws = [];
  const pathDraws = [];
  for (let i = 0; i < H.objects.length; i++) {
    const rec = H.objects[i];
    if (rec.hidden || rec.role === 'rain_emitter' || rec.role === 'rain_prototype') continue;
    if (rec.count > 0) {
      if (rec.transparent) transDraws.push(i);
      else opaqueDraws.push(i);
    }
    if (rec.lines && rec.lines.length > 0) pathDraws.push(i);
  }

  // Geometry Nodes rain instancing setup
  const rainCfg = H.rain || { enabled: false };
  const rainProto = rainCfg.enabled && rainCfg.prototype >= 0 ? H.objects[rainCfg.prototype] : null;
  const rainFaces = new Float32Array(24 * 3);
  let rainFaceCount = 0;
  if (rainCfg.enabled && rainCfg.faces) {
    rainFaceCount = Math.min(rainCfg.faces.length, 6);
    for (let f = 0; f < rainFaceCount; f++) {
      for (let v = 0; v < 4; v++) {
        const pt = rainCfg.faces[f][v];
        const idx = (f * 4 + v) * 3;
        rainFaces[idx] = pt[0]; rainFaces[idx + 1] = pt[1]; rainFaces[idx + 2] = pt[2];
      }
    }
  }
  const rainEmitters = new Float32Array(32);

  return {
    gl: gl,
    progScene: progScene, progWire: progWire, progRain: progRain, progLines: progLines,
    progSky: progSky, progBright: progBright, progBlur: progBlur, progStreak: progStreak,
    progComposite: progComposite,
    vaoScene: vaoScene, vaoWire: vaoWire, vaoLines: vaoLines, vaoPost: vaoPost,
    texMat: texMat, matTableBase: matTableBase, matTableLive: matTableLive,
    updateMaterialTex: updateMaterialTex,
    ensureTargets: ensureTargets,
    getTargets: () => ({
      w: fboW, h: fboH,
      fboMain: fboMain, texMainColor: texMainColor, texMainNeon: texMainNeon, texMainAux: texMainAux,
      fboMirror: fboMirror, texMirror: texMirror, mw: Math.max(32, fboW >> 1), mh: Math.max(32, fboH >> 1),
      fboBloomA0: fboBloomA0, texBloomA0: texBloomA0, fboBloomA1: fboBloomA1, texBloomA1: texBloomA1, bw: Math.max(32, fboW >> 1), bh: Math.max(32, fboH >> 1),
      fboBloomB0: fboBloomB0, texBloomB0: texBloomB0, fboBloomB1: fboBloomB1, texBloomB1: texBloomB1, qw: Math.max(16, fboW >> 2), qh: Math.max(16, fboH >> 2),
      fboStreak0: fboStreak0, texStreak0: texStreak0, fboStreak1: fboStreak1, texStreak1: texStreak1,
    }),
    opaqueDraws: opaqueDraws, transDraws: transDraws, pathDraws: pathDraws,
    rainCfg: rainCfg, rainProto: rainProto, rainFaces: rainFaces, rainFaceCount: rainFaceCount, rainEmitters: rainEmitters,
  };
}

// ---------------------------------------------------------------------------
// 4. Scene Graph & 240-Frame Animation Evaluator
// ---------------------------------------------------------------------------
function createSceneEvaluator(scene, renderer) {
  const H = scene.header;
  const S = scene.S;
  const numObjs = H.objects.length;
  const numFrames = H.frames;
  const localMats = new Float32Array(numObjs * 16);
  const worldMats = new Float32Array(numObjs * 16);
  const normalMats = new Float32Array(numObjs * 9);
  const interp12 = new Float32Array(12);

  for (let i = 0; i < numObjs; i++) {
    m34ToM4(S.node_locals, i * 12, localMats, i * 16);
    m34ToM4(S.node_locals, i * 12, worldMats, i * 16);
  }

  const armMatrices = H.armatures.map(() => {
    const buf = new Float32Array(16 * 16);
    for (let b = 0; b < 16; b++) {
      buf[b * 16] = buf[b * 16 + 5] = buf[b * 16 + 10] = buf[b * 16 + 15] = 1;
    }
    return buf;
  });

  const lightPos = new Float32Array(16 * 4);
  const lightCol = new Float32Array(16 * 4);
  const lightDir = new Float32Array(16 * 4);
  const lightAux = new Float32Array(16 * 4);
  const cutFrames = new Set(H.markers.map((m) => m.frame));

  function sample34(arr, base0, base1, alpha) {
    if (alpha <= 1e-4) {
      for (let k = 0; k < 12; k++) interp12[k] = arr[base0 + k];
    } else {
      const ia = 1.0 - alpha;
      for (let k = 0; k < 12; k++) interp12[k] = arr[base0 + k] * ia + arr[base1 + k] * alpha;
    }
  }

  function evaluateFrame(frameF, audioBands, audioReactive) {
    const fClamped = clamp(frameF, 1.0, numFrames);
    const fInt = Math.floor(fClamped);
    const f0 = clamp(fInt - 1, 0, numFrames - 1);
    const f1 = (f0 + 1) % numFrames;
    const acrossCut = cutFrames.has(fInt + 1);
    const alpha = acrossCut ? 0.0 : (fClamped - fInt);

    // 1. Update animated object tracks
    for (let t = 0; t < H.tracks.length; t++) {
      const node = H.tracks[t].node;
      const base = t * numFrames * 12;
      sample34(S.tracks, base + f0 * 12, base + f1 * 12, alpha);
      m34ToM4(interp12, 0, localMats, node * 16);
    }

    // 2. Topological world matrix propagation
    const order = H.order;
    for (let k = 0; k < order.length; k++) {
      const i = order[k];
      const rec = H.objects[i];
      if (rec.ws === 1 || rec.parent < 0) {
        for (let m = 0; m < 16; m++) worldMats[i * 16 + m] = localMats[i * 16 + m];
      } else {
        M4.mulOff(worldMats, i * 16, worldMats, rec.parent * 16, localMats, i * 16);
      }
      if (rec.count > 0) normalMat3(worldMats, i * 16, normalMats, i * 9);
    }

    // 3. Armature bone skinning matrices
    for (let a = 0; a < H.armatures.length; a++) {
      const arm = H.armatures[a];
      const nb = arm.bones.length;
      const dst = armMatrices[a];
      for (let b = 0; b < nb; b++) {
        const b0 = arm.offset + (f0 * nb + b) * 12;
        const b1 = arm.offset + (f1 * nb + b) * 12;
        sample34(S.bone_tracks, b0, b1, alpha);
        m34ToM4(interp12, 0, dst, b * 16);
      }
    }

    // 4. Lights (positions, directions, animated energies + sub-bass thruster boost)
    const numLights = Math.min(H.lights.length, 16);
    for (let i = 0; i < numLights; i++) {
      const L = H.lights[i];
      const wo = L.node * 16;
      const wx = worldMats[wo + 12], wy = worldMats[wo + 13], wz = worldMats[wo + 14];
      let dx = -worldMats[wo + 8], dy = -worldMats[wo + 9], dz = -worldMats[wo + 10];
      const dlen = Math.hypot(dx, dy, dz) || 1.0;
      dx /= dlen; dy /= dlen; dz /= dlen;

      let energy = L.energy;
      if (L.track >= 0) {
        const e0 = S.light_tracks[L.track + f0];
        const e1 = S.light_tracks[L.track + f1];
        energy = e0 * (1.0 - alpha) + e1 * alpha;
      }
      if (audioReactive > 0.5 && L.name.indexOf('UnderHull') >= 0) {
        energy *= 1.0 + audioBands[0] * 2.2;
      }
      const scale = L.type === 'AREA' ? 0.032 : (L.type === 'SPOT' ? 0.042 : 0.055);
      const rad = energy * scale;
      const ltype = L.type === 'SPOT' ? 1.0 : (L.type === 'AREA' ? 2.0 : 0.0);

      lightPos[i * 4] = wx;
      lightPos[i * 4 + 1] = wy;
      lightPos[i * 4 + 2] = wz;
      lightPos[i * 4 + 3] = ltype;

      lightCol[i * 4] = L.color[0] * rad;
      lightCol[i * 4 + 1] = L.color[1] * rad;
      lightCol[i * 4 + 2] = L.color[2] * rad;
      lightCol[i * 4 + 3] = Math.max(L.size || 0.25, L.soft || 0.15, 0.35);

      const halfCone = (L.spot_size || 0.785398) * 0.5;
      const cosOuter = Math.cos(halfCone);
      const cosInner = Math.cos(halfCone * (1.0 - (L.spot_blend || 0.15)));
      lightDir[i * 4] = dx;
      lightDir[i * 4 + 1] = dy;
      lightDir[i * 4 + 2] = dz;
      lightDir[i * 4 + 3] = cosOuter;

      lightAux[i * 4] = cosInner;
      lightAux[i * 4 + 1] = L.size || 0.25;
      lightAux[i * 4 + 2] = L.size_y || 0.25;
      lightAux[i * 4 + 3] = 0.0;
    }

    // 5. Material animation tracks (neon flicker + UV scroll)
    const baseMat = renderer.matTableBase;
    const liveMat = renderer.matTableLive;
    liveMat.set(baseMat);
    for (let m = 0; m < H.material_tracks.length; m++) {
      const mt = H.material_tracks[m];
      const texBase = mt.mat * H.material_texels * 4;
      if (mt.param === 'emission_scale') {
        const v0 = S.material_tracks[mt.offset + f0];
        const v1 = S.material_tracks[mt.offset + f1];
        const s = v0 * (1.0 - alpha) + v1 * alpha;
        liveMat[texBase + 7] = baseMat[texBase + 7] * s;
      } else if (mt.param === 'scroll') {
        const o0 = mt.offset + f0 * 3;
        const o1 = mt.offset + f1 * 3;
        liveMat[texBase + 20] = S.material_tracks[o0] * (1.0 - alpha) + S.material_tracks[o1] * alpha;
        liveMat[texBase + 21] = S.material_tracks[o0 + 1] * (1.0 - alpha) + S.material_tracks[o1 + 1] * alpha;
        liveMat[texBase + 22] = S.material_tracks[o0 + 2] * (1.0 - alpha) + S.material_tracks[o1 + 2] * alpha;
      }
    }
    renderer.updateMaterialTex();

    // 6. Rain emitter world matrices
    if (renderer.rainCfg.enabled && renderer.rainCfg.emitters) {
      for (let e = 0; e < Math.min(2, renderer.rainCfg.emitters.length); e++) {
        const emNode = renderer.rainCfg.emitters[e];
        for (let k = 0; k < 16; k++) renderer.rainEmitters[e * 16 + k] = worldMats[emNode * 16 + k];
      }
    }
  }

  function directorMarkerForFrame(frameF) {
    let chosen = H.markers[0] || { name: 'Shot_01_85mm_Orbit', frame: 1, camera: 0 };
    for (let i = 0; i < H.markers.length; i++) {
      if (frameF >= H.markers[i].frame) chosen = H.markers[i];
    }
    return chosen;
  }

  return {
    worldMats: worldMats,
    normalMats: normalMats,
    armMatrices: armMatrices,
    lightPos: lightPos,
    lightCol: lightCol,
    lightDir: lightDir,
    lightAux: lightAux,
    numLights: Math.min(H.lights.length, 16),
    evaluateFrame: evaluateFrame,
    directorMarkerForFrame: directorMarkerForFrame,
  };
}

// ---------------------------------------------------------------------------
// 5. Interactive Camera Controller (Orbit / Pan / WASD Fly + 3 Director Cams)
// ---------------------------------------------------------------------------
function createCameraSystem(canvas, header, evaluator, onStateChange) {
  const free = {
    yaw: -1.18,
    pitch: 0.24,
    dist: 13.8,
    target: new Float32Array([0.0, 0.0, 2.35]),
    fovy: 36.0 * Math.PI / 180.0,
    near: 0.12,
    far: 260.0,
  };
  function resetFree() {
    free.yaw = -1.18;
    free.pitch = 0.24;
    free.dist = 13.8;
    free.target[0] = 0.0; free.target[1] = 0.0; free.target[2] = 2.35;
  }

  const keys = {};
  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    keys[e.code] = true;
  });
  window.addEventListener('keyup', (e) => { keys[e.code] = false; });
  window.addEventListener('blur', () => { for (const k in keys) keys[k] = false; });

  let dragMode = 0, lastX = 0, lastY = 0;
  const pointers = new Map();
  let lastPinchDist = 0;

  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('pointerdown', (e) => {
    canvas.focus();
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    canvas.classList.add('dragging');
    if (pointers.size === 1) {
      lastX = e.clientX;
      lastY = e.clientY;
      dragMode = (e.button === 2 || e.button === 1 || e.shiftKey) ? 2 : 1;
      onStateChange('interact');
    } else if (pointers.size === 2) {
      const pts = Array.from(pointers.values());
      lastPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      lastX = (pts[0].x + pts[1].x) * 0.5;
      lastY = (pts[0].y + pts[1].y) * 0.5;
    }
  });
  canvas.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 1 && dragMode > 0) {
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (dragMode === 1) {
        free.yaw -= dx * 0.0055;
        free.pitch = clamp(free.pitch + dy * 0.0045, -1.48, 1.48);
      } else {
        panByPixels(dx, dy);
      }
    } else if (pointers.size === 2) {
      const pts = Array.from(pointers.values());
      const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
      const cx = (pts[0].x + pts[1].x) * 0.5;
      const cy = (pts[0].y + pts[1].y) * 0.5;
      if (lastPinchDist > 0) {
        free.dist = clamp(free.dist * (lastPinchDist / Math.max(dist, 10)), 1.2, 95.0);
      }
      panByPixels(cx - lastX, cy - lastY);
      lastPinchDist = dist;
      lastX = cx;
      lastY = cy;
    }
  });
  function endPointer(e) {
    pointers.delete(e.pointerId);
    if (pointers.size === 0) {
      dragMode = 0;
      canvas.classList.remove('dragging');
    }
  }
  canvas.addEventListener('pointerup', endPointer);
  canvas.addEventListener('pointercancel', endPointer);

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    onStateChange('interact');
    const factor = Math.exp(e.deltaY * 0.0011);
    free.dist = clamp(free.dist * factor, 1.1, 95.0);
  }, { passive: false });

  function basisFromAngles(yaw, pitch) {
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    // Forward vector from eye to target in Z-up world
    const fx = -cp * cy, fy = -cp * sy, fz = -sp;
    // Right = normalize(forward x [0,0,1])
    let rx = fy, ry = -fx;
    const rlen = Math.hypot(rx, ry) || 1.0;
    rx /= rlen; ry /= rlen;
    // Up = right x forward
    const ux = -ry * fz, uy = rx * fz, uz = rx * fy - ry * fx;
    return { fx: fx, fy: fy, fz: fz, rx: rx, ry: ry, ux: ux, uy: uy, uz: uz };
  }

  function panByPixels(dx, dy) {
    const b = basisFromAngles(free.yaw, free.pitch);
    const scale = free.dist * 0.0014;
    free.target[0] += (-b.rx * dx + b.ux * dy) * scale;
    free.target[1] += (-b.ry * dx + b.uy * dy) * scale;
    free.target[2] = clamp(free.target[2] + b.uz * dy * scale, 0.15, 28.0);
  }

  function updateFly(dt) {
    const moveF = (keys.KeyW || keys.ArrowUp ? 1 : 0) - (keys.KeyS || keys.ArrowDown ? 1 : 0);
    const moveR = (keys.KeyD || keys.ArrowRight ? 1 : 0) - (keys.KeyA || keys.ArrowLeft ? 1 : 0);
    const moveU = (keys.KeyE ? 1 : 0) - (keys.KeyQ ? 1 : 0);
    if (!moveF && !moveR && !moveU) return;
    onStateChange('interact');
    const b = basisFromAngles(free.yaw, free.pitch);
    const speed = (keys.ShiftLeft || keys.ShiftRight ? 14.0 : 5.5) * dt;
    free.target[0] += (b.fx * moveF + b.rx * moveR) * speed;
    free.target[1] += (b.fy * moveF + b.ry * moveR) * speed;
    free.target[2] = clamp(free.target[2] + (b.fz * moveF + moveU) * speed, 0.15, 32.0);
  }

  const viewMat = M4.create();
  const projMat = M4.create();
  const viewProj = M4.create();
  const invViewProj = M4.create();
  const camWorld = M4.create();
  const eye = new Float32Array(3);

  function computeCamera(mode, frameF, aspect) {
    const mk = evaluator.directorMarkerForFrame(frameF);
    let camIdx = -1;
    let label = 'FREE ORBIT';
    let shotName = mk.name;
    if (mode === 'director') {
      camIdx = mk.camera >= 0 ? mk.camera : 0;
      label = 'DIRECTOR // ' + (header.cameras[camIdx] ? header.cameras[camIdx].name.replace('Cinematic_Camera_', '') : '85mm');
    } else if (mode === '0' || mode === '1' || mode === '2') {
      camIdx = parseInt(mode, 10);
      label = header.cameras[camIdx] ? header.cameras[camIdx].name.replace('Cinematic_Camera_', '') : ('CAM ' + mode);
    }

    if (camIdx >= 0 && header.cameras[camIdx]) {
      const cam = header.cameras[camIdx];
      const wo = cam.node * 16;
      for (let k = 0; k < 16; k++) camWorld[k] = evaluator.worldMats[wo + k];
      // Orthonormalize camera basis columns before inverting
      for (let c = 0; c < 3; c++) {
        const o = c * 4;
        const len = Math.hypot(camWorld[o], camWorld[o + 1], camWorld[o + 2]) || 1.0;
        camWorld[o] /= len; camWorld[o + 1] /= len; camWorld[o + 2] /= len;
      }
      eye[0] = camWorld[12];
      eye[1] = camWorld[13];
      eye[2] = Math.max(0.08, camWorld[14]);
      camWorld[14] = eye[2];
      M4.invert(viewMat, camWorld);
      const sensorH = (cam.sensor || 36.0) / (16.0 / 9.0);
      const fovy = 2.0 * Math.atan(sensorH / (2.0 * (cam.lens || 50.0)));
      const near = Math.max(0.1, cam.clip_start || 0.1);
      const far = Math.min(280.0, cam.clip_end || 260.0);
      M4.perspective(projMat, fovy, aspect, near, far);
      M4.mul(viewProj, projMat, viewMat);
      M4.invert(invViewProj, viewProj);
      return {
        view: viewMat, proj: projMat, viewProj: viewProj, invViewProj: invViewProj,
        eye: eye, far: far, label: label, shotName: shotName, camIdx: camIdx,
      };
    }

    const cp = Math.cos(free.pitch), sp = Math.sin(free.pitch);
    const cy = Math.cos(free.yaw), sy = Math.sin(free.yaw);
    eye[0] = free.target[0] + free.dist * cp * cy;
    eye[1] = free.target[1] + free.dist * cp * sy;
    eye[2] = Math.max(0.12, free.target[2] + free.dist * sp);
    M4.lookAt(viewMat, eye, free.target, [0, 0, 1]);
    M4.perspective(projMat, free.fovy, aspect, free.near, free.far);
    M4.mul(viewProj, projMat, viewMat);
    M4.invert(invViewProj, viewProj);
    return {
      view: viewMat, proj: projMat, viewProj: viewProj, invViewProj: invViewProj,
      eye: eye, far: free.far, label: 'FREE ORBIT', shotName: shotName, camIdx: -1,
    };
  }

  return {
    free: free,
    resetFree: resetFree,
    updateFly: updateFly,
    computeCamera: computeCamera,
  };
}

// ---------------------------------------------------------------------------
// 6. WebAudio API Engine & 64-Band Audio-Reactive FFT Visualizer
// ---------------------------------------------------------------------------
function createAudioEngine(wavBytes, wavB64) {
  // Parse 16-bit PCM stereo WAV header for deterministic fallback & instant AudioBuffer creation
  const dv = new DataView(wavBytes.buffer, wavBytes.byteOffset, wavBytes.byteLength);
  let sampleRate = 48000, numChannels = 2, dataOffset = 44, dataBytes = wavBytes.byteLength - 44;
  let off = 12;
  while (off + 8 <= wavBytes.byteLength) {
    const id = String.fromCharCode(wavBytes[off], wavBytes[off + 1], wavBytes[off + 2], wavBytes[off + 3]);
    const sz = dv.getUint32(off + 4, true);
    if (id === 'fmt ') {
      numChannels = dv.getUint16(off + 10, true) || 2;
      sampleRate = dv.getUint32(off + 12, true) || 48000;
    } else if (id === 'data') {
      dataOffset = off + 8;
      dataBytes = Math.min(sz, wavBytes.byteLength - dataOffset);
      break;
    }
    off += 8 + ((sz + 1) & ~1);
  }
  const totalFrames = Math.floor(dataBytes / (numChannels * 2));
  const pcmL = new Float32Array(totalFrames);
  const pcmR = new Float32Array(totalFrames);
  let p = dataOffset;
  for (let i = 0; i < totalFrames; i++) {
    pcmL[i] = dv.getInt16(p, true) / 32768.0;
    pcmR[i] = (numChannels > 1 ? dv.getInt16(p + 2, true) : dv.getInt16(p, true)) / 32768.0;
    p += numChannels * 2;
  }
  const duration = totalFrames / sampleRate;

  // WebAudio API nodes: AudioContext + AnalyserNode (fftSize = 2048, smoothingTimeConstant = 0.82)
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  let audioCtx = null, analyser = null, analyserL = null, analyserR = null;
  let splitter = null, gainNode = null, mediaDest = null, audioBuffer = null, activeSource = null;
  const dataUri = 'data:audio/wav;base64,' + wavB64;

  function ensureContext() {
    if (audioCtx || !AudioCtx) return audioCtx;
    try {
      audioCtx = new AudioCtx({ sampleRate: sampleRate });
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.82;
      analyser.minDecibels = -92;
      analyser.maxDecibels = -10;

      analyserL = audioCtx.createAnalyser();
      analyserL.fftSize = 1024;
      analyserL.smoothingTimeConstant = 0.6;
      analyserR = audioCtx.createAnalyser();
      analyserR.fftSize = 1024;
      analyserR.smoothingTimeConstant = 0.6;

      splitter = audioCtx.createChannelSplitter(2);
      gainNode = audioCtx.createGain();
      gainNode.gain.value = state.muted ? 0.0 : 0.9;
      if (typeof audioCtx.createMediaStreamDestination === 'function') {
        mediaDest = audioCtx.createMediaStreamDestination();
        gainNode.connect(mediaDest);
      }

      analyser.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      splitter.connect(analyserL, 0);
      splitter.connect(analyserR, 1);

      audioBuffer = audioCtx.createBuffer(2, totalFrames, sampleRate);
      audioBuffer.getChannelData(0).set(pcmL);
      audioBuffer.getChannelData(1).set(pcmR);
    } catch (e) {
      warn('AudioContext init warning: ' + e);
    }
    return audioCtx;
  }

  const state = {
    playing: false,
    muted: false,
    speed: 1.0,
    startedAtCtxTime: 0,
    startedAtOffset: 0,
    dataUri: dataUri,
  };

  function stopSource() {
    if (activeSource) {
      try { activeSource.stop(); } catch (e) {}
      try { activeSource.disconnect(); } catch (e) {}
      activeSource = null;
    }
  }

  function startSourceAt(offsetSec) {
    ensureContext();
    stopSource();
    if (!audioCtx || !audioBuffer || !state.playing) return;
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }
    const src = audioCtx.createBufferSource();
    src.buffer = audioBuffer;
    src.loop = true;
    src.loopStart = 0;
    src.loopEnd = duration;
    src.playbackRate.value = state.speed;
    src.connect(analyser);
    src.connect(splitter);
    const wrapped = ((offsetSec % duration) + duration) % duration;
    src.start(0, wrapped);
    state.startedAtCtxTime = audioCtx.currentTime;
    state.startedAtOffset = wrapped;
    activeSource = src;
  }

  function unlock() {
    ensureContext();
    if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume().then(() => {
        if (state.playing && !activeSource) startSourceAt(state.startedAtOffset);
      }).catch(() => {});
    }
  }

  function setPlaying(play, offsetSec) {
    state.playing = !!play;
    state.startedAtOffset = offsetSec || 0;
    if (state.playing) startSourceAt(state.startedAtOffset);
    else stopSource();
  }

  function seek(offsetSec) {
    state.startedAtOffset = offsetSec;
    if (state.playing) startSourceAt(offsetSec);
  }

  function setSpeed(spd, currentOffsetSec) {
    state.speed = spd;
    state.startedAtOffset = currentOffsetSec;
    if (activeSource && audioCtx) {
      activeSource.playbackRate.setValueAtTime(spd, audioCtx.currentTime);
      state.startedAtCtxTime = audioCtx.currentTime;
    }
  }

  function setMuted(m) {
    state.muted = !!m;
    if (gainNode && audioCtx) {
      gainNode.gain.setValueAtTime(state.muted ? 0.0 : 0.9, audioCtx.currentTime);
    }
  }

  // Built-in radix-2 2048-point FFT for frame-accurate spectrum when paused/scrubbing or before AudioContext unlock
  const FFT_N = 2048;
  const HALF_N = FFT_N >> 1;
  const hann = new Float32Array(FFT_N);
  const bitRev = new Uint16Array(FFT_N);
  const cosTbl = new Float32Array(HALF_N);
  const sinTbl = new Float32Array(HALF_N);
  for (let i = 0; i < FFT_N; i++) {
    hann[i] = 0.5 * (1.0 - Math.cos(2.0 * Math.PI * i / (FFT_N - 1)));
    let x = i, r = 0;
    for (let b = 0; b < 11; b++) { r = (r << 1) | (x & 1); x >>= 1; }
    bitRev[i] = r;
  }
  for (let i = 0; i < HALF_N; i++) {
    const a = -2.0 * Math.PI * i / FFT_N;
    cosTbl[i] = Math.cos(a);
    sinTbl[i] = Math.sin(a);
  }
  const fftRe = new Float32Array(FFT_N);
  const fftIm = new Float32Array(FFT_N);
  const smoothBins = new Float32Array(HALF_N);
  const byteFreq = new Uint8Array(HALF_N);
  const waveMono = new Float32Array(FFT_N);
  const waveL = new Float32Array(1024);
  const waveR = new Float32Array(1024);

  function computePcmSpectrum(offsetSec) {
    const centerSample = Math.floor((((offsetSec % duration) + duration) % duration) * sampleRate);
    for (let i = 0; i < FFT_N; i++) {
      const idx = (centerSample + i - HALF_N + totalFrames) % totalFrames;
      const mono = 0.5 * (pcmL[idx] + pcmR[idx]);
      waveMono[i] = mono;
      const br = bitRev[i];
      fftRe[br] = mono * hann[i];
      fftIm[br] = 0.0;
    }
    for (let i = 0; i < 1024; i++) {
      const idx = (centerSample + i - 512 + totalFrames) % totalFrames;
      waveL[i] = pcmL[idx];
      waveR[i] = pcmR[idx];
    }
    for (let size = 2; size <= FFT_N; size <<= 1) {
      const half = size >> 1;
      const step = FFT_N / size;
      for (let i = 0; i < FFT_N; i += size) {
        for (let j = 0; j < half; j++) {
          const k = j * step;
          const wr = cosTbl[k], wi = sinTbl[k];
          const a = i + j, b = a + half;
          const tr = wr * fftRe[b] - wi * fftIm[b];
          const ti = wr * fftIm[b] + wi * fftRe[b];
          fftRe[b] = fftRe[a] - tr;
          fftIm[b] = fftIm[a] - ti;
          fftRe[a] += tr;
          fftIm[a] += ti;
        }
      }
    }
    const smooth = 0.82;
    for (let k = 0; k < HALF_N; k++) {
      const mag = Math.hypot(fftRe[k], fftIm[k]) / HALF_N;
      const db = 20.0 * Math.log10(Math.max(mag, 1e-6));
      const norm = clamp((db + 92.0) / 82.0, 0.0, 1.0);
      smoothBins[k] = smoothBins[k] * smooth + norm * (1.0 - smooth);
      byteFreq[k] = Math.round(smoothBins[k] * 255);
    }
  }

  // 64-band logarithmic frequency table (20Hz .. 18000Hz)
  const NUM_BARS = 64;
  const barBinLo = new Uint16Array(NUM_BARS);
  const barBinHi = new Uint16Array(NUM_BARS);
  const barValues = new Float32Array(NUM_BARS);
  const barPeaks = new Float32Array(NUM_BARS);
  const binHz = sampleRate / FFT_N;
  const minHz = 20.0, maxHz = 18000.0;
  for (let b = 0; b < NUM_BARS; b++) {
    const f0 = minHz * Math.pow(maxHz / minHz, b / NUM_BARS);
    const f1 = minHz * Math.pow(maxHz / minHz, (b + 1) / NUM_BARS);
    const lo = clamp(Math.floor(f0 / binHz), 1, HALF_N - 1);
    const hi = clamp(Math.max(lo, Math.ceil(f1 / binHz)), lo, HALF_N - 1);
    barBinLo[b] = lo;
    barBinHi[b] = hi;
  }

  const bands = new Float32Array(3); // [sub, mid, high]
  function bandAverage(hzLo, hzHi, boost) {
    const i0 = clamp(Math.floor(hzLo / binHz), 1, HALF_N - 1);
    const i1 = clamp(Math.ceil(hzHi / binHz), i0, HALF_N - 1);
    let sum = 0.0;
    for (let i = i0; i <= i1; i++) sum += byteFreq[i] / 255.0;
    return clamp((sum / (i1 - i0 + 1)) * boost, 0.0, 1.0);
  }

  const hudCanvas = $('hud-audio');
  const hudCtx = hudCanvas.getContext('2d');

  function sampleAndRenderHUD(offsetSec) {
    const liveWebAudio = audioCtx && audioCtx.state === 'running' && state.playing && !state.muted && analyser;
    if (liveWebAudio) {
      analyser.getByteFrequencyData(byteFreq);
      analyser.getFloatTimeDomainData(waveMono);
      analyserL.getFloatTimeDomainData(waveL);
      analyserR.getFloatTimeDomainData(waveR);
    } else {
      computePcmSpectrum(offsetSec);
    }

    for (let b = 0; b < NUM_BARS; b++) {
      let peak = 0.0, sum = 0.0;
      const lo = barBinLo[b], hi = barBinHi[b];
      for (let k = lo; k <= hi; k++) {
        const v = byteFreq[k] / 255.0;
        if (v > peak) peak = v;
        sum += v;
      }
      const val = 0.65 * peak + 0.35 * (sum / (hi - lo + 1));
      barValues[b] = val;
      if (val >= barPeaks[b]) barPeaks[b] = val;
      else barPeaks[b] = Math.max(0.0, barPeaks[b] - 0.008);
    }

    // Sub-Bass (20..120Hz), Mid (250..2000Hz), High (4..16kHz)
    bands[0] = bandAverage(20.0, 120.0, 1.25);
    bands[1] = bandAverage(250.0, 2000.0, 1.35);
    bands[2] = bandAverage(4000.0, 16000.0, 1.65);

    $('b-sub').textContent = bands[0].toFixed(2);
    $('b-mid').textContent = bands[1].toFixed(2);
    $('b-high').textContent = bands[2].toFixed(2);
    $('bb-sub').style.width = Math.round(bands[0] * 100) + '%';
    $('bb-mid').style.width = Math.round(bands[1] * 100) + '%';
    $('bb-high').style.width = Math.round(bands[2] * 100) + '%';

    // Render 60fps HUD canvas: 64-Band Spectrum + Stereo Lissajous Phase Scope + Waveform Oscilloscope
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(240, Math.floor(hudCanvas.clientWidth * dpr));
    const ch = Math.max(140, Math.floor(hudCanvas.clientHeight * dpr));
    if (hudCanvas.width !== cw || hudCanvas.height !== ch) {
      hudCanvas.width = cw;
      hudCanvas.height = ch;
    }
    const ctx = hudCtx;
    ctx.clearRect(0, 0, cw, ch);

    const specH = Math.floor(ch * 0.53);
    const botY = specH + Math.floor(6 * dpr);
    const botH = ch - botY;
    const lissW = Math.floor(cw * 0.34);
    const scopeX = lissW + Math.floor(6 * dpr);
    const scopeW = cw - scopeX;

    // 1. 64-Bar Logarithmic Frequency Spectrum
    ctx.strokeStyle = 'rgba(0, 234, 255, 0.10)';
    ctx.lineWidth = 1;
    for (let g = 1; g <= 3; g++) {
      const gy = Math.floor((specH - 14 * dpr) * (g / 4)) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(cw, gy); ctx.stroke();
    }
    const barPitch = cw / NUM_BARS;
    const barW = Math.max(1.5, barPitch - 1.5 * dpr);
    const plotH = specH - 14 * dpr;
    const grad = ctx.createLinearGradient(0, plotH, 0, 2);
    grad.addColorStop(0.0, 'rgba(0, 234, 255, 0.85)');
    grad.addColorStop(0.65, 'rgba(255, 43, 214, 0.92)');
    grad.addColorStop(1.0, 'rgba(255, 179, 64, 0.98)');
    ctx.fillStyle = grad;

    for (let b = 0; b < NUM_BARS; b++) {
      const x = b * barPitch + 0.5 * dpr;
      const bh = Math.max(1.5, barValues[b] * plotH);
      ctx.fillRect(x, plotH - bh, barW, bh);
      const py = plotH - barPeaks[b] * plotH;
      ctx.fillStyle = '#d4f3ff';
      ctx.fillRect(x, Math.max(0, py - 1.5 * dpr), barW, 1.5 * dpr);
      ctx.fillStyle = grad;
    }

    ctx.fillStyle = 'rgba(120, 149, 173, 0.88)';
    ctx.font = Math.round(9 * dpr) + 'px Cascadia Mono, Consolas, monospace';
    const labels = [['20Hz', 0.01], ['120Hz', 0.25], ['1kHz', 0.56], ['4kHz', 0.77], ['16kHz', 0.91]];
    for (let i = 0; i < labels.length; i++) {
      ctx.fillText(labels[i][0], Math.floor(cw * labels[i][1]), specH - 2 * dpr);
    }

    // 2. Stereo Lissajous Phase Scope (Goniometer)
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, botY, lissW, botH);
    ctx.clip();
    ctx.fillStyle = 'rgba(2, 6, 14, 0.55)';
    ctx.fillRect(0, botY, lissW, botH);
    ctx.strokeStyle = 'rgba(0, 234, 255, 0.22)';
    ctx.strokeRect(0.5, botY + 0.5, lissW - 1, botH - 1);

    const cx = lissW * 0.5, cy = botY + botH * 0.52;
    const rad = Math.min(lissW, botH) * 0.42;
    ctx.strokeStyle = 'rgba(0, 234, 255, 0.16)';
    ctx.beginPath();
    ctx.moveTo(cx, cy - rad); ctx.lineTo(cx + rad, cy);
    ctx.lineTo(cx, cy + rad); ctx.lineTo(cx - rad, cy);
    ctx.closePath();
    ctx.moveTo(cx - rad, cy); ctx.lineTo(cx + rad, cy);
    ctx.moveTo(cx, cy - rad); ctx.lineTo(cx, cy + rad);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255, 43, 214, 0.82)';
    ctx.lineWidth = 1.2 * dpr;
    ctx.beginPath();
    const scaleLiss = rad * 1.35;
    for (let i = 0; i < 512; i += 2) {
      const l = waveL[i], r = waveR[i];
      const side = (l - r) * 0.7071;
      const mid = (l + r) * 0.7071;
      const px = cx + clamp(side * scaleLiss, -rad, rad);
      const py = cy - clamp(mid * scaleLiss, -rad, rad);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.fillStyle = 'rgba(0, 234, 255, 0.75)';
    ctx.fillText('LISSAJOUS L/R', 4 * dpr, botY + 10 * dpr);
    ctx.restore();

    // 3. Time-Domain Waveform Oscilloscope
    ctx.save();
    ctx.beginPath();
    ctx.rect(scopeX, botY, scopeW, botH);
    ctx.clip();
    ctx.fillStyle = 'rgba(2, 6, 14, 0.55)';
    ctx.fillRect(scopeX, botY, scopeW, botH);
    ctx.strokeStyle = 'rgba(0, 234, 255, 0.22)';
    ctx.strokeRect(scopeX + 0.5, botY + 0.5, scopeW - 1, botH - 1);

    const midY = botY + botH * 0.52;
    ctx.strokeStyle = 'rgba(0, 234, 255, 0.14)';
    ctx.beginPath();
    ctx.moveTo(scopeX, midY); ctx.lineTo(scopeX + scopeW, midY);
    ctx.stroke();

    // Trigger stabilization on rising zero-crossing
    let trig = 0;
    for (let i = 16; i < 512; i++) {
      if (waveMono[i - 1] < 0.0 && waveMono[i] >= 0.0) { trig = i; break; }
    }
    const samplesToShow = 384;
    const amp = botH * 0.44;

    // Stereo side difference trace (magenta)
    ctx.strokeStyle = 'rgba(255, 43, 214, 0.50)';
    ctx.lineWidth = 1.0 * dpr;
    ctx.beginPath();
    for (let i = 0; i < samplesToShow; i++) {
      const idx = (trig + i) & 1023;
      const x = scopeX + (i / (samplesToShow - 1)) * scopeW;
      const y = midY - clamp((waveL[idx] - waveR[idx]) * 0.8, -1.0, 1.0) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Primary time-domain trace (cyan)
    ctx.strokeStyle = '#00eaff';
    ctx.lineWidth = 1.45 * dpr;
    ctx.beginPath();
    for (let i = 0; i < samplesToShow; i++) {
      const x = scopeX + (i / (samplesToShow - 1)) * scopeW;
      const y = midY - clamp(waveMono[trig + i] * 1.15, -1.0, 1.0) * amp;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = 'rgba(0, 234, 255, 0.75)';
    ctx.fillText('TIME-DOMAIN OSCILLOSCOPE', scopeX + 5 * dpr, botY + 10 * dpr);
    ctx.restore();

    return bands;
  }

  return {
    state: state,
    duration: duration,
    sampleRate: sampleRate,
    ensureContext: ensureContext,
    unlock: unlock,
    setPlaying: setPlaying,
    seek: seek,
    setSpeed: setSpeed,
    setMuted: setMuted,
    sampleAndRenderHUD: sampleAndRenderHUD,
    getMediaStream: () => (mediaDest ? mediaDest.stream : null),
  };
}

// ---------------------------------------------------------------------------
// 7. Multi-Pass Scene Render Pass + Timeline Scrubber + Video Deck & Exporters
// ---------------------------------------------------------------------------
function renderViewportPass(renderer, evaluator, header, cam, rect, opts) {
  const gl = renderer.gl;
  const T = renderer.getTargets();
  const H = header;
  let drawCalls = 0;

  // Bind shared material lookup texture on unit 0
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, renderer.texMat);

  function drawSceneObjects(prog, list, mirrorPass, flipFacing) {
    const u = prog.u;
    gl.useProgram(prog.prog);
    gl.uniformMatrix4fv(u.uViewProj, false, mirrorPass ? viewProjMirror : cam.viewProj);
    gl.uniform1f(u.uFrameF, opts.frameF);
    if (u.uMatTex !== null) gl.uniform1i(u.uMatTex, 0);
    if (u.uCamPos !== null) gl.uniform3fv(u.uCamPos, cam.eye);
    if (u.uFar !== null) gl.uniform1f(u.uFar, cam.far);
    if (u.uTime !== null) gl.uniform1f(u.uTime, opts.timeSec);

    if (prog === renderer.progScene) {
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, T.texMirror);
      gl.uniform1i(u.uReflTex, 1);
      gl.uniform1i(u.uUseRefl, opts.mirror && !mirrorPass ? 1 : 0);
      gl.uniform1i(u.uMirrorPass, mirrorPass ? 1 : 0);
      gl.uniform1i(u.uFlipFacing, flipFacing ? 1 : 0);
      gl.uniform4f(u.uViewport, 0, 0, T.w, T.h);
      gl.uniform1i(u.uLightCount, evaluator.numLights);
      gl.uniform4fv(u.uLightPos, evaluator.lightPos);
      gl.uniform4fv(u.uLightCol, evaluator.lightCol);
      gl.uniform4fv(u.uLightDir, evaluator.lightDir);
      gl.uniform4fv(u.uLightAux, evaluator.lightAux);
      gl.uniform3fv(u.uSkyTop, H.world.sky_top);
      gl.uniform3fv(u.uSkyBottom, H.world.sky_bottom);
      gl.uniform1f(u.uAmbient, H.world.strength || 0.32);
      gl.uniform3fv(u.uAudio, opts.audioBands);
      gl.uniform1f(u.uAudioReactive, opts.audioReactive ? 1.0 : 0.0);
      gl.uniform1f(u.uRainWet, opts.rain ? opts.rainDensity : 0.0);
    }

    const isWire = prog === renderer.progWire;
    gl.bindVertexArray(isWire ? renderer.vaoWire : renderer.vaoScene);

    for (let idx = 0; idx < list.length; idx++) {
      const i = list[idx];
      const rec = H.objects[i];
      if (mirrorPass && rec.bounds && rec.bounds[2] + rec.bounds[3] < -0.1 && evaluator.worldMats[i * 16 + 14] < 0.01) {
        continue;
      }
      gl.uniformMatrix4fv(u.uModel, false, evaluator.worldMats, i * 16);
      gl.uniformMatrix3fv(u.uNormalMat, false, evaluator.normalMats, i * 9);
      if (rec.arm !== undefined && rec.arm >= 0) {
        gl.uniform1i(u.uSkinned, 1);
        gl.uniformMatrix4fv(u.uBones, false, evaluator.armMatrices[rec.arm]);
      } else {
        gl.uniform1i(u.uSkinned, 0);
      }
      if (rec.wave) {
        gl.uniform4f(u.uWave, rec.wave[0], rec.wave[1], rec.wave[2], rec.wave[3]);
        gl.uniform1f(u.uWaveNormal, rec.wave[4] || 0.0);
      } else {
        gl.uniform4f(u.uWave, 0.0, 1.0, 0.0, 1.0);
        gl.uniform1f(u.uWaveNormal, 0.0);
      }
      if (isWire) {
        gl.drawElements(gl.LINES, rec.count * 2, gl.UNSIGNED_INT, rec.first * 2 * 4);
      } else {
        gl.drawElements(gl.TRIANGLES, rec.count, gl.UNSIGNED_INT, rec.first * 4);
      }
      drawCalls++;
    }
  }

  const viewProjMirror = M4.create();
  M4.mul(viewProjMirror, cam.viewProj, MIRROR_Z);

  // PASS 1: Wet-Asphalt Planar Mirror Reflection Pass (Z = 0)
  if (opts.mirror && opts.aov !== 4) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboMirror);
    gl.viewport(0, 0, T.mw, T.mh);
    gl.disable(gl.SCISSOR_TEST);
    gl.enable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    gl.clearColor(0.005, 0.008, 0.018, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    drawSceneObjects(renderer.progScene, renderer.opaqueDraws, true, true);
    gl.bindTexture(gl.TEXTURE_2D, T.texMirror);
    gl.generateMipmap(gl.TEXTURE_2D);
  }

  // PASS 2: Main MRT Pass (Color0 = HDR PBR, Color1 = AOV_Neon_Mask, Color2 = Normals + Linear Depth)
  gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboMain);
  gl.viewport(0, 0, T.w, T.h);
  gl.disable(gl.SCISSOR_TEST);
  gl.depthMask(true);
  gl.disable(gl.BLEND);
  gl.clearColor(0.0, 0.0, 0.0, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  // Sky gradient + volumetric horizon stage
  gl.disable(gl.DEPTH_TEST);
  gl.depthMask(false);
  gl.useProgram(renderer.progSky.prog);
  gl.uniformMatrix4fv(renderer.progSky.u.uInvViewProj, false, cam.invViewProj);
  gl.uniform3fv(renderer.progSky.u.uCamPos, cam.eye);
  gl.uniform3fv(renderer.progSky.u.uSkyTop, H.world.sky_top);
  gl.uniform3fv(renderer.progSky.u.uSkyBottom, H.world.sky_bottom);
  gl.uniform3fv(renderer.progSky.u.uFogColor, H.world.fog_color);
  gl.uniform1f(renderer.progSky.u.uSkyStrength, H.world.strength || 0.32);
  gl.uniform1f(renderer.progSky.u.uTime, opts.timeSec);
  gl.bindVertexArray(renderer.vaoPost);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  drawCalls++;

  // Opaque geometry
  gl.enable(gl.DEPTH_TEST);
  gl.depthFunc(gl.LEQUAL);
  gl.depthMask(true);
  if (opts.aov === 4) {
    drawSceneObjects(renderer.progWire, renderer.opaqueDraws, false, false);
    drawSceneObjects(renderer.progWire, renderer.transDraws, false, false);
  } else {
    drawSceneObjects(renderer.progScene, renderer.opaqueDraws, false, false);
  }

  // Optional curve paths overlay
  if (opts.paths || opts.aov === 4) {
    gl.useProgram(renderer.progLines.prog);
    gl.uniformMatrix4fv(renderer.progLines.u.uViewProj, false, cam.viewProj);
    gl.uniform4f(renderer.progLines.u.uColor, 0.0, 0.92, 1.0, 0.85);
    gl.bindVertexArray(renderer.vaoLines);
    for (let p = 0; p < renderer.pathDraws.length; p++) {
      const idx = renderer.pathDraws[p];
      const rec = H.objects[idx];
      gl.uniformMatrix4fv(renderer.progLines.u.uModel, false, evaluator.worldMats, idx * 16);
      for (let s = 0; s < rec.lines.length; s++) {
        gl.drawArrays(gl.LINE_STRIP, rec.lines[s][0], rec.lines[s][1]);
        drawCalls++;
      }
    }
  }

  // Transparent geometry (holograms, glass rain, broth/steam, light trails)
  if (opts.aov !== 4 && renderer.transDraws.length > 0) {
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    drawSceneObjects(renderer.progScene, renderer.transDraws, false, false);
  }

  // Geometry Nodes Instanced Rain Field (GN_Cyber_Rain_Field)
  if (opts.rain && opts.rainDensity > 0.01 && renderer.rainProto && renderer.rainFaceCount > 0) {
    const u = renderer.progRain.u;
    gl.useProgram(renderer.progRain.prog);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.depthMask(false);
    gl.uniformMatrix4fv(u.uViewProj, false, cam.viewProj);
    gl.uniformMatrix4fv(u.uEmitter, false, renderer.rainEmitters);
    gl.uniform3fv(u.uFaces, renderer.rainFaces);
    gl.uniform1i(u.uFaceCount, renderer.rainFaceCount);
    const perEmitter = Math.max(64, Math.floor((renderer.rainCfg.points_per_emitter || 4000) * opts.rainDensity * 0.45));
    gl.uniform1i(u.uPerEmitter, perEmitter);
    gl.uniform1ui(u.uSeed, renderer.rainCfg.seed || 2099);
    gl.uniform3fv(u.uScaleMin, renderer.rainCfg.scale_min || [0.75, 0.75, 0.65]);
    gl.uniform3fv(u.uScaleMax, renderer.rainCfg.scale_max || [1.25, 1.25, 1.65]);
    gl.uniform1f(u.uRainPhase, opts.rainPhase);
    gl.uniform1f(u.uWidth, 1.25);
    gl.uniform3fv(u.uCamPos, cam.eye);
    gl.uniform1f(u.uIntensity, 0.65 + opts.audioBands[2] * (opts.audioReactive ? 0.85 : 0.0));
    gl.bindVertexArray(renderer.vaoScene);
    gl.drawElementsInstanced(gl.TRIANGLES, renderer.rainProto.count, gl.UNSIGNED_INT, renderer.rainProto.first * 4, perEmitter * 2);
    drawCalls++;
  }

  gl.disable(gl.BLEND);
  gl.disable(gl.DEPTH_TEST);
  gl.depthMask(true);
  gl.bindVertexArray(renderer.vaoPost);

  // PASS 3: Bright-Pass Emissive Threshold Extraction (AOV_Neon_Mask -> fboBloomA0)
  if (opts.bloom) {
    const uB = renderer.progBright.u;
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboBloomA0);
    gl.viewport(0, 0, T.bw, T.bh);
    gl.useProgram(renderer.progBright.prog);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.texMainColor);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, T.texMainNeon);
    gl.uniform1i(uB.uColor, 0);
    gl.uniform1i(uB.uNeon, 1);
    gl.uniform2f(uB.uTexel, 1.0 / T.w, 1.0 / T.h);
    gl.uniform1f(uB.uThreshold, 0.85);
    gl.uniform1f(uB.uKnee, 0.45);
    const neonAudioGain = 1.0 + opts.audioBands[1] * (opts.audioReactive ? 0.95 : 0.0);
    gl.uniform1f(uB.uNeonGain, 1.35 * neonAudioGain);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // PASS 4: Two-Pass Separable Gaussian Neon Bloom FBO (Half-Res + Quarter-Res Pyramid)
    const uG = renderer.progBlur.u;
    gl.useProgram(renderer.progBlur.prog);
    gl.uniform1i(uG.uSrc, 0);

    // Half-res horizontal + vertical
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboBloomA1);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.texBloomA0);
    gl.uniform2f(uG.uTexel, 1.0 / T.bw, 1.0 / T.bh);
    gl.uniform2f(uG.uDir, 1.0, 0.0);
    gl.uniform1f(uG.uRadius, 1.35);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboBloomA0);
    gl.bindTexture(gl.TEXTURE_2D, T.texBloomA1);
    gl.uniform2f(uG.uDir, 0.0, 1.0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Quarter-res wide halo horizontal + vertical
    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboBloomB1);
    gl.viewport(0, 0, T.qw, T.qh);
    gl.bindTexture(gl.TEXTURE_2D, T.texBloomA0);
    gl.uniform2f(uG.uTexel, 1.0 / T.qw, 1.0 / T.qh);
    gl.uniform2f(uG.uDir, 1.0, 0.0);
    gl.uniform1f(uG.uRadius, 2.1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboBloomB0);
    gl.bindTexture(gl.TEXTURE_2D, T.texBloomB1);
    gl.uniform2f(uG.uDir, 0.0, 1.0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Anamorphic Streak Glare stage
    if (opts.streaks) {
      const uS = renderer.progStreak.u;
      gl.useProgram(renderer.progStreak.prog);
      gl.viewport(0, 0, T.bw, T.bh);
      gl.uniform1i(uS.uSrc, 0);
      gl.uniform2f(uS.uTexel, 1.0 / T.bw, 1.0 / T.bh);
      const ang = 0.26;
      gl.uniform2f(uS.uDir, Math.cos(ang), Math.sin(ang));
      gl.uniform1f(uS.uFade, 0.90);

      gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboStreak1);
      gl.bindTexture(gl.TEXTURE_2D, T.texBloomA0);
      gl.uniform1f(uS.uStep, 1.5);
      gl.uniform1f(uS.uThreshold, 1.2);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.bindFramebuffer(gl.FRAMEBUFFER, T.fboStreak0);
      gl.bindTexture(gl.TEXTURE_2D, T.texStreak1);
      gl.uniform1f(uS.uStep, 4.5);
      gl.uniform1f(uS.uThreshold, 0.0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  // PASS 5: Final AgX High-Contrast Tone Mapping + Mist + Chromatic Aberration Composite
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(rect.x, rect.y, rect.w, rect.h);
  gl.enable(gl.SCISSOR_TEST);
  gl.scissor(rect.x, rect.y, rect.w, rect.h);

  const uC = renderer.progComposite.u;
  gl.useProgram(renderer.progComposite.prog);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, T.texMainColor);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, T.texMainNeon);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, T.texMainAux);
  gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, T.texBloomA0);
  gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, T.texBloomB0);
  gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, T.texStreak0);
  gl.uniform1i(uC.uColor, 0);
  gl.uniform1i(uC.uNeon, 1);
  gl.uniform1i(uC.uAux, 2);
  gl.uniform1i(uC.uBloomA, 3);
  gl.uniform1i(uC.uBloomB, 4);
  gl.uniform1i(uC.uStreak, 5);
  gl.uniform1i(uC.uAov, opts.aov);
  gl.uniform1f(uC.uExposure, (H.world.exposure || 0.15) + (H.compositor.exposure || 0.06) + opts.exposureBias);
  const bloomAudioBoost = 1.0 + opts.audioBands[1] * (opts.audioReactive ? 0.55 : 0.0);
  gl.uniform1f(uC.uBloomStrength, opts.bloom ? 0.58 * opts.bloomGain * bloomAudioBoost : 0.0);
  gl.uniform1f(uC.uStreakStrength, opts.bloom && opts.streaks ? 0.24 * opts.bloomGain * bloomAudioBoost : 0.0);
  gl.uniform1f(uC.uChroma, opts.chroma ? (H.compositor.dispersion || 0.012) * 0.38 : 0.0);
  gl.uniform1f(uC.uDistort, opts.chroma ? (H.compositor.distortion || -0.015) * 0.45 : 0.0);
  gl.uniform1f(uC.uMistOn, opts.mist ? 1.0 : 0.0);
  gl.uniform1f(uC.uMistStart, H.world.mist_start || 8.0);
  gl.uniform1f(uC.uMistDepth, H.world.mist_depth || 28.0);
  gl.uniform1f(uC.uMistMix, H.compositor.mist_mix || 0.42);
  gl.uniform1f(uC.uFar, cam.far);
  gl.uniform1f(uC.uTime, opts.timeSec);
  gl.uniform1f(uC.uGrain, 0.012);
  gl.uniform1f(uC.uVignette, 0.42);
  gl.uniform3fv(uC.uMistTint, H.compositor.mist_tint || [0.038, 0.065, 0.115]);
  gl.uniform3fv(uC.uFogColor, H.world.fog_color || [0.32, 0.45, 0.62]);
  gl.uniform3fv(uC.uLift, H.compositor.lift || [1.0, 1.0, 1.0]);
  gl.uniform3fv(uC.uGamma, H.compositor.gamma || [1.0, 1.0, 1.0]);
  gl.uniform3fv(uC.uGain, H.compositor.gain || [1.0, 1.0, 1.0]);
  gl.uniform2f(uC.uResolution, rect.w, rect.h);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.disable(gl.SCISSOR_TEST);

  return drawCalls;
}

// ---------------------------------------------------------------------------
// 8. Studio Application Bootstrap, Timeline Deck, PiP/Split & Exporters
// ---------------------------------------------------------------------------
async function bootStudio() {
  const tBoot = performance.now();
  loaderStep('[1/5] Decoding Base64 CPKS scene & WAV payloads', 0.15);
  await yieldUI();

  const sceneEl = $('scene-payload');
  const audioEl = $('audio-payload');
  const expectedSceneSha = sceneEl.dataset.sha256;
  const expectedAudioSha = audioEl.dataset.sha256;

  const compressedScene = b64ToBytes(sceneEl.textContent);
  const wavBytes = b64ToBytes(audioEl.textContent);

  loaderStep('[2/5] Inflating zlib stream (' + fmtBytes(compressedScene.byteLength) + ')', 0.38);
  await yieldUI();
  const tInf0 = performance.now();
  const inflated = await inflateZlib(compressedScene);
  const inflateMs = (performance.now() - tInf0).toFixed(1);
  $('st-inflate').textContent = inflated.method + ' (' + inflateMs + 'ms)';

  loaderStep('[3/5] Verifying SHA-256 payload integrity', 0.60);
  await yieldUI();
  const sceneDigest = await sha256Hex(inflated.data);
  const audioDigest = await sha256Hex(wavBytes);
  const shaOk = sceneDigest.hex === expectedSceneSha && audioDigest.hex === expectedAudioSha;
  const stInt = $('st-integrity');
  stInt.textContent = (shaOk ? 'OK ' : 'MISMATCH ') + sceneDigest.hex.slice(0, 12);
  stInt.className = shaOk ? 'ok' : 'bad';
  if (!shaOk) throw new Error('Payload SHA-256 mismatch');

  loaderStep('[4/5] Unpacking 380 meshes + 154 curves & compiling WebGL2 GLSL ES 3.00', 0.82);
  await yieldUI();
  const scene = parseContainer(inflated.data);
  const H = scene.header;
  const glCanvas = $('gl');
  const renderer = createRenderer(glCanvas, scene);
  const evaluator = createSceneEvaluator(scene, renderer);

  loaderStep('[5/5] Initializing WebAudio FFT 2048 & 240-frame video deck', 0.96);
  await yieldUI();
  const audio = createAudioEngine(wavBytes, audioEl.textContent.trim());

  // Populate header telemetry
  $('st-tris').textContent = H.counts.triangles.toLocaleString();
  $('st-payload').textContent = fmtBytes(compressedScene.byteLength) + ' -> ' + fmtBytes(inflated.data.byteLength);
  $('st-geo').textContent = H.counts.meshes + 'M / ' + H.counts.curves + 'C / ' + H.counts.materials + 'Mat';

  // Secondary PiP 2D canvas inside #pip-view for zero-dependency Director Cut monitor
  const pipView = $('pip-view');
  const deckVideo = $('deck-video');
  const pipCanvas = document.createElement('canvas');
  pipCanvas.id = 'pip-canvas';
  pipCanvas.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block';
  pipView.insertBefore(pipCanvas, deckVideo);
  const pipCtx = pipCanvas.getContext('2d');

  // Optional external MP4 sync (auto-probes DEFAULT_VIDEO_NAME or user file upload)
  let videoLoaded = false;
  deckVideo.addEventListener('loadedmetadata', () => {
    videoLoaded = true;
    pipView.classList.add('mp4');
    $('pip-src').textContent = 'SYNCED MP4 // ' + (deckVideo.dataset.name || DEFAULT_VIDEO_NAME);
  });
  deckVideo.addEventListener('error', () => {
    videoLoaded = false;
    pipView.classList.remove('mp4');
    $('pip-src').textContent = 'LIVE WEBGL2 DIRECTOR FEED';
  });
  $('file-video').addEventListener('change', (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    deckVideo.dataset.name = file.name;
    deckVideo.src = URL.createObjectURL(file);
    deckVideo.load();
    toast('Loaded video: ' + file.name);
  });

  // Playback & UI state
  const ui = {
    playing: true,
    loop: true,
    speed: 1.0,
    frameF: 1.0,
    camMode: 'director',
    aov: 0,
    pipMode: 'pip', // 'pip' | 'split' | 'off'
    bloom: true,
    mist: true,
    chroma: true,
    mirror: true,
    rain: true,
    audioReactive: true,
    paths: false,
    streaks: true,
    exposureBias: 0.0,
    bloomGain: 1.0,
    rainDensity: 0.6,
    rainPhase: 0.0,
  };

  function setCamMode(mode) {
    ui.camMode = mode;
    const btns = $('cam-btns').querySelectorAll('button');
    for (let i = 0; i < btns.length; i++) {
      btns[i].classList.toggle('on', btns[i].dataset.cam === mode);
    }
  }
  function setAov(aov) {
    ui.aov = aov;
    const btns = $('aov-btns').querySelectorAll('button');
    for (let i = 0; i < btns.length; i++) {
      btns[i].classList.toggle('on', parseInt(btns[i].dataset.aov, 10) === aov);
    }
  }
  setCamMode('director');

  const camSys = createCameraSystem(glCanvas, H, evaluator, (ev) => {
    audio.unlock();
    if (ev === 'interact' && ui.camMode !== 'free') {
      setCamMode('free');
    }
  });

  // Wire up camera & AOV buttons
  $('cam-btns').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-cam]');
    if (!btn) return;
    audio.unlock();
    setCamMode(btn.dataset.cam);
  });
  $('aov-btns').addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-aov]');
    if (!btn) return;
    audio.unlock();
    setAov(parseInt(btn.dataset.aov, 10));
  });

  // Checkboxes & sliders
  const bindCheck = (id, key) => {
    const el = $(id);
    el.addEventListener('change', () => { ui[key] = el.checked; });
  };
  bindCheck('t-bloom', 'bloom');
  bindCheck('t-mist', 'mist');
  bindCheck('t-chroma', 'chroma');
  bindCheck('t-mirror', 'mirror');
  bindCheck('t-rain', 'rain');
  bindCheck('t-react', 'audioReactive');
  bindCheck('t-paths', 'paths');
  bindCheck('t-streaks', 'streaks');

  $('s-exposure').addEventListener('input', (e) => {
    ui.exposureBias = parseFloat(e.target.value);
    $('o-exposure').textContent = (ui.exposureBias >= 0 ? '+' : '') + ui.exposureBias.toFixed(2);
  });
  $('s-bloom').addEventListener('input', (e) => {
    ui.bloomGain = parseFloat(e.target.value);
    $('o-bloom').textContent = ui.bloomGain.toFixed(2);
  });
  $('s-rain').addEventListener('input', (e) => {
    ui.rainDensity = parseFloat(e.target.value);
    $('o-rain').textContent = Math.round(ui.rainDensity * 100) + '%';
  });

  // Transport Deck Controls
  function syncExternalMedia() {
    const sec = (ui.frameF - 1.0) / FPS;
    audio.seek(sec);
    if (videoLoaded) {
      try { deckVideo.currentTime = sec; } catch (e) {}
    }
  }
  function setPlaying(play) {
    ui.playing = !!play;
    const btn = $('btn-play');
    btn.classList.toggle('on', ui.playing);
    btn.innerHTML = ui.playing ? '&#10074;&#10074; PAUSE' : '&#9654; PLAY';
    const sec = (ui.frameF - 1.0) / FPS;
    audio.setPlaying(ui.playing, sec);
    if (videoLoaded) {
      deckVideo.playbackRate = ui.speed;
      if (ui.playing) deckVideo.play().catch(() => {});
      else deckVideo.pause();
    }
  }
  function jumpToFrame(f, pauseFirst) {
    if (pauseFirst && ui.playing) setPlaying(false);
    ui.frameF = clamp(f, 1.0, H.frames);
    syncExternalMedia();
  }

  $('btn-play').addEventListener('click', () => { audio.unlock(); setPlaying(!ui.playing); });
  $('btn-start').addEventListener('click', () => { audio.unlock(); jumpToFrame(1.0, false); });
  $('btn-end').addEventListener('click', () => { audio.unlock(); jumpToFrame(H.frames, false); });
  $('btn-back').addEventListener('click', () => {
    audio.unlock();
    const prev = Math.round(ui.frameF) - 1;
    jumpToFrame(prev < 1 ? (ui.loop ? H.frames : 1) : prev, true);
  });
  $('btn-fwd').addEventListener('click', () => {
    audio.unlock();
    const next = Math.round(ui.frameF) + 1;
    jumpToFrame(next > H.frames ? (ui.loop ? 1 : H.frames) : next, true);
  });
  $('btn-loop').addEventListener('click', () => {
    ui.loop = !ui.loop;
    $('btn-loop').classList.toggle('on', ui.loop);
  });
  $('sel-speed').addEventListener('change', (e) => {
    ui.speed = parseFloat(e.target.value) || 1.0;
    audio.setSpeed(ui.speed, (ui.frameF - 1.0) / FPS);
    if (videoLoaded) deckVideo.playbackRate = ui.speed;
  });
  $('btn-mute').addEventListener('click', () => {
    audio.unlock();
    const nextMute = !audio.state.muted;
    audio.setMuted(nextMute);
    $('btn-mute').textContent = nextMute ? 'MUTED' : 'AUDIO ON';
    $('btn-mute').classList.toggle('on', !nextMute);
  });
  $('btn-help').addEventListener('click', () => $('help').classList.toggle('show'));

  // PiP / Split-Screen switcher
  function setPipMode(mode) {
    ui.pipMode = mode;
    $('btn-pip').classList.toggle('on', mode === 'pip');
    $('btn-split').classList.toggle('on', mode === 'split');
    $('btn-pipoff').classList.toggle('on', mode === 'off');
    pipView.style.display = mode === 'off' ? 'none' : 'block';
  }
  $('btn-pip').addEventListener('click', () => setPipMode('pip'));
  $('btn-split').addEventListener('click', () => setPipMode('split'));
  $('btn-pipoff').addEventListener('click', () => setPipMode('off'));

  // One-click PNG Snapshot Export
  function exportPngSnapshot() {
    const fStr = String(Math.round(ui.frameF)).padStart(3, '0');
    glCanvas.toBlob((blob) => {
      if (!blob) return;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'cyberpunk_ramen_f' + fStr + '.png';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
      toast('Saved snapshot: cyberpunk_ramen_f' + fStr + '.png');
    }, 'image/png');
  }
  $('btn-png').addEventListener('click', exportPngSnapshot);

  // One-click .webm Canvas + WebAudio Recorder
  let mediaRecorder = null;
  let recChunks = [];
  function toggleWebmRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      return;
    }
    if (typeof glCanvas.captureStream !== 'function' || typeof MediaRecorder === 'undefined') {
      toast('MediaRecorder unavailable in this browser');
      return;
    }
    audio.unlock();
    const stream = glCanvas.captureStream(FPS);
    const aStream = audio.getMediaStream();
    if (aStream) {
      const tracks = aStream.getAudioTracks();
      for (let i = 0; i < tracks.length; i++) stream.addTrack(tracks[i]);
    }
    const mime = MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
      ? 'video/webm;codecs=vp9,opus'
      : (MediaRecorder.isTypeSupported('video/webm;codecs=vp8,opus') ? 'video/webm;codecs=vp8,opus' : 'video/webm');
    recChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12000000 });
    mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      $('btn-rec').classList.remove('on');
      $('btn-rec').textContent = 'REC .WEBM';
      $('pip-rec').classList.remove('on');
      const blob = new Blob(recChunks, { type: mime });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'cyberpunk_diorama_240f.webm';
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 8000);
      toast('Exported video: cyberpunk_diorama_240f.webm (' + fmtBytes(blob.size) + ')');
    };
    mediaRecorder.start(200);
    $('btn-rec').classList.add('on');
    $('btn-rec').textContent = 'STOP REC';
    $('pip-rec').classList.add('on');
    if (!ui.playing) setPlaying(true);
    toast('Recording WebGL2 + WebAudio to .webm');
  }
  $('btn-rec').addEventListener('click', toggleWebmRecording);

  // Interactive 240-Frame Scrubber Canvas with TimelineMarker flags at f1, f97, f169
  const scrubCanvas = $('scrubber');
  const scrubCtx = scrubCanvas.getContext('2d');
  let scrubbing = false;
  function scrubFromEvent(e) {
    const rect = scrubCanvas.getBoundingClientRect();
    const pad = 12;
    const u = clamp((e.clientX - rect.left - pad) / Math.max(1, rect.width - pad * 2), 0.0, 1.0);
    jumpToFrame(1.0 + u * (H.frames - 1), false);
  }
  scrubCanvas.addEventListener('pointerdown', (e) => {
    audio.unlock();
    scrubbing = true;
    scrubCanvas.setPointerCapture(e.pointerId);
    scrubFromEvent(e);
  });
  scrubCanvas.addEventListener('pointermove', (e) => { if (scrubbing) scrubFromEvent(e); });
  scrubCanvas.addEventListener('pointerup', () => { scrubbing = false; });
  scrubCanvas.addEventListener('pointercancel', () => { scrubbing = false; });

  function drawScrubber() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(240, Math.floor(scrubCanvas.clientWidth * dpr));
    const h = Math.max(40, Math.floor(scrubCanvas.clientHeight * dpr));
    if (scrubCanvas.width !== w || scrubCanvas.height !== h) {
      scrubCanvas.width = w;
      scrubCanvas.height = h;
    }
    const ctx = scrubCtx;
    ctx.clearRect(0, 0, w, h);
    const pad = 12 * dpr;
    const trackW = w - pad * 2;
    const xOf = (f) => pad + ((f - 1) / (H.frames - 1)) * trackW;

    // Shot range bands (f1..96, f97..168, f169..240)
    const shotColors = ['rgba(0, 234, 255, 0.09)', 'rgba(255, 43, 214, 0.09)', 'rgba(255, 179, 64, 0.09)'];
    for (let i = 0; i < H.markers.length; i++) {
      const fStart = H.markers[i].frame;
      const fEnd = i + 1 < H.markers.length ? H.markers[i + 1].frame : H.frames;
      const x0 = xOf(fStart), x1 = xOf(fEnd);
      ctx.fillStyle = shotColors[i % shotColors.length];
      ctx.fillRect(x0, 18 * dpr, x1 - x0, h - 22 * dpr);
    }

    // Frame ruler ticks
    ctx.strokeStyle = 'rgba(120, 149, 173, 0.28)';
    ctx.lineWidth = 1;
    ctx.font = Math.round(9 * dpr) + 'px Cascadia Mono, Consolas, monospace';
    ctx.fillStyle = 'rgba(120, 149, 173, 0.80)';
    for (let f = 1; f <= H.frames; f += 8) {
      const x = Math.floor(xOf(f)) + 0.5;
      const major = (f === 1 || f % 24 === 0);
      ctx.beginPath();
      ctx.moveTo(x, major ? h - 18 * dpr : h - 10 * dpr);
      ctx.lineTo(x, h - 2 * dpr);
      ctx.stroke();
      if (major) ctx.fillText('f' + f, x - 8 * dpr, h - 20 * dpr);
    }

    // Baked keyframe diamonds
    ctx.fillStyle = 'rgba(0, 234, 255, 0.65)';
    for (let i = 0; i < H.keyframes.length; i++) {
      const kx = xOf(H.keyframes[i]);
      const ky = h - 7 * dpr;
      ctx.fillRect(kx - 1.5 * dpr, ky - 1.5 * dpr, 3 * dpr, 3 * dpr);
    }

    // TimelineMarker flags at f1, f97, f169
    const flagColors = ['#00eaff', '#ff2bd6', '#ffb340'];
    for (let i = 0; i < H.markers.length; i++) {
      const mk = H.markers[i];
      const mx = Math.floor(xOf(mk.frame)) + 0.5;
      const col = flagColors[i % flagColors.length];
      ctx.strokeStyle = col;
      ctx.lineWidth = 1.5 * dpr;
      ctx.beginPath();
      ctx.moveTo(mx, 2 * dpr);
      ctx.lineTo(mx, h - 2 * dpr);
      ctx.stroke();

      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.moveTo(mx, 2 * dpr);
      ctx.lineTo(mx + 8 * dpr, 6 * dpr);
      ctx.lineTo(mx, 10 * dpr);
      ctx.closePath();
      ctx.fill();
      ctx.fillText(mk.name + ' (f' + mk.frame + ')', mx + 10 * dpr, 11 * dpr);
    }

    // Playhead cursor
    const px = xOf(ui.frameF);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
    ctx.fillStyle = '#00eaff';
    ctx.fillRect(px - 4 * dpr, 0, 8 * dpr, 5 * dpr);
  }

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
    audio.unlock();
    if (e.code === 'Space') { e.preventDefault(); setPlaying(!ui.playing); }
    else if (e.code === 'Comma') { e.preventDefault(); $('btn-back').click(); }
    else if (e.code === 'Period') { e.preventDefault(); $('btn-fwd').click(); }
    else if (e.code === 'Home') { e.preventDefault(); jumpToFrame(1.0, false); }
    else if (e.code === 'End') { e.preventDefault(); jumpToFrame(H.frames, false); }
    else if (e.code === 'KeyL') $('btn-loop').click();
    else if (e.code === 'KeyF') setCamMode('free');
    else if (e.code === 'KeyC') setCamMode('director');
    else if (e.code === 'Digit1') setCamMode('0');
    else if (e.code === 'Digit2') setCamMode('1');
    else if (e.code === 'Digit3') setCamMode('2');
    else if (e.code === 'KeyZ') setAov(0);
    else if (e.code === 'KeyX') setAov(1);
    else if (e.code === 'KeyV') setAov(2);
    else if (e.code === 'KeyN') setAov(3);
    else if (e.code === 'KeyB') setAov(4);
    else if (e.code === 'KeyR') { camSys.resetFree(); setCamMode('free'); }
    else if (e.code === 'KeyP') exportPngSnapshot();
    else if (e.code === 'KeyM') $('btn-mute').click();
    else if (e.code === 'KeyU') document.body.classList.toggle('hidden-ui');
    else if (e.code === 'KeyH' || e.key === '?') $('help').classList.toggle('show');
  });

  // Unlock WebAudio on first pointer click anywhere
  window.addEventListener('pointerdown', () => audio.unlock(), { once: true });
  setPlaying(true);

  // Hide loader
  loaderStep('READY // ' + (performance.now() - tBoot).toFixed(0) + 'ms boot', 1.0, 'ok');
  $('loader').classList.add('done');

  // Expose runtime inspection hook for automated verification
  STUDIO.ready = true;
  STUDIO.header = H;
  STUDIO.ui = ui;
  STUDIO.audio = audio;
  STUDIO.renderer = renderer;

  // Main 60fps Render & Audio-Reactive Loop
  let lastNow = performance.now();
  let fpsAccum = 0, fpsFrames = 0, fpsDisplay = 60.0;

  function frameLoop(now) {
    const dt = Math.min(0.1, Math.max(0.0001, (now - lastNow) * 0.001));
    lastNow = now;
    fpsAccum += dt;
    fpsFrames++;
    if (fpsAccum >= 0.35) {
      fpsDisplay = fpsFrames / fpsAccum;
      $('st-fps').textContent = fpsDisplay.toFixed(1);
      fpsAccum = 0;
      fpsFrames = 0;
    }

    if (ui.playing && !scrubbing) {
      ui.frameF += dt * FPS * ui.speed;
      if (ui.frameF > H.frames + 0.999) {
        if (ui.loop) {
          ui.frameF = 1.0 + ((ui.frameF - 1.0) % H.frames);
          syncExternalMedia();
        } else {
          ui.frameF = H.frames;
          setPlaying(false);
        }
      }
    }

    camSys.updateFly(dt);

    const timeSec = (ui.frameF - 1.0) / FPS;
    const audioBands = audio.sampleAndRenderHUD(timeSec);
    const rainSpeed = 11.5 * (1.0 + (ui.audioReactive ? audioBands[2] * 1.45 : 0.0));
    ui.rainPhase = (ui.rainPhase + dt * rainSpeed) % 1000.0;

    evaluator.evaluateFrame(ui.frameF, audioBands, ui.audioReactive ? 1.0 : 0.0);

    // Resize WebGL2 canvas & FBOs
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const cw = Math.max(64, Math.floor(glCanvas.clientWidth * dpr));
    const ch = Math.max(64, Math.floor(glCanvas.clientHeight * dpr));
    if (glCanvas.width !== cw || glCanvas.height !== ch) {
      glCanvas.width = cw;
      glCanvas.height = ch;
    }
    renderer.ensureTargets(cw, ch);

    const passOpts = {
      frameF: ui.frameF,
      timeSec: timeSec,
      aov: ui.aov,
      bloom: ui.bloom,
      mist: ui.mist,
      chroma: ui.chroma,
      mirror: ui.mirror,
      rain: ui.rain,
      audioReactive: ui.audioReactive,
      paths: ui.paths,
      streaks: ui.streaks,
      exposureBias: ui.exposureBias,
      bloomGain: ui.bloomGain,
      rainDensity: ui.rainDensity,
      rainPhase: ui.rainPhase,
      audioBands: audioBands,
    };

    let totalDraws = 0;
    let activeCamInfo = null;

    if (ui.pipMode === 'split') {
      const halfW = cw >> 1;
      const leftCam = camSys.computeCamera(ui.camMode === 'director' ? 'free' : ui.camMode, ui.frameF, halfW / ch);
      totalDraws += renderViewportPass(renderer, evaluator, H, leftCam, { x: 0, y: 0, w: halfW, h: ch }, passOpts);
      const rightCam = camSys.computeCamera('director', ui.frameF, (cw - halfW) / ch);
      totalDraws += renderViewportPass(renderer, evaluator, H, rightCam, { x: halfW, y: 0, w: cw - halfW, h: ch }, passOpts);
      activeCamInfo = rightCam;
      if (!videoLoaded) {
        const pw = Math.max(160, Math.floor(pipCanvas.clientWidth * dpr));
        const ph = Math.max(90, Math.floor(pipCanvas.clientHeight * dpr));
        if (pipCanvas.width !== pw || pipCanvas.height !== ph) { pipCanvas.width = pw; pipCanvas.height = ph; }
        pipCtx.drawImage(glCanvas, halfW, 0, cw - halfW, ch, 0, 0, pw, ph);
      }
    } else {
      if (ui.pipMode === 'pip' && !videoLoaded && ui.camMode !== 'director') {
        const pw = Math.min(cw, Math.max(160, Math.floor(pipCanvas.clientWidth * dpr)));
        const ph = Math.min(ch, Math.max(90, Math.floor(pipCanvas.clientHeight * dpr)));
        if (pipCanvas.width !== pw || pipCanvas.height !== ph) { pipCanvas.width = pw; pipCanvas.height = ph; }
        const dirCam = camSys.computeCamera('director', ui.frameF, pw / ph);
        const pipOpts = Object.assign({}, passOpts, { mirror: false, streaks: false });
        totalDraws += renderViewportPass(renderer, evaluator, H, dirCam, { x: 0, y: 0, w: pw, h: ph }, pipOpts);
        pipCtx.drawImage(glCanvas, 0, ch - ph, pw, ph, 0, 0, pw, ph);
      }
      const mainCam = camSys.computeCamera(ui.camMode, ui.frameF, cw / ch);
      totalDraws += renderViewportPass(renderer, evaluator, H, mainCam, { x: 0, y: 0, w: cw, h: ch }, passOpts);
      activeCamInfo = mainCam;
      if (ui.pipMode === 'pip' && !videoLoaded && ui.camMode === 'director') {
        const pw = Math.max(160, Math.floor(pipCanvas.clientWidth * dpr));
        const ph = Math.max(90, Math.floor(pipCanvas.clientHeight * dpr));
        if (pipCanvas.width !== pw || pipCanvas.height !== ph) { pipCanvas.width = pw; pipCanvas.height = ph; }
        pipCtx.drawImage(glCanvas, 0, 0, cw, ch, 0, 0, pw, ph);
      }
    }

    // Update HUD readouts & timeline scrubber
    const fInt = clamp(Math.round(ui.frameF), 1, H.frames);
    const mk = evaluator.directorMarkerForFrame(ui.frameF);
    $('st-frame').textContent = fInt + ' / ' + H.frames;
    $('st-draws').textContent = String(totalDraws);
    $('st-cam').textContent = activeCamInfo ? activeCamInfo.label : ui.camMode.toUpperCase();
    $('st-shot').textContent = mk.name;
    $('pip-shot').textContent = mk.name + ' // F' + String(fInt).padStart(3, '0');
    $('tc').innerHTML = 'F' + String(fInt).padStart(3, '0') + ' / ' + H.frames +
      '<small id="tc-time">00:' + timeSec.toFixed(2).padStart(5, '0') + '</small>';
    STUDIO.frame = fInt;
    STUDIO.stats = { fps: fpsDisplay, draws: totalDraws, shot: mk.name, audioBands: Array.from(audioBands) };

    drawScrubber();
    requestAnimationFrame(frameLoop);
  }
  requestAnimationFrame(frameLoop);
}

bootStudio().catch((err) => {
  report(err);
  fatal('STUDIO INIT FAILED: ' + ((err && err.message) || err));
});
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())

