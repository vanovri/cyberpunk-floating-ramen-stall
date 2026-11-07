#!/usr/bin/env python3
"""
render_diorama_animation.py
===========================

GOD MODE BLENDER PHASE 7A -- 240-frame animation evaluator & multiprocess
software Z-buffer / planar-reflection / neon-bloom render engine for
``cyberpunk_floating_ramen_stall.py``.

Pure Python 3.14 standard library only (math, struct, zlib, bisect,
multiprocessing, hashlib, json, pickle, tempfile).

Pipeline
--------
1. TRACK 25 -- scene evaluation
   * The Blender scene script is executed once against the stdlib ``bpy`` /
     ``bmesh`` / ``mathutils`` mock that ships in ``verify_blender_script.py``.
     The resulting data-blocks are flattened into a picklable snapshot
     (meshes, curves -> tube meshes, FONT -> bitmap-glyph meshes, ARRAY and
     LATTICE modifiers baked, WAVE / ARMATURE deformers recorded, materials
     resolved by walking the shader-node links, F-curves with per-key
     interpolation, constraints, bones, lights, cameras, timeline markers).
   * ``evaluate_scene_at_frame(snapshot, frame)`` evaluates every F-curve
     (BEZIER auto-clamped ease-in-out / LINEAR / CONSTANT), the object parent
     hierarchy, FOLLOW_PATH (arc-length parametrised Bezier/NURBS/POLY paths,
     optional curve-follow) and TRACK_TO constraints, PoseBone armature
     kinematics, light energies, neon ballast flicker and the TimelineMarker
     director cut (85mm orbit f1-96, 50mm hero-low f97-168, 105mm macro
     f169-240).

2. TRACK 26 -- rendering (one frame per worker via multiprocessing.Pool)
   * Near-plane clipped, perspective-correct barycentric span rasteriser with
     a float 1/z buffer (deferred G-buffer: triangle id + depth).
   * Multi-light shading: point / spot / one-sided area lights, Lambert
     diffuse + GGX specular (Smith-Schlick visibility, Schlick Fresnel),
     hemispheric ambient, emission.
   * Planar mirror reflection of the scene across Z = 0 (half-resolution
     mirrored-camera pass) composited onto the wet asphalt with Fresnel and a
     procedural puddle mask.
   * Sorted alpha / additive transparent pass (rain glass, steam, umbrellas,
     holograms), GN rain streaks, quadratic volumetric mist, threshold +
     separable box (3x ~ Gaussian) neon emission bloom, AgX base + High
     Contrast look tone mapping, vignette.
   * Stills ``render_f001_cam1.png``, ``render_f120_cam2.png``,
     ``render_f200_cam3.png``; all 240 RGB buffers are handed to
     ``video_muxer.py`` (AVI + MP4) and cached in a zlib frame cache.

Usage::

    python render_diorama_animation.py                  # full 240 frames + mux
    python render_diorama_animation.py --frames 1,120,200 --no-mux
    python render_diorama_animation.py --width 512 --height 288 --workers 8
"""

import bisect
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import shutil
import struct
import sys
import tempfile
import time
import zlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_SCRIPT_NAME = "cyberpunk_floating_ramen_stall.py"
WAV_NAME = "cyberpunk_ramen_soundtrack.wav"
RENDER_REPORT_NAME = "cyberpunk_diorama_render_report.json"

FPS = 24
FRAME_START = 1
FRAME_END = 240
DEFAULT_W = 320   # multiple of 16 -> exact H.264 macroblock grid (20 x 12 MBs)
DEFAULT_H = 192
STILLS = (
    (1, "render_f001_cam1.png"),
    (120, "render_f120_cam2.png"),
    (200, "render_f200_cam3.png"),
)

PI = math.pi
INV_PI = 1.0 / math.pi

# Material kinds
K_OPAQUE = 0
K_ASPHALT = 1
K_WINDOWS = 2
K_GLASS = 3
K_HOLO = 4
K_LANTERN = 5
K_NEON = 6

# Light kinds
L_POINT = 0
L_AREA = 1
L_SPOT = 2

# Look / calibration constants
EXPOSURE_EV = 0.15            # scene.view_settings.exposure
EXPOSURE_CALIBRATION = 1.35   # radiometric -> display calibration of this software model
AMB_SKY = (0.020, 0.024, 0.042)
AMB_GROUND = (0.030, 0.020, 0.036)
ENV_SPEC = (0.030, 0.034, 0.060)
MIST_START = 8.0              # world.mist_settings.start
MIST_DEPTH = 28.0             # world.mist_settings.depth
MIST_COLOR = (0.030, 0.038, 0.066)
MIST_MAX = 0.78
BLOOM_THRESHOLD = 1.0
BLOOM_SMALL = 0.55
BLOOM_LARGE = 0.45
RAIN_FALL_SPEED = 9.0         # m/s terminal velocity of GN streak instances
RAIN_DROPS_PER_SHEET = 1800

# ==============================================================================
# 1. AFFINE MATH (3x4 row-major tuples)
# ==============================================================================

IDENT = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
MIRROR_Z = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0)


def m_mul(a, b):
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = a
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11 = b
    return (
        a0 * b0 + a1 * b4 + a2 * b8, a0 * b1 + a1 * b5 + a2 * b9, a0 * b2 + a1 * b6 + a2 * b10, a0 * b3 + a1 * b7 + a2 * b11 + a3,
        a4 * b0 + a5 * b4 + a6 * b8, a4 * b1 + a5 * b5 + a6 * b9, a4 * b2 + a5 * b6 + a6 * b10, a4 * b3 + a5 * b7 + a6 * b11 + a7,
        a8 * b0 + a9 * b4 + a10 * b8, a8 * b1 + a9 * b5 + a10 * b9, a8 * b2 + a9 * b6 + a10 * b10, a8 * b3 + a9 * b7 + a10 * b11 + a11,
    )


def m_compose(loc, rot, scale):
    """Blender object basis: T @ R(Euler XYZ = Rz @ Ry @ Rx) @ S."""
    cx, sx = math.cos(rot[0]), math.sin(rot[0])
    cy, sy = math.cos(rot[1]), math.sin(rot[1])
    cz, sz = math.cos(rot[2]), math.sin(rot[2])
    kx, ky, kz = scale
    return (
        cy * cz * kx, (cz * sx * sy - cx * sz) * ky, (cx * cz * sy + sx * sz) * kz, loc[0],
        cy * sz * kx, (cx * cz + sx * sy * sz) * ky, (cx * sy * sz - cz * sx) * kz, loc[1],
        -sy * kx, cy * sx * ky, cx * cy * kz, loc[2],
    )


def m_inv(m):
    a, b, c, tx, d, e, f, ty, g, h, i, tz = m
    c00 = e * i - f * h
    c01 = -(d * i - f * g)
    c02 = d * h - e * g
    det = a * c00 + b * c01 + c * c02
    if abs(det) < 1e-20:
        return IDENT
    inv = 1.0 / det
    r00, r01, r02 = c00 * inv, -(b * i - c * h) * inv, (b * f - c * e) * inv
    r10, r11, r12 = c01 * inv, (a * i - c * g) * inv, -(a * f - c * d) * inv
    r20, r21, r22 = c02 * inv, -(a * h - b * g) * inv, (a * e - b * d) * inv
    return (
        r00, r01, r02, -(r00 * tx + r01 * ty + r02 * tz),
        r10, r11, r12, -(r10 * tx + r11 * ty + r12 * tz),
        r20, r21, r22, -(r20 * tx + r21 * ty + r22 * tz),
    )


def m_cofactor3(m):
    """Cofactor matrix of the linear part (inverse-transpose up to det scale)."""
    a, b, c, _tx, d, e, f, _ty, g, h, i, _tz = m
    return (
        e * i - f * h, -(d * i - f * g), d * h - e * g,
        -(b * i - c * h), a * i - c * g, -(a * h - b * g),
        b * f - c * e, -(a * f - c * d), a * e - b * d,
    )


def m_apply(m, p):
    x, y, z = p
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    )


def m_apply_dir(m, p):
    x, y, z = p
    return (m[0] * x + m[1] * y + m[2] * z, m[4] * x + m[5] * y + m[6] * z, m[8] * x + m[9] * y + m[10] * z)


def m_translation(m):
    return (m[3], m[7], m[11])


def m_max_scale(m):
    return max(
        math.sqrt(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]),
        math.sqrt(m[1] * m[1] + m[5] * m[5] + m[9] * m[9]),
        math.sqrt(m[2] * m[2] + m[6] * m[6] + m[10] * m[10]),
    )


def v_norm(v):
    x, y, z = v
    ln = math.sqrt(x * x + y * y + z * z)
    if ln < 1e-20:
        return (0.0, 0.0, 1.0)
    return (x / ln, y / ln, z / ln)


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def xform_points(m, pts):
    a, b, c, d, e, f, g, h, i, j, k, l = m
    return [(a * x + b * y + c * z + d, e * x + f * y + g * z + h, i * x + j * y + k * z + l) for (x, y, z) in pts]


def xform_normals(cof, nrm):
    a, b, c, d, e, f, g, h, i = cof
    out = []
    ap = out.append
    sq = math.sqrt
    for (x, y, z) in nrm:
        nx = a * x + b * y + c * z
        ny = d * x + e * y + f * z
        nz = g * x + h * y + i * z
        ln = sq(nx * nx + ny * ny + nz * nz)
        if ln > 1e-20:
            ap((nx / ln, ny / ln, nz / ln))
        else:
            ap((0.0, 0.0, 1.0))
    return out


def _axis_code(name):
    """'TRACK_NEGATIVE_Z' / 'FORWARD_Y' / 'UP_Z' -> '-Z' | 'Y' | 'Z' ..."""
    s = str(name).upper()
    neg = "NEGATIVE" in s
    return ("-" if neg else "") + s[-1]


def rot_from_axes(track_code, tvec, up_code, uvec):
    """Right-handed rotation whose `track` axis points along tvec and `up` axis toward uvec."""
    if track_code[-1] == up_code[-1]:
        up_code = 'Y' if track_code[-1] != 'Y' else 'Z'
    t = v_norm(tvec)
    axes = {}
    sgn = -1.0 if track_code.startswith('-') else 1.0
    axes[track_code[-1]] = (t[0] * sgn, t[1] * sgn, t[2] * sgn)
    d = v_dot(uvec, t)
    u = (uvec[0] - d * t[0], uvec[1] - d * t[1], uvec[2] - d * t[2])
    if math.sqrt(v_dot(u, u)) < 1e-6:
        alt = (0.0, 1.0, 0.0) if abs(t[1]) < 0.9 else (1.0, 0.0, 0.0)
        d = v_dot(alt, t)
        u = (alt[0] - d * t[0], alt[1] - d * t[1], alt[2] - d * t[2])
    u = v_norm(u)
    usgn = -1.0 if up_code.startswith('-') else 1.0
    axes[up_code[-1]] = (u[0] * usgn, u[1] * usgn, u[2] * usgn)
    missing = ({'X', 'Y', 'Z'} - set(axes)).pop()
    if missing == 'X':
        axes['X'] = v_cross(axes['Y'], axes['Z'])
    elif missing == 'Y':
        axes['Y'] = v_cross(axes['Z'], axes['X'])
    else:
        axes['Z'] = v_cross(axes['X'], axes['Y'])
    return axes['X'], axes['Y'], axes['Z']


def m_from_cols(cx, cy, cz, t):
    return (cx[0], cy[0], cz[0], t[0], cx[1], cy[1], cz[1], t[1], cx[2], cy[2], cz[2], t[2])


# ==============================================================================
# 2. F-CURVE PREPARATION & EVALUATION (BEZIER auto-clamped / LINEAR / CONSTANT)
# ==============================================================================

def prepare_fcurve(raw_keys):
    """raw_keys: iterable of (frame, value, interpolation). Duplicate frames: last write wins."""
    dedup = {}
    for f, v, ip in raw_keys:
        dedup[float(f)] = (v, ip)
    frames = sorted(dedup)
    if not frames:
        return None
    first_val = dedup[frames[0]][0]
    scalar = not isinstance(first_val, (tuple, list))
    values = []
    interps = []
    for f in frames:
        v, ip = dedup[f]
        values.append((float(v),) if scalar else tuple(float(c) for c in v))
        interps.append(str(ip))
    dim = len(values[0])
    n = len(frames)
    slopes = []
    for i in range(n):
        if i == 0 or i == n - 1:
            slopes.append((0.0,) * dim)   # AUTO_CLAMPED end handles are flat
            continue
        f0, f1, f2 = frames[i - 1], frames[i], frames[i + 1]
        sl = []
        for c in range(dim):
            v0, v1, v2 = values[i - 1][c], values[i][c], values[i + 1][c]
            d1 = (v1 - v0) / (f1 - f0)
            d2 = (v2 - v1) / (f2 - f1)
            if d1 * d2 <= 0.0:
                sl.append(0.0)        # local extremum -> clamped flat handle (no overshoot)
            else:
                m = (v2 - v0) / (f2 - f0)
                lim = 3.0 * min(abs(d1), abs(d2))
                if abs(m) > lim:
                    m = math.copysign(lim, m)
                sl.append(m)
        slopes.append(tuple(sl))
    return {"frames": frames, "values": values, "interp": interps, "slopes": slopes, "scalar": scalar}


def eval_fcurve(fc, frame):
    fr = fc["frames"]
    vals = fc["values"]
    if frame <= fr[0]:
        out = vals[0]
    elif frame >= fr[-1]:
        out = vals[-1]
    else:
        i = bisect.bisect_right(fr, frame) - 1
        f0, f1 = fr[i], fr[i + 1]
        v0, v1 = vals[i], vals[i + 1]
        h = f1 - f0
        s = (frame - f0) / h
        ip = fc["interp"][i]
        if ip == 'CONSTANT':
            out = v0
        elif ip == 'LINEAR':
            out = tuple(a + (b - a) * s for a, b in zip(v0, v1))
        else:
            # Cubic Bezier with 1/3-interval handles == Hermite on (value, auto-clamped slope)
            m0, m1 = fc["slopes"][i], fc["slopes"][i + 1]
            s2 = s * s
            s3 = s2 * s
            h00 = 2 * s3 - 3 * s2 + 1
            h10 = s3 - 2 * s2 + s
            h01 = -2 * s3 + 3 * s2
            h11 = s3 - s2
            out = tuple(h00 * a + h10 * h * ma + h01 * b + h11 * h * mb for a, b, ma, mb in zip(v0, v1, m0, m1))
    return out[0] if fc["scalar"] else out


# ==============================================================================
# 3. CURVE SAMPLING (POLY / NURBS / BEZIER) & ARC-LENGTH PATHS
# ==============================================================================

def _nurbs_eval(ctrl, order, endpoint, cyclic, samples_per_span):
    n = len(ctrl)
    if n < 2:
        return [tuple(ctrl[0])] if ctrl else []
    k = max(2, min(order, n))
    pts = list(ctrl)
    if cyclic:
        pts = pts + pts[:k - 1]
        n = len(pts)
        knots = [float(i) for i in range(n + k)]
        t0, t1 = float(k - 1), float(n)
    elif endpoint:
        knots = [0.0] * k + [float(i) for i in range(1, n - k + 1)] + [float(n - k + 1)] * k
        t0, t1 = 0.0, float(n - k + 1)
    else:
        knots = [float(i) for i in range(n + k)]
        t0, t1 = float(k - 1), float(n)
    spans = max(1, int(round(t1 - t0)))
    count = spans * samples_per_span + 1
    out = []
    for si in range(count):
        t = t0 + (t1 - t0) * si / (count - 1)
        if t >= t1:
            t = t1 - 1e-9
        basis = [1.0 if knots[i] <= t < knots[i + 1] else 0.0 for i in range(len(knots) - 1)]
        for d in range(2, k + 1):
            nb = []
            for i in range(len(knots) - d):
                a = 0.0
                den1 = knots[i + d - 1] - knots[i]
                den2 = knots[i + d] - knots[i + 1]
                if den1 > 0.0 and basis[i] != 0.0:
                    a += (t - knots[i]) / den1 * basis[i]
                if den2 > 0.0 and basis[i + 1] != 0.0:
                    a += (knots[i + d] - t) / den2 * basis[i + 1]
                nb.append(a)
            basis = nb
        x = y = z = 0.0
        for i in range(n):
            w = basis[i]
            if w:
                x += w * pts[i][0]
                y += w * pts[i][1]
                z += w * pts[i][2]
        out.append((x, y, z))
    if cyclic and out:
        out = out[:-1]
    return out


def _bezier_eval(bpts, cyclic, res):
    out = []
    n = len(bpts)
    segs = n if cyclic else n - 1
    for s in range(segs):
        p0, h0r = bpts[s][0], bpts[s][2]
        nxt = bpts[(s + 1) % n]
        p3, h3l = nxt[0], nxt[1]
        for i in range(res):
            t = i / res
            u = 1.0 - t
            b0, b1, b2, b3 = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
            out.append(tuple(b0 * p0[c] + b1 * h0r[c] + b2 * h3l[c] + b3 * p3[c] for c in range(3)))
    if not cyclic and n:
        out.append(tuple(bpts[-1][0]))
    return out


def sample_spline(sp, res_hint):
    """Return (points, cyclic) for a flattened spline record."""
    st = sp["type"]
    cyc = sp["cyclic"]
    if st == 'BEZIER':
        return _bezier_eval(sp["bezier"], cyc, max(4, res_hint)), cyc
    ctrl = sp["points"]
    if st == 'NURBS':
        return _nurbs_eval(ctrl, sp["order"], sp["endpoint"], cyc, max(2, res_hint)), cyc
    return list(ctrl), cyc


def build_path_table(spline_rec):
    pts, cyc = sample_spline(spline_rec, 48)
    if cyc and pts:
        pts = pts + [pts[0]]
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + math.dist(pts[i - 1], pts[i]))
    return {"pts": pts, "cum": cum, "cyclic": cyc, "length": cum[-1]}


def path_point(table, u):
    """Arc-length parametrised point + unit tangent on a sampled path (offset_factor u in [0,1])."""
    pts, cum = table["pts"], table["cum"]
    total = table["length"]
    if total <= 0.0 or len(pts) < 2:
        return (pts[0] if pts else (0.0, 0.0, 0.0)), (0.0, 1.0, 0.0)
    if table["cyclic"]:
        u = u % 1.0
    else:
        u = min(1.0, max(0.0, u))
    s = u * total
    i = bisect.bisect_right(cum, s) - 1
    i = max(0, min(len(pts) - 2, i))
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg <= 0 else (s - cum[i]) / seg
    a, b = pts[i], pts[i + 1]
    p = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)
    return p, v_norm(v_sub(b, a))


# ==============================================================================
# 4. 5x7 BITMAP FONT FOR FONT CURVES
# ==============================================================================

FONT_5X7 = {
    'A': "01110 10001 10001 11111 10001 10001 10001", 'B': "11110 10001 10001 11110 10001 10001 11110",
    'C': "01110 10001 10000 10000 10000 10001 01110", 'D': "11110 10001 10001 10001 10001 10001 11110",
    'E': "11111 10000 10000 11110 10000 10000 11111", 'F': "11111 10000 10000 11110 10000 10000 10000",
    'G': "01110 10001 10000 10111 10001 10001 01111", 'H': "10001 10001 10001 11111 10001 10001 10001",
    'I': "01110 00100 00100 00100 00100 00100 01110", 'J': "00111 00010 00010 00010 00010 10010 01100",
    'K': "10001 10010 10100 11000 10100 10010 10001", 'L': "10000 10000 10000 10000 10000 10000 11111",
    'M': "10001 11011 10101 10101 10001 10001 10001", 'N': "10001 10001 11001 10101 10011 10001 10001",
    'O': "01110 10001 10001 10001 10001 10001 01110", 'P': "11110 10001 10001 11110 10000 10000 10000",
    'Q': "01110 10001 10001 10001 10101 10010 01101", 'R': "11110 10001 10001 11110 10100 10010 10001",
    'S': "01111 10000 10000 01110 00001 00001 11110", 'T': "11111 00100 00100 00100 00100 00100 00100",
    'U': "10001 10001 10001 10001 10001 10001 01110", 'V': "10001 10001 10001 10001 10001 01010 00100",
    'W': "10001 10001 10001 10101 10101 10101 01010", 'X': "10001 10001 01010 00100 01010 10001 10001",
    'Y': "10001 10001 01010 00100 00100 00100 00100", 'Z': "11111 00001 00010 00100 01000 10000 11111",
    '0': "01110 10001 10011 10101 11001 10001 01110", '1': "00100 01100 00100 00100 00100 00100 01110",
    '2': "01110 10001 00001 00010 00100 01000 11111", '3': "11111 00010 00100 00010 00001 10001 01110",
    '4': "00010 00110 01010 10010 11111 00010 00010", '5': "11111 10000 11110 00001 00001 10001 01110",
    '6': "00110 01000 10000 11110 10001 10001 01110", '7': "11111 00001 00010 00100 01000 01000 01000",
    '8': "01110 10001 10001 01110 10001 10001 01110", '9': "01110 10001 10001 01111 00001 00010 01100",
    '-': "00000 00000 00000 11111 00000 00000 00000", '/': "00001 00010 00010 00100 01000 01000 10000",
    '.': "00000 00000 00000 00000 00000 01100 01100", ':': "00000 01100 01100 00000 01100 01100 00000",
    ' ': "00000 00000 00000 00000 00000 00000 00000",
}


def font_glyph_quads(body, size, extrude):
    """Merge horizontal runs of lit glyph dots into front-facing quads (local XY plane, centred)."""
    ds = size * 0.1
    text = str(body).upper()
    total_w = (len(text) * 6 - 1) * ds
    x_origin = -total_w * 0.5
    y_origin = -3.5 * ds
    zf = max(extrude, 0.0005)
    verts = []
    quads = []
    for ci, ch in enumerate(text):
        rows = FONT_5X7.get(ch, FONT_5X7[' ']).split()
        gx = x_origin + ci * 6 * ds
        for r, row in enumerate(rows):
            y0 = y_origin + (6 - r) * ds
            c = 0
            while c < 5:
                if row[c] == '1':
                    c0 = c
                    while c < 5 and row[c] == '1':
                        c += 1
                    x0 = gx + c0 * ds
                    x1 = gx + c * ds
                    base = len(verts)
                    verts.extend([(x0, y0, zf), (x1, y0, zf), (x1, y0 + ds, zf), (x0, y0 + ds, zf)])
                    quads.append((base, base + 1, base + 2, base + 3))
                else:
                    c += 1
    return verts, quads


# ==============================================================================
# 5. SCENE SNAPSHOT EXTRACTION (executes the Blender script on the stdlib mock)
# ==============================================================================

def _accelerate_mock_bmesh(vbs):
    """Replace the mock's O(n) EdgeSeq.get scan with a hashed lookup (renderer process only)."""
    BMEdge = vbs.BMEdge
    if getattr(vbs.EdgeSeq, "_diorama_accelerated", False):
        return
    orig_get = vbs.EdgeSeq.get

    def fast_new(self, verts):
        v1, v2 = verts
        e = BMEdge(v1, v2, len(self))
        self.append(e)
        a, b = id(v1), id(v2)
        self.__dict__.setdefault("_edge_lut", {})[(a, b) if a < b else (b, a)] = e
        return e

    def fast_get(self, verts):
        vs = tuple(verts)
        if len(vs) != 2:
            return orig_get(self, verts)
        lut = self.__dict__.get("_edge_lut")
        if not lut:
            return None
        a, b = id(vs[0]), id(vs[1])
        return lut.get((a, b) if a < b else (b, a))

    vbs.EdgeSeq.new = fast_new
    vbs.EdgeSeq.get = fast_get
    vbs.EdgeSeq._diorama_accelerated = True


def _rgb(v, fallback=None):
    if v is None:
        return fallback
    try:
        t = tuple(float(c) for c in v)
    except TypeError:
        f = float(v)
        return (f, f, f)
    if len(t) >= 3:
        return t[:3]
    return fallback


def resolve_material(mat):
    """Walk the mock ShaderNodeTree links and derive render parameters."""
    nt = mat.node_tree
    owner = {}
    for n in nt.nodes:
        for s in n.inputs:
            owner[id(s)] = n
        for s in n.outputs:
            owner[id(s)] = n
    feed = {id(i): o for (o, i) in nt.links}

    def src(sock):
        o = feed.get(id(sock))
        return owner.get(id(o)) if o is not None else None

    def color_of(sock, emissive=False, depth=0):
        node = src(sock)
        if node is None or depth > 6:
            return _rgb(sock.default_value)
        t = node.bl_idname
        if t == "ShaderNodeValToRGB":
            cols = [tuple(e.color[:3]) for e in node.color_ramp.elements]
            if emissive:
                return max(cols, key=lambda c: c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722)
            return tuple(sum(c[i] for c in cols) / len(cols) for i in range(3))
        if t == "ShaderNodeMix":
            a = color_of(node.inputs[6], emissive, depth + 1)
            b = color_of(node.inputs[7], emissive, depth + 1)
            if a and b:
                return tuple((x + y) * 0.5 for x, y in zip(a, b))
            return a or b
        if t == "ShaderNodeTexBrick":
            return color_of(node.inputs["Color1"], emissive, depth + 1)
        for nm in ("Color", "Color1", "Base Color"):
            if nm in node.inputs:
                return color_of(node.inputs[nm], emissive, depth + 1)
        return None

    def value_of(sock, depth=0):
        node = src(sock)
        if node is None or depth > 6:
            v = sock.default_value
            if v is None:
                return None
            try:
                return float(v)
            except TypeError:
                return None
        t = node.bl_idname
        if t == "ShaderNodeValToRGB":
            cols = [e.color for e in node.color_ramp.elements]
            return sum(c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722 for c in cols) / len(cols)
        if t == "ShaderNodeMix":
            vals = [value_of(node.inputs[2], depth + 1), value_of(node.inputs[3], depth + 1)]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None
        if t == "ShaderNodeMath":
            for s in node.inputs:
                v = value_of(s, depth + 1)
                if v is not None:
                    return v
        return None

    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    emis_node = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeEmission"), None)
    transl = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfTranslucent"), None)
    base = (0.5, 0.5, 0.5)
    metal, rough, alpha, transm = 0.0, 0.5, 1.0, 0.0
    ecol, estr = None, 0.0
    if bsdf is not None:
        base = color_of(bsdf.inputs["Base Color"]) or base
        metal = value_of(bsdf.inputs["Metallic"]) or 0.0
        r = value_of(bsdf.inputs["Roughness"])
        rough = 0.5 if r is None else r
        a = value_of(bsdf.inputs["Alpha"])
        alpha = 1.0 if a is None else a
        if "Transmission Weight" in bsdf.inputs:
            transm = value_of(bsdf.inputs["Transmission Weight"]) or 0.0
        estr = value_of(bsdf.inputs["Emission Strength"]) or 0.0
        ecol = color_of(bsdf.inputs["Emission Color"], emissive=True)
    if emis_node is not None:
        ecol = color_of(emis_node.inputs["Color"], emissive=True) or ecol
        estr = max(estr, value_of(emis_node.inputs["Strength"]) or 0.0)
    if estr > 0.0 and ecol is None:
        ecol = base
    ecol = ecol or (0.0, 0.0, 0.0)

    name = mat.name
    kind = K_OPAQUE
    if "Wet_Asphalt" in name:
        kind = K_ASPHALT
    elif "Night_Windows" in name:
        kind = K_WINDOWS
        ecol = (0.0, 0.0, 0.0)
        estr = 0.0
    elif "Hologram" in name:
        kind = K_HOLO
        alpha = 0.30 if "Mega" in name else 0.42
    elif "Lantern_Washi" in name:
        kind = K_LANTERN
    elif "Steam" in name:
        kind = K_GLASS
        alpha = 0.13
    elif alpha < 0.95 or transm > 0.85:
        kind = K_GLASS
        alpha = min(alpha, 0.45)
    elif estr * max(ecol) >= 2.0:
        kind = K_NEON
    if kind == K_LANTERN and transl is not None:
        tcol = _rgb(transl.inputs["Color"].default_value, (1.0, 0.5, 0.2))
        ecol = tuple(ecol[i] * estr + tcol[i] * 3.2 for i in range(3))
        estr = 1.0
    return {
        "name": name, "base": tuple(base), "metallic": float(min(1.0, max(0.0, metal))),
        "roughness": float(min(1.0, max(0.02, rough))), "alpha": float(alpha),
        "emission": tuple(c * estr for c in ecol), "kind": kind,
        "flicker": "Flicker" in name,
    }


def _extract_fcurves(anim):
    out = {}
    if anim is None or getattr(anim, "action", None) is None:
        return out
    for fc in anim.action.fcurves:
        raw = [(kp.co[0], kp.co[1], kp.interpolation) for kp in fc.keyframe_points]
        prep = prepare_fcurve(raw)
        if prep is not None:
            out[fc.data_path] = prep
    return out


def _spline_record(sp):
    rec = {"type": sp.type, "cyclic": bool(sp.use_cyclic_u), "order": int(getattr(sp, "order_u", 4)),
           "endpoint": bool(getattr(sp, "use_endpoint_u", True)), "res": int(getattr(sp, "resolution_u", 12))}
    if sp.type == 'BEZIER':
        rec["bezier"] = [(tuple(bp.co), tuple(bp.handle_left), tuple(bp.handle_right)) for bp in sp.bezier_points]
        rec["points"] = [b[0] for b in rec["bezier"]]
    else:
        rec["points"] = [tuple(float(c) for c in p.co[:3]) for p in sp.points]
    return rec


def _poly_newell(verts, idx):
    nx = ny = nz = 0.0
    n = len(idx)
    for i in range(n):
        x0, y0, z0 = verts[idx[i]]
        x1, y1, z1 = verts[idx[(i + 1) % n]]
        nx += (y0 - y1) * (z0 + z1)
        ny += (z0 - z1) * (x0 + x1)
        nz += (x0 - x1) * (y0 + y1)
    return (nx, ny, nz)


def _bsphere(verts):
    if not verts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    c = ((min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5, (min(zs) + max(zs)) * 0.5)
    return (c[0], c[1], c[2], max(math.dist(c, v) for v in verts))


def finalize_geometry(name, verts, polys, mats_kind, custom_vnormals=None, smooth_angle_deg=34.0):
    """polys: list of (vertex index tuple, material id, smooth flag). Builds triangle lists with
    deduplicated corner normals (auto-smooth by angle over position-welded vertices)."""
    cos_lim = math.cos(math.radians(smooth_angle_deg))
    fnorm = [_poly_newell(verts, p[0]) for p in polys]
    fhat = [v_norm(n) for n in fnorm]
    adj = {}
    if custom_vnormals is None:
        for pi, (idx, _m, _s) in enumerate(polys):
            for vi in idx:
                x, y, z = verts[vi]
                adj.setdefault((round(x * 2e4), round(y * 2e4), round(z * 2e4)), []).append(pi)
    nlut = {}
    normals = []

    def nid(n):
        key = (round(n[0], 3), round(n[1], 3), round(n[2], 3))
        j = nlut.get(key)
        if j is None:
            j = len(normals)
            nlut[key] = j
            normals.append(n)
        return j

    tris, ttris = [], []
    for pi, (idx, mid, smooth) in enumerate(polys):
        if len(idx) < 3:
            continue
        corner = []
        fh = fhat[pi]
        for vi in idx:
            if custom_vnormals is not None:
                corner.append(nid(custom_vnormals[vi]))
            elif smooth:
                x, y, z = verts[vi]
                sx = sy = sz = 0.0
                for q in adj[(round(x * 2e4), round(y * 2e4), round(z * 2e4))]:
                    qh = fhat[q]
                    if fh[0] * qh[0] + fh[1] * qh[1] + fh[2] * qh[2] >= cos_lim:
                        qn = fnorm[q]
                        sx += qn[0]
                        sy += qn[1]
                        sz += qn[2]
                corner.append(nid(v_norm((sx, sy, sz)) if (sx or sy or sz) else fh))
            else:
                corner.append(nid(fh))
        target = ttris if mats_kind[mid] in (K_GLASS, K_HOLO) else tris
        for k in range(1, len(idx) - 1):
            target.append((idx[0], idx[k], idx[k + 1], mid, corner[0], corner[k], corner[k + 1]))
    return {"name": name, "verts": verts, "normals": normals, "tris": tris, "ttris": ttris,
            "bsphere": _bsphere(verts), "skin": None, "wave": None, "vnormals": None,
            "no_reflect": False, "emit_mode": None, "emit_ref": None}


def tube_geometry(name, spline_pts, cyclic, radius, mid, mats_kind):
    """Sweep a circular bevel profile along a sampled spline (parallel-transport frames)."""
    pts = []
    for p in spline_pts:
        if not pts or math.dist(pts[-1], p) > 1e-6:
            pts.append(tuple(p))
    if cyclic and len(pts) > 2 and math.dist(pts[0], pts[-1]) < 1e-6:
        pts.pop()
    if len(pts) < 2:
        return None
    sides = 8 if radius >= 0.02 else (6 if radius >= 0.008 else 4)
    n = len(pts)
    tans = []
    for i in range(n):
        if cyclic:
            a, b = pts[(i - 1) % n], pts[(i + 1) % n]
        else:
            a, b = pts[max(0, i - 1)], pts[min(n - 1, i + 1)]
        tans.append(v_norm(v_sub(b, a)))
    t0 = tans[0]
    ref = (0.0, 0.0, 1.0) if abs(t0[2]) < 0.9 else (1.0, 0.0, 0.0)
    nrm = v_norm(v_cross(t0, ref))
    verts, vnorms, polys = [], [], []
    for i in range(n):
        t = tans[i]
        d = v_dot(nrm, t)
        nrm = v_norm((nrm[0] - d * t[0], nrm[1] - d * t[1], nrm[2] - d * t[2]))
        bn = v_cross(t, nrm)
        p = pts[i]
        for s in range(sides):
            a = 2.0 * PI * s / sides
            ca, sa = math.cos(a), math.sin(a)
            rd = (nrm[0] * ca + bn[0] * sa, nrm[1] * ca + bn[1] * sa, nrm[2] * ca + bn[2] * sa)
            verts.append((p[0] + rd[0] * radius, p[1] + rd[1] * radius, p[2] + rd[2] * radius))
            vnorms.append(rd)
    rings = n if cyclic else n - 1
    for i in range(rings):
        j = (i + 1) % n
        for s in range(sides):
            s2 = (s + 1) % sides
            polys.append(((i * sides + s, j * sides + s, j * sides + s2, i * sides + s2), mid, True))
    return finalize_geometry(name, verts, polys, mats_kind, custom_vnormals=vnorms)


def build_scene_snapshot(verbose=True):
    """Execute the Blender scene script under the stdlib mock and flatten it into plain data."""
    t0 = time.time()
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import verify_blender_script as vbs
    _accelerate_mock_bmesh(vbs)
    bpy, bmesh_mod, mathutils_mod = vbs.make_mock_blender((4, 3, 0))
    saved = {k: sys.modules.get(k) for k in ("bpy", "bmesh", "mathutils")}
    sys.modules["bpy"], sys.modules["bmesh"], sys.modules["mathutils"] = bpy, bmesh_mod, mathutils_mod
    tmpdir = tempfile.mkdtemp(prefix="diorama_eval_")
    try:
        src_path = os.path.join(SCRIPT_DIR, SCENE_SCRIPT_NAME)
        with open(src_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        # __file__ points into a temp dir so the scene's WAV/manifest exports never touch the project files
        ns = {"__name__": "__diorama_render__", "__file__": os.path.join(tmpdir, "scene_eval.py")}
        exec(compile(src, src_path, "exec"), ns)
        ns["build_cyberpunk_floating_ramen_diorama"]()
        snap = _flatten_scene(bpy)
        snap["scene_source_sha256"] = hashlib.sha256(src.encode("utf-8")).hexdigest()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        shutil.rmtree(tmpdir, ignore_errors=True)
    snap["build_seconds"] = round(time.time() - t0, 2)
    if verbose:
        print(f"[scene] evaluated Blender script on stdlib mock in {snap['build_seconds']}s: "
              f"{len(snap['objects'])} objects, {len(snap['geoms'])} render geometries, "
              f"{sum(len(g['tris']) + len(g['ttris']) for g in snap['geoms']):,} triangles")
    return snap


def _flatten_scene(bpy):
    scene = bpy.context.scene
    # ---- materials
    mats = [{"name": "__default__", "base": (0.5, 0.5, 0.5), "metallic": 0.0, "roughness": 0.5,
             "alpha": 1.0, "emission": (0.0, 0.0, 0.0), "kind": K_OPAQUE, "flicker": False}]
    mat_index = {"__default__": 0}
    mat_fcurves = {}
    for m in bpy.data.materials:
        mat_index[m.name] = len(mats)
        mats.append(resolve_material(m))
        fcs = _extract_fcurves(getattr(m.node_tree, "animation_data", None))
        if fcs:
            mat_fcurves[m.name] = sorted(fcs)
    mats_kind = [m["kind"] for m in mats]

    # ---- objects (transforms, animation, constraints)
    objects = {}
    paths = {}
    for o in bpy.data.objects:
        rec = {
            "name": o.name, "type": o.type, "parent": o.parent.name if o.parent is not None else None,
            "loc": tuple(float(c) for c in o.location), "rot": tuple(float(c) for c in o.rotation_euler),
            "scale": tuple(float(c) for c in o.scale), "hide_render": bool(o.hide_render),
            "fcurves": _extract_fcurves(o.animation_data),
            "data_fcurves": _extract_fcurves(getattr(o.data, "animation_data", None)) if o.data is not None else {},
            "constraints": [],
        }
        for c in o.constraints:
            rec["constraints"].append({
                "type": c.type, "name": c.name, "target": c.target.name if c.target is not None else None,
                "use_fixed_location": bool(c.use_fixed_location), "use_curve_follow": bool(c.use_curve_follow),
                "offset_factor": float(c.offset_factor), "forward_axis": c.forward_axis,
                "up_axis": c.up_axis, "track_axis": c.track_axis,
            })
        if o.type == 'LIGHT':
            d = o.data
            rec["light"] = {"type": d.type, "energy": float(d.energy), "color": tuple(float(c) for c in d.color),
                            "spot_size": float(d.spot_size), "spot_blend": float(d.spot_blend)}
        elif o.type == 'CAMERA':
            d = o.data
            rec["camera"] = {"lens": float(d.lens), "sensor_width": float(d.sensor_width),
                             "clip_start": float(d.clip_start), "clip_end": float(d.clip_end),
                             "fstop": float(d.dof.aperture_fstop),
                             "focus_object": d.dof.focus_object.name if d.dof.focus_object is not None else None}
        elif o.type == 'ARMATURE':
            rec["bones"] = [(eb.name, tuple(eb.head), tuple(eb.tail), float(eb.roll),
                             eb.parent.name if eb.parent is not None else None) for eb in o.data.edit_bones]
        elif o.type == 'CURVE' and o.data.splines:
            paths[o.name] = build_path_table(_spline_record(o.data.splines[0]))
        objects[o.name] = rec

    markers = sorted((int(m.frame), m.name, m.camera.name if m.camera is not None else None)
                     for m in scene.timeline_markers)

    snap = {"objects": objects, "paths": paths, "materials": mats, "markers": markers,
            "default_camera": scene.camera.name if scene.camera is not None else None,
            "material_fcurves": mat_fcurves, "geoms": [], "rain": []}

    rest = evaluate_scene_at_frame(snap, 1.0, need_bones=False)["world"]
    lantern_lights = [(n, m_translation(rest[n])) for n, r in objects.items()
                      if r["type"] == 'LIGHT' and n.startswith("Lantern_Internal_PointLight_")]

    geoms = []
    rng_state = [2099]   # GN_Cyber_Rain_Field DistributePointsOnFaces seed

    def rnd():
        rng_state[0] = (rng_state[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return rng_state[0] / 2147483648.0

    for o in bpy.data.objects:
        if o.hide_render:
            continue
        g = None
        if o.type == 'MESH':
            if any(m.type == 'NODES' for m in o.modifiers):
                # Geometry-Nodes rain emitter: streak instances replace the emitter mesh at render time
                layers = sorted({round(v.co.z, 4) for v in o.data.vertices})
                xs = [v.co.x for v in o.data.vertices]
                ys = [v.co.y for v in o.data.vertices]
                drops = []
                for _ in range(RAIN_DROPS_PER_SHEET):
                    drops.append((min(xs) + (max(xs) - min(xs)) * rnd(), min(ys) + (max(ys) - min(ys)) * rnd(),
                                  layers[int(rnd() * len(layers)) % len(layers)], 0.65 + rnd(), rnd()))
                snap["rain"].append({"emitter": o.name, "drops": drops, "length": 0.46})
                continue
            me = o.data
            verts = [tuple(float(c) for c in v.co) for v in me.vertices]
            slots = [mat_index.get(m.name, 0) for m in me.materials] or [0]
            polys = [(tuple(p.vertices), slots[min(int(p.material_index), len(slots) - 1)], bool(p.use_smooth))
                     for p in me.polygons]
            wave = None
            skin = None
            for mod in o.modifiers:
                if mod.type == 'ARRAY':
                    count = int(getattr(mod, "count", 2))
                    off = [0.0, 0.0, 0.0]
                    if getattr(mod, "use_relative_offset", True):
                        rel = tuple(getattr(mod, "relative_offset_displace", (1.0, 0.0, 0.0)))
                        for ax in range(3):
                            cs = [v[ax] for v in verts]
                            off[ax] += rel[ax] * ((max(cs) - min(cs)) if cs else 0.0)
                    if getattr(mod, "use_constant_offset", False):
                        cst = tuple(getattr(mod, "constant_offset_displace", (0.0, 0.0, 0.0)))
                        for ax in range(3):
                            off[ax] += cst[ax]
                    nv = len(verts)
                    base_v, base_p = list(verts), list(polys)
                    for k in range(1, count):
                        verts.extend((x + off[0] * k, y + off[1] * k, z + off[2] * k) for (x, y, z) in base_v)
                        polys.extend((tuple(i + nv * k for i in idx), m, s) for (idx, m, s) in base_p)
                elif mod.type == 'LATTICE' and mod.object is not None:
                    verts = _apply_lattice(verts, rest[o.name], rest[mod.object.name], mod.object.data)
                elif mod.type == 'WAVE':
                    wave = {
                        "height": float(getattr(mod, "height", 0.5)), "width": float(getattr(mod, "width", 1.5)),
                        "narrow": float(getattr(mod, "narrowness", 1.5)), "speed": float(getattr(mod, "speed", 0.25)),
                        "use_x": bool(getattr(mod, "use_x", True)), "use_y": bool(getattr(mod, "use_y", True)),
                        "cyclic": bool(getattr(mod, "use_cyclic", True)), "normal": bool(getattr(mod, "use_normal", False)),
                    }
                elif mod.type == 'ARMATURE' and mod.object is not None:
                    bone_names = [b[0] for b in objects[mod.object.name]["bones"]]
                    vmap = [-1] * len(verts)
                    for vg in o.vertex_groups:
                        if vg.name in bone_names:
                            bi = bone_names.index(vg.name)
                            for vi, w in vg.weights.items():
                                if w > 0.0 and vi < len(vmap):
                                    vmap[vi] = bi
                    skin = {"armature": mod.object.name, "bones": bone_names, "vmap": vmap}
            g = finalize_geometry(o.name, verts, polys, mats_kind)
            g["wave"] = wave
            g["skin"] = skin
            if wave is not None and wave["normal"]:
                acc = [[0.0, 0.0, 0.0] for _ in verts]
                for (idx, _m, _s) in polys:
                    n = _poly_newell(verts, idx)
                    for vi in idx:
                        a = acc[vi]
                        a[0] += n[0]
                        a[1] += n[1]
                        a[2] += n[2]
                g["vnormals"] = [v_norm(tuple(a)) for a in acc]
        elif o.type == 'CURVE':
            cd = o.data
            if cd.bevel_depth > 0.0 and cd.splines:
                mid = mat_index.get(cd.materials[0].name, 0) if cd.materials else 0
                parts = []
                for sp in cd.splines:
                    rec = _spline_record(sp)
                    span_res = 6 if cd.bevel_depth >= 0.01 else 3
                    if rec["type"] == 'NURBS':
                        p = rec["points"]
                        length = sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1))
                        spans = max(1, len(p) - min(rec["order"], len(p)) + 1)
                        span_res = max(span_res, min(24, int(length / spans / 0.6) + 3))
                    pts, cyc = sample_spline(rec, span_res)
                    tg = tube_geometry(o.name, pts, cyc, float(cd.bevel_depth), mid, mats_kind)
                    if tg is not None:
                        parts.append(tg)
                if parts:
                    g = parts[0] if len(parts) == 1 else _merge_geoms(o.name, parts)
        elif o.type == 'FONT':
            cd = o.data
            mid = mat_index.get(cd.materials[0].name, 0) if cd.materials else 0
            verts, quads = font_glyph_quads(cd.body, float(cd.size), float(cd.extrude))
            if quads:
                g = finalize_geometry(o.name, verts, [(q, mid, False) for q in quads], mats_kind)
        if g is None or (not g["tris"] and not g["ttris"]):
            continue
        if o.name == "Wet_Asphalt_Diorama_Plinth" or o.name.startswith("Puddle_Splash_Ripple_"):
            g["no_reflect"] = True
        if any(mats[t[3]]["flicker"] for t in g["tris"] + g["ttris"]):
            g["emit_mode"] = "flicker"
        elif o.name.startswith("Paper_Lantern_") and lantern_lights:
            p = m_translation(rest[o.name])
            g["emit_mode"] = "light"
            g["emit_ref"] = min(lantern_lights, key=lambda nl: math.dist(nl[1], p))[0]
        geoms.append(g)
    snap["geoms"] = geoms
    return snap


def _merge_geoms(name, parts):
    verts, normals, tris, ttris = [], [], [], []
    for g in parts:
        vo, no = len(verts), len(normals)
        verts.extend(g["verts"])
        normals.extend(g["normals"])
        tris.extend((a + vo, b + vo, c + vo, m, x + no, y + no, z + no) for (a, b, c, m, x, y, z) in g["tris"])
        ttris.extend((a + vo, b + vo, c + vo, m, x + no, y + no, z + no) for (a, b, c, m, x, y, z) in g["ttris"])
    out = dict(parts[0])
    out.update({"name": name, "verts": verts, "normals": normals, "tris": tris, "ttris": ttris,
                "bsphere": _bsphere(verts)})
    return out


def _apply_lattice(verts, obj_world, lat_world, lat_data):
    """Trilinear lattice deformation (displacement field co_deform - co over the U*V*W grid)."""
    u, v, w = int(lat_data.points_u), int(lat_data.points_v), int(lat_data.points_w)
    disp = [tuple(p.co_deform[i] - p.co[i] for i in range(3)) for p in lat_data.points]
    to_lat = m_mul(m_inv(lat_world), obj_world)
    from_lat = m_inv(to_lat)
    out = []
    for p in verts:
        lx, ly, lz = m_apply(to_lat, p)
        fu = min(max((lx + 0.5) * (u - 1), 0.0), u - 1.0)
        fv = min(max((ly + 0.5) * (v - 1), 0.0), v - 1.0)
        fw = min(max((lz + 0.5) * (w - 1), 0.0), w - 1.0)
        i0, j0, k0 = min(int(fu), u - 2), min(int(fv), v - 2), min(int(fw), w - 2)
        tu, tv, tw = fu - i0, fv - j0, fw - k0
        dx = dy = dz = 0.0
        for dk, wk in ((0, 1 - tw), (1, tw)):
            for dj, wj in ((0, 1 - tv), (1, tv)):
                for di, wi in ((0, 1 - tu), (1, tu)):
                    wt = wi * wj * wk
                    if wt:
                        d = disp[(k0 + dk) * u * v + (j0 + dj) * u + (i0 + di)]
                        dx += d[0] * wt
                        dy += d[1] * wt
                        dz += d[2] * wt
        out.append(m_apply(from_lat, (lx + dx, ly + dy, lz + dz)))
    return out


# ==============================================================================
# 6. TRACK 25: evaluate_scene_at_frame
# ==============================================================================

def _bone_rest_matrix(head, tail, roll):
    """Blender vec_roll_to_mat3 + head translation (armature space)."""
    nor = v_norm(v_sub(tail, head))
    x, y, z = nor
    theta = 1.0 + y
    if theta > 1e-5 or (x * x + z * z) > 1e-10:
        if theta <= 1e-5:
            theta = 1e-5
        col_x = (1.0 - x * x / theta, -x, -x * z / theta)
        col_z = (-x * z / theta, -z, 1.0 - z * z / theta)
    else:
        col_x = (-1.0, 0.0, 0.0)
        col_z = (0.0, 0.0, 1.0)
    m = m_from_cols(col_x, nor, col_z, head)
    if abs(roll) > 1e-9:
        c, s = math.cos(roll), math.sin(roll)
        m = m_mul(m, (c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0))
    return m


def evaluate_scene_at_frame(snap, frame, need_bones=True):
    """Evaluate the full animated scene state at an arbitrary (fractional) frame.

    Returns world matrices of every object, the TimelineMarker-selected active camera,
    light states, armature bone deform matrices and emission multipliers.
    """
    frame = float(frame)
    objs = snap["objects"]
    paths = snap["paths"]
    world = {}
    channel_values = {}

    def local_of(rec):
        fcs = rec["fcurves"]
        loc = eval_fcurve(fcs["location"], frame) if "location" in fcs else rec["loc"]
        rot = eval_fcurve(fcs["rotation_euler"], frame) if "rotation_euler" in fcs else rec["rot"]
        scl = eval_fcurve(fcs["scale"], frame) if "scale" in fcs else rec["scale"]
        if fcs:
            channel_values[rec["name"]] = {"location": tuple(loc), "rotation_euler": tuple(rot), "scale": tuple(scl)}
        return m_compose(loc, rot, scl)

    def world_of(name, stack=()):
        m = world.get(name)
        if m is not None:
            return m
        rec = objs[name]
        m = local_of(rec)
        if rec["parent"] is not None and rec["parent"] not in stack:
            m = m_mul(world_of(rec["parent"], stack + (name,)), m)
        for c in rec["constraints"]:
            tgt = c["target"]
            if tgt is None or tgt not in objs:
                continue
            if c["type"] == 'FOLLOW_PATH' and tgt in paths:
                key = f'constraints["{c["name"]}"].offset_factor'
                off = eval_fcurve(rec["fcurves"][key], frame) if key in rec["fcurves"] else c["offset_factor"]
                channel_values.setdefault(name, {})[key] = off
                p, tan = path_point(paths[tgt], off)
                if c["use_curve_follow"]:
                    cx, cy, cz = rot_from_axes(_axis_code(c["forward_axis"]), tan, _axis_code(c["up_axis"]), (0.0, 0.0, 1.0))
                    follow = m_from_cols(cx, cy, cz, p)
                else:
                    follow = (1.0, 0.0, 0.0, p[0], 0.0, 1.0, 0.0, p[1], 0.0, 0.0, 1.0, p[2])
                m = m_mul(m_mul(world_of(tgt, stack + (name,)), follow), m)
            elif c["type"] in ('TRACK_TO', 'DAMPED_TRACK'):
                tpos = m_translation(world_of(tgt, stack + (name,)))
                pos = m_translation(m)
                d = v_sub(tpos, pos)
                if v_dot(d, d) < 1e-12:
                    continue
                sx = math.sqrt(m[0] ** 2 + m[4] ** 2 + m[8] ** 2)
                sy = math.sqrt(m[1] ** 2 + m[5] ** 2 + m[9] ** 2)
                sz = math.sqrt(m[2] ** 2 + m[6] ** 2 + m[10] ** 2)
                up_code = _axis_code(c["up_axis"]) if c["type"] == 'TRACK_TO' else 'Y'
                cx, cy, cz = rot_from_axes(_axis_code(c["track_axis"]), d, up_code, (0.0, 0.0, 1.0))
                m = m_from_cols((cx[0] * sx, cx[1] * sx, cx[2] * sx), (cy[0] * sy, cy[1] * sy, cy[2] * sy),
                                (cz[0] * sz, cz[1] * sz, cz[2] * sz), pos)
        world[name] = m
        return m

    for name in objs:
        world_of(name)

    # ---- TimelineMarker director cut
    active_cam = snap["default_camera"]
    marker_name = None
    for mf, mname, mcam in snap["markers"]:
        if mf <= frame + 1e-6 and mcam is not None:
            active_cam, marker_name = mcam, mname
    cam_rec = objs[active_cam]["camera"] if active_cam in objs else None

    # ---- lights
    lights = []
    energies = {}
    for name, rec in objs.items():
        if rec["type"] != 'LIGHT':
            continue
        L = rec["light"]
        energy = eval_fcurve(rec["data_fcurves"]["energy"], frame) if "energy" in rec["data_fcurves"] else L["energy"]
        energies[name] = energy
        m = world[name]
        lights.append({"name": name, "type": L["type"], "pos": m_translation(m), "dir": v_norm((-m[2], -m[6], -m[10])),
                       "energy": energy, "color": L["color"], "spot_size": L["spot_size"],
                       "spot_blend": L["spot_blend"]})

    # ---- neon ballast flicker (shares the Lantern_Internal_PointLight_05 stutter F-curve)
    flicker = 1.0
    fl = objs.get("Lantern_Internal_PointLight_05")
    if fl is not None and fl["light"]["energy"] > 0:
        e = energies.get("Lantern_Internal_PointLight_05", fl["light"]["energy"])
        flicker = min(1.2, max(0.0, (e / 36.0 - 0.35) / 0.65))

    # ---- armature PoseBone kinematics
    bone_deform = {}
    if need_bones:
        for name, rec in objs.items():
            if rec["type"] != 'ARMATURE':
                continue
            rest_m, pose_m, deform = {}, {}, {}
            fcs = rec["fcurves"]
            for bname, head, tail, roll, parent in rec["bones"]:
                rest_m[bname] = _bone_rest_matrix(head, tail, roll)
                pfx = f'pose.bones["{bname}"].'
                loc = eval_fcurve(fcs[pfx + "location"], frame) if pfx + "location" in fcs else (0.0, 0.0, 0.0)
                rot = eval_fcurve(fcs[pfx + "rotation_euler"], frame) if pfx + "rotation_euler" in fcs else (0.0, 0.0, 0.0)
                scl = eval_fcurve(fcs[pfx + "scale"], frame) if pfx + "scale" in fcs else (1.0, 1.0, 1.0)
                basis = m_compose(loc, rot, scl)
                if parent is not None and parent in pose_m:
                    offs = m_mul(m_inv(rest_m[parent]), rest_m[bname])
                    pose_m[bname] = m_mul(m_mul(pose_m[parent], offs), basis)
                else:
                    pose_m[bname] = m_mul(rest_m[bname], basis)
                deform[bname] = m_mul(pose_m[bname], m_inv(rest_m[bname]))
            bone_deform[name] = deform

    return {
        "frame": frame, "world": world, "active_camera": active_cam, "active_marker": marker_name,
        "camera": cam_rec, "lights": lights, "light_energy": energies, "flicker": flicker,
        "bone_deform": bone_deform, "channels": channel_values,
    }


# ==============================================================================
# 7. TRACK 26: RASTERISATION, SHADING, REFLECTION, BLOOM, AgX
# ==============================================================================

_WORKER = {}


def _prepare_render_context(snap, width, height):
    mats = []
    for m in snap["materials"]:
        br, bg, bb = m["base"]
        metal = m["metallic"]
        alpha_r = max(0.02, m["roughness"] ** 2)
        er, eg, eb = m["emission"]
        mats.append((
            br * (1 - metal) * INV_PI, bg * (1 - metal) * INV_PI, bb * (1 - metal) * INV_PI,
            0.04 * (1 - metal) + br * metal, 0.04 * (1 - metal) + bg * metal, 0.04 * (1 - metal) + bb * metal,
            alpha_r * alpha_r, alpha_r * 0.5, er, eg, eb, m["alpha"], m["kind"], br, bg, bb,
        ))
    return {"snap": snap, "mats": mats, "W": width, "H": height}


def _init_worker(snap, width, height):
    _WORKER.clear()
    _WORKER.update(_prepare_render_context(snap, width, height))


def _lights_to_camera(state_lights, view):
    out = []
    for L in state_lights:
        p = m_apply(view, L["pos"])
        d = v_norm(m_apply_dir(view, L["dir"]))
        e = L["energy"]
        cr, cg, cb = L["color"]
        if L["type"] == 'AREA':
            kind, inten = L_AREA, e * INV_PI             # Lambertian emitter: I0 = P / pi
        elif L["type"] == 'SPOT':
            kind, inten = L_SPOT, e / (4.0 * PI)
        else:
            kind, inten = L_POINT, e / (4.0 * PI)
        half = L["spot_size"] * 0.5
        co = math.cos(half)
        ci = math.cos(half * (1.0 - min(0.99, max(0.0, L["spot_blend"]))))
        rng2 = inten * max(cr, cg, cb) / 0.018
        if kind == L_POINT and e <= 60.0 and rng2 > 22.0:
            rng2 = 22.0
        out.append((kind, p[0], p[1], p[2], cr * inten, cg * inten, cb * inten, d[0], d[1], d[2], co, max(ci, co + 1e-4), rng2))
    return out


def _shade(P0, P1, P2, N0, N1, N2, mat, emult, lights, upx, upy, upz):
    """Two-sided Lambert + GGX (Smith-Schlick, Schlick Fresnel) multi-light shading in camera space."""
    dr, dg, db, f0r, f0g, f0b, a2, kg, er, eg, eb, _alpha, _kind, _br, _bg, _bb = mat
    il = 1.0 / math.sqrt(P0 * P0 + P1 * P1 + P2 * P2)
    V0, V1, V2 = -P0 * il, -P1 * il, -P2 * il
    ndv = N0 * V0 + N1 * V1 + N2 * V2
    if ndv < 0.0:
        N0, N1, N2, ndv = -N0, -N1, -N2, -ndv
    if ndv < 1e-3:
        ndv = 1e-3
    if _kind == K_NEON:
        return dr + er * emult, dg + eg * emult, db + eb * emult, N0, N1, N2, ndv
    hu = 0.5 + 0.5 * (N0 * upx + N1 * upy + N2 * upz)
    if _kind == K_WINDOWS:
        return (
            dr * PI * (AMB_GROUND[0] + (AMB_SKY[0] - AMB_GROUND[0]) * hu),
            dg * PI * (AMB_GROUND[1] + (AMB_SKY[1] - AMB_GROUND[1]) * hu),
            db * PI * (AMB_GROUND[2] + (AMB_SKY[2] - AMB_GROUND[2]) * hu),
            N0, N1, N2, ndv,
        )
    omv = 1.0 - ndv
    omv2 = omv * omv
    fv = omv2 * omv2 * omv
    r = dr * PI * (AMB_GROUND[0] + (AMB_SKY[0] - AMB_GROUND[0]) * hu) + (f0r + (1.0 - f0r) * fv) * ENV_SPEC[0]
    g = dg * PI * (AMB_GROUND[1] + (AMB_SKY[1] - AMB_GROUND[1]) * hu) + (f0g + (1.0 - f0g) * fv) * ENV_SPEC[1]
    b = db * PI * (AMB_GROUND[2] + (AMB_SKY[2] - AMB_GROUND[2]) * hu) + (f0b + (1.0 - f0b) * fv) * ENV_SPEC[2]
    gv = 0.25 / (ndv * (1.0 - kg) + kg)
    sqrt = math.sqrt
    for (lk, lx, ly, lz, cr, cg, cb, ddx, ddy, ddz, co, ci, rng2) in lights:
        Lx, Ly, Lz = lx - P0, ly - P1, lz - P2
        d2 = Lx * Lx + Ly * Ly + Lz * Lz
        if d2 > rng2:
            continue
        ndl_raw = N0 * Lx + N1 * Ly + N2 * Lz
        if ndl_raw <= 0.0:
            continue
        if lk == L_AREA and (Lx * ddx + Ly * ddy + Lz * ddz) >= 0.0:
            continue
        inv = 1.0 / sqrt(d2) if d2 > 1e-12 else 0.0
        Lx *= inv
        Ly *= inv
        Lz *= inv
        ndl = ndl_raw * inv
        att = 1.0 / (d2 if d2 > 0.0025 else 0.0025)
        if lk == L_AREA:
            att *= -(Lx * ddx + Ly * ddy + Lz * ddz)
        elif lk == L_SPOT:
            c = -(Lx * ddx + Ly * ddy + Lz * ddz)
            if c <= co:
                continue
            if c < ci:
                t = (c - co) / (ci - co)
                att *= t * t * (3.0 - 2.0 * t)
        Hx, Hy, Hz = Lx + V0, Ly + V1, Lz + V2
        hl2 = Hx * Hx + Hy * Hy + Hz * Hz
        if hl2 > 1e-18:
            inv_h = 1.0 / sqrt(hl2)
            ndh = (N0 * Hx + N1 * Hy + N2 * Hz) * inv_h
            vdh = (V0 * Hx + V1 * Hy + V2 * Hz) * inv_h
        else:
            ndh = 0.0
            vdh = 0.0
        if ndh < 0.0:
            ndh = 0.0
        if vdh < 0.0:
            vdh = 0.0
        dd = ndh * ndh * (a2 - 1.0) + 1.0
        D = a2 / (PI * dd * dd)
        if D > 400.0:
            D = 400.0
        spec = D * gv / (ndl * (1.0 - kg) + kg)
        omh = 1.0 - vdh
        omh2 = omh * omh
        fw = omh2 * omh2 * omh
        e = ndl * att
        r += (dr + (f0r + (1.0 - f0r) * fw) * spec) * cr * e
        g += (dg + (f0g + (1.0 - f0g) * fw) * spec) * cg * e
        b += (db + (f0b + (1.0 - f0b) * fw) * spec) * cb * e
    if er or eg or eb:
        r += er * emult
        g += eg * emult
        b += eb * emult
    return r, g, b, N0, N1, N2, ndv


def _raster_tri(zb, tb, W, H, tid, x0, y0, z0, x1, y1, z1, x2, y2, z2):
    """Span rasteriser writing nearest 1/z into zb and the triangle id into tb.
    Returns the screen-space barycentric planes (A1,B1,C1,A2,B2,C2) if any pixel was written."""
    minx = x0 if x0 < x1 else x1
    if x2 < minx:
        minx = x2
    maxx = x0 if x0 > x1 else x1
    if x2 > maxx:
        maxx = x2
    miny = y0 if y0 < y1 else y1
    if y2 < miny:
        miny = y2
    maxy = y0 if y0 > y1 else y1
    if y2 > maxy:
        maxy = y2
    xs = math.ceil(minx - 0.5)
    xe = math.floor(maxx - 0.5)
    ys = math.ceil(miny - 0.5)
    ye = math.floor(maxy - 0.5)
    if xs < 0:
        xs = 0
    if ys < 0:
        ys = 0
    if xe > W - 1:
        xe = W - 1
    if ye > H - 1:
        ye = H - 1
    if xs > xe or ys > ye:
        return None
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if -1e-10 < area < 1e-10:
        return None
    inv = 1.0 / area
    A0, B0, C0 = (y1 - y2) * inv, (x2 - x1) * inv, (x1 * y2 - x2 * y1) * inv
    A1, B1, C1 = (y2 - y0) * inv, (x0 - x2) * inv, (x2 * y0 - x0 * y2) * inv
    A2, B2, C2 = (y0 - y1) * inv, (x1 - x0) * inv, (x0 * y1 - x1 * y0) * inv
    if xs == xe and ys == ye:
        px, py = xs + 0.5, ys + 0.5
        l0 = A0 * px + B0 * py + C0
        if l0 >= -1e-7:
            l1 = A1 * px + B1 * py + C1
            if l1 >= -1e-7:
                l2 = A2 * px + B2 * py + C2
                if l2 >= -1e-7:
                    v = l0 * z0 + l1 * z1 + l2 * z2
                    i = ys * W + xs
                    if v > zb[i]:
                        zb[i] = v
                        tb[i] = tid
                        return (A1, B1, C1, A2, B2, C2)
        return None
    Az = A0 * z0 + A1 * z1 + A2 * z2
    Bz = B0 * z0 + B1 * z1 + B2 * z2
    Cz = C0 * z0 + C1 * z1 + C2 * z2
    wrote = False
    ceil = math.ceil
    floor = math.floor
    for y in range(ys, ye + 1):
        py = y + 0.5
        lo, hi = xs, xe
        skip = False
        for A, K in ((A0, B0 * py + C0), (A1, B1 * py + C1), (A2, B2 * py + C2)):
            if A > 1e-12:
                v = ceil(-K / A - 0.5 - 1e-7)
                if v > lo:
                    lo = v
            elif A < -1e-12:
                v = floor(-K / A - 0.5 + 1e-7)
                if v < hi:
                    hi = v
            elif K < -1e-9:
                skip = True
                break
        if skip or lo > hi:
            continue
        izr = Bz * py + Cz + Az * 0.5
        base = y * W
        for x in range(lo, hi + 1):
            v = Az * x + izr
            i = base + x
            if v > zb[i]:
                zb[i] = v
                tb[i] = tid
                wrote = True
    if wrote:
        return (A1, B1, C1, A2, B2, C2)
    return None


def _clip_near(poly, near):
    zc = -near
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        pin, qin = p[2] < zc, q[2] < zc
        if pin:
            out.append(p)
        if pin != qin:
            t = (zc - p[2]) / (q[2] - p[2])
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, zc - 1e-6))
    return out


def _face_normal(a, b, c):
    return v_norm(v_cross(v_sub(b, a), v_sub(c, a)))


def _frustum_visible(M, bs, tx, ty, near, fpx=0.0):
    cx, cy, cz = m_apply(M, bs[:3])
    r = bs[3] * m_max_scale(M) + 1e-4
    if cz - r > -near:
        return False
    if fpx > 0.0 and cz < -near and r * fpx < 0.32 * (-cz):
        return False
    sx = math.sqrt(1.0 + tx * tx)
    sy = math.sqrt(1.0 + ty * ty)
    if (cx + tx * cz) / sx > r or (-cx + tx * cz) / sx > r:
        return False
    if (cy + ty * cz) / sy > r or (-cy + ty * cz) / sy > r:
        return False
    return True


def _deformed_local_verts(g, state):
    """Apply ARMATURE skinning (rigid vertex groups) and WAVE modifier displacement for this frame."""
    verts = g["verts"]
    skin = g["skin"]
    if skin is not None:
        deform = state["bone_deform"].get(skin["armature"], {})
        mats = [deform.get(b, IDENT) for b in skin["bones"]]
        out = []
        for vi, p in enumerate(verts):
            bi = skin["vmap"][vi]
            out.append(m_apply(mats[bi], p) if bi >= 0 else p)
        verts = out
    wave = g["wave"]
    if wave is not None:
        frame = state["frame"]
        h, w, nar, spd = wave["height"], wave["width"], wave["narrow"], wave["speed"]
        minfac = 1.0 / math.exp(w * nar * w * nar)
        vn = g["vnormals"]
        out = []
        for vi, (x, y, z) in enumerate(verts):
            if wave["use_x"] and wave["use_y"]:
                amp = math.sqrt(x * x + y * y)
            elif wave["use_x"]:
                amp = x
            elif wave["use_y"]:
                amp = y
            else:
                amp = 0.0
            amp -= frame * spd
            if wave["cyclic"]:
                amp = math.fmod(amp - w, 2.0 * w) + w
            if -w < amp < w:
                a = amp * nar
                disp = (1.0 / math.exp(a * a) - minfac) * h
                if vn is not None:
                    n = vn[vi]
                    out.append((x + n[0] * disp, y + n[1] * disp, z + n[2] * disp))
                else:
                    out.append((x, y, z + disp))
            else:
                out.append((x, y, z))
        verts = out
    return verts


def _emit_mult(g, state):
    mode = g["emit_mode"]
    if mode == "flicker":
        return state["flicker"]
    if mode == "light":
        return state["light_energy"].get(g["emit_ref"], 36.0) / 36.0
    return 1.0


def _raster_pass(passgeo, W, H, fpx, cxp, cyp, near):
    N = W * H
    zb = [0.0] * N
    tb = [-1] * N
    recs = []
    trans = []
    raster = _raster_tri
    for (cv, cn, tris, ttris, emult, flatdyn) in passgeo:
        sxl, syl, izl, ixl, iyl = [], [], [], [], []
        ax, ay, az, aix, aiy = sxl.append, syl.append, izl.append, ixl.append, iyl.append
        for (x, y, z) in cv:
            if z < -near:
                w = -1.0 / z
                sx = cxp + fpx * x * w
                sy = cyp - fpx * y * w
                ax(sx)
                ay(sy)
                az(w)
                aix(int(sx + 0.5) if sx >= -0.5 else -1)
                aiy(int(sy + 0.5) if sy >= -0.5 else -1)
            else:
                ax(0.0)
                ay(0.0)
                az(-1.0)
                aix(-2)
                aiy(-2)
        for (a, b, c, m, na, nb, nc) in tris:
            wa, wb, wc = izl[a], izl[b], izl[c]
            if wa > 0.0 and wb > 0.0 and wc > 0.0:
                if ixl[a] == ixl[b] == ixl[c] or iyl[a] == iyl[b] == iyl[c]:
                    continue
                coef = raster(zb, tb, W, H, len(recs), sxl[a], syl[a], wa, sxl[b], syl[b], wb, sxl[c], syl[c], wc)
                if coef is not None:
                    if flatdyn:
                        n0 = n1 = n2 = _face_normal(cv[a], cv[b], cv[c])
                    else:
                        n0, n1, n2 = cn[na], cn[nb], cn[nc]
                    recs.append(coef + (wa, wb, wc, m, n0, n1, n2, emult))
            elif wa > 0.0 or wb > 0.0 or wc > 0.0:
                poly = _clip_near([cv[a], cv[b], cv[c]], near)
                if len(poly) < 3:
                    continue
                fn = _face_normal(cv[a], cv[b], cv[c])
                proj = [(cxp + fpx * p[0] / -p[2], cyp - fpx * p[1] / -p[2], -1.0 / p[2]) for p in poly]
                for k in range(1, len(proj) - 1):
                    p0, p1, p2 = proj[0], proj[k], proj[k + 1]
                    coef = raster(zb, tb, W, H, len(recs), p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
                    if coef is not None:
                        recs.append(coef + (p0[2], p1[2], p2[2], m, fn, fn, fn, emult))
        for (a, b, c, m, na, nb, nc) in ttris:
            trans.append((cv[a], cv[b], cv[c], m, emult))
    return zb, tb, recs, trans


def _window_emission(px, py, pz, nwx, nwy):
    """Procedural night-window matrix (MAT_Building_Night_Windows palette & 4.8 emission)."""
    along = (py if abs(nwx) > abs(nwy) else px) / 1.30
    hz = pz / 1.05
    iu = math.floor(along)
    iv = math.floor(hz)
    fu = along - iu
    fv = hz - iv
    if fu < 0.16 or fu > 0.84 or fv < 0.22 or fv > 0.80:
        return None
    h = math.sin(iu * 127.1 + iv * 311.7) * 43758.5453
    h -= math.floor(h)
    if h < 0.46:
        return (0.010 * 4.8, 0.012 * 4.8, 0.018 * 4.8)
    if h < 0.68:
        c = (0.05, 0.85, 1.0)
    elif h < 0.88:
        c = (1.0, 0.56, 0.14)
    else:
        c = (1.0, 0.08, 0.54)
    k = 4.8 * (0.55 + 0.45 * ((h * 7.31) % 1.0))
    return (c[0] * k, c[1] * k, c[2] * k)


def _sky(dz):
    t = 0.5 + 0.5 * dz
    t = t * t * (3 - 2 * t)
    r = (0.005 + (0.016 - 0.005) * t) * 0.32
    g = (0.009 + (0.007 - 0.009) * t) * 0.32
    b = (0.022 + (0.030 - 0.022) * t) * 0.32
    haze = max(0.0, 1.0 - abs(dz)) ** 4
    return (r + MIST_COLOR[0] * 1.6 * haze, g + MIST_COLOR[1] * 1.6 * haze, b + MIST_COLOR[2] * 1.6 * haze)


def _shade_pass(W, H, fpx, cxp, cyp, zb, tb, recs, mats, lights, up, inv_view, mirror, refl, y_start=0, y_end=None):
    N = W * H
    R = [0.0] * N
    G = [0.0] * N
    B = [0.0] * N
    upx, upy, upz = up
    shade = _shade
    sqrt = math.sqrt
    if refl is not None:
        RR, RG, RB, RW, RH = refl
    if y_end is None:
        y_end = H
    for y in range(y_start, y_end):
        py = y + 0.5
        vv = -(py - cyp) / fpx
        base = y * W
        prev_t = -2
        prev_r = prev_g = prev_b = 0.0
        for x in range(W):
            i = base + x
            t = tb[i]
            px = x + 0.5
            uu = (px - cxp) / fpx
            if t < 0:
                dl = sqrt(uu * uu + vv * vv + 1.0)
                dz = (uu * upx + vv * upy - upz) / dl
                if mirror:
                    dz = -dz
                R[i], G[i], B[i] = _sky(dz)
                prev_t = -2
                continue
            A1, B1, C1, A2, B2, C2, w0, w1, w2, m, n0, n1, n2, em = recs[t]
            mat = mats[m]
            kind = mat[12]
            if t == prev_t and (x & 3) != 0 and kind == K_OPAQUE and n0 is n1 and n1 is n2:
                R[i] = prev_r
                G[i] = prev_g
                B[i] = prev_b
                continue
            if n0 is n1 and n1 is n2:
                nx, ny, nz = n0
            else:
                l1 = A1 * px + B1 * py + C1
                l2 = A2 * px + B2 * py + C2
                q1 = l1 * w1
                q2 = l2 * w2
                q0 = (1.0 - l1 - l2) * w0
                s = q0 + q1 + q2
                if s <= 0.0:
                    s = 1e-12
                q0 /= s
                q1 /= s
                q2 /= s
                nx = q0 * n0[0] + q1 * n1[0] + q2 * n2[0]
                ny = q0 * n0[1] + q1 * n1[1] + q2 * n2[1]
                nz = q0 * n0[2] + q1 * n1[2] + q2 * n2[2]
                nl = sqrt(nx * nx + ny * ny + nz * nz)
                if nl > 1e-12:
                    nx /= nl
                    ny /= nl
                    nz /= nl
            d = 1.0 / zb[i]
            P0, P1, P2 = uu * d, vv * d, -d
            r, g, b, nx, ny, nz, ndv = shade(P0, P1, P2, nx, ny, nz, mat, em, lights, upx, upy, upz)
            if kind == K_WINDOWS or (kind == K_ASPHALT and refl is not None):
                wx, wy, wz = m_apply(inv_view, (P0, P1, P2))
                if kind == K_WINDOWS:
                    nwx, nwy, _nwz = m_apply_dir(inv_view, (nx, ny, nz))
                    we = _window_emission(wx, wy, wz, nwx, nwy)
                    if we is not None:
                        r += we[0]
                        g += we[1]
                        b += we[2]
                elif nx * upx + ny * upy + nz * upz > 0.80:
                    nn = (math.sin(1.9 * wx + 0.4) * math.cos(1.6 * wy - 0.8)
                          + 0.45 * math.sin(4.3 * wx - 2.9 * wy + 1.7) + 0.25 * math.sin(7.7 * wy + 3.1 * wx))
                    tt = min(1.0, max(0.0, (nn + 0.2) / 0.7))
                    wet = 0.45 + 0.55 * tt * tt * (3 - 2 * tt)
                    k = wet * (0.30 + 0.70 * (0.04 + 0.96 * (1.0 - ndv) ** 5))
                    if k > 0.92:
                        k = 0.92
                    fx = (px + 0.6 * math.sin(wx * 9.0 + wy * 5.0)) * 0.5 - 0.5
                    fy = py * 0.5 - 0.5
                    ix = min(max(int(fx), 0), RW - 2)
                    iy = min(max(int(fy), 0), RH - 2)
                    tx = min(1.0, max(0.0, fx - ix))
                    ty = min(1.0, max(0.0, fy - iy))
                    j = iy * RW + ix
                    w00 = (1 - tx) * (1 - ty)
                    w10 = tx * (1 - ty)
                    w01 = (1 - tx) * ty
                    w11 = tx * ty
                    rr = RR[j] * w00 + RR[j + 1] * w10 + RR[j + RW] * w01 + RR[j + RW + 1] * w11
                    rg = RG[j] * w00 + RG[j + 1] * w10 + RG[j + RW] * w01 + RG[j + RW + 1] * w11
                    rb = RB[j] * w00 + RB[j + 1] * w10 + RB[j + RW] * w01 + RB[j + RW + 1] * w11
                    r = r * (1.0 - 0.6 * k) + rr * k
                    g = g * (1.0 - 0.6 * k) + rg * k
                    b = b * (1.0 - 0.6 * k) + rb * k
            if d > MIST_START:
                f = (d - MIST_START) / MIST_DEPTH
                if f > 1.0:
                    f = 1.0
                f = f * f * MIST_MAX
                r = r * (1 - f) + MIST_COLOR[0] * f
                g = g * (1 - f) + MIST_COLOR[1] * f
                b = b * (1 - f) + MIST_COLOR[2] * f
            R[i] = r
            G[i] = g
            B[i] = b
            prev_t = t
            prev_r = r
            prev_g = g
            prev_b = b
    return R, G, B


def _tri_spans(W, H, x0, y0, x1, y1, x2, y2):
    minx, maxx = min(x0, x1, x2), max(x0, x1, x2)
    miny, maxy = min(y0, y1, y2), max(y0, y1, y2)
    xs, xe = max(0, math.ceil(minx - 0.5)), min(W - 1, math.floor(maxx - 0.5))
    ys, ye = max(0, math.ceil(miny - 0.5)), min(H - 1, math.floor(maxy - 0.5))
    if xs > xe or ys > ye:
        return None
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(area) < 1e-10:
        return None
    inv = 1.0 / area
    E = ((y1 - y2) * inv, (x2 - x1) * inv, (x1 * y2 - x2 * y1) * inv,
         (y2 - y0) * inv, (x0 - x2) * inv, (x2 * y0 - x0 * y2) * inv,
         (y0 - y1) * inv, (x1 - x0) * inv, (x0 * y1 - x1 * y0) * inv)
    spans = []
    for y in range(ys, ye + 1):
        py = y + 0.5
        lo, hi = xs, xe
        ok = True
        for A, K in ((E[0], E[1] * py + E[2]), (E[3], E[4] * py + E[5]), (E[6], E[7] * py + E[8])):
            if A > 1e-12:
                lo = max(lo, math.ceil(-K / A - 0.5 - 1e-7))
            elif A < -1e-12:
                hi = min(hi, math.floor(-K / A - 0.5 + 1e-7))
            elif K < -1e-9:
                ok = False
                break
        if ok and lo <= hi:
            spans.append((y, lo, hi))
    return spans, E


def _transparent_pass(trans, W, H, fpx, cxp, cyp, near, zb, R, G, B, mats, lights, up):
    """Back-to-front alpha (glass/steam/umbrella) and additive scan-line (hologram) blending."""
    trans.sort(key=lambda t: t[0][2] + t[1][2] + t[2][2])
    upx, upy, upz = up
    for (va, vb, vc, m, em) in trans:
        if va[2] >= -near or vb[2] >= -near or vc[2] >= -near:
            continue
        mat = mats[m]
        kind = mat[12]
        alpha = mat[11]
        P = ((va[0] + vb[0] + vc[0]) / 3.0, (va[1] + vb[1] + vc[1]) / 3.0, (va[2] + vb[2] + vc[2]) / 3.0)
        if kind == K_HOLO:
            cr, cg, cb = mat[8] * em, mat[9] * em, mat[10] * em
        else:
            n = _face_normal(va, vb, vc)
            cr, cg, cb, _a, _b, _c, ndv = _shade(P[0], P[1], P[2], n[0], n[1], n[2], mat, em, lights, upx, upy, upz)
            alpha = min(0.95, alpha + 0.35 * (1.0 - ndv) ** 3)
        d = -P[2]
        if d > MIST_START:
            f = min(1.0, (d - MIST_START) / MIST_DEPTH)
            f = f * f * MIST_MAX
            cr, cg, cb = cr * (1 - f) + MIST_COLOR[0] * f, cg * (1 - f) + MIST_COLOR[1] * f, cb * (1 - f) + MIST_COLOR[2] * f
            if kind == K_HOLO:
                alpha *= (1 - f)
        pr = [(cxp + fpx * v[0] / -v[2], cyp - fpx * v[1] / -v[2], -1.0 / v[2]) for v in (va, vb, vc)]
        sp = _tri_spans(W, H, pr[0][0], pr[0][1], pr[1][0], pr[1][1], pr[2][0], pr[2][1])
        if sp is None:
            continue
        spans, E = sp
        z0, z1, z2 = pr[0][2], pr[1][2], pr[2][2]
        Az = E[0] * z0 + E[3] * z1 + E[6] * z2
        Bz = E[1] * z0 + E[4] * z1 + E[7] * z2
        Cz = E[2] * z0 + E[5] * z1 + E[8] * z2
        additive = kind == K_HOLO
        for (y, lo, hi) in spans:
            py = y + 0.5
            izr = Bz * py + Cz + Az * 0.5
            base = y * W
            if additive:
                sc = alpha * (0.55 + 0.45 * ((y >> 1) & 1))
                ar, ag, ab = cr * sc, cg * sc, cb * sc
                for x in range(lo, hi + 1):
                    i = base + x
                    if Az * x + izr > zb[i]:
                        R[i] += ar
                        G[i] += ag
                        B[i] += ab
            else:
                ia = 1.0 - alpha
                ar, ag, ab = cr * alpha, cg * alpha, cb * alpha
                for x in range(lo, hi + 1):
                    i = base + x
                    if Az * x + izr > zb[i]:
                        R[i] = R[i] * ia + ar
                        G[i] = G[i] * ia + ag
                        B[i] = B[i] * ia + ab


def _rain_pass(snap, state, view, W, H, fpx, cxp, cyp, near, zb, R, G, B):
    """GN rain: instances spawn on the (keyframed) emitter layers and fall to Z = 0 at terminal velocity."""
    t = (state["frame"] - 1.0) / FPS
    drawn = 0
    for sheet in snap["rain"]:
        Mw = state["world"][sheet["emitter"]]
        L = sheet["length"]
        for (x, y, lz, scl, phase) in sheet["drops"]:
            wx, wy, wz = m_apply(Mw, (x, y, lz))
            if wz <= 0.05:
                continue
            fall = (RAIN_FALL_SPEED * t + phase * wz) % wz
            ztop = wz - fall
            zbot = max(0.0, ztop - L * scl)
            a = m_apply(view, (wx, wy, ztop))
            b = m_apply(view, (wx, wy, zbot))
            if a[2] >= -near or b[2] >= -near:
                continue
            ia, ib = -1.0 / a[2], -1.0 / b[2]
            xa, ya = cxp + fpx * a[0] * ia, cyp - fpx * a[1] * ia
            xb, yb = cxp + fpx * b[0] * ib, cyp - fpx * b[1] * ib
            if (xa < 0 and xb < 0) or (xa >= W and xb >= W) or (ya < 0 and yb < 0) or (ya >= H and yb >= H):
                continue
            steps = min(160, int(max(abs(xb - xa), abs(yb - ya))) + 1)
            k = 0.16 / (1.0 + 0.08 * -a[2]) * min(1.0, 8.0 / steps + 0.35)
            cr, cg, cb = 0.55 * k, 0.70 * k, 0.88 * k
            drawn += 1
            for s in range(steps):
                u = s / steps
                xi = int(xa + (xb - xa) * u)
                yi = int(ya + (yb - ya) * u)
                if 0 <= xi < W and 0 <= yi < H:
                    i = yi * W + xi
                    if ia + (ib - ia) * u > zb[i]:
                        R[i] += cr
                        G[i] += cg
                        B[i] += cb
    return drawn


def _blur_h(src, W, H, r):
    out = [0.0] * (W * H)
    inv = 1.0 / (2 * r + 1)
    for y in range(H):
        o = y * W
        row = src[o:o + W]
        pad = [row[0]] * r + row + [row[-1]] * (r + 1)
        acc = sum(pad[0:2 * r + 1])
        for x in range(W):
            out[o + x] = acc * inv
            acc += pad[x + 2 * r + 1] - pad[x]
    return out


def _blur_v(src, W, H, r):
    out = [0.0] * (W * H)
    inv = 1.0 / (2 * r + 1)
    for x in range(W):
        col = src[x::W]
        pad = [col[0]] * r + col + [col[-1]] * (r + 1)
        acc = sum(pad[0:2 * r + 1])
        res = [0.0] * H
        for y in range(H):
            res[y] = acc * inv
            acc += pad[y + 2 * r + 1] - pad[y]
        out[x::W] = res
    return out


def _gauss3(ch, W, H, r):
    """Three separable box passes ~ Gaussian."""
    for _ in range(3):
        ch = _blur_v(_blur_h(ch, W, H, r), W, H, r)
    return ch


def _bloom(R, G, B, W, H):
    W2, H2 = W // 2, H // 2
    br, bg, bb = [0.0] * (W2 * H2), [0.0] * (W2 * H2), [0.0] * (W2 * H2)
    for y in range(H2):
        for x in range(W2):
            i = 2 * y * W + 2 * x
            r = (R[i] + R[i + 1] + R[i + W] + R[i + W + 1]) * 0.25
            g = (G[i] + G[i + 1] + G[i + W] + G[i + W + 1]) * 0.25
            b = (B[i] + B[i + 1] + B[i + W] + B[i + W + 1]) * 0.25
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            if lum > BLOOM_THRESHOLD:
                k = (lum - BLOOM_THRESHOLD) / lum
                j = y * W2 + x
                br[j], bg[j], bb[j] = r * k, g * k, b * k
    comb = []
    for ch in (br, bg, bb):
        small = _gauss3(ch, W2, H2, 2)
        large = _gauss3(small, W2, H2, 7)
        comb.append([s * BLOOM_SMALL + l * BLOOM_LARGE for s, l in zip(small, large)])
    cr, cg, cb = comb
    for y in range(H):
        fy = min(max((y + 0.5) * 0.5 - 0.5, 0.0), H2 - 1.001)
        iy = int(fy)
        ty = fy - iy
        for x in range(W):
            fx = min(max((x + 0.5) * 0.5 - 0.5, 0.0), W2 - 1.001)
            ix = int(fx)
            tx = fx - ix
            j = iy * W2 + ix
            w00, w10, w01, w11 = (1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty
            i = y * W + x
            R[i] += cr[j] * w00 + cr[j + 1] * w10 + cr[j + W2] * w01 + cr[j + W2 + 1] * w11
            G[i] += cg[j] * w00 + cg[j + 1] * w10 + cg[j + W2] * w01 + cg[j + W2 + 1] * w11
            B[i] += cb[j] * w00 + cb[j + 1] * w10 + cb[j + W2] * w01 + cb[j + W2 + 1] * w11


def _blur_buffer(R, G, B, W, H):
    return tuple(_blur_v(_blur_h(c, W, H, 1), W, H, 1) for c in (R, G, B))


_AGX_MIN_EV = -12.47393
_AGX_MAX_EV = 4.026069
_AGX_RANGE = _AGX_MAX_EV - _AGX_MIN_EV


def _agx_curve(v):
    if v < 1e-10:
        v = 1e-10
    v = (math.log2(v) - _AGX_MIN_EV) / _AGX_RANGE
    if v <= 0.0:
        return 0.0
    if v >= 1.0:
        v = 1.0
    v2 = v * v
    v4 = v2 * v2
    return 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232


def agx_tonemap_to_bytes(R, G, B, W, H, exposure):
    """AgX base (inset -> log2 encode -> sigmoid) + 'High Contrast' look + outset; returns RGB bytes."""
    out = bytearray(W * H * 3)
    cx_, cy_ = W * 0.5, H * 0.5
    inv_d2 = 1.0 / (cx_ * cx_ + cy_ * cy_)
    power, sat = 1.30, 1.12
    curve = _agx_curve
    o = 0
    for y in range(H):
        dy = (y + 0.5 - cy_)
        dy2 = dy * dy
        base = y * W
        for x in range(W):
            i = base + x
            dx = (x + 0.5 - cx_)
            vig = (1.0 - 0.28 * (dx * dx + dy2) * inv_d2) * exposure
            r = R[i] * vig
            g = G[i] * vig
            b = B[i] * vig
            a0 = curve(0.842479062253094 * r + 0.0784335999999992 * g + 0.0792237451477643 * b)
            a1 = curve(0.0423282422610123 * r + 0.878468636469772 * g + 0.0791661274605434 * b)
            a2 = curve(0.0423756549057051 * r + 0.0784336 * g + 0.879142973793104 * b)
            luma = 0.2126 * a0 + 0.7152 * a1 + 0.0722 * a2
            a0 = luma + sat * ((a0 if a0 > 0.0 else 0.0) ** power - luma)
            a1 = luma + sat * ((a1 if a1 > 0.0 else 0.0) ** power - luma)
            a2 = luma + sat * ((a2 if a2 > 0.0 else 0.0) ** power - luma)
            q0 = int((1.19687900512017 * a0 - 0.0980208811401368 * a1 - 0.0990297440797205 * a2) * 255.0 + 0.5)
            q1 = int((-0.0528968517574562 * a0 + 1.15190312990417 * a1 - 0.0989611768448433 * a2) * 255.0 + 0.5)
            q2 = int((-0.0529716355144438 * a0 - 0.0980434501171241 * a1 + 1.15107367264116 * a2) * 255.0 + 0.5)
            out[o] = 0 if q0 < 0 else (255 if q0 > 255 else q0)
            out[o + 1] = 0 if q1 < 0 else (255 if q1 > 255 else q1)
            out[o + 2] = 0 if q2 < 0 else (255 if q2 > 255 else q2)
            o += 3
    return bytes(out)


def render_frame(frame, ctx=None):
    """Render one frame; returns (frame, rgb_bytes, stats)."""
    ctx = ctx or _WORKER
    snap, mats, W, H = ctx["snap"], ctx["mats"], ctx["W"], ctx["H"]
    t_start = time.time()
    state = evaluate_scene_at_frame(snap, frame)
    cam_name = state["active_camera"]
    cam = state["camera"]
    cam_world = state["world"][cam_name]
    view = m_inv(cam_world)
    near = max(0.05, cam["clip_start"])
    fpx = cam["lens"] / cam["sensor_width"] * W     # sensor_fit AUTO, landscape -> horizontal
    cxp, cyp = W * 0.5, H * 0.5
    tx, ty = cxp / fpx, cyp / fpx
    W2, H2 = W // 2, H // 2
    fpx2, cxp2, cyp2 = fpx * 0.5, W2 * 0.5, H2 * 0.5
    up = m_apply_dir(view, (0.0, 0.0, 1.0))

    main_geo, mirror_candidates = [], []
    tri_count = 0
    view_m = m_mul(view, MIRROR_Z)
    for g in snap["geoms"]:
        Wm = state["world"][g["name"]]
        M = m_mul(view, Wm)
        Mm = m_mul(view_m, Wm)
        vis_main = _frustum_visible(M, g["bsphere"], tx, ty, near, fpx)
        vis_mirror = (not g["no_reflect"]) and g["bsphere"][3] >= 0.045 and _frustum_visible(Mm, g["bsphere"], tx, ty, near, fpx2)
        if not (vis_main or vis_mirror):
            continue
        verts = _deformed_local_verts(g, state) if (g["skin"] is not None or g["wave"] is not None) else g["verts"]
        em = _emit_mult(g, state)
        flatdyn = g["skin"] is not None
        if vis_main:
            main_geo.append((xform_points(M, verts), [] if flatdyn else xform_normals(m_cofactor3(M), g["normals"]),
                             g["tris"], g["ttris"], em, flatdyn))
            tri_count += len(g["tris"]) + len(g["ttris"])
        if vis_mirror:
            mirror_candidates.append((Mm, verts, g["normals"], g["tris"], em, flatdyn))

    lights_main = _lights_to_camera(state["lights"], view)

    # ---- main opaque raster pass first (so we know whether wet asphalt Z=0 is visible)
    zb, tb, recs, trans = _raster_pass(main_geo, W, H, fpx, cxp, cyp, near)
    asph_tids = {ti for ti, rec in enumerate(recs) if mats[rec[9]][12] == K_ASPHALT}
    recm = []
    refl_buf = None
    if asph_tids:
        asph_ymin, asph_ymax = H, 0
        for y in range(H):
            row = tb[y * W:(y + 1) * W]
            if any(t in asph_tids for t in row if t >= 0):
                if y < asph_ymin:
                    asph_ymin = y
                if y > asph_ymax:
                    asph_ymax = y
        if asph_ymin <= asph_ymax:
            ym0 = max(0, asph_ymin // 2 - 2)
            ym1 = min(H2, asph_ymax // 2 + 3)
            mirror_geo = [(xform_points(Mm, v), [] if fd else xform_normals(m_cofactor3(Mm), nrm), tr, [], em, fd)
                          for (Mm, v, nrm, tr, em, fd) in mirror_candidates]
            mirror_lights = [dict(L, pos=(L["pos"][0], L["pos"][1], -L["pos"][2]), dir=(L["dir"][0], L["dir"][1], -L["dir"][2]))
                             for L in state["lights"][:8]]
            lights_mirror = _lights_to_camera(mirror_lights, view)
            zbm, tbm, recm, _ = _raster_pass(mirror_geo, W2, H2, fpx2, cxp2, cyp2, near)
            RRm, RGm, RBm = _shade_pass(W2, H2, fpx2, cxp2, cyp2, zbm, tbm, recm, mats, lights_mirror, up, cam_world, True, None, ym0, ym1)
            RRm, RGm, RBm = _blur_buffer(RRm, RGm, RBm, W2, H2)
            refl_buf = (RRm, RGm, RBm, W2, H2)

    # ---- main shade + reflection composite, transparency, rain, bloom, AgX
    R, G, B = _shade_pass(W, H, fpx, cxp, cyp, zb, tb, recs, mats, lights_main, up, cam_world, False, refl_buf)
    _transparent_pass(trans, W, H, fpx, cxp, cyp, near, zb, R, G, B, mats, lights_main, up)
    drops = _rain_pass(snap, state, view, W, H, fpx, cxp, cyp, near, zb, R, G, B)
    _bloom(R, G, B, W, H)
    rgb = agx_tonemap_to_bytes(R, G, B, W, H, (2.0 ** EXPOSURE_EV) * EXPOSURE_CALIBRATION)
    stats = {
        "frame": int(frame), "camera": cam_name, "marker": state["active_marker"],
        "visible_triangles": tri_count, "rasterised_records": len(recs), "mirror_records": len(recm),
        "transparent_triangles": len(trans), "rain_streaks": drops,
        "seconds": round(time.time() - t_start, 2),
        "sha256": hashlib.sha256(rgb).hexdigest(),
    }
    return int(frame), rgb, stats


def _render_worker(frame):
    return render_frame(frame)


# ==============================================================================
# 8. OUTPUT: PNG writer & driver
# ==============================================================================

def write_png(path, width, height, rgb):
    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as fh:
        fh.write(png)
    return path


def _parse_args(argv):
    opts = {"frames": list(range(FRAME_START, FRAME_END + 1)), "width": DEFAULT_W, "height": DEFAULT_H,
            "workers": max(1, (os.cpu_count() or 2)), "mux": True, "snapshot_cache": None, "out_dir": SCRIPT_DIR}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--frames":
            frames = []
            for part in argv[i + 1].split(","):
                if "-" in part:
                    lo, hi = part.split("-")
                    frames.extend(range(int(lo), int(hi) + 1))
                else:
                    frames.append(int(part))
            opts["frames"] = frames
            i += 1
        elif a == "--width":
            opts["width"] = int(argv[i + 1])
            i += 1
        elif a == "--height":
            opts["height"] = int(argv[i + 1])
            i += 1
        elif a == "--workers":
            opts["workers"] = int(argv[i + 1])
            i += 1
        elif a == "--no-mux":
            opts["mux"] = False
        elif a == "--snapshot-cache":
            opts["snapshot_cache"] = argv[i + 1]
            i += 1
        elif a == "--out-dir":
            opts["out_dir"] = argv[i + 1]
            i += 1
        i += 1
    return opts


def load_or_build_snapshot(cache_path=None):
    with open(os.path.join(SCRIPT_DIR, SCENE_SCRIPT_NAME), "rb") as fh:
        src_hash = hashlib.sha256(fh.read()).hexdigest()
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, "rb") as fh:
            snap = pickle.load(fh)
        if snap.get("scene_source_sha256") == src_hash:
            print(f"[scene] loaded snapshot cache {cache_path}")
            return snap
    snap = build_scene_snapshot()
    if cache_path:
        with open(cache_path, "wb") as fh:
            pickle.dump(snap, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return snap


def main(argv=None):
    opts = _parse_args(sys.argv[1:] if argv is None else argv)
    W, H = opts["width"], opts["height"]
    out_dir = opts["out_dir"]
    t0 = time.time()
    snap = load_or_build_snapshot(opts["snapshot_cache"])
    frames = opts["frames"]
    workers = max(1, min(opts["workers"], len(frames)))
    print(f"[render] {len(frames)} frame(s) at {W}x{H} on {workers} worker process(es)", flush=True)
    results = {}
    stats = []
    if workers == 1:
        ctx = _prepare_render_context(snap, W, H)
        for f in frames:
            fr, rgb, st = render_frame(f, ctx)
            results[fr] = rgb
            stats.append(st)
            print(f"  f{fr:03d} {st['camera']:<38} {st['seconds']:6.2f}s  tris={st['visible_triangles']}", flush=True)
    else:
        mp = multiprocessing.get_context("spawn")
        with mp.Pool(workers, initializer=_init_worker, initargs=(snap, W, H)) as pool:
            for fr, rgb, st in pool.imap(_render_worker, frames, chunksize=1):
                results[fr] = rgb
                stats.append(st)
                print(f"  f{fr:03d} {st['camera']:<38} {st['seconds']:6.2f}s  tris={st['visible_triangles']}", flush=True)
    render_seconds = round(time.time() - t0, 1)

    stills = {}
    for f, fname in STILLS:
        if f in results:
            path = write_png(os.path.join(out_dir, fname), W, H, results[f])
            with open(path, "rb") as fh:
                stills[fname] = {"frame": f, "sha256": hashlib.sha256(fh.read()).hexdigest()}
            print(f"[still] {fname}")

    ordered_stats = sorted(stats, key=lambda s: s["frame"])
    report = {
        "renderer": "render_diorama_animation.py (pure-Python software Z-buffer, stdlib only)",
        "resolution": [W, H], "fps": FPS, "frames_rendered": len(results),
        "frame_range": [min(frames), max(frames)], "workers": workers,
        "render_wall_seconds": render_seconds, "scene_build_seconds": snap.get("build_seconds"),
        "scene_source_sha256": snap.get("scene_source_sha256"),
        "triangles_total": sum(len(g["tris"]) + len(g["ttris"]) for g in snap["geoms"]),
        "geometries": len(snap["geoms"]),
        "director_cut": [{"frame": m[0], "marker": m[1], "camera": m[2]} for m in snap["markers"]],
        "pipeline": ["fcurve_bezier_auto_clamped+linear", "parent_hierarchy", "FOLLOW_PATH", "TRACK_TO",
                     "posebone_skinning", "lattice", "array", "wave", "zbuffer_span_raster",
                     "perspective_correct_barycentrics", "ggx_multi_light", "planar_mirror_z0",
                     "alpha_additive_transparency", "gn_rain_streaks", "quadratic_mist",
                     "separable_box_gaussian_bloom", "agx_high_contrast"],
        "stills": stills,
        "frame_sha256": {str(s["frame"]): s["sha256"] for s in ordered_stats},
        "frame_cameras": {str(s["frame"]): s["camera"] for s in ordered_stats},
        "mean_seconds_per_frame_per_worker": round(sum(s["seconds"] for s in stats) / max(1, len(stats)), 2),
    }

    if opts["mux"] and len(results) == FRAME_END - FRAME_START + 1:
        import video_muxer
        ordered = [results[f] for f in range(FRAME_START, FRAME_END + 1)]
        video_muxer.write_frame_cache(os.path.join(out_dir, video_muxer.FRAME_CACHE_NAME), W, H, ordered)
        report["mux"] = video_muxer.mux_all(ordered, W, H, FPS, os.path.join(SCRIPT_DIR, WAV_NAME), out_dir)
    elif opts["mux"]:
        print("[mux] skipped: muxing requires all 240 frames (use --frames 1-240)")

    with open(os.path.join(out_dir, RENDER_REPORT_NAME), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    manifest_path = os.path.join(out_dir, "cyberpunk_diorama_manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as mfh:
            manifest_data = json.load(mfh)
        p7_art = {}
        for aname in ("cyberpunk_diorama_240f.avi", "cyberpunk_diorama_240f.mp4",
                      "render_f001_cam1.png", "render_f120_cam2.png", "render_f200_cam3.png"):
            apath = os.path.join(out_dir, aname)
            if os.path.isfile(apath):
                with open(apath, "rb") as afh:
                    p7_art[aname] = {
                        "filepath": aname,
                        "byte_size": os.path.getsize(apath),
                        "sha256_digest": hashlib.sha256(afh.read()).hexdigest(),
                    }
        manifest_data["phase7_animation_and_video"] = {
            "artifacts": p7_art,
            "frame_pixel_sha256": {
                k: report["frame_sha256"][k]
                for k in ("1", "120", "200")
                if k in report["frame_sha256"]
            },
        }
        with open(manifest_path, "w", encoding="utf-8") as mfh:
            json.dump(manifest_data, mfh, indent=2)

    print(f"[done] {len(results)} frames in {render_seconds}s -> {RENDER_REPORT_NAME}")
    return report


if __name__ == "__main__":
    main()
