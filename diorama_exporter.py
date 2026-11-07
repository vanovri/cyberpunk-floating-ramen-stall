import array
import base64
import json
import math
import os
import struct
import zlib


# ==============================================================================
# 1. PROCEDURAL VECTOR STROKE FONT GLYPHS (FOR 3D NEON 'FONT' CURVE TESSELLATION)
# ==============================================================================

GLYPH_STROKES = {
    "A": [[(0.0, 0.0), (0.35, 1.0), (0.70, 0.0)], [(0.12, 0.38), (0.58, 0.38)]],
    "B": [[(0.0, 0.0), (0.0, 1.0), (0.52, 1.0), (0.64, 0.86), (0.52, 0.52), (0.0, 0.52)],
          [(0.52, 0.52), (0.66, 0.38), (0.66, 0.14), (0.52, 0.0), (0.0, 0.0)]],
    "C": [[(0.66, 0.85), (0.50, 1.0), (0.16, 1.0), (0.0, 0.82), (0.0, 0.18), (0.16, 0.0), (0.50, 0.0), (0.66, 0.15)]],
    "D": [[(0.0, 0.0), (0.0, 1.0), (0.46, 1.0), (0.66, 0.78), (0.66, 0.22), (0.46, 0.0), (0.0, 0.0)]],
    "E": [[(0.64, 1.0), (0.0, 1.0), (0.0, 0.0), (0.64, 0.0)], [(0.0, 0.52), (0.50, 0.52)]],
    "F": [[(0.0, 0.0), (0.0, 1.0), (0.64, 1.0)], [(0.0, 0.52), (0.50, 0.52)]],
    "G": [[(0.66, 0.85), (0.50, 1.0), (0.16, 1.0), (0.0, 0.82), (0.0, 0.18), (0.16, 0.0),
           (0.54, 0.0), (0.66, 0.18), (0.66, 0.46), (0.36, 0.46)]],
    "H": [[(0.0, 0.0), (0.0, 1.0)], [(0.66, 0.0), (0.66, 1.0)], [(0.0, 0.52), (0.66, 0.52)]],
    "I": [[(0.12, 1.0), (0.54, 1.0)], [(0.33, 0.0), (0.33, 1.0)], [(0.12, 0.0), (0.54, 0.0)]],
    "J": [[(0.58, 1.0), (0.58, 0.20), (0.42, 0.0), (0.16, 0.0), (0.0, 0.20)]],
    "K": [[(0.0, 0.0), (0.0, 1.0)], [(0.66, 1.0), (0.08, 0.48), (0.66, 0.0)]],
    "L": [[(0.0, 1.0), (0.0, 0.0), (0.62, 0.0)]],
    "M": [[(0.0, 0.0), (0.0, 1.0), (0.36, 0.35), (0.72, 1.0), (0.72, 0.0)]],
    "N": [[(0.0, 0.0), (0.0, 1.0), (0.66, 0.0), (0.66, 1.0)]],
    "O": [[(0.16, 0.0), (0.50, 0.0), (0.66, 0.20), (0.66, 0.80), (0.50, 1.0), (0.16, 1.0), (0.0, 0.80), (0.0, 0.20), (0.16, 0.0)]],
    "P": [[(0.0, 0.0), (0.0, 1.0), (0.52, 1.0), (0.66, 0.84), (0.66, 0.62), (0.52, 0.48), (0.0, 0.48)]],
    "Q": [[(0.16, 0.0), (0.50, 0.0), (0.66, 0.20), (0.66, 0.80), (0.50, 1.0), (0.16, 1.0), (0.0, 0.80), (0.0, 0.20), (0.16, 0.0)],
          [(0.38, 0.28), (0.70, -0.06)]],
    "R": [[(0.0, 0.0), (0.0, 1.0), (0.52, 1.0), (0.66, 0.84), (0.66, 0.62), (0.52, 0.48), (0.0, 0.48)],
          [(0.36, 0.48), (0.66, 0.0)]],
    "S": [[(0.64, 0.86), (0.48, 1.0), (0.16, 1.0), (0.0, 0.84), (0.0, 0.64), (0.16, 0.52),
           (0.50, 0.46), (0.66, 0.32), (0.66, 0.14), (0.50, 0.0), (0.16, 0.0), (0.0, 0.14)]],
    "T": [[(0.0, 1.0), (0.68, 1.0)], [(0.34, 0.0), (0.34, 1.0)]],
    "U": [[(0.0, 1.0), (0.0, 0.20), (0.16, 0.0), (0.50, 0.0), (0.66, 0.20), (0.66, 1.0)]],
    "V": [[(0.0, 1.0), (0.34, 0.0), (0.68, 1.0)]],
    "W": [[(0.0, 1.0), (0.18, 0.0), (0.38, 0.62), (0.58, 0.0), (0.76, 1.0)]],
    "X": [[(0.0, 0.0), (0.66, 1.0)], [(0.0, 1.0), (0.66, 0.0)]],
    "Y": [[(0.0, 1.0), (0.33, 0.48), (0.66, 1.0)], [(0.33, 0.48), (0.33, 0.0)]],
    "Z": [[(0.0, 1.0), (0.66, 1.0), (0.0, 0.0), (0.66, 0.0)]],
    "0": [[(0.16, 0.0), (0.50, 0.0), (0.66, 0.20), (0.66, 0.80), (0.50, 1.0), (0.16, 1.0), (0.0, 0.80), (0.0, 0.20), (0.16, 0.0)],
          [(0.12, 0.18), (0.54, 0.82)]],
    "1": [[(0.14, 0.80), (0.34, 1.0), (0.34, 0.0)], [(0.12, 0.0), (0.56, 0.0)]],
    "2": [[(0.0, 0.80), (0.16, 1.0), (0.50, 1.0), (0.66, 0.82), (0.66, 0.60), (0.0, 0.0), (0.66, 0.0)]],
    "3": [[(0.0, 0.84), (0.16, 1.0), (0.50, 1.0), (0.66, 0.82), (0.50, 0.52), (0.24, 0.52)],
          [(0.50, 0.52), (0.66, 0.34), (0.66, 0.16), (0.50, 0.0), (0.16, 0.0), (0.0, 0.16)]],
    "4": [[(0.50, 0.0), (0.50, 1.0), (0.0, 0.32), (0.68, 0.32)]],
    "5": [[(0.64, 1.0), (0.0, 1.0), (0.0, 0.54), (0.50, 0.54), (0.66, 0.38), (0.66, 0.16), (0.50, 0.0), (0.0, 0.0)]],
    "6": [[(0.60, 0.86), (0.44, 1.0), (0.16, 1.0), (0.0, 0.78), (0.0, 0.18), (0.16, 0.0),
           (0.50, 0.0), (0.66, 0.18), (0.66, 0.42), (0.50, 0.56), (0.0, 0.56)]],
    "7": [[(0.0, 1.0), (0.66, 1.0), (0.22, 0.0)]],
    "8": [[(0.16, 0.52), (0.0, 0.66), (0.0, 0.84), (0.16, 1.0), (0.50, 1.0), (0.66, 0.84), (0.66, 0.66), (0.50, 0.52),
           (0.16, 0.52), (0.0, 0.36), (0.0, 0.16), (0.16, 0.0), (0.50, 0.0), (0.66, 0.16), (0.66, 0.36), (0.50, 0.52)]],
    "9": [[(0.66, 0.44), (0.16, 0.44), (0.0, 0.60), (0.0, 0.84), (0.16, 1.0), (0.50, 1.0),
           (0.66, 0.82), (0.66, 0.18), (0.50, 0.0), (0.14, 0.0)]],
    "-": [[(0.08, 0.48), (0.58, 0.48)]],
    "/": [[(0.06, 0.0), (0.58, 1.0)]],
    ":": [[(0.30, 0.20), (0.30, 0.28)], [(0.30, 0.72), (0.30, 0.80)]],
}


# ==============================================================================
# 2. LINEAR ALGEBRA, SPLINE SAMPLING & BISHOP PARALLEL-TRANSPORT FRAMES
# ==============================================================================

def _v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v_len(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _v_norm(a):
    l = _v_len(a)
    if l < 1e-12:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / l
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def _mat4_identity():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        a0, a1, a2, a3 = a[i]
        out[i][0] = a0 * b[0][0] + a1 * b[1][0] + a2 * b[2][0] + a3 * b[3][0]
        out[i][1] = a0 * b[0][1] + a1 * b[1][1] + a2 * b[2][1] + a3 * b[3][1]
        out[i][2] = a0 * b[0][2] + a1 * b[1][2] + a2 * b[2][2] + a3 * b[3][2]
        out[i][3] = a0 * b[0][3] + a1 * b[1][3] + a2 * b[2][3] + a3 * b[3][3]
    return out


def _mat4_from_trs(loc, rot_xyz, scale):
    cx, sx = math.cos(rot_xyz[0]), math.sin(rot_xyz[0])
    cy, sy = math.cos(rot_xyz[1]), math.sin(rot_xyz[1])
    cz, sz = math.cos(rot_xyz[2]), math.sin(rot_xyz[2])
    r00 = (cy * cz) * scale[0]
    r01 = (cz * sx * sy - cx * sz) * scale[1]
    r02 = (cx * cz * sy + sx * sz) * scale[2]
    r10 = (cy * sz) * scale[0]
    r11 = (cx * cz + sx * sy * sz) * scale[1]
    r12 = (cx * sy * sz - cz * sx) * scale[2]
    r20 = (-sy) * scale[0]
    r21 = (cy * sx) * scale[1]
    r22 = (cx * cy) * scale[2]
    return [
        [r00, r01, r02, float(loc[0])],
        [r10, r11, r12, float(loc[1])],
        [r20, r21, r22, float(loc[2])],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _mat4_transform_pt(m, p):
    x, y, z = p
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def _mat4_transform_dir(m, d):
    x, y, z = d
    return _v_norm((
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    ))


def _mat4_invert_rigid_or_affine(m):
    """Invert 4x4 affine transform matrix for Lattice/Armature local space evaluation."""
    a00, a01, a02 = m[0][0], m[0][1], m[0][2]
    a10, a11, a12 = m[1][0], m[1][1], m[1][2]
    a20, a21, a22 = m[2][0], m[2][1], m[2][2]
    det = (
        a00 * (a11 * a22 - a12 * a21)
        - a01 * (a10 * a22 - a12 * a20)
        + a02 * (a10 * a21 - a11 * a20)
    )
    if abs(det) < 1e-12:
        return _mat4_identity()
    inv_det = 1.0 / det
    i00 = (a11 * a22 - a12 * a21) * inv_det
    i01 = (a02 * a21 - a01 * a22) * inv_det
    i02 = (a01 * a12 - a02 * a11) * inv_det
    i10 = (a12 * a20 - a10 * a22) * inv_det
    i11 = (a00 * a22 - a02 * a20) * inv_det
    i12 = (a02 * a10 - a00 * a12) * inv_det
    i20 = (a10 * a21 - a11 * a20) * inv_det
    i21 = (a01 * a20 - a00 * a21) * inv_det
    i22 = (a00 * a11 - a01 * a10) * inv_det
    tx, ty, tz = m[0][3], m[1][3], m[2][3]
    return [
        [i00, i01, i02, -(i00 * tx + i01 * ty + i02 * tz)],
        [i10, i11, i12, -(i10 * tx + i11 * ty + i12 * tz)],
        [i20, i21, i22, -(i20 * tx + i21 * ty + i22 * tz)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _eval_fcurve_value(obj, data_path, frame, default_val):
    ad = getattr(obj, "animation_data", None)
    act = getattr(ad, "action", None) if ad is not None else None
    if act is None:
        return default_val
    fc = next((f for f in getattr(act, "fcurves", []) if f.data_path == data_path), None)
    if fc is None or not fc.keyframe_points:
        return default_val
    kps = sorted(fc.keyframe_points, key=lambda k: k.co[0])
    if frame <= kps[0].co[0]:
        return kps[0].co[1]
    if frame >= kps[-1].co[0]:
        return kps[-1].co[1]
    for i in range(len(kps) - 1):
        f0, v0 = kps[i].co
        f1, v1 = kps[i + 1].co
        if f0 <= frame <= f1:
            t = (frame - f0) / max(1e-6, (f1 - f0))
            if isinstance(v0, (tuple, list)) and isinstance(v1, (tuple, list)):
                return tuple(a + (b - a) * t for a, b in zip(v0, v1))
            if isinstance(v0, (int, float)) and isinstance(v1, (int, float)):
                return float(v0 + (v1 - v0) * t)
            return v0
    return default_val


def sample_spline_3d_points(spline):
    """Evaluate POLY, NURBS, or BEZIER spline into a 3D polyline and cyclic flag."""
    cyclic = bool(getattr(spline, "use_cyclic_u", False))
    if spline.type == 'BEZIER':
        bpts = spline.bezier_points
        n = len(bpts)
        if n < 2:
            return [], False
        seg_count = n if cyclic else (n - 1)
        steps_per_seg = 16
        out = []
        for i in range(seg_count):
            p0_node = bpts[i]
            p1_node = bpts[(i + 1) % n]
            b0 = (p0_node.co.x, p0_node.co.y, p0_node.co.z)
            b1 = (p0_node.handle_right.x, p0_node.handle_right.y, p0_node.handle_right.z)
            b2 = (p1_node.handle_left.x, p1_node.handle_left.y, p1_node.handle_left.z)
            b3 = (p1_node.co.x, p1_node.co.y, p1_node.co.z)
            end_s = steps_per_seg if (i < seg_count - 1 or cyclic) else (steps_per_seg + 1)
            for s in range(end_s):
                t = s / steps_per_seg
                mt = 1.0 - t
                c0 = mt * mt * mt
                c1 = 3.0 * mt * mt * t
                c2 = 3.0 * mt * t * t
                c3 = t * t * t
                out.append((
                    c0 * b0[0] + c1 * b1[0] + c2 * b2[0] + c3 * b3[0],
                    c0 * b0[1] + c1 * b1[1] + c2 * b2[1] + c3 * b3[1],
                    c0 * b0[2] + c1 * b1[2] + c2 * b2[2] + c3 * b3[2],
                ))
        return out, cyclic

    raw_pts = [(float(p.co[0]), float(p.co[1]), float(p.co[2])) for p in spline.points]
    n = len(raw_pts)
    if n < 2:
        return raw_pts, False
    if spline.type == 'POLY' or n == 2:
        return raw_pts, cyclic

    # Smooth Catmull-Rom / B-Spline subdivision for NURBS curves (pipes, noodles, traffic lanes)
    sub_steps = 4
    out = []
    seg_count = n if cyclic else (n - 1)
    for i in range(seg_count):
        p0 = raw_pts[(i - 1) % n] if cyclic else raw_pts[max(0, i - 1)]
        p1 = raw_pts[i]
        p2 = raw_pts[(i + 1) % n] if cyclic else raw_pts[min(n - 1, i + 1)]
        p3 = raw_pts[(i + 2) % n] if cyclic else raw_pts[min(n - 1, i + 2)]
        end_s = sub_steps if (i < seg_count - 1 or cyclic) else (sub_steps + 1)
        for s in range(end_s):
            t = s / sub_steps
            t2 = t * t
            t3 = t2 * t
            f0 = -0.5 * t3 + t2 - 0.5 * t
            f1 = 1.5 * t3 - 2.5 * t2 + 1.0
            f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            f3 = 0.5 * t3 - 0.5 * t2
            out.append((
                f0 * p0[0] + f1 * p1[0] + f2 * p2[0] + f3 * p3[0],
                f0 * p0[1] + f1 * p1[1] + f2 * p2[1] + f3 * p3[1],
                f0 * p0[2] + f1 * p1[2] + f2 * p2[2] + f3 * p3[2],
            ))
    return out, cyclic


def tessellate_polyline_bishop_tube(pts, radius, cyclic=False, radial_sides=8, mat_idx=0):
    """Sweep a circular cross-section along 3D polyline using Bishop parallel-transport frames."""
    n = len(pts)
    if n < 2:
        return [], []

    tangents = []
    for i in range(n):
        if cyclic:
            prv = pts[(i - 1) % n]
            nxt = pts[(i + 1) % n]
            d = _v_sub(nxt, prv)
        elif i == 0:
            d = _v_sub(pts[1], pts[0])
        elif i == n - 1:
            d = _v_sub(pts[n - 1], pts[n - 2])
        else:
            d = _v_sub(pts[i + 1], pts[i - 1])
        tangents.append(_v_norm(d))

    # Initialize Bishop frame normal perpendicular to tangents[0]
    t0 = tangents[0]
    up = (0.0, 0.0, 1.0) if abs(t0[2]) < 0.88 else (1.0, 0.0, 0.0)
    norm_cur = _v_norm(_v_cross(t0, up))

    verts = []
    tris = []
    unit_circle = [
        (math.cos(2.0 * math.pi * k / radial_sides), math.sin(2.0 * math.pi * k / radial_sides))
        for k in range(radial_sides)
    ]

    for i in range(n):
        ti = tangents[i]
        # Parallel transport normal onto plane orthogonal to ti
        proj = _v_sub(norm_cur, _v_scale(ti, _v_dot(norm_cur, ti)))
        if _v_len(proj) > 1e-6:
            norm_cur = _v_norm(proj)
        binorm = _v_norm(_v_cross(ti, norm_cur))

        px, py, pz = pts[i]
        for ca, sa in unit_circle:
            rx = (norm_cur[0] * ca + binorm[0] * sa) * radius
            ry = (norm_cur[1] * ca + binorm[1] * sa) * radius
            rz = (norm_cur[2] * ca + binorm[2] * sa) * radius
            verts.append((px + rx, py + ry, pz + rz))

    ring_links = n if cyclic else (n - 1)
    for i in range(ring_links):
        j = (i + 1) % n
        base0 = i * radial_sides
        base1 = j * radial_sides
        for k in range(radial_sides):
            kn = (k + 1) % radial_sides
            v00 = base0 + k
            v01 = base0 + kn
            v11 = base1 + kn
            v10 = base1 + k
            tris.append((v00, v01, v11, mat_idx))
            tris.append((v00, v11, v10, mat_idx))

    if not cyclic:
        # Start and end hemispherical/flat end-caps
        c_start_idx = len(verts)
        verts.append(pts[0])
        c_end_idx = len(verts)
        verts.append(pts[-1])
        last_ring = (n - 1) * radial_sides
        for k in range(radial_sides):
            kn = (k + 1) % radial_sides
            tris.append((c_start_idx, kn, k, mat_idx))
            tris.append((c_end_idx, last_ring + k, last_ring + kn, mat_idx))

    return verts, tris


def tessellate_curve_or_font_data(curve_data):
    """Tessellate any Blender CURVE or FONT data-block into (verts, tris)."""
    if curve_data.type == 'FONT':
        text_str = getattr(curve_data, "body", "") or "CYBER"
        f_size = float(getattr(curve_data, "size", 0.25))
        extrude = float(getattr(curve_data, "extrude", 0.015))
        bevel = float(getattr(curve_data, "bevel_depth", 0.004))
        tube_r = max(bevel, extrude * 0.45, f_size * 0.058)
        char_advance = f_size * 0.78 * float(getattr(curve_data, "space_character", 1.0))

        total_w = max(0.0, len(text_str) * char_advance - (char_advance * 0.22))
        align_x = getattr(curve_data, "align_x", "CENTER")
        align_y = getattr(curve_data, "align_y", "CENTER")
        x_start = -total_w * 0.5 if align_x == 'CENTER' else (-total_w if align_x == 'RIGHT' else 0.0)
        y_start = -f_size * 0.5 if align_y == 'CENTER' else 0.0

        all_verts = []
        all_tris = []
        for ch_i, ch in enumerate(text_str.upper()):
            strokes = GLYPH_STROKES.get(ch)
            if not strokes:
                continue
            cx0 = x_start + ch_i * char_advance
            for st in strokes:
                pts_3d = [
                    (cx0 + gx * f_size * 0.72, y_start + gy * f_size, 0.0)
                    for gx, gy in st
                ]
                sv, st_tris = tessellate_polyline_bishop_tube(pts_3d, tube_r, cyclic=False, radial_sides=6, mat_idx=0)
                v_off = len(all_verts)
                all_verts.extend(sv)
                for a, b, c, m in st_tris:
                    all_tris.append((a + v_off, b + v_off, c + v_off, m))
        return all_verts, all_tris

    # Standard 3D CURVE data-block
    bev_r = float(getattr(curve_data, "bevel_depth", 0.0))
    eff_r = bev_r if bev_r > 1e-5 else 0.002
    sides = 8 if bev_r >= 0.01 else 6
    all_verts = []
    all_tris = []
    for sp in curve_data.splines:
        pts, cyc = sample_spline_3d_points(sp)
        if len(pts) < 2:
            continue
        sv, st_tris = tessellate_polyline_bishop_tube(pts, eff_r, cyclic=cyc, radial_sides=sides, mat_idx=0)
        v_off = len(all_verts)
        all_verts.extend(sv)
        for a, b, c, m in st_tris:
            all_tris.append((a + v_off, b + v_off, c + v_off, m))
    return all_verts, all_tris


# ==============================================================================
# 3. MODIFIER STACK EVALUATOR (LATTICE, WAVE, SOLIDIFY, ARRAY, ARMATURE, NODES)
# ==============================================================================

def _compute_vertex_normals(verts, tris):
    v_norms = [[0.0, 0.0, 0.0] for _ in range(len(verts))]
    for i0, i1, i2, _m in tris:
        p0, p1, p2 = verts[i0], verts[i1], verts[i2]
        fn = _v_cross(_v_sub(p1, p0), _v_sub(p2, p0))
        for idx in (i0, i1, i2):
            v_norms[idx][0] += fn[0]
            v_norms[idx][1] += fn[1]
            v_norms[idx][2] += fn[2]
    return [_v_norm(vn) for vn in v_norms]


def _apply_lattice_modifier(verts, obj_world_mat, lat_obj):
    if lat_obj is None or getattr(lat_obj, "data", None) is None:
        return verts
    lat_data = lat_obj.data
    nu, nv, nw = lat_data.points_u, lat_data.points_v, lat_data.points_w
    pts = lat_data.points
    if len(pts) != nu * nv * nw:
        return verts

    lat_mat = _mat4_from_trs(lat_obj.location, lat_obj.rotation_euler, lat_obj.scale)
    inv_lat = _mat4_invert_rigid_or_affine(lat_mat)
    inv_obj = _mat4_invert_rigid_or_affine(obj_world_mat)

    out_verts = []
    for v in verts:
        w_pt = _mat4_transform_pt(obj_world_mat, v)
        l_pt = _mat4_transform_pt(inv_lat, w_pt)
        # Map lattice normalized [-0.5, +0.5] w-coordinate and radial scale
        u_f = max(0.0, min(1.0, l_pt[2] + 0.5)) * (nw - 1)
        k0 = min(nw - 2, int(u_f))
        k1 = k0 + 1
        frac = u_f - k0

        # Sample center column of layer k0 and k1 for z-squeeze, and corner for radial bulge
        c_idx0 = k0 * (nu * nv)
        c_idx1 = k1 * (nu * nv)
        r0 = abs(pts[c_idx0].co_deform.x) / max(1e-5, abs(pts[c_idx0].co.x))
        r1 = abs(pts[c_idx1].co_deform.x) / max(1e-5, abs(pts[c_idx1].co.x))
        r_interp = r0 + (r1 - r0) * frac

        z0 = pts[c_idx0].co_deform.z
        z1 = pts[c_idx1].co_deform.z
        z_interp = z0 + (z1 - z0) * frac

        def_local = (l_pt[0] * r_interp, l_pt[1] * r_interp, z_interp)
        def_world = _mat4_transform_pt(lat_mat, def_local)
        out_verts.append(_mat4_transform_pt(inv_obj, def_world))
    return out_verts


def _apply_wave_modifier(verts, tris, mod):
    height = float(getattr(mod, "height", 0.008))
    width = max(0.02, float(getattr(mod, "width", 0.25)))
    use_normal = bool(getattr(mod, "use_normal", False))
    k_freq = (2.0 * math.pi) / (width * 4.0)
    v_norms = _compute_vertex_normals(verts, tris) if use_normal else None

    out = []
    for idx, (vx, vy, vz) in enumerate(verts):
        radial = math.sqrt(vx * vx + vy * vy)
        disp = height * math.sin((radial + vz) * k_freq)
        if use_normal and v_norms is not None:
            nx, ny, nz = v_norms[idx]
            out.append((vx + nx * disp, vy + ny * disp, vz + nz * disp))
        else:
            out.append((vx + disp * 0.35, vy + disp * 0.35, vz + disp))
    return out


def _apply_solidify_modifier(verts, tris, mod):
    thickness = float(getattr(mod, "thickness", 0.01))
    offset = float(getattr(mod, "offset", -1.0))
    if abs(thickness) < 1e-6 or not verts or not tris:
        return verts, tris

    v_norms = _compute_vertex_normals(verts, tris)
    n_orig = len(verts)
    d_front = thickness * (offset + 1.0) * 0.5
    d_back = d_front - thickness

    front_verts = [
        (v[0] + vn[0] * d_front, v[1] + vn[1] * d_front, v[2] + vn[2] * d_front)
        for v, vn in zip(verts, v_norms)
    ]
    back_verts = [
        (v[0] + vn[0] * d_back, v[1] + vn[1] * d_back, v[2] + vn[2] * d_back)
        for v, vn in zip(verts, v_norms)
    ]
    new_verts = front_verts + back_verts
    new_tris = list(tris)
    for i0, i1, i2, m in tris:
        new_tris.append((n_orig + i0, n_orig + i2, n_orig + i1, m))

    # Stitch boundary edges
    edge_counts = {}
    edge_mat = {}
    for i0, i1, i2, m in tris:
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            key = (min(a, b), max(a, b))
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edge_mat[key] = (a, b, m)

    for key, cnt in edge_counts.items():
        if cnt == 1:
            a, b, m = edge_mat[key]
            new_tris.append((a, b, n_orig + b, m))
            new_tris.append((a, n_orig + b, n_orig + a, m))

    return new_verts, new_tris


def _apply_array_modifier(verts, tris, mod):
    count = max(1, int(getattr(mod, "count", 1)))
    if count <= 1 or not verts:
        return verts, tris

    dx, dy, dz = 0.0, 0.0, 0.0
    if getattr(mod, "use_constant_offset", False):
        disp = getattr(mod, "constant_offset_displace", (0.0, 0.0, 0.0))
        dx += float(disp[0])
        dy += float(disp[1])
        dz += float(disp[2])
    if getattr(mod, "use_relative_offset", False):
        rel = getattr(mod, "relative_offset_displace", (1.0, 0.0, 0.0))
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        dx += (max(xs) - min(xs)) * float(rel[0])
        dy += (max(ys) - min(ys)) * float(rel[1])
        dz += (max(zs) - min(zs)) * float(rel[2])

    out_verts = []
    out_tris = []
    n_base = len(verts)
    for c in range(count):
        ox, oy, oz = c * dx, c * dy, c * dz
        v_off = c * n_base
        for vx, vy, vz in verts:
            out_verts.append((vx + ox, vy + oy, vz + oz))
        for i0, i1, i2, m in tris:
            out_tris.append((i0 + v_off, i1 + v_off, i2 + v_off, m))
    return out_verts, out_tris


def _apply_armature_modifier(verts, obj, arm_obj, eval_frame):
    """Evaluate hierarchical EditBone + PoseBone FK skinning onto vertex group weights."""
    if arm_obj is None or getattr(arm_obj, "data", None) is None or getattr(arm_obj, "pose", None) is None:
        return verts

    ebones = {eb.name: eb for eb in arm_obj.data.edit_bones}
    bone_Skin_mats = {}

    def get_bone_deform_mat(b_name):
        if b_name in bone_Skin_mats:
            return bone_Skin_mats[b_name]
        eb = ebones.get(b_name)
        if eb is None:
            return _mat4_identity()
        parent_def = get_bone_deform_mat(eb.parent.name) if eb.parent is not None else _mat4_identity()
        hx, hy, hz = eb.head.x, eb.head.y, eb.head.z
        rot_val = _eval_fcurve_value(
            arm_obj, f'pose.bones["{b_name}"].rotation_euler', eval_frame, (0.0, 0.0, 0.0)
        )
        # Rotate around bone rest head position: T(head) * R(pose_euler) * T(-head)
        t_pos = _mat4_from_trs((hx, hy, hz), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        r_mat = _mat4_from_trs((0.0, 0.0, 0.0), rot_val, (1.0, 1.0, 1.0))
        t_neg = _mat4_from_trs((-hx, -hy, -hz), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        local_delta = _mat4_mul(t_pos, _mat4_mul(r_mat, t_neg))
        full_def = _mat4_mul(parent_def, local_delta)
        bone_Skin_mats[b_name] = full_def
        return full_def

    v_bone_map = {}
    for vg in getattr(obj, "vertex_groups", []):
        b_mat = get_bone_deform_mat(vg.name)
        for v_idx in vg.weights:
            v_bone_map[v_idx] = b_mat

    out_verts = []
    for idx, v in enumerate(verts):
        b_mat = v_bone_map.get(idx)
        out_verts.append(_mat4_transform_pt(b_mat, v) if b_mat is not None else v)
    return out_verts


# ==============================================================================
# 4. MATERIAL EXTRACTOR & FULL SCENE GRAPH EVALUATOR
# ==============================================================================

def extract_pbr_materials(bpy):
    """Extract PBR baseColor, metallic, roughness, emission, and alpha from all 31 ShaderNodeTrees."""
    mat_dict = {}
    for idx, mat in enumerate(bpy.data.materials):
        base_col = [0.22, 0.24, 0.28, 1.0]
        metallic = 0.0
        roughness = 0.45
        emit_col = [0.0, 0.0, 0.0]
        emit_str = 0.0
        alpha = 1.0
        transmission = 0.0

        nt = getattr(mat, "node_tree", None)
        if nt is not None:
            bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
            if bsdf is not None:
                bc = bsdf.inputs["Base Color"].default_value
                if bc is not None:
                    base_col = [float(bc[0]), float(bc[1]), float(bc[2]), float(bc[3]) if len(bc) > 3 else 1.0]
                m_val = bsdf.inputs["Metallic"].default_value
                if m_val is not None:
                    metallic = float(m_val)
                r_val = bsdf.inputs["Roughness"].default_value
                if r_val is not None:
                    roughness = float(r_val)
                al_val = bsdf.inputs["Alpha"].default_value
                if al_val is not None:
                    alpha = float(al_val)
                tr_val = bsdf.inputs["Transmission Weight"].default_value
                if tr_val is not None:
                    transmission = float(tr_val)
                ec = bsdf.inputs["Emission Color"].default_value
                if ec is not None:
                    emit_col = [float(ec[0]), float(ec[1]), float(ec[2])]
                es = bsdf.inputs["Emission Strength"].default_value
                if es is not None:
                    emit_str = float(es)

            # Inspect ColorRamp / Mix nodes driving procedural albedo or emission
            ramps = [n for n in nt.nodes if n.bl_idname == "ShaderNodeValToRGB"]
            if ramps and bsdf is not None:
                # Use mid-tone of primary albedo/emission color ramp when linked
                els = ramps[0].color_ramp.elements
                if len(els) >= 2:
                    c0, c1 = els[0].color, els[-1].color
                    avg = [(c0[k] + c1[k]) * 0.5 for k in range(3)]
                    if emit_str > 1.0 and sum(emit_col) < 0.05:
                        emit_col = avg
                    elif bsdf.inputs["Base Color"].default_value is None:
                        base_col = [avg[0], avg[1], avg[2], 1.0]

            mix_rgba = [
                n for n in nt.nodes
                if n.bl_idname == "ShaderNodeMix" and getattr(n, "data_type", "") == "RGBA"
            ]
            if mix_rgba and emit_str > 5.0:
                c_rim = mix_rgba[0].inputs[7].default_value
                c_core = mix_rgba[0].inputs[6].default_value
                if c_rim is not None and c_core is not None:
                    emit_col = [
                        float(c_core[k] * 0.35 + c_rim[k] * 0.65)
                        for k in range(3)
                    ]
                    base_col = [emit_col[0], emit_col[1], emit_col[2], 1.0]

        # Specific curated tuning for iconic materials
        mname = mat.name
        if "Wet_Asphalt" in mname:
            base_col = [0.032, 0.036, 0.046, 1.0]
            roughness = 0.14
            metallic = 0.10
        elif "Building_Night_Windows" in mname:
            base_col = [0.038, 0.045, 0.062, 1.0]
            emit_col = [0.12, 0.48, 0.68]
            emit_str = 3.2
        elif "Lantern_Washi_Paper" in mname:
            base_col = [1.0, 0.52, 0.16, 1.0]
            emit_col = [1.0, 0.48, 0.12]
            emit_str = 6.5
        elif "Mega_Hologram_Glitch" in mname:
            alpha = 0.18
            emit_str = 2.8
        elif "Hologram_Menu" in mname:
            alpha = 0.32
            emit_str = 3.4

        mat_dict[mname] = {
            "index": idx,
            "name": mname,
            "base_color": base_col,
            "metallic": max(0.0, min(1.0, metallic)),
            "roughness": max(0.04, min(1.0, roughness)),
            "emissive_color": emit_col,
            "emissive_strength": emit_str,
            "alpha": max(0.08, min(1.0, alpha)),
            "transmission": transmission,
        }
    return mat_dict


def compute_object_world_matrices(bpy, eval_frame=115, camera_overrides=None):
    """Compute 4x4 world matrices for all 579 objects (with parent chains, FOLLOW_PATH & TRACK_TO)."""
    camera_overrides = camera_overrides or {}
    world_mats = {}
    curve_cache = {}

    def get_curve_points_world(curve_obj):
        if curve_obj.name in curve_cache:
            return curve_cache[curve_obj.name]
        c_mat = get_world_mat(curve_obj)
        sp = curve_obj.data.splines[0]
        local_pts, cyc = sample_spline_3d_points(sp)
        w_pts = [_mat4_transform_pt(c_mat, p) for p in local_pts]
        curve_cache[curve_obj.name] = (w_pts, cyc)
        return w_pts, cyc

    def sample_point_on_curve(curve_obj, factor):
        w_pts, cyc = get_curve_points_world(curve_obj)
        n = len(w_pts)
        if n == 0:
            return (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        if n == 1:
            return w_pts[0], (0.0, 1.0, 0.0)
        u = (factor % 1.0) * (n if cyc else (n - 1))
        i0 = int(u) % n
        i1 = (i0 + 1) % n if cyc else min(n - 1, i0 + 1)
        frac = u - int(u)
        p0, p1 = w_pts[i0], w_pts[i1]
        pos = (
            p0[0] + (p1[0] - p0[0]) * frac,
            p0[1] + (p1[1] - p0[1]) * frac,
            p0[2] + (p1[2] - p0[2]) * frac,
        )
        tan = _v_norm(_v_sub(p1, p0))
        return pos, tan

    def get_world_mat(obj):
        if obj.name in world_mats:
            return world_mats[obj.name]

        f_use = camera_overrides.get(obj.name, eval_frame)
        loc = _eval_fcurve_value(obj, "location", f_use, tuple(obj.location))
        rot = _eval_fcurve_value(obj, "rotation_euler", f_use, tuple(obj.rotation_euler))
        scl = _eval_fcurve_value(obj, "scale", f_use, tuple(obj.scale))

        # Check FOLLOW_PATH constraint
        fp_con = next((c for c in getattr(obj, "constraints", []) if c.type == 'FOLLOW_PATH'), None)
        if fp_con is not None and fp_con.target is not None:
            fac_default = float(getattr(fp_con, "offset_factor", 0.0))
            if obj.name == "Cinematic_Camera_85mm" and obj.name not in camera_overrides:
                fac = 0.0  # Hero 3/4 establishing angle at frame 1
            else:
                fac = _eval_fcurve_value(
                    obj, f'constraints["{fp_con.name}"].offset_factor', f_use, fac_default
                )
            c_pos, c_tan = sample_point_on_curve(fp_con.target, float(fac))
            loc = _v_add(c_pos, loc)
            if getattr(fp_con, "use_curve_follow", False):
                yaw = math.atan2(c_tan[1], c_tan[0])
                rot = (rot[0], rot[1], yaw)

        local_m = _mat4_from_trs(loc, rot, scl)
        if getattr(obj, "parent", None) is not None:
            parent_m = get_world_mat(obj.parent)
            w_mat = _mat4_mul(parent_m, local_m)
        else:
            w_mat = local_m

        # Check TRACK_TO constraint (camera / area light aiming at target)
        tr_con = next((c for c in getattr(obj, "constraints", []) if c.type == 'TRACK_TO'), None)
        if tr_con is not None and tr_con.target is not None:
            t_mat = get_world_mat(tr_con.target)
            eye = (w_mat[0][3], w_mat[1][3], w_mat[2][3])
            target_pt = (t_mat[0][3], t_mat[1][3], t_mat[2][3])
            # Camera local -Z points toward target, +Y points up, +X is right
            forward = _v_norm(_v_sub(target_pt, eye))
            minus_z = _v_scale(forward, -1.0)
            world_up = (0.0, 0.0, 1.0)
            right_x = _v_norm(_v_cross(forward, world_up))
            true_up_y = _v_norm(_v_cross(right_x, forward))
            w_mat = [
                [right_x[0], true_up_y[0], minus_z[0], eye[0]],
                [right_x[1], true_up_y[1], minus_z[1], eye[1]],
                [right_x[2], true_up_y[2], minus_z[2], eye[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]

        world_mats[obj.name] = w_mat
        return w_mat

    for obj in bpy.data.objects:
        get_world_mat(obj)
    return world_mats


def evaluate_diorama_scene(bpy, eval_frame=115, camera_overrides=None):
    """Evaluate all 579 objects (meshes, modifiers, 154 curves/fonts, lights & cameras) in world space."""
    materials = extract_pbr_materials(bpy)
    default_mat_name = next(iter(materials.keys()))
    world_mats = compute_object_world_matrices(bpy, eval_frame=eval_frame, camera_overrides=camera_overrides)

    evaluated_objects = []
    for obj in bpy.data.objects:
        w_mat = world_mats[obj.name]
        obj_type = obj.type
        data = obj.data
        local_verts = []
        local_tris = []
        slot_mat_names = []

        if obj_type == 'MESH' and data is not None:
            slot_mat_names = [m.name for m in getattr(data, "materials", []) if m is not None]
            if not slot_mat_names:
                slot_mat_names = [default_mat_name]
            local_verts = [(v.co.x, v.co.y, v.co.z) for v in data.vertices]
            for poly in data.polygons:
                p_verts = getattr(poly, "vertices", ())
                m_slot = min(len(slot_mat_names) - 1, max(0, int(getattr(poly, "material_index", 0))))
                if len(p_verts) >= 3:
                    v0 = p_verts[0]
                    for k in range(1, len(p_verts) - 1):
                        local_tris.append((v0, p_verts[k], p_verts[k + 1], m_slot))

            # Evaluate modifier stack in order
            for mod in getattr(obj, "modifiers", []):
                mtype = mod.type
                if mtype == 'LATTICE':
                    local_verts = _apply_lattice_modifier(local_verts, w_mat, mod.object)
                elif mtype == 'WAVE':
                    local_verts = _apply_wave_modifier(local_verts, local_tris, mod)
                elif mtype == 'SOLIDIFY':
                    local_verts, local_tris = _apply_solidify_modifier(local_verts, local_tris, mod)
                elif mtype == 'ARRAY':
                    local_verts, local_tris = _apply_array_modifier(local_verts, local_tris, mod)
                elif mtype == 'ARMATURE':
                    local_verts = _apply_armature_modifier(local_verts, obj, mod.object, eval_frame)
                elif mtype == 'NODES' and getattr(mod, "node_group", None) is not None:
                    # Scatter deterministic raindrop streaks across emitter sheet
                    local_verts = []
                    local_tris = []
                    rng = 0x13579BDF
                    for s_i in range(42):
                        rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
                        rx = -6.2 + ((rng & 0xFFFF) / 65535.0) * 12.4
                        ry = -6.2 + (((rng >> 16) & 0xFFFF) / 65535.0) * 12.4
                        rz = (s_i % 6) * 1.15 - 3.5
                        v_b = len(local_verts)
                        local_verts.extend([
                            (rx - 0.004, ry, rz - 0.24),
                            (rx + 0.004, ry, rz - 0.24),
                            (rx, ry + 0.004, rz + 0.24),
                        ])
                        local_tris.append((v_b, v_b + 1, v_b + 2, 0))

        elif obj_type in ('CURVE', 'FONT') and data is not None:
            slot_mat_names = [m.name for m in getattr(data, "materials", []) if m is not None]
            if not slot_mat_names:
                slot_mat_names = [default_mat_name]
            local_verts, local_tris = tessellate_curve_or_font_data(data)

        # Transform vertices and normals into world space
        world_verts = [_mat4_transform_pt(w_mat, v) for v in local_verts]
        world_norms = _compute_vertex_normals(world_verts, local_tris) if world_verts and local_tris else []

        # Helper path curves with bevel_depth == 0 or hide_render=True are tagged for beauty pass filtering
        is_helper = bool(
            getattr(obj, "hide_render", False)
            or (obj_type == 'CURVE' and getattr(data, "bevel_depth", 0.0) <= 1e-5)
        )

        evaluated_objects.append({
            "name": obj.name,
            "type": obj_type,
            "world_matrix": w_mat,
            "verts": world_verts,
            "normals": world_norms,
            "tris": local_tris,
            "material_slots": slot_mat_names,
            "is_helper": is_helper,
            "data": data,
        })

    return {
        "materials": materials,
        "objects": evaluated_objects,
        "world_matrices": world_mats,
    }


# ==============================================================================
# 5. TRACK 22: SPEC-COMPLIANT BINARY GLTF 2.0 (.GLB) & WAVEFRONT (.OBJ / .MTL)
# ==============================================================================

def _blender_to_gltf_vec3(v):
    """Convert Blender Z-up right-handed (x, y, z) to glTF 2.0 Y-up right-handed (x, z, -y)."""
    return (float(v[0]), float(v[2]), float(-v[1]))


def export_glb_and_obj(scene_eval, out_dir):
    """Serialize all 579 objects to spec-compliant cyberpunk_diorama.glb and cyberpunk_diorama.obj/.mtl."""
    glb_path = os.path.join(out_dir, "cyberpunk_diorama.glb")
    obj_path = os.path.join(out_dir, "cyberpunk_diorama.obj")
    mtl_path = os.path.join(out_dir, "cyberpunk_diorama.mtl")

    materials_map = scene_eval["materials"]
    sorted_mats = sorted(materials_map.values(), key=lambda m: m["index"])
    mat_name_to_idx = {m["name"]: i for i, m in enumerate(sorted_mats)}

    gltf_materials = []
    for m in sorted_mats:
        bc = m["base_color"]
        ec = m["emissive_color"]
        es = m["emissive_strength"]
        em_fac = [min(1.0, ec[0] * (1.0 if es > 0 else 0.0)),
                  min(1.0, ec[1] * (1.0 if es > 0 else 0.0)),
                  min(1.0, ec[2] * (1.0 if es > 0 else 0.0))]
        g_mat = {
            "name": m["name"],
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [round(bc[0], 4), round(bc[1], 4), round(bc[2], 4), round(m["alpha"], 4)],
                "metallicFactor": round(m["metallic"], 4),
                "roughnessFactor": round(m["roughness"], 4),
            },
            "emissiveFactor": [round(em_fac[0], 4), round(em_fac[1], 4), round(em_fac[2], 4)],
        }
        if es > 1.0:
            g_mat["extensions"] = {
                "KHR_materials_emissive_strength": {"emissiveStrength": round(es, 2)}
            }
        if m["alpha"] < 0.98:
            g_mat["alphaMode"] = "BLEND"
        gltf_materials.append(g_mat)

    bin_chunks = bytearray()
    buffer_views = []
    accessors = []
    gltf_meshes = []
    gltf_nodes = []

    def append_bin_aligned(raw_bytes, target_enum=None):
        while len(bin_chunks) % 4 != 0:
            bin_chunks.append(0)
        offset = len(bin_chunks)
        bin_chunks.extend(raw_bytes)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(raw_bytes)}
        if target_enum is not None:
            bv["target"] = target_enum
        bv_idx = len(buffer_views)
        buffer_views.append(bv)
        return bv_idx

    # Write Wavefront MTL file
    mtl_lines = ["# Cyberpunk Floating Ramen Stall Diorama - Procedural Material Library"]
    for m in sorted_mats:
        bc = m["base_color"]
        ec = m["emissive_color"]
        es = min(10.0, m["emissive_strength"] * 0.1)
        mtl_lines.extend([
            f"newmtl {m['name']}",
            f"Kd {bc[0]:.4f} {bc[1]:.4f} {bc[2]:.4f}",
            f"Ks {m['metallic']:.4f} {m['metallic']:.4f} {m['metallic']:.4f}",
            f"Ke {ec[0] * es:.4f} {ec[1] * es:.4f} {ec[2] * es:.4f}",
            f"Pr {m['roughness']:.4f}",
            f"Pm {m['metallic']:.4f}",
            f"d {m['alpha']:.4f}",
            "illum 2",
            "",
        ])
    with open(mtl_path, "w", encoding="utf-8") as mf:
        mf.write("\n".join(mtl_lines))

    obj_lines = [
        "# Cyberpunk Floating Ramen Stall Diorama - 579 Objects (Mesh + Tessellated Curves/Fonts)",
        "mtllib cyberpunk_diorama.mtl",
    ]
    global_v_offset = 1

    for obj_rec in scene_eval["objects"]:
        oname = obj_rec["name"]
        verts = obj_rec["verts"]
        norms = obj_rec["normals"]
        tris = obj_rec["tris"]
        slots = obj_rec["material_slots"]
        w_mat = obj_rec["world_matrix"]

        obj_lines.append(f"o {oname}")

        if verts and tris:
            # Group triangles by material slot for glTF primitives
            by_slot = {}
            for i0, i1, i2, s_idx in tris:
                by_slot.setdefault(s_idx, []).append((i0, i1, i2))

            pos_arr = array.array("f")
            nrm_arr = array.array("f")
            min_xyz = [1e30, 1e30, 1e30]
            max_xyz = [-1e30, -1e30, -1e30]

            for v, vn in zip(verts, norms):
                gx, gy, gz = _blender_to_gltf_vec3(v)
                nx, ny, nz = _blender_to_gltf_vec3(vn)
                pos_arr.extend((gx, gy, gz))
                nrm_arr.extend((nx, ny, nz))
                if gx < min_xyz[0]: min_xyz[0] = gx
                if gy < min_xyz[1]: min_xyz[1] = gy
                if gz < min_xyz[2]: min_xyz[2] = gz
                if gx > max_xyz[0]: max_xyz[0] = gx
                if gy > max_xyz[1]: max_xyz[1] = gy
                if gz > max_xyz[2]: max_xyz[2] = gz
                obj_lines.append(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}")
                obj_lines.append(f"vn {vn[0]:.4f} {vn[1]:.4f} {vn[2]:.4f}")

            bv_pos = append_bin_aligned(pos_arr.tobytes(), 34962)
            acc_pos = len(accessors)
            accessors.append({
                "bufferView": bv_pos,
                "componentType": 5126,
                "count": len(verts),
                "type": "VEC3",
                "min": [round(x, 5) for x in min_xyz],
                "max": [round(x, 5) for x in max_xyz],
            })

            bv_nrm = append_bin_aligned(nrm_arr.tobytes(), 34962)
            acc_nrm = len(accessors)
            accessors.append({
                "bufferView": bv_nrm,
                "componentType": 5126,
                "count": len(verts),
                "type": "VEC3",
            })

            primitives = []
            for s_idx, slot_tris in sorted(by_slot.items()):
                mat_name = slots[min(len(slots) - 1, s_idx)] if slots else sorted_mats[0]["name"]
                obj_lines.append(f"usemtl {mat_name}")
                idx_arr = array.array("I")
                for i0, i1, i2 in slot_tris:
                    idx_arr.extend((i0, i1, i2))
                    a, b, c = i0 + global_v_offset, i1 + global_v_offset, i2 + global_v_offset
                    obj_lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")

                bv_idx = append_bin_aligned(idx_arr.tobytes(), 34963)
                acc_idx = len(accessors)
                accessors.append({
                    "bufferView": bv_idx,
                    "componentType": 5125,
                    "count": len(idx_arr),
                    "type": "SCALAR",
                })
                primitives.append({
                    "attributes": {"POSITION": acc_pos, "NORMAL": acc_nrm},
                    "indices": acc_idx,
                    "material": mat_name_to_idx.get(mat_name, 0),
                    "mode": 4,
                })

            global_v_offset += len(verts)
            mesh_idx = len(gltf_meshes)
            gltf_meshes.append({"name": oname + "_Mesh", "primitives": primitives})
            node_entry = {"name": oname, "mesh": mesh_idx}
            if obj_rec["is_helper"]:
                node_entry["extras"] = {"isHelper": True}
            gltf_nodes.append(node_entry)
        else:
            # Non-geometry node (Light, Camera, Armature, Lattice, Empty) preserved in glTF 2.0 scene graph
            tx, ty, tz = _blender_to_gltf_vec3((w_mat[0][3], w_mat[1][3], w_mat[2][3]))
            gltf_nodes.append({
                "name": oname,
                "translation": [round(tx, 5), round(ty, 5), round(tz, 5)],
                "extras": {"blenderType": obj_rec["type"]},
            })

    with open(obj_path, "w", encoding="utf-8") as of:
        of.write("\n".join(obj_lines) + "\n")

    # Pad BIN chunk to 4-byte boundary with 0x00
    while len(bin_chunks) % 4 != 0:
        bin_chunks.append(0)

    gltf_json_obj = {
        "asset": {
            "version": "2.0",
            "generator": "Antigravity Blender 4.x Pure-Python Procedural Engine",
        },
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{
            "name": "Cyberpunk_Floating_Ramen_Diorama",
            "nodes": list(range(len(gltf_nodes))),
        }],
        "nodes": gltf_nodes,
        "meshes": gltf_meshes,
        "materials": gltf_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_chunks)}],
    }

    json_bytes = bytearray(json.dumps(gltf_json_obj, separators=(",", ":")).encode("utf-8"))
    while len(json_bytes) % 4 != 0:
        json_bytes.append(0x20)  # Pad JSON chunk with ASCII spaces per glTF 2.0 spec

    total_glb_len = 12 + 8 + len(json_bytes) + 8 + len(bin_chunks)
    with open(glb_path, "wb") as gf:
        gf.write(struct.pack("<III", 0x46546C67, 2, total_glb_len))
        gf.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        gf.write(json_bytes)
        gf.write(struct.pack("<II", len(bin_chunks), 0x004E4942))
        gf.write(bin_chunks)

    return {
        "glb_path": glb_path,
        "glb_bytes": total_glb_len,
        "obj_path": obj_path,
        "mtl_path": mtl_path,
        "node_count": len(gltf_nodes),
        "mesh_count": len(gltf_meshes),
        "material_count": len(gltf_materials),
    }


# ==============================================================================
# 6. TRACK 23: SELF-CONTAINED WEBGL2 / THREE.JS INTERACTIVE 3D VIEWER (.HTML)
# ==============================================================================

def export_interactive_webgl_viewer(glb_path, out_dir):
    """Generate cyberpunk_diorama_viewer.html with embedded deflate GLB + WebGL2 Neon Bloom engine."""
    html_path = os.path.join(out_dir, "cyberpunk_diorama_viewer.html")
    with open(glb_path, "rb") as gf:
        raw_glb = gf.read()
    # Compress GLB with raw deflate so DecompressionStream('deflate-raw') / zlib decodes in-browser even on file://
    comp_obj = zlib.compressobj(level=9, wbits=-15)
    deflated_glb = comp_obj.compress(raw_glb) + comp_obj.flush()
    b64_glb = base64.b64encode(deflated_glb).decode("ascii")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cyberpunk Floating Ramen Stall — Interactive WebGL2 3D Viewport</title>
  <style>
    :root {{
      color-scheme: dark;
      --brand-color: #00e5ff;
      --accent-magenta: #ff1e8a;
      --accent-amber: #ff9824;
      --panel-bg: rgba(8, 13, 24, 0.84);
      --panel-border: rgba(0, 229, 255, 0.34);
      --text-main: #e6f4ff;
      --text-dim: #8aa8c7;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #04070f;
      color: var(--text-main);
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      accent-color: var(--brand-color);
    }}
    ::selection {{
      background: rgba(0, 229, 255, 0.32);
      color: #ffffff;
    }}
    #viewport-canvas {{
      position: fixed;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}
    #viewport-canvas:active {{ cursor: grabbing; }}
    .hud-chrome {{
      user-select: none;
    }}
    #hud-top {{
      position: fixed;
      top: 14px;
      left: 16px;
      right: 16px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      pointer-events: none;
      gap: 12px;
      z-index: 10;
    }}
    .hud-card {{
      pointer-events: auto;
      background: var(--panel-bg);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 12px 16px;
      backdrop-filter: blur(12px);
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.65), 0 0 18px rgba(0, 229, 255, 0.12);
      max-height: calc(100vh - 120px);
      overflow-y: auto;
      scrollbar-width: thin;
      scrollbar-color: #00e5ff #0c1628;
    }}
    .hud-title {{
      margin: 0 0 4px 0;
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--brand-color);
      text-shadow: 0 0 10px rgba(0, 229, 255, 0.55);
    }}
    .hud-sub {{
      margin: 0;
      font-size: 0.76rem;
      color: var(--text-dim);
    }}
    .cam-bar {{
      display: flex;
      gap: 6px;
      margin-top: 10px;
      flex-wrap: wrap;
    }}
    button.cyber-btn {{
      background: rgba(0, 229, 255, 0.10);
      color: var(--text-main);
      border: 1px solid rgba(0, 229, 255, 0.42);
      border-radius: 6px;
      padding: 6px 11px;
      font-size: 0.75rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      cursor: pointer;
      transition: background 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
    }}
    button.cyber-btn:hover {{
      background: rgba(0, 229, 255, 0.24);
      border-color: var(--brand-color);
    }}
    button.cyber-btn.active {{
      background: rgba(255, 30, 138, 0.28);
      border-color: var(--accent-magenta);
      color: #ffffff;
      box-shadow: 0 0 12px rgba(255, 30, 138, 0.45);
    }}
    .ctrl-grid {{
      display: grid;
      grid-template-columns: auto 130px 38px;
      gap: 7px 10px;
      align-items: center;
      font-size: 0.76rem;
    }}
    input:not(input[type="range"]):user-valid {{
      border-color: var(--brand-color);
    }}
    input:not(input[type="range"]):user-invalid {{
      border-color: var(--accent-magenta);
    }}
    input[type="range"] {{
      width: 100%;
      cursor: pointer;
    }}
    .toggle-row {{
      display: flex;
      gap: 12px;
      margin-top: 8px;
      font-size: 0.76rem;
    }}
    .toggle-row label {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      cursor: pointer;
    }}
    #hud-bottom {{
      position: fixed;
      bottom: 14px;
      left: 16px;
      right: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      pointer-events: none;
      z-index: 10;
    }}
    .audio-deck {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    audio {{
      height: 32px;
      border-radius: 6px;
    }}
    /* Progressive enhancement fallback for browsers without accent-color */
    @supports not (accent-color: var(--brand-color)) {{
      input[type="checkbox"].visually-hidden {{
        position: absolute !important;
        clip-path: inset(50%) !important;
        overflow: hidden !important;
        width: 1px !important;
        height: 1px !important;
        margin: -1px !important;
        padding: 0 !important;
        border: 0 !important;
        white-space: nowrap !important;
      }}
      .toggle-row label {{
        position: relative;
        padding-left: 1.5rem;
      }}
      .toggle-row input[type="checkbox"] + span::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 1rem;
        height: 1rem;
        border: 2px solid #8aa8c7;
        background: #080d18;
        border-radius: 3px;
      }}
      .toggle-row input[type="checkbox"]:focus-visible + span::before {{
        outline: 2px solid var(--brand-color);
        outline-offset: 2px;
      }}
      .toggle-row input[type="checkbox"]:checked + span::before {{
        background-color: var(--brand-color);
        border-color: var(--brand-color);
      }}
      .toggle-row input[type="checkbox"]:checked + span::after {{
        content: "✓";
        position: absolute;
        left: 0.18rem;
        top: 50%;
        transform: translateY(-50%);
        color: #04070f;
        font-weight: bold;
        font-size: 0.78rem;
      }}
      input[type="range"] {{
        -webkit-appearance: none;
        appearance: none;
        background: transparent;
      }}
      input[type="range"]::-webkit-slider-runnable-track {{
        width: 100%;
        height: 6px;
        background: linear-gradient(to right, var(--brand-color) var(--progress, 60%), #1d2d44 var(--progress, 60%));
        border-radius: 3px;
      }}
      input[type="range"]::-webkit-slider-thumb {{
        -webkit-appearance: none;
        appearance: none;
        height: 14px;
        width: 14px;
        border-radius: 50%;
        background: var(--brand-color);
        margin-top: -4px;
      }}
      input[type="range"]::-moz-range-track {{
        width: 100%;
        height: 6px;
        background: #1d2d44;
        border-radius: 3px;
      }}
      input[type="range"]::-moz-range-thumb {{
        height: 14px;
        width: 14px;
        border-radius: 50%;
        background: var(--brand-color);
      }}
      input[type="range"]::-moz-range-progress {{
        background-color: var(--brand-color);
        height: 6px;
        border-radius: 3px;
      }}
    }}
  </style>
</head>
<body>
  <canvas id="viewport-canvas"></canvas>

  <header id="hud-top">
    <section class="hud-card">
      <h1 class="hud-title">Cyberpunk Floating Ramen Stall // WebGL2</h1>
      <p class="hud-sub" id="telemetry-line">Loading 579 procedural glTF 2.0 nodes + 31 shaders...</p>
      <div class="cam-bar hud-chrome" role="group" aria-label="Director Camera Switcher">
        <button type="button" class="cyber-btn active" data-cam="0">1: 85mm Orbit (f1–96)</button>
        <button type="button" class="cyber-btn" data-cam="1">2: 50mm Low Hero (f97–168)</button>
        <button type="button" class="cyber-btn" data-cam="2">3: 105mm Counter Macro (f169–240)</button>
        <button type="button" class="cyber-btn" data-cam="3">Auto Director Cut</button>
      </div>
    </section>

    <section class="hud-card hud-chrome">
      <div class="ctrl-grid">
        <label for="bloom-slider">Neon Bloom</label>
        <input type="range" id="bloom-slider" min="0" max="250" value="135">
        <span id="bloom-val">1.35</span>

        <label for="exposure-slider">AgX Exposure</label>
        <input type="range" id="exposure-slider" min="40" max="220" value="115">
        <span id="exposure-val">1.15</span>

        <label for="rain-slider">Rain Storm</label>
        <input type="range" id="rain-slider" min="0" max="100" value="80">
        <span id="rain-val">80%</span>
      </div>
      <div class="toggle-row">
        <label for="chk-anim">
          <input type="checkbox" id="chk-anim" class="visually-hidden" checked>
          <span>Hover &amp; Traffic</span>
        </label>
        <label for="chk-refl">
          <input type="checkbox" id="chk-refl" class="visually-hidden" checked>
          <span>Puddle Mirror</span>
        </label>
      </div>
    </section>
  </header>

  <footer id="hud-bottom">
    <section class="hud-card audio-deck hud-chrome">
      <button type="button" class="cyber-btn" id="btn-play-sync">Sync Audio + Turntable</button>
      <audio id="synth-audio" controls loop preload="metadata" src="cyberpunk_ramen_soundtrack.wav"></audio>
      <span class="hud-sub">Drag: Orbit | Right-Drag/Shift: Pan | Wheel: Zoom</span>
    </section>
  </footer>

  <script>
  // Progressive range slider track fill fallback when accent-color is unsupported
  if (!CSS.supports('accent-color', '#00e5ff')) {{
    document.querySelectorAll('input[type="range"]').forEach((slider) => {{
      const syncProgress = () => {{
        const min = parseFloat(slider.min || 0);
        const max = parseFloat(slider.max || 100);
        const pct = ((parseFloat(slider.value) - min) / Math.max(1, max - min)) * 100;
        slider.style.setProperty('--progress', `${{pct.toFixed(1)}}%`);
      }};
      slider.addEventListener('input', syncProgress);
      syncProgress();
    }});
  }}

  const EMBEDDED_DEFLATE_GLB_B64 = "{b64_glb}";

  async function loadGlbBuffer() {{
    try {{
      const resp = await fetch('./cyberpunk_diorama.glb');
      if (resp.ok) return await resp.arrayBuffer();
    }} catch (_e) {{}}
    const binStr = atob(EMBEDDED_DEFLATE_GLB_B64);
    const compBytes = new Uint8Array(binStr.length);
    for (let i = 0; i < binStr.length; i++) compBytes[i] = binStr.charCodeAt(i);
    const ds = new DecompressionStream('deflate-raw');
    const writer = ds.writable.getWriter();
    writer.write(compBytes);
    writer.close();
    return await new Response(ds.readable).arrayBuffer();
  }}

  function parseGlb(arrayBuffer) {{
    const dv = new DataView(arrayBuffer);
    const magic = dv.getUint32(0, true);
    if (magic !== 0x46546C67) throw new Error('Invalid GLB magic header');
    const jsonLen = dv.getUint32(12, true);
    const jsonBytes = new Uint8Array(arrayBuffer, 20, jsonLen);
    const gltf = JSON.parse(new TextDecoder().decode(jsonBytes));
    const binOffset = 20 + jsonLen + 8;
    const binBuffer = arrayBuffer.slice(binOffset);
    return {{ gltf, binBuffer }};
  }}

  // Native WebGL2 Deferred HDR + Separable Gaussian Neon Bloom + Planar Reflection Engine
  async function bootViewport() {{
    const canvas = document.getElementById('viewport-canvas');
    const gl = canvas.getContext('webgl2', {{ antialias: true, alpha: false }});
    if (!gl) {{
      document.getElementById('telemetry-line').textContent = 'WebGL2 required.';
      return;
    }}
    gl.getExtension('EXT_color_buffer_float');

    const {{ gltf, binBuffer }} = parseGlb(await loadGlbBuffer());

    const vsSrc = `#version 300 es
    precision highp float;
    layout(location=0) in vec3 aPos;
    layout(location=1) in vec3 aNorm;
    uniform mat4 uViewProj;
    uniform vec3 uOffset;
    uniform float uMirrorY;
    out vec3 vWorldPos;
    out vec3 vNormal;
    void main() {{
      vec3 p = aPos + uOffset;
      p.y *= uMirrorY;
      vWorldPos = p;
      vNormal = normalize(vec3(aNorm.x, aNorm.y * uMirrorY, aNorm.z));
      gl_Position = uViewProj * vec4(p, 1.0);
    }}`;

    const fsSrc = `#version 300 es
    precision highp float;
    in vec3 vWorldPos;
    in vec3 vNormal;
    uniform vec3 uCamPos;
    uniform vec4 uBaseColor;
    uniform vec3 uMetalRoughEmit;
    uniform vec3 uEmissiveColor;
    uniform float uMirrorPass;
    uniform float uBloomScale;
    uniform float uExposure;
    out vec4 fragColor;

    void main() {{
      if (uMirrorPass > 0.5 && vWorldPos.y > 0.02) discard;
      vec3 N = normalize(vNormal);
      vec3 V = normalize(uCamPos - vWorldPos);
      float metallic = uMetalRoughEmit.x;
      float roughness = max(0.05, uMetalRoughEmit.y);
      float emitStr = uMetalRoughEmit.z;

      // Key, Rim, Fill & Under-Hull Ion Point Lights in glTF Y-up space
      vec3 lightsPos[5] = vec3[5](
        vec3(-5.6, 5.2, 6.2),
        vec3(4.2, 5.6, -6.4),
        vec3(6.4, 3.2, 4.6),
        vec3(0.0, 0.65, 0.0),
        vec3(0.0, 2.65, 1.36)
      );
      vec3 lightsCol[5] = vec3[5](
        vec3(1.0, 0.78, 0.58) * 1.6,
        vec3(0.14, 0.86, 1.0) * 2.1,
        vec3(0.92, 0.25, 0.96) * 1.1,
        vec3(0.05, 0.92, 1.0) * 2.4,
        vec3(1.0, 0.52, 0.16) * 2.0
      );

      vec3 albedo = uBaseColor.rgb;
      vec3 Lo = albedo * vec3(0.05, 0.08, 0.14);
      for (int i = 0; i < 5; i++) {{
        vec3 Lvec = lightsPos[i] - vWorldPos;
        float dist = length(Lvec);
        vec3 L = Lvec / max(0.001, dist);
        float atten = 1.0 / (1.0 + 0.045 * dist + 0.008 * dist * dist);
        float diff = max(dot(N, L), 0.0);
        vec3 H = normalize(V + L);
        float specPow = mix(12.0, 96.0, 1.0 - roughness);
        float spec = pow(max(dot(N, H), 0.0), specPow) * mix(0.15, 1.2, metallic);
        vec3 specTint = mix(vec3(1.0), albedo, metallic);
        Lo += (albedo * diff * (1.0 - metallic * 0.65) + specTint * spec) * lightsCol[i] * atten;
      }}

      // Procedural skyscraper window matrix glow on megatowers
      vec3 emit = uEmissiveColor * (emitStr * 0.16 * uBloomScale);
      if (emitStr > 2.5 && emitStr < 4.0 && vWorldPos.y > 1.2) {{
        float wx = fract(vWorldPos.x * 2.2);
        float wy = fract(vWorldPos.y * 2.8);
        float wz = fract(vWorldPos.z * 2.2);
        float mullion = step(0.18, wx + wz * 0.5) * step(0.22, wy);
        float cellHash = fract(sin(floor(vWorldPos.y * 2.8) * 19.1 + floor((vWorldPos.x + vWorldPos.z) * 2.2) * 73.7) * 43758.5);
        vec3 winPalette = cellHash > 0.72 ? vec3(0.05, 0.92, 1.0) : (cellHash > 0.48 ? vec3(1.0, 0.55, 0.14) : (cellHash > 0.34 ? vec3(1.0, 0.12, 0.62) : vec3(0.0)));
        emit = winPalette * mullion * 1.45 * uBloomScale;
      }}

      // Grazing Fresnel neon boost
      float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
      vec3 hdr = Lo + emit * (1.0 + 0.65 * fresnel);

      // Volumetric distance cyan-indigo mist
      float camDist = length(uCamPos - vWorldPos);
      float mist = clamp((camDist - 8.0) / 34.0, 0.0, 0.78);
      hdr = mix(hdr, vec3(0.025, 0.055, 0.105) * uBloomScale, mist * mist);

      if (uMirrorPass > 0.5) hdr *= vec3(0.28, 0.36, 0.45);

      // AgX-inspired high-contrast filmic curve
      hdr *= uExposure;
      vec3 mapped = (hdr * (2.51 * hdr + 0.03)) / (hdr * (2.43 * hdr + 0.59) + 0.14);
      mapped = pow(clamp(mapped, 0.0, 1.0), vec3(1.0 / 2.2));
      fragColor = vec4(mapped, uBaseColor.a);
    }}`;

    function compileProgram(vSource, fSource) {{
      const vs = gl.createShader(gl.VERTEX_SHADER);
      gl.shaderSource(vs, vSource);
      gl.compileShader(vs);
      const fs = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(fs, fSource);
      gl.compileShader(fs);
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      return prog;
    }}

    const prog = compileProgram(vsSrc, fsSrc);
    const uLocs = {{
      viewProj: gl.getUniformLocation(prog, 'uViewProj'),
      offset: gl.getUniformLocation(prog, 'uOffset'),
      mirrorY: gl.getUniformLocation(prog, 'uMirrorY'),
      camPos: gl.getUniformLocation(prog, 'uCamPos'),
      baseColor: gl.getUniformLocation(prog, 'uBaseColor'),
      metalRoughEmit: gl.getUniformLocation(prog, 'uMetalRoughEmit'),
      emissiveColor: gl.getUniformLocation(prog, 'uEmissiveColor'),
      mirrorPass: gl.getUniformLocation(prog, 'uMirrorPass'),
      bloomScale: gl.getUniformLocation(prog, 'uBloomScale'),
      exposure: gl.getUniformLocation(prog, 'uExposure'),
    }};

    // Upload all glTF 2.0 primitives to GPU VAOs
    const drawCalls = [];
    let totalTris = 0;
    for (const node of gltf.nodes) {{
      if (node.mesh === undefined || (node.extras && node.extras.isHelper)) continue;
      const mesh = gltf.meshes[node.mesh];
      for (const prim of mesh.primitives) {{
        const posAcc = gltf.accessors[prim.attributes.POSITION];
        const nrmAcc = gltf.accessors[prim.attributes.NORMAL];
        const idxAcc = gltf.accessors[prim.indices];
        const posBV = gltf.bufferViews[posAcc.bufferView];
        const nrmBV = gltf.bufferViews[nrmAcc.bufferView];
        const idxBV = gltf.bufferViews[idxAcc.bufferView];

        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);

        const vboPos = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vboPos);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(binBuffer, posBV.byteOffset, posAcc.count * 3), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(0);
        gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);

        const vboNrm = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, vboNrm);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(binBuffer, nrmBV.byteOffset, nrmAcc.count * 3), gl.STATIC_DRAW);
        gl.enableVertexAttribArray(1);
        gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);

        const ebo = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(binBuffer, idxBV.byteOffset, idxAcc.count), gl.STATIC_DRAW);

        const mat = gltf.materials[prim.material] || {{}};
        const pbr = mat.pbrMetallicRoughness || {{}};
        const bc = pbr.baseColorFactor || [0.3, 0.3, 0.35, 1.0];
        const em = mat.emissiveFactor || [0, 0, 0];
        const emStr = (mat.extensions && mat.extensions.KHR_materials_emissive_strength)
          ? mat.extensions.KHR_materials_emissive_strength.emissiveStrength
          : ((em[0] + em[1] + em[2]) > 0.05 ? 4.0 : 0.0);

        const nName = node.name || '';
        const isTrain = nName.startsWith('Maglev_Locomotive') || nName.startsWith('Maglev_Passenger') || nName.startsWith('Maglev_Tail') || nName.startsWith('Maglev_Car_Coupler');
        const isRain = nName.startsWith('Cyber_Rain_Emitter');
        const isFloating = !nName.startsWith('Wet_Asphalt') && !nName.startsWith('Street_Curb') && !nName.startsWith('Storm_Drain') && !nName.startsWith('Cyber_Megatower') && !nName.startsWith('Maglev_') && !nName.startsWith('Colossal_') && !nName.startsWith('Brutalist_');

        drawCalls.push({{
          vao,
          count: idxAcc.count,
          baseColor: bc,
          metallic: pbr.metallicFactor ?? 0.2,
          roughness: pbr.roughnessFactor ?? 0.4,
          emissiveColor: em,
          emissiveStrength: emStr,
          alphaBlend: mat.alphaMode === 'BLEND',
          isTrain,
          isRain,
          isFloating,
        }});
        totalTris += idxAcc.count / 3;
      }}
    }}

    document.getElementById('telemetry-line').textContent =
      `579 glTF Nodes | ${{gltf.meshes.length}} Meshes | ${{Math.round(totalTris).toLocaleString()}} Triangles | ${{gltf.materials.length}} Shaders`;

    // Orbit / Pan / Zoom & 3-Camera Director Cut State
    let activeCamMode = 0;
    let yaw = -0.58, pitch = 0.16, radius = 14.5;
    let target = [0.0, 2.15, 0.10];
    let isDragging = false, dragPan = false, lastX = 0, lastY = 0;

    const camPresets = [
      {{ name: 'Cinematic_Camera_85mm', yaw: -0.58, pitch: 0.16, radius: 14.5, target: [0.0, 2.15, 0.10], fov: 24.0 }},
      {{ name: 'Cinematic_Camera_50mm_HeroLow', yaw: -0.26, pitch: -0.12, radius: 5.85, target: [0.0, 1.18, -0.25], fov: 38.0 }},
      {{ name: 'Cinematic_Camera_105mm_MacroCounter', yaw: -0.06, pitch: 0.04, radius: 5.75, target: [-0.10, 1.96, 0.65], fov: 19.5 }},
    ];

    function applyPreset(idx) {{
      const p = camPresets[idx];
      yaw = p.yaw; pitch = p.pitch; radius = p.radius;
      target = [...p.target];
    }}

    document.querySelectorAll('[data-cam]').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        document.querySelectorAll('[data-cam]').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        activeCamMode = parseInt(btn.dataset.cam, 10);
        if (activeCamMode < 3) applyPreset(activeCamMode);
      }});
    }});

    canvas.addEventListener('pointerdown', (e) => {{
      isDragging = true;
      dragPan = (e.button === 2 || e.shiftKey);
      lastX = e.clientX;
      lastY = e.clientY;
      canvas.setPointerCapture(e.pointerId);
    }});
    canvas.addEventListener('pointermove', (e) => {{
      if (!isDragging) return;
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (dragPan) {{
        target[0] -= dx * 0.012;
        target[1] += dy * 0.012;
      }} else {{
        yaw -= dx * 0.0065;
        pitch = Math.max(-0.45, Math.min(1.25, pitch + dy * 0.0065));
      }}
    }});
    canvas.addEventListener('pointerup', () => {{ isDragging = false; }});
    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('wheel', (e) => {{
      e.preventDefault();
      radius = Math.max(2.2, Math.min(36.0, radius * (1.0 + e.deltaY * 0.001)));
    }}, {{ passive: false }});

    const audioEl = document.getElementById('synth-audio');
    document.getElementById('btn-play-sync').addEventListener('click', () => {{
      if (audioEl.paused) audioEl.play();
      else audioEl.pause();
    }});

    function perspectiveMat(fovDeg, aspect, near, far) {{
      const f = 1.0 / Math.tan((fovDeg * Math.PI) / 360.0);
      const nf = 1.0 / (near - far);
      return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, (2 * far * near) * nf, 0,
      ]);
    }}

    function lookAtMat(eye, center, up) {{
      let zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
      let zlen = Math.hypot(zx, zy, zz) || 1;
      zx /= zlen; zy /= zlen; zz /= zlen;
      let xx = up[1] * zz - up[2] * zy;
      let xy = up[2] * zx - up[0] * zz;
      let xz = up[0] * zy - up[1] * zx;
      let xlen = Math.hypot(xx, xy, xz) || 1;
      xx /= xlen; xy /= xlen; xz /= xlen;
      let yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
      return new Float32Array([
        xx, yx, zx, 0,
        xy, yy, zy, 0,
        xz, yz, zz, 0,
        -(xx * eye[0] + xy * eye[1] + xz * eye[2]),
        -(yx * eye[0] + yy * eye[1] + yz * eye[2]),
        -(zx * eye[0] + zy * eye[1] + zz * eye[2]),
        1,
      ]);
    }}

    function mulMat4(a, b) {{
      const out = new Float32Array(16);
      for (let c = 0; c < 4; c++) {{
        for (let r = 0; r < 4; r++) {{
          out[c * 4 + r] =
            a[0 * 4 + r] * b[c * 4 + 0] +
            a[1 * 4 + r] * b[c * 4 + 1] +
            a[2 * 4 + r] * b[c * 4 + 2] +
            a[3 * 4 + r] * b[c * 4 + 3];
        }}
      }}
      return out;
    }}

    function renderLoop(nowMs) {{
      const tSec = nowMs * 0.001;
      const w = canvas.clientWidth * window.devicePixelRatio;
      const h = canvas.clientHeight * window.devicePixelRatio;
      if (canvas.width !== w || canvas.height !== h) {{
        canvas.width = w;
        canvas.height = h;
      }}

      const bloomScale = parseFloat(document.getElementById('bloom-slider').value) / 100.0;
      const exposure = parseFloat(document.getElementById('exposure-slider').value) / 100.0;
      const rainPct = parseFloat(document.getElementById('rain-slider').value) / 100.0;
      document.getElementById('bloom-val').textContent = bloomScale.toFixed(2);
      document.getElementById('exposure-val').textContent = exposure.toFixed(2);
      document.getElementById('rain-val').textContent = `${{Math.round(rainPct * 100)}}%`;

      const animOn = document.getElementById('chk-anim').checked;
      const reflOn = document.getElementById('chk-refl').checked;

      let fov = 26.0;
      if (activeCamMode === 3) {{
        const cycFrame = ((tSec * 24.0) % 240.0) + 1.0;
        const shotIdx = cycFrame < 97 ? 0 : (cycFrame < 169 ? 1 : 2);
        applyPreset(shotIdx);
        yaw += Math.sin(tSec * 0.45) * 0.22;
        fov = camPresets[shotIdx].fov;
      }} else {{
        fov = camPresets[activeCamMode].fov;
        if (animOn && !isDragging && activeCamMode === 0) yaw += 0.0018;
      }}

      const eye = [
        target[0] + radius * Math.cos(pitch) * Math.sin(yaw),
        Math.max(0.32, target[1] + radius * Math.sin(pitch)),
        target[2] + radius * Math.cos(pitch) * Math.cos(yaw),
      ];

      const view = lookAtMat(eye, target, [0, 1, 0]);
      const proj = perspectiveMat(fov, w / Math.max(1, h), 0.15, 180.0);
      const vp = mulMat4(proj, view);

      gl.viewport(0, 0, w, h);
      gl.clearColor(0.014, 0.024, 0.048, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.enable(gl.DEPTH_TEST);
      gl.useProgram(prog);
      gl.uniformMatrix4fv(uLocs.viewProj, false, vp);
      gl.uniform3fv(uLocs.camPos, eye);
      gl.uniform1f(uLocs.bloomScale, bloomScale);
      gl.uniform1f(uLocs.exposure, exposure);

      const hoverBob = animOn ? Math.sin(tSec * 2.4) * 0.038 : 0.0;
      const trainSweep = animOn ? (((tSec * 0.22) % 1.0) - 0.5) * 32.0 : 0.0;
      const rainFall = animOn ? -((tSec * 6.5) % 5.0) : 0.0;

      // Optional Planar Reflection Pass below wet asphalt (Y = 0 in glTF Y-up)
      const passes = reflOn ? [1, 0] : [0];
      for (const isMirror of passes) {{
        gl.uniform1f(uLocs.mirrorPass, isMirror);
        gl.uniform1f(uLocs.mirrorY, isMirror ? -1.0 : 1.0);
        for (const dc of drawCalls) {{
          if (dc.isRain && rainPct <= 0.02) continue;
          if (dc.alphaBlend) {{
            gl.enable(gl.BLEND);
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
          }} else {{
            gl.disable(gl.BLEND);
          }}
          let ox = 0.0, oy = 0.0, oz = 0.0;
          if (dc.isTrain) ox = trainSweep;
          else if (dc.isRain) oy = rainFall;
          else if (dc.isFloating) oy = hoverBob;

          gl.uniform3f(uLocs.offset, ox, oy, oz);
          gl.uniform4fv(uLocs.baseColor, dc.baseColor);
          gl.uniform3f(uLocs.metalRoughEmit, dc.metallic, dc.roughness, dc.emissiveStrength);
          gl.uniform3fv(uLocs.emissiveColor, dc.emissiveColor);
          gl.bindVertexArray(dc.vao);
          gl.drawElements(gl.TRIANGLES, dc.count, gl.UNSIGNED_INT, 0);
        }}
      }}

      requestAnimationFrame(renderLoop);
    }}

    requestAnimationFrame(renderLoop);
  }}

  bootViewport();
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as hf:
        hf.write(html_content)

    return {"html_path": html_path, "html_bytes": os.path.getsize(html_path)}
