import array
import math
import os
import struct
import zlib


# ==============================================================================
# TRACK 24: PURE-PYTHON 3D PERSPECTIVE RASTERIZER, PLANAR PUDDLE REFLECTION,
#           14-LIGHT PBR/PHONG SHADING & MULTI-PASS NEON BLOOM PNG ENGINE
# ==============================================================================

def _v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _v_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _v_norm(a):
    l = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if l < 1e-12:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / l
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def _look_at_camera_basis(eye, target, up=(0.0, 0.0, 1.0)):
    """Return orthonormal camera basis (right_x, true_up_y, forward) for world->camera projection."""
    forward = _v_norm(_v_sub(target, eye))
    right_x = _v_norm(_v_cross(forward, up))
    true_up_y = _v_norm(_v_cross(right_x, forward))
    return right_x, true_up_y, forward


def _extract_scene_lights(scene_eval):
    """Collect all 14 scene LIGHT objects with world positions, RGB colors, and calibrated night falloff."""
    lights = []
    for obj in scene_eval["objects"]:
        if obj["type"] == "LIGHT" and obj["data"] is not None:
            w_mat = obj["world_matrix"]
            lpos = (w_mat[0][3], w_mat[1][3], w_mat[2][3])
            ldata = obj["data"]
            col = tuple(getattr(ldata, "color", (1.0, 1.0, 1.0)))
            energy = float(getattr(ldata, "energy", 50.0))
            ltype = getattr(ldata, "type", "POINT")
            # Calibrate light range so night skyline stays moody while foreground stall & bowls pop
            if ltype == "AREA":
                intensity = min(1.35, math.sqrt(max(1.0, energy)) * 0.032)
                radius_falloff = 0.065
            elif ltype == "SPOT":
                intensity = min(1.55, math.sqrt(max(1.0, energy)) * 0.042)
                radius_falloff = 0.095
            else:
                intensity = min(1.25, math.sqrt(max(1.0, energy)) * 0.075)
                radius_falloff = 0.38
            lights.append({
                "name": obj["name"],
                "pos": lpos,
                "color": (float(col[0]), float(col[1]), float(col[2])),
                "intensity": intensity,
                "falloff": radius_falloff,
            })
    return lights


def _puddle_mask(wx, wy):
    """Procedural wet-asphalt puddle reflection mask in [0.22, 0.78]."""
    w1 = math.sin(wx * 1.45 + math.cos(wy * 1.65))
    w2 = math.cos(wx * 2.85 - wy * 2.25)
    raw = 0.5 + 0.32 * w1 + 0.18 * w2
    if raw < 0.38:
        return 0.22
    return min(0.78, 0.22 + (raw - 0.38) * 1.05)


def _building_window_emission(wx, wy, wz):
    """Procedural parallax apartment/office window matrix for MAT_Building_Night_Windows."""
    if wz < 0.85:
        return (0.0, 0.0, 0.0)
    u_coord = wx + wy * 0.73
    cell_u = int(math.floor(abs(u_coord) * 2.6))
    cell_z = int(math.floor(abs(wz) * 2.1))
    frac_u = (abs(u_coord) * 2.6) - cell_u
    frac_z = (abs(wz) * 2.1) - cell_z
    if frac_u < 0.18 or frac_u > 0.84 or frac_z < 0.20 or frac_z > 0.82:
        return (0.0, 0.0, 0.0)
    h = ((cell_u * 73856093) ^ (cell_z * 19349663)) & 0xFFFFFFFF
    prob = (h & 0xFF) / 255.0
    if prob < 0.46:
        return (0.0, 0.0, 0.0)
    palette_sel = (h >> 8) & 0x07
    brightness = 0.95 + ((h >> 12) & 0xFF) / 255.0 * 1.65
    if palette_sel in (0, 1, 2):
        return (0.06 * brightness, 0.88 * brightness, 1.18 * brightness)  # Cyber Cyan
    if palette_sel in (3, 4, 5):
        return (1.18 * brightness, 0.62 * brightness, 0.14 * brightness)  # Warm Sodium Amber
    return (1.18 * brightness, 0.14 * brightness, 0.68 * brightness)      # Neon Magenta


def _shade_triangle_color_and_emission(v0, v1, v2, fn, mat, obj_name, lights, eye):
    """Compute HDR RGB surface color and bright-pass emission RGB for a triangle."""
    cx = (v0[0] + v1[0] + v2[0]) * (1.0 / 3.0)
    cy = (v0[1] + v1[1] + v2[1]) * (1.0 / 3.0)
    cz = (v0[2] + v1[2] + v2[2]) * (1.0 / 3.0)

    vx = eye[0] - cx
    vy = eye[1] - cy
    vz = eye[2] - cz
    vlen = math.sqrt(vx * vx + vy * vy + vz * vz)
    if vlen > 1e-9:
        inv_v = 1.0 / vlen
        vx, vy, vz = vx * inv_v, vy * inv_v, vz * inv_v
    else:
        vx, vy, vz = 0.0, -1.0, 0.0

    nx, ny, nz = fn
    ndotv = nx * vx + ny * vy + nz * vz
    if ndotv < 0.0:
        nx, ny, nz = -nx, -ny, -nz
        ndotv = -ndotv

    bc = list(mat["base_color"])
    roughness = mat["roughness"]
    metallic = mat["metallic"]
    ec = mat["emissive_color"]
    es = mat["emissive_strength"]
    mname = mat["name"]

    # Curate surface albedo contrast so dark hulls, megatowers, and cyber-chef stand out clearly
    if "Building_Night_Windows" in mname or obj_name.startswith("Cyber_Megatower") or obj_name.startswith("Brutalist_"):
        bc = [0.025, 0.032, 0.048, 1.0]
        roughness = 0.55
        metallic = 0.35
    elif "Brushed_Metal" in mname:
        if obj_name.startswith("Chef_"):
            bc = [0.36, 0.42, 0.50, 1.0]
            metallic = 0.88
            roughness = 0.18
        elif "Partition" in obj_name or "Boiler" in obj_name or "Hood" in obj_name:
            bc = [0.055, 0.062, 0.078, 1.0]
            metallic = 0.72
            roughness = 0.38
        else:
            bc = [0.11, 0.13, 0.16, 1.0]
            metallic = 0.78
            roughness = 0.28
    elif "Hinoki_Wood" in mname:
        bc = [0.46, 0.28, 0.14, 1.0]
        roughness = 0.36
        metallic = 0.04
    elif "Noren_Fabric" in mname:
        bc = [0.12, 0.16, 0.28, 1.0]
        roughness = 0.65

    radial_r = math.sqrt(cx * cx + cy * cy)
    ao = min(1.0, max(0.28, 0.42 + 0.15 * min(3.0, max(0.0, cz)) + 0.04 * min(4.0, radial_r)))

    hemi = 0.5 + 0.5 * nz
    amb_r = (0.015 * (1.0 - hemi) + 0.036 * hemi) * ao
    amb_g = (0.020 * (1.0 - hemi) + 0.054 * hemi) * ao
    amb_b = (0.034 * (1.0 - hemi) + 0.095 * hemi) * ao

    diff_r, diff_g, diff_b = amb_r, amb_g, amb_b
    spec_r, spec_g, spec_b = 0.0, 0.0, 0.0

    shininess = max(6.0, min(96.0, (2.0 / max(0.05, roughness * roughness)) - 2.0))
    spec_scale = (0.12 + 0.85 * metallic + 0.35 * (1.0 - roughness)) * ao

    # Distant background megastructures receive attenuated local stall lighting
    bg_atten = 0.18 if cy > 3.8 else 1.0

    for lgt in lights:
        lx, ly, lz = lgt["pos"]
        dx = lx - cx
        dy = ly - cy
        dz = lz - cz
        dist_sq = dx * dx + dy * dy + dz * dz
        dist = math.sqrt(dist_sq) + 1e-6
        inv_d = 1.0 / dist
        ldx, ldy, ldz = dx * inv_d, dy * inv_d, dz * inv_d

        ndotl = nx * ldx + ny * ldy + nz * ldz
        if ndotl > 0.0:
            att = (lgt["intensity"] * bg_atten) / (1.0 + lgt["falloff"] * dist_sq)
            lcr, lcg, lcb = lgt["color"]
            contrib = ndotl * att
            diff_r += lcr * contrib
            diff_g += lcg * contrib
            diff_b += lcb * contrib

            hx = ldx + vx
            hy = ldy + vy
            hz = ldz + vz
            hlen = math.sqrt(hx * hx + hy * hy + hz * hz)
            if hlen > 1e-9:
                ndoth = max(0.0, (nx * hx + ny * hy + nz * hz) / hlen)
                if ndoth > 0.0:
                    sp = (ndoth ** shininess) * att * spec_scale
                    spec_r += lcr * sp
                    spec_g += lcg * sp
                    spec_b += lcb * sp

    # Add edge rim-light on metallic & character silhouettes
    fresnel = (1.0 - max(0.0, ndotv)) ** 2.4
    rim_boost = fresnel * (0.12 + 0.35 * metallic) * bg_atten
    spec_r += 0.08 * rim_boost
    spec_g += 0.42 * rim_boost
    spec_b += 0.65 * rim_boost

    kd = (1.0 - 0.68 * metallic)
    col_r = bc[0] * diff_r * kd + (spec_r * (1.0 - metallic + metallic * bc[0]))
    col_g = bc[1] * diff_g * kd + (spec_g * (1.0 - metallic + metallic * bc[1]))
    col_b = bc[2] * diff_b * kd + (spec_b * (1.0 - metallic + metallic * bc[2]))

    em_r, em_g, em_b = 0.0, 0.0, 0.0
    if "Building_Night_Windows" in mname:
        wr, wg, wb = _building_window_emission(cx, cy, cz)
        col_r += wr
        col_g += wg
        col_b += wb
        em_r += wr * 0.65
        em_g += wg * 0.65
        em_b += wb * 0.65
    elif "Wet_Asphalt" in mname:
        pm = _puddle_mask(cx, cy)
        col_r = col_r * (1.0 - 0.45 * pm) + spec_r * (0.75 + 1.35 * pm)
        col_g = col_g * (1.0 - 0.45 * pm) + spec_g * (0.75 + 1.35 * pm)
        col_b = col_b * (1.0 - 0.45 * pm) + spec_b * (0.75 + 1.35 * pm)
    elif es > 0.15:
        gain = min(3.8, 0.50 + math.log1p(es) * 0.68)
        core_boost = 0.38 * max(0.0, ndotv)
        er = ec[0] * gain + core_boost + 0.28 * fresnel * ec[0]
        eg = ec[1] * gain + core_boost + 0.28 * fresnel * ec[1]
        eb = ec[2] * gain + core_boost + 0.28 * fresnel * ec[2]
        col_r = col_r * 0.20 + er
        col_g = col_g * 0.20 + eg
        col_b = col_b * 0.20 + eb
        if es >= 1.2:
            bloom_scale = min(2.2, 0.28 + math.log1p(es) * 0.36)
            em_r = ec[0] * bloom_scale
            em_g = ec[1] * bloom_scale
            em_b = ec[2] * bloom_scale

    return (col_r, col_g, col_b), (em_r, em_g, em_b)


def _rasterize_triangle_scanline(
    sx0, sy0, z0,
    sx1, sy1, z1,
    sx2, sy2, z2,
    cr, cg, cb,
    er, eg, eb,
    alpha,
    mark_asphalt,
    only_on_asphalt,
    width, height,
    zbuf, buf_r, buf_g, buf_b, emit_r, emit_g, emit_b, asphalt_mask,
):
    """Fast edge-walking scanline triangle rasterizer with Z-buffer and optional alpha/reflection."""
    if sy0 > sy1:
        sx0, sy0, z0, sx1, sy1, z1 = sx1, sy1, z1, sx0, sy0, z0
    if sy1 > sy2:
        sx1, sy1, z1, sx2, sy2, z2 = sx2, sy2, z2, sx1, sy1, z1
    if sy0 > sy1:
        sx0, sy0, z0, sx1, sy1, z1 = sx1, sy1, z1, sx0, sy0, z0

    y_start = max(0, int(math.ceil(sy0)))
    y_end = min(height - 1, int(math.floor(sy2)))
    if y_start > y_end:
        return

    dy02 = sy2 - sy0
    if dy02 < 1e-6:
        return
    inv_dy02 = 1.0 / dy02
    dy01 = sy1 - sy0
    inv_dy01 = 1.0 / dy01 if dy01 > 1e-6 else 0.0
    dy12 = sy2 - sy1
    inv_dy12 = 1.0 / dy12 if dy12 > 1e-6 else 0.0

    is_opaque = (alpha >= 0.96) and not only_on_asphalt
    one_minus_a = 1.0 - alpha

    for y in range(y_start, y_end + 1):
        fy = float(y)
        t02 = (fy - sy0) * inv_dy02
        xa = sx0 + (sx2 - sx0) * t02
        za = z0 + (z2 - z0) * t02

        if fy < sy1:
            if dy01 > 1e-6:
                t01 = (fy - sy0) * inv_dy01
                xb = sx0 + (sx1 - sx0) * t01
                zb = z0 + (z1 - z0) * t01
            else:
                xb, zb = sx1, z1
        else:
            if dy12 > 1e-6:
                t12 = (fy - sy1) * inv_dy12
                xb = sx1 + (sx2 - sx1) * t12
                zb = z1 + (z2 - z1) * t12
            else:
                xb, zb = sx2, z2

        if xa > xb:
            xa, xb = xb, xa
            za, zb = zb, za

        x_start = max(0, int(math.ceil(xa)))
        x_end = min(width - 1, int(math.floor(xb)))
        if x_start > x_end:
            continue

        span = xb - xa
        dz_dx = (zb - za) / span if span > 1e-6 else 0.0
        z_cur = za + (float(x_start) - xa) * dz_dx
        row_offset = y * width

        if is_opaque:
            for x in range(x_start, x_end + 1):
                idx = row_offset + x
                if z_cur < zbuf[idx]:
                    zbuf[idx] = z_cur
                    buf_r[idx] = cr
                    buf_g[idx] = cg
                    buf_b[idx] = cb
                    emit_r[idx] = er
                    emit_g[idx] = eg
                    emit_b[idx] = eb
                    asphalt_mask[idx] = mark_asphalt
                z_cur += dz_dx
        elif only_on_asphalt:
            for x in range(x_start, x_end + 1):
                idx = row_offset + x
                if asphalt_mask[idx] == 1 and z_cur < zbuf[idx]:
                    zbuf[idx] = z_cur
                    buf_r[idx] = buf_r[idx] * one_minus_a + cr * alpha
                    buf_g[idx] = buf_g[idx] * one_minus_a + cg * alpha
                    buf_b[idx] = buf_b[idx] * one_minus_a + cb * alpha
                    emit_r[idx] += er * alpha
                    emit_g[idx] += eg * alpha
                    emit_b[idx] += eb * alpha
                z_cur += dz_dx
        else:
            for x in range(x_start, x_end + 1):
                idx = row_offset + x
                if z_cur < zbuf[idx]:
                    buf_r[idx] = buf_r[idx] * one_minus_a + cr * alpha
                    buf_g[idx] = buf_g[idx] * one_minus_a + cg * alpha
                    buf_b[idx] = buf_b[idx] * one_minus_a + cb * alpha
                    emit_r[idx] += er * alpha
                    emit_g[idx] += eg * alpha
                    emit_b[idx] += eb * alpha
                z_cur += dz_dx


def _box_blur_1d_horizontal(src, dst, w, h, radius):
    """O(W*H) sliding-window 1D horizontal box blur."""
    inv = 1.0 / float(2 * radius + 1)
    for y in range(h):
        row = y * w
        acc = src[row] * float(radius + 1)
        for rx in range(1, radius + 1):
            acc += src[row + min(w - 1, rx)]
        for x in range(w):
            dst[row + x] = acc * inv
            left_x = max(0, x - radius)
            right_x = min(w - 1, x + radius + 1)
            acc += src[row + right_x] - src[row + left_x]


def _box_blur_1d_vertical(src, dst, w, h, radius):
    """O(W*H) sliding-window 1D vertical box blur."""
    inv = 1.0 / float(2 * radius + 1)
    for x in range(w):
        acc = src[x] * float(radius + 1)
        for ry in range(1, radius + 1):
            acc += src[min(h - 1, ry) * w + x]
        for y in range(h):
            dst[y * w + x] = acc * inv
            top_y = max(0, y - radius)
            bot_y = min(h - 1, y + radius + 1)
            acc += src[bot_y * w + x] - src[top_y * w + x]


def _apply_neon_bloom_and_streaks(buf_r, buf_g, buf_b, emit_r, emit_g, emit_b, width, height):
    """Downsample 4x, run 3-pass Gaussian box blur + wide anamorphic horizontal streak pass, and upsample."""
    dw = width // 4
    dh = height // 4
    n_low = dw * dh

    lr = array.array("f", [0.0]) * n_low
    lg = array.array("f", [0.0]) * n_low
    lb = array.array("f", [0.0]) * n_low
    tr = array.array("f", [0.0]) * n_low
    tg = array.array("f", [0.0]) * n_low
    tb = array.array("f", [0.0]) * n_low

    for dy in range(dh):
        sy_base = dy * 4
        d_row = dy * dw
        for dx in range(dw):
            sx_base = dx * 4
            sr, sg, sb = 0.0, 0.0, 0.0
            for oy in range(4):
                s_row = (sy_base + oy) * width + sx_base
                for ox in range(4):
                    idx = s_row + ox
                    er = emit_r[idx]
                    eg = emit_g[idx]
                    eb = emit_b[idx]
                    hr = max(0.0, buf_r[idx] - 1.25) * 0.35
                    hg = max(0.0, buf_g[idx] - 1.25) * 0.35
                    hb = max(0.0, buf_b[idx] - 1.25) * 0.35
                    sr += er + hr
                    sg += eg + hg
                    sb += eb + hb
            out_idx = d_row + dx
            lr[out_idx] = sr * 0.0625
            lg[out_idx] = sg * 0.0625
            lb[out_idx] = sb * 0.0625

    str_r = array.array("f", [0.0]) * n_low
    str_g = array.array("f", [0.0]) * n_low
    str_b = array.array("f", [0.0]) * n_low
    _box_blur_1d_horizontal(lr, tr, dw, dh, 20)
    _box_blur_1d_horizontal(tr, str_r, dw, dh, 20)
    _box_blur_1d_horizontal(lg, tg, dw, dh, 20)
    _box_blur_1d_horizontal(tg, str_g, dw, dh, 20)
    _box_blur_1d_horizontal(lb, tb, dw, dh, 20)
    _box_blur_1d_horizontal(tb, str_b, dw, dh, 20)

    for rad in (4, 7, 10):
        _box_blur_1d_horizontal(lr, tr, dw, dh, rad)
        _box_blur_1d_vertical(tr, lr, dw, dh, rad)
        _box_blur_1d_horizontal(lg, tg, dw, dh, rad)
        _box_blur_1d_vertical(tg, lg, dw, dh, rad)
        _box_blur_1d_horizontal(lb, tb, dw, dh, rad)
        _box_blur_1d_vertical(tb, lb, dw, dh, rad)

    for i in range(n_low):
        lr[i] = lr[i] * 1.15 + str_r[i] * 0.48
        lg[i] = lg[i] * 1.15 + str_g[i] * 0.54
        lb[i] = lb[i] * 1.15 + str_b[i] * 0.62

    for y in range(height):
        gy = (y + 0.5) * 0.25 - 0.5
        y0 = max(0, min(dh - 1, int(math.floor(gy))))
        y1 = min(dh - 1, y0 + 1)
        wy = max(0.0, min(1.0, gy - y0))
        one_wy = 1.0 - wy
        r0_off = y0 * dw
        r1_off = y1 * dw
        dst_row = y * width

        for x in range(width):
            gx = (x + 0.5) * 0.25 - 0.5
            x0 = max(0, min(dw - 1, int(math.floor(gx))))
            x1 = min(dw - 1, x0 + 1)
            wx = max(0.0, min(1.0, gx - x0))
            one_wx = 1.0 - wx

            w00 = one_wx * one_wy
            w10 = wx * one_wy
            w01 = one_wx * wy
            w11 = wx * wy

            i00 = r0_off + x0
            i10 = r0_off + x1
            i01 = r1_off + x0
            i11 = r1_off + x1

            br = lr[i00] * w00 + lr[i10] * w10 + lr[i01] * w01 + lr[i11] * w11
            bg = lg[i00] * w00 + lg[i10] * w10 + lg[i01] * w01 + lg[i11] * w11
            bb = lb[i00] * w00 + lb[i10] * w10 + lb[i01] * w01 + lb[i11] * w11

            d_idx = dst_row + x
            buf_r[d_idx] += br
            buf_g[d_idx] += bg
            buf_b[d_idx] += bb


def _agx_filmic_tonemap_to_srgb_byte(val):
    """AgX High-Contrast Filmic tone-mapping curve + 1/2.2 sRGB encoding -> [0..255]."""
    x = max(0.0, val * 1.12)
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    clamped = max(0.0, min(1.0, mapped))
    srgb = clamped ** (1.0 / 2.2)
    return int(srgb * 255.0 + 0.5)


def write_png_rgb24(filepath, width, height, rgb_bytes):
    """Encode raw 24-bit RGB scanlines into a spec-compliant PNG file using struct & zlib only."""
    raw_scanlines = bytearray()
    row_stride = width * 3
    for y in range(height):
        raw_scanlines.append(0)
        start = y * row_stride
        raw_scanlines.extend(rgb_bytes[start : start + row_stride])

    compressed_idat = zlib.compress(bytes(raw_scanlines), level=6)

    def make_png_chunk(chunk_type_bytes, payload_bytes):
        crc = zlib.crc32(chunk_type_bytes + payload_bytes) & 0xFFFFFFFF
        return (
            struct.pack(">I", len(payload_bytes))
            + chunk_type_bytes
            + payload_bytes
            + struct.pack(">I", crc)
        )

    ihdr_payload = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_data = (
        b"\x89PNG\r\n\x1a\n"
        + make_png_chunk(b"IHDR", ihdr_payload)
        + make_png_chunk(b"IDAT", compressed_idat)
        + make_png_chunk(b"IEND", b"")
    )
    with open(filepath, "wb") as pf:
        pf.write(png_data)
    return len(png_data)


def render_camera_cut(scene_eval, lights, cam_spec, out_png_path, width=1280, height=720):
    """Render a 1280x720 24-bit PNG from a director camera with Z-buffer, planar reflection & Neon Bloom."""
    eye = cam_spec["eye"]
    target = cam_spec["target"]
    lens_mm = cam_spec["lens_mm"]
    sensor_width_mm = cam_spec.get("sensor_width_mm", 36.0)

    right_x, true_up_y, forward = _look_at_camera_basis(eye, target)
    fov_x_tan = (sensor_width_mm * 0.5) / lens_mm
    fov_y_tan = fov_x_tan * (float(height) / float(width))
    inv_fov_x = 1.0 / fov_x_tan
    inv_fov_y = 1.0 / fov_y_tan
    half_w = width * 0.5
    half_h = height * 0.5

    npix = width * height
    zbuf = array.array("f", [1e30]) * npix
    buf_r = array.array("f", [0.0]) * npix
    buf_g = array.array("f", [0.0]) * npix
    buf_b = array.array("f", [0.0]) * npix
    emit_r = array.array("f", [0.0]) * npix
    emit_g = array.array("f", [0.0]) * npix
    emit_b = array.array("f", [0.0]) * npix
    asphalt_mask = array.array("B", [0]) * npix

    # 1. Initialize atmospheric cyberpunk night sky gradient + horizon haze
    for y in range(height):
        v_norm = float(y) / float(height - 1)
        horizon_band = math.exp(-((v_norm - 0.58) ** 2) * 14.0)
        sr = 0.010 + 0.038 * horizon_band + 0.008 * v_norm
        sg = 0.016 + 0.052 * horizon_band + 0.018 * v_norm
        sb = 0.034 + 0.098 * horizon_band + 0.036 * v_norm
        row = y * width
        for x in range(width):
            u_norm = (float(x) / float(width - 1)) - 0.5
            city_glow = math.exp(-(u_norm * u_norm) * 3.5) * horizon_band
            idx = row + x
            buf_r[idx] = sr + 0.032 * city_glow
            buf_g[idx] = sg + 0.018 * city_glow
            buf_b[idx] = sb + 0.046 * city_glow

    materials = scene_eval["materials"]
    default_mat = next(iter(materials.values()))

    opaque_queue = []
    reflection_candidates = []
    translucent_queue = []

    rx0, rx1, rx2 = right_x
    uy0, uy1, uy2 = true_up_y
    fw0, fw1, fw2 = forward
    ex, ey, ez = eye

    # 2. Project & shade all scene triangles
    for obj in scene_eval["objects"]:
        if obj["is_helper"]:
            continue
        verts = obj["verts"]
        tris = obj["tris"]
        if not verts or not tris:
            continue
        slots = obj["material_slots"]
        oname = obj["name"]

        cam_verts = []
        for vx, vy, vz in verts:
            dx = vx - ex
            dy = vy - ey
            dz = vz - ez
            cz = dx * fw0 + dy * fw1 + dz * fw2
            cx = dx * rx0 + dy * rx1 + dz * rx2
            cy = dx * uy0 + dy * uy1 + dz * uy2
            if cz > 0.12:
                inv_z = 1.0 / cz
                ndc_x = cx * inv_z * inv_fov_x
                ndc_y = cy * inv_z * inv_fov_y
                sx = (ndc_x + 1.0) * half_w
                sy = (1.0 - ndc_y) * half_h
                cam_verts.append((sx, sy, cz, ndc_x, ndc_y))
            else:
                cam_verts.append(None)

        for i0, i1, i2, s_idx in tris:
            m_name = slots[min(len(slots) - 1, s_idx)] if slots else default_mat["name"]
            mat = materials.get(m_name, default_mat)
            w0 = verts[i0]
            w1 = verts[i1]
            w2 = verts[i2]

            e1 = (w1[0] - w0[0], w1[1] - w0[1], w1[2] - w0[2])
            e2 = (w2[0] - w0[0], w2[1] - w0[1], w2[2] - w0[2])
            fn = _v_norm(_v_cross(e1, e2))

            p0 = cam_verts[i0]
            p1 = cam_verts[i1]
            p2 = cam_verts[i2]
            if p0 is not None and p1 is not None and p2 is not None:
                min_sx = min(p0[0], p1[0], p2[0])
                max_sx = max(p0[0], p1[0], p2[0])
                min_sy = min(p0[1], p1[1], p2[1])
                max_sy = max(p0[1], p1[1], p2[1])
                if max_sx >= 0.0 and min_sx < width and max_sy >= 0.0 and min_sy < height:
                    col_rgb, em_rgb = _shade_triangle_color_and_emission(w0, w1, w2, fn, mat, oname, lights, eye)
                    is_asph = 1 if ("Wet_Asphalt" in m_name and fn[2] > 0.7) else 0
                    alpha = mat["alpha"]
                    tri_entry = (
                        p0[0], p0[1], p0[2],
                        p1[0], p1[1], p1[2],
                        p2[0], p2[1], p2[2],
                        col_rgb[0], col_rgb[1], col_rgb[2],
                        em_rgb[0], em_rgb[1], em_rgb[2],
                        alpha, is_asph,
                    )
                    if alpha < 0.96:
                        translucent_queue.append(tri_entry)
                    else:
                        opaque_queue.append(tri_entry)

                    if not is_asph and w0[2] > 0.03 and w1[2] > 0.03 and w2[2] > 0.03:
                        if mat["emissive_strength"] > 0.8 or (
                            abs(w0[0]) < 5.8 and abs(w0[1]) < 6.8 and w0[2] < 5.5
                        ):
                            reflection_candidates.append((w0, w1, w2, fn, col_rgb, em_rgb))

    # 3. Rasterize primary opaque geometry
    for tri in opaque_queue:
        _rasterize_triangle_scanline(
            tri[0], tri[1], tri[2],
            tri[3], tri[4], tri[5],
            tri[6], tri[7], tri[8],
            tri[9], tri[10], tri[11],
            tri[12], tri[13], tri[14],
            tri[15], tri[16], False,
            width, height,
            zbuf, buf_r, buf_g, buf_b, emit_r, emit_g, emit_b, asphalt_mask,
        )

    # 4. Planar Wet-Asphalt Mirror Reflection Pass (Z = 0 plane) onto asphalt pixels
    refl_zbuf = array.array("f", [1e30]) * npix
    for w0, w1, w2, fn, col_rgb, em_rgb in reflection_candidates:
        rw0 = (w0[0], w0[1], -w0[2])
        rw1 = (w1[0], w1[1], -w1[2])
        rw2 = (w2[0], w2[1], -w2[2])

        rp = []
        for vx, vy, vz in (rw0, rw1, rw2):
            dx = vx - ex
            dy = vy - ey
            dz = vz - ez
            cz = dx * fw0 + dy * fw1 + dz * fw2
            if cz <= 0.15:
                break
            cx = dx * rx0 + dy * rx1 + dz * rx2
            cy = dx * uy0 + dy * uy1 + dz * uy2
            inv_z = 1.0 / cz
            sx = (cx * inv_z * inv_fov_x + 1.0) * half_w
            sy = (1.0 - cy * inv_z * inv_fov_y) * half_h
            rp.append((sx, sy, cz))
        if len(rp) == 3:
            p0, p1, p2 = rp
            if max(p0[0], p1[0], p2[0]) >= 0.0 and min(p0[0], p1[0], p2[0]) < width:
                if max(p0[1], p1[1], p2[1]) >= 0.0 and min(p0[1], p1[1], p2[1]) < height:
                    refl_fresnel = _puddle_mask((w0[0] + w1[0] + w2[0]) / 3.0, (w0[1] + w1[1] + w2[1]) / 3.0)
                    _rasterize_triangle_scanline(
                        p0[0], p0[1], p0[2],
                        p1[0], p1[1], p1[2],
                        p2[0], p2[1], p2[2],
                        col_rgb[0] * 0.90, col_rgb[1] * 0.96, col_rgb[2] * 1.12,
                        em_rgb[0] * 0.78, em_rgb[1] * 0.85, em_rgb[2] * 0.95,
                        refl_fresnel, 0, True,
                        width, height,
                        refl_zbuf, buf_r, buf_g, buf_b, emit_r, emit_g, emit_b, asphalt_mask,
                    )

    # 5. Rasterize translucent geometry (back-to-front sorted by average depth)
    translucent_queue.sort(key=lambda t: (t[2] + t[5] + t[8]), reverse=True)
    for tri in translucent_queue:
        _rasterize_triangle_scanline(
            tri[0], tri[1], tri[2],
            tri[3], tri[4], tri[5],
            tri[6], tri[7], tri[8],
            tri[9], tri[10], tri[11],
            tri[12], tri[13], tri[14],
            tri[15], 0, False,
            width, height,
            zbuf, buf_r, buf_g, buf_b, emit_r, emit_g, emit_b, asphalt_mask,
        )

    # 6. Volumetric Distance Mist Pass (start=9.5m, depth=28.0m, quadratic falloff)
    mist_start = 9.5
    inv_mist_depth = 1.0 / 28.0
    for i in range(npix):
        d = zbuf[i]
        if d < 1e20 and d > mist_start:
            t = min(1.0, (d - mist_start) * inv_mist_depth)
            m_fac = (t * t) * 0.58
            one_m = 1.0 - m_fac
            buf_r[i] = buf_r[i] * one_m + 0.032 * m_fac
            buf_g[i] = buf_g[i] * one_m + 0.055 * m_fac
            buf_b[i] = buf_b[i] * one_m + 0.105 * m_fac

    # 7. Crisp Motion-Blurred Diagonal Cyberpunk Rain Streaks Overlay
    rng = 0x9E3779B9
    for _ in range(240):
        rng = (rng * 1664525 + 1013904223) & 0xFFFFFFFF
        rx = int((rng & 0xFFFF) * width // 65536)
        ry = int(((rng >> 16) & 0xFFFF) * (height - 40) // 65536)
        s_len = 14 + (rng % 22)
        s_bright = 0.09 + ((rng >> 8) & 0xFF) / 255.0 * 0.16
        for step in range(s_len):
            px = rx + (step // 5)
            py = ry + step
            if 0 <= px < width and 0 <= py < height:
                p_idx = py * width + px
                t_fade = math.sin(math.pi * (step / float(s_len)))
                add_v = s_bright * t_fade
                buf_r[p_idx] += add_v * 0.55
                buf_g[p_idx] += add_v * 0.85
                buf_b[p_idx] += add_v * 1.15

    # 8. Multi-Pass Separable Box/Gaussian Neon Emission Bloom + Anamorphic Streaks
    _apply_neon_bloom_and_streaks(buf_r, buf_g, buf_b, emit_r, emit_g, emit_b, width, height)

    # 9. AgX High-Contrast Filmic Tone-Mapping & sRGB 24-bit Packing
    rgb_bytes = bytearray(npix * 3)
    lum_sum = 0.0
    lum_sq_sum = 0.0
    bright_neon_pixels = 0

    for i in range(npix):
        r8 = _agx_filmic_tonemap_to_srgb_byte(buf_r[i])
        g8 = _agx_filmic_tonemap_to_srgb_byte(buf_g[i])
        b8 = _agx_filmic_tonemap_to_srgb_byte(buf_b[i])
        b_off = i * 3
        rgb_bytes[b_off] = r8
        rgb_bytes[b_off + 1] = g8
        rgb_bytes[b_off + 2] = b8

        lum = 0.2126 * r8 + 0.7152 * g8 + 0.0722 * b8
        lum_sum += lum
        lum_sq_sum += lum * lum
        if r8 > 205 or g8 > 205 or b8 > 205:
            bright_neon_pixels += 1

    png_size = write_png_rgb24(out_png_path, width, height, rgb_bytes)
    mean_lum = lum_sum / float(npix)
    std_lum = math.sqrt(max(0.0, (lum_sq_sum / float(npix)) - mean_lum * mean_lum))

    return {
        "filepath": out_png_path,
        "byte_size": png_size,
        "width": width,
        "height": height,
        "mean_luminance": round(mean_lum, 2),
        "std_luminance": round(std_lum, 2),
        "bright_neon_pixels": bright_neon_pixels,
    }


def render_all_camera_previews(scene_eval, out_dir, width=1280, height=720):
    """Render all 3 director camera cuts to spec-compliant 1280x720 24-bit RGB .png files."""
    lights = _extract_scene_lights(scene_eval)

    camera_cuts = [
        {
            "filename": "render_cam1_85mm_establishing.png",
            "camera_name": "Cinematic_Camera_85mm",
            "eye": (-12.80, -19.20, 3.55),
            "target": (0.0, 0.20, 2.42),
            "lens_mm": 85.0,
            "sensor_width_mm": 36.0,
        },
        {
            "filename": "render_cam2_50mm_low_hero.png",
            "camera_name": "Cinematic_Camera_50mm_HeroLow",
            "eye": (-2.35, -7.10, 0.46),
            "target": (0.0, 0.15, 1.12),
            "lens_mm": 50.0,
            "sensor_width_mm": 36.0,
        },
        {
            "filename": "render_cam3_105mm_counter_macro.png",
            "camera_name": "Cinematic_Camera_105mm_MacroCounter",
            "eye": (-0.38, -6.35, 1.82),
            "target": (-0.05, -0.35, 1.92),
            "lens_mm": 105.0,
            "sensor_width_mm": 36.0,
        },
    ]

    results = []
    for cut in camera_cuts:
        out_path = os.path.join(out_dir, cut["filename"])
        stats = render_camera_cut(scene_eval, lights, cut, out_path, width=width, height=height)
        stats["camera_name"] = cut["camera_name"]
        stats["lens_mm"] = cut["lens_mm"]
        results.append(stats)

    return results
