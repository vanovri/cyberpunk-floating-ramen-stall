"""Fast stdlib tests for small geometry, timeline and binary pipeline fixtures.

No bpy runtime, renderer, browser, ffmpeg process or network service is started.
"""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave
import zlib


ROOT = Path(__file__).resolve().parents[1]


def load_module(filename):
    spec = importlib.util.spec_from_file_location("audit_" + filename[:-3], ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exporter = load_module("diorama_exporter.py")
animation = load_module("render_diorama_animation.py")
studio = load_module("build_interactive_studio_html.py")
preview = load_module("render_diorama_previews.py")
video = load_module("video_muxer.py")


def mp4_boxes(raw, start=0, end=None):
    """Independent bounded reader for the ordinary-size boxes emitted here."""
    end = len(raw) if end is None else end
    result = []
    while start < end:
        if end - start < 8:
            raise AssertionError("truncated MP4 box header")
        size, tag = struct.unpack_from(">I4s", raw, start)
        if size < 8 or start + size > end:
            raise AssertionError("MP4 box crosses its parent")
        result.append((tag, start, start + 8, start + size))
        start += size
    return result


def mp4_descendants(raw, start=0, end=None):
    containers = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}
    for box in mp4_boxes(raw, start, end):
        yield box
        if box[0] in containers:
            yield from mp4_descendants(raw, box[2], box[3])


def riff_chunks(raw, start, end):
    while start < end:
        if end - start < 8:
            raise AssertionError("truncated RIFF chunk")
        tag, size = struct.unpack_from("<4sI", raw, start)
        payload = start + 8
        if payload + size > end:
            raise AssertionError("RIFF chunk crosses its parent")
        yield tag, start, payload, payload + size
        start = payload + size + (size & 1)
    if start != end:
        raise AssertionError("bad RIFF padding")


class GeometryTimelineTests(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=9)

    def test_affine_inverse_with_rotation_translation_and_nonuniform_scale(self):
        matrix = exporter._mat4_from_trs((3, -7, 2), (0.3, -0.2, 0.7), (2, 3, 0.5))
        point = (-4, 2, 5)
        transformed = exporter._mat4_transform_pt(matrix, point)
        inverse = exporter._mat4_invert_rigid_or_affine(matrix)
        self.assertVectorAlmostEqual(exporter._mat4_transform_pt(inverse, transformed), point)

    def test_bishop_tube_has_closed_index_topology_and_requested_radius(self):
        vertices, triangles = exporter.tessellate_polyline_bishop_tube(
            [(0, 0, 0), (0, 0, 2)], 0.25, radial_sides=8, mat_idx=3
        )
        self.assertEqual(len(vertices), 18)
        self.assertEqual(len(triangles), 32)
        edges = {}
        for a, b, c, material in triangles:
            self.assertEqual(material, 3)
            self.assertEqual(len({a, b, c}), 3)
            for x, y in ((a, b), (b, c), (c, a)):
                self.assertTrue(0 <= x < len(vertices))
                key = tuple(sorted((x, y)))
                edges[key] = edges.get(key, 0) + 1
        self.assertTrue(all(count == 2 for count in edges.values()))
        for x, y, _ in vertices[:16]:
            self.assertAlmostEqual(math.hypot(x, y), 0.25)

    def test_linear_timeline_deduplicates_last_write_and_clamps_endpoints(self):
        fc = animation.prepare_fcurve([(10, 99, "LINEAR"), (0, 0, "LINEAR"), (10, 20, "LINEAR")])
        self.assertEqual(animation.eval_fcurve(fc, -1), 0)
        self.assertEqual(animation.eval_fcurve(fc, 5), 10)
        self.assertEqual(animation.eval_fcurve(fc, 30), 20)
        self.assertIsNone(animation.prepare_fcurve([]))

    def test_constant_vector_curve_switches_at_the_key(self):
        fc = animation.prepare_fcurve([(1, (0, 1, 2), "CONSTANT"), (3, (2, 3, 4), "LINEAR")])
        self.assertEqual(animation.eval_fcurve(fc, 2.999), (0, 1, 2))
        self.assertEqual(animation.eval_fcurve(fc, 3), (2, 3, 4))

    def test_auto_clamped_curve_stays_inside_adjacent_key_values(self):
        fc = animation.prepare_fcurve([(0, 0, "BEZIER"), (2, 3, "BEZIER"), (4, 1, "BEZIER")])
        for index in range(81):
            frame = index / 20
            value = animation.eval_fcurve(fc, frame)
            self.assertGreaterEqual(value, 0 if frame <= 2 else 1)
            self.assertLessEqual(value, 3)

    def test_scene_parent_animation_and_camera_cuts_without_mutating_snapshot(self):
        def rec(name, loc=(0, 0, 0), parent=None, camera=None):
            return {"name": name, "loc": loc, "rot": (0, 0, 0), "scale": (1, 1, 1),
                    "parent": parent, "fcurves": {}, "constraints": [],
                    "type": "CAMERA" if camera else "EMPTY", "camera": camera}

        objects = {"parent": rec("parent", (10, 0, 0)), "child": rec("child", parent="parent"),
                   "camA": rec("camA", camera={"lens": 50}), "camB": rec("camB", camera={"lens": 85})}
        objects["child"]["fcurves"]["location"] = animation.prepare_fcurve(
            [(1, (0, 0, 0), "LINEAR"), (3, (2, 0, 0), "LINEAR")]
        )
        snapshot = {"objects": objects, "paths": {}, "default_camera": "camA",
                    "markers": [(1, "first", "camA"), (3, "second", "camB")]}
        before = copy.deepcopy(snapshot)
        state = animation.evaluate_scene_at_frame(snapshot, 2)
        self.assertVectorAlmostEqual(animation.m_translation(state["world"]["child"]), (11, 0, 0))
        self.assertEqual(state["active_camera"], "camA")
        self.assertEqual(animation.evaluate_scene_at_frame(snapshot, 3)["active_camera"], "camB")
        self.assertEqual(snapshot, before)


class BinaryArtifactTests(unittest.TestCase):
    def test_cpks_sections_roundtrip_with_alignment_and_independent_hashes(self):
        header = {"counts": {"vertices": 1}}
        sections = [("odd", b"abc", "u8", 1), ("positions", struct.pack("<3f", 1, 2, 3), "f32", 4)]
        blob, packed_header = studio.assemble_container(header, sections)
        version, parsed, unpacked = studio.parse_container(blob)
        self.assertEqual(version, 1)
        self.assertEqual(header, {"counts": {"vertices": 1}})
        self.assertEqual(parsed, packed_header)
        for name, raw, _, _ in sections:
            self.assertEqual(unpacked[name], raw)
            self.assertEqual(parsed["sections"][name]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(parsed["sections"][name]["offset"] % 4, 0)

    def test_cpks_rejects_invalid_magic_and_shuffle_handles_short_data(self):
        with self.assertRaisesRegex(ValueError, "magic"):
            studio.parse_container(b"wrong")
        for raw in (b"", b"abc", bytes(range(16))):
            for width in (1, 2, 4):
                self.assertEqual(studio.byte_unshuffle(studio.byte_shuffle(raw, width), width), raw)

    def test_html_substitution_requires_each_named_token(self):
        self.assertEqual(studio.render_html("x@@ONE@@y", {"ONE": "<scene>"}), "x<scene>y")
        with self.assertRaisesRegex(KeyError, "MISSING"):
            studio.render_html("plain", {"MISSING": "x"})

    def test_png_crc_dimensions_and_actual_pixels(self):
        rgb = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.png"
            count = preview.write_png_rgb24(path, 2, 2, rgb)
            raw = path.read_bytes()
        self.assertEqual(count, len(raw))
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n")
        offset, chunks = 8, {}
        while offset < len(raw):
            size = struct.unpack_from(">I", raw, offset)[0]
            tag = raw[offset + 4:offset + 8]
            payload = raw[offset + 8:offset + 8 + size]
            crc = struct.unpack_from(">I", raw, offset + 8 + size)[0]
            self.assertEqual(crc, zlib.crc32(tag + payload) & 0xFFFFFFFF)
            chunks[tag] = payload
            offset += 12 + size
        self.assertEqual(offset, len(raw))
        self.assertEqual(struct.unpack(">IIBBBBB", chunks[b"IHDR"]), (2, 2, 8, 2, 0, 0, 0))
        self.assertEqual(zlib.decompress(chunks[b"IDAT"]), b"\0" + rgb[:6] + b"\0" + rgb[6:])

    def test_glb_and_obj_reopen_with_axis_conversion_and_global_indices(self):
        material = {"name": "neon", "index": 0, "base_color": (0.2, 0.3, 0.4),
                    "emissive_color": (1, 0, 0), "emissive_strength": 2,
                    "alpha": 1, "metallic": 0.5, "roughness": 0.25}
        triangle = {"name": "first", "verts": [(0, 0, 0), (1, 0, 0), (0, 1, 2)],
                    "normals": [(0, 0, 1)] * 3, "tris": [(0, 1, 2, 0)],
                    "material_slots": ["neon"], "world_matrix": exporter._mat4_identity(),
                    "is_helper": False}
        second = {**triangle, "name": "second"}
        scene = {"materials": {"neon": material}, "objects": [triangle, second]}
        before = copy.deepcopy(scene)
        with tempfile.TemporaryDirectory() as tmp:
            report = exporter.export_glb_and_obj(scene, tmp)
            raw = Path(report["glb_path"]).read_bytes()
            obj = Path(report["obj_path"]).read_text(encoding="utf-8")
            self.assertIn("newmtl neon", Path(report["mtl_path"]).read_text(encoding="utf-8"))
        self.assertEqual(struct.unpack_from("<III", raw), (0x46546C67, 2, len(raw)))
        length, kind = struct.unpack_from("<I4s", raw, 12)
        self.assertEqual(kind, b"JSON")
        gltf = json.loads(raw[20:20 + length])
        binary_length, kind = struct.unpack_from("<I4s", raw, 20 + length)
        self.assertEqual(kind, b"BIN\0")
        binary = raw[28 + length:]
        self.assertEqual(binary_length, len(binary))
        self.assertEqual(gltf["buffers"][0]["byteLength"], len(binary))
        self.assertEqual(gltf["nodes"][1]["mesh"], 1)
        for view in gltf["bufferViews"]:
            self.assertEqual(view["byteOffset"] % 4, 0)
            self.assertLessEqual(view["byteOffset"] + view["byteLength"], len(binary))
        first_view = gltf["bufferViews"][gltf["accessors"][0]["bufferView"]]
        self.assertEqual(struct.unpack_from("<9f", binary, first_view["byteOffset"]),
                         (0, 0, 0, 1, 0, 0, 0, 2, -1))
        self.assertIn("f 1//1 2//2 3//3", obj)
        self.assertIn("f 4//4 5//5 6//6", obj)
        self.assertEqual(scene, before)


class VideoPipelineTests(unittest.TestCase):
    def test_rgbz_roundtrip_and_bad_magic(self):
        frames = [bytes(range(12)), bytes(reversed(range(12)))]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frames.rgbz"
            video.write_frame_cache(path, 2, 2, frames)
            self.assertEqual(video.read_frame_cache(path), (2, 2, frames))
            path.write_bytes(b"NOPE")
            with self.assertRaisesRegex(ValueError, "not an RGBZ"):
                video.read_frame_cache(path)

    def test_audio_fractional_boundaries_preserve_samples_and_pad_short_track(self):
        pcm = struct.pack("<7h", *range(7))
        chunks = video._audio_slices(pcm, 3, 3, 7, 2)
        self.assertEqual([len(c) for c in chunks], [4, 4, 6])
        self.assertEqual(b"".join(chunks), pcm)
        padded = video._audio_slices(pcm[:6], 3, 3, 7, 2)
        self.assertEqual(b"".join(padded), pcm[:6] + b"\0" * 8)

    def test_bgr_conversion_reverses_rows_swaps_channels_and_pads(self):
        rgb = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255))
        expected = bytes((255, 0, 0, 255, 255, 255, 0, 0, 0, 0, 255, 0, 255, 0, 0, 0))
        self.assertEqual(video.rgb_to_bottom_up_bgr(rgb, 2, 2), expected)

    def test_wav_loading_and_unsupported_sample_width(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.wav"
            for width, pcm in ((2, struct.pack("<3h", 1, 2, 3)), (1, b"abc")):
                with wave.open(str(path), "wb") as writer:
                    writer.setnchannels(1)
                    writer.setsampwidth(width)
                    writer.setframerate(7)
                    writer.writeframes(pcm)
                if width == 2:
                    self.assertEqual(video.load_wav(str(path)), (pcm, 7, 1, 2))
                else:
                    with self.assertRaisesRegex(ValueError, "16-bit"):
                        video.load_wav(str(path))

    def test_unsigned_exp_golomb_codes_decode_independently(self):
        values = (0, 1, 2, 3, 15, 255, 1023)
        writer = video.BitWriter()
        for value in values:
            writer.ue(value)
        writer.align_zero()
        bits = "".join(f"{b:08b}" for b in writer.tobytes())
        offset = 0
        for expected in values:
            zeros = 0
            while bits[offset + zeros] == "0":
                zeros += 1
            decoded = int(bits[offset + zeros:offset + 2 * zeros + 1], 2) - 1
            self.assertEqual(decoded, expected)
            offset += 2 * zeros + 1

    def test_emulation_prevention_retains_payload_when_removed(self):
        original = b"\x00\x00\x00\x01\x02\x00\x00\x03\x04\x00\x00\x04"
        encoded = video.emulation_prevent(original)
        decoded = bytearray()
        zeros = 0
        for value in encoded:
            if zeros >= 2 and value == 3:
                zeros = 0
                continue
            decoded.append(value)
            zeros = zeros + 1 if value == 0 else 0
        self.assertEqual(bytes(decoded), original)

    def test_avi_roundtrip_chunk_sizes_frame_order_and_index_offsets(self):
        frames = [bytes((255, 0, 0)) * 4, bytes((0, 255, 0)) * 4]
        pcm = struct.pack("<7h", *range(7))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.avi"
            report = video.mux_avi(frames, 2, 2, 3, pcm, 7, 1, str(path))
            raw = path.read_bytes()
        self.assertEqual(struct.unpack_from("<4sI4s", raw), (b"RIFF", len(raw) - 8, b"AVI "))
        top = list(riff_chunks(raw, 12, len(raw)))
        movi = next(c for c in top if c[0] == b"LIST" and raw[c[2]:c[2] + 4] == b"movi")
        chunks = list(riff_chunks(raw, movi[2] + 4, movi[3]))
        self.assertEqual([c[0] for c in chunks], [b"00db", b"01wb", b"00db", b"01wb"])
        self.assertEqual(raw[chunks[0][2]:chunks[0][3]], video.rgb_to_bottom_up_bgr(frames[0], 2, 2))
        index = next(c for c in top if c[0] == b"idx1")
        entries = list(struct.iter_unpack("<4sIII", raw[index[2]:index[3]]))
        self.assertEqual(len(entries), report["idx1_entries"])
        for (tag, flags, offset, size), chunk in zip(entries, chunks):
            self.assertEqual((tag, flags, movi[2] + offset, size),
                             (chunk[0], 0x10, chunk[1], chunk[3] - chunk[2]))

    def test_mp4_track_sample_tables_reference_exact_mdat_payloads(self):
        frames = [bytes((255, 0, 0)) * 256, bytes((0, 0, 255)) * 256]
        pcm = struct.pack("<7h", *range(7))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.mp4"
            report = video.mux_mp4(frames, 16, 16, 3, pcm, 7, 1, str(path))
            raw = path.read_bytes()
        top = mp4_boxes(raw)
        self.assertEqual([b[0] for b in top], [b"ftyp", b"mdat", b"moov"])
        mdat = top[1]
        all_boxes = list(mp4_descendants(raw))
        sizes = next(b for b in all_boxes if b[0] == b"stsz")
        default_size, count = struct.unpack_from(">II", raw, sizes[2] + 4)
        self.assertEqual((default_size, count), (0, 2))
        lengths = struct.unpack_from(">2I", raw, sizes[2] + 12)
        offsets_box = next(b for b in all_boxes if b[0] == b"stco")
        self.assertEqual(struct.unpack_from(">I", raw, offsets_box[2] + 4)[0], 2)
        offsets = struct.unpack_from(">2I", raw, offsets_box[2] + 8)
        for offset, length in zip(offsets, lengths):
            self.assertGreaterEqual(offset, mdat[2])
            self.assertLessEqual(offset + length, mdat[3])
            nal_length = struct.unpack_from(">I", raw, offset)[0]
            self.assertEqual(nal_length + 4, length)
            self.assertEqual(raw[offset + 4] & 31, 5)
        self.assertEqual(report["video_samples"], 2)
        self.assertEqual(report["audio_pcm_frames"], 4)

    def test_mp4_rejects_non_macroblock_dimensions_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.mp4"
            with self.assertRaisesRegex(ValueError, "multiples of 16"):
                video.mux_mp4([b""], 17, 16, 24, b"", 48000, 2, str(path))
            self.assertFalse(path.exists())

    def test_optional_transcoder_missing_path_does_not_start_a_process(self):
        with patch.object(video.shutil, "which", return_value=None), patch.object(video.subprocess, "run") as run:
            self.assertEqual(video.ffmpeg_transcode("input.avi", "audio.wav", "output.mp4"), {"available": False})
        run.assert_not_called()

    def test_bt709_yuv420_coefficients_and_isobmff_colr_nclx_box(self):
        # Pure BT.709 limited-range (16..235 / 16..240) check on primary green (0, 255, 0)
        rgb = bytes((0, 255, 0)) * 4
        yuv = video.rgb_to_yuv420(rgb, 2, 2)
        # BT.709 Y for (0,255,0): ((47*0 + 157*255 + 16*0 + 128) >> 8) + 16 = 156 + 16 = 172 (BT.601 was 145)
        self.assertEqual(list(yuv[0]), [172, 172, 172, 172])
        self.assertEqual(len(yuv[1]), 1)
        self.assertEqual(len(yuv[2]), 1)
        # Verify colr(nclx) box (primaries=1, transfer=1, matrix=1, full_range=0) and SPS VUI matrix=1
        self.assertEqual(video._colr_nclx(1, 1, 1, 0), b"\x00\x00\x00\x13colrnclx\x00\x01\x00\x01\x00\x01\x00")
        sps = video.build_sps(16, 16, 24)
        pps = video.build_pps()
        avc1 = video._avc1(16, 16, sps, pps)
        self.assertIn(b"colrnclx\x00\x01\x00\x01\x00\x01\x00", avc1)
        # Independently decode the SPS fields up to colour_description.
        rbsp = sps[1:].replace(b"\x00\x00\x03", b"\x00\x00")
        bits = "".join(f"{byte:08b}" for byte in rbsp)
        cursor = 0
        def take(count):
            nonlocal cursor
            result = int(bits[cursor:cursor + count], 2)
            cursor += count
            return result
        def ue():
            zeroes = 0
            while take(1) == 0:
                zeroes += 1
            return (1 << zeroes) - 1 + (take(zeroes) if zeroes else 0)
        self.assertEqual(take(8), 66)
        take(16)  # constraint flags and level
        self.assertEqual([ue() for _ in range(4)], [0, 0, 2, 1])
        self.assertEqual(take(1), 0)  # gaps flag
        self.assertEqual([ue(), ue()], [0, 0])  # one 16x16 macroblock
        self.assertEqual([take(1) for _ in range(4)], [1, 1, 0, 1])
        self.assertEqual([take(1) for _ in range(3)], [0, 0, 1])
        self.assertEqual(take(3), 5)
        self.assertEqual(take(1), 0)  # limited range
        self.assertEqual(take(1), 1)  # colour_description present
        self.assertEqual([take(8) for _ in range(3)], [1, 1, 1])


class HeadlessLockAndShaderGroupTests(unittest.TestCase):
    def test_verify_and_lock_scene_for_batch_render_and_node_groups(self):
        verifier = load_module("verify_blender_script.py")
        bpy, bmesh_mod, mathutils_mod = verifier.make_mock_blender((4, 3, 0))
        import sys
        old_mods = {k: sys.modules.get(k) for k in ("bpy", "bmesh", "mathutils")}
        temporary_scene = tempfile.TemporaryDirectory()
        try:
            sys.modules["bpy"] = bpy
            sys.modules["bmesh"] = bmesh_mod
            sys.modules["mathutils"] = mathutils_mod
            script_path = ROOT / "cyberpunk_floating_ramen_stall.py"
            ns = {"__name__": "__main__", "__file__": str(Path(temporary_scene.name) / script_path.name)}
            exec(compile(script_path.read_text(encoding="utf-8"), str(script_path), "exec"), ns)

            # Verify 4 shared ShaderNodeGroups + GN_Cyber_Rain_Field
            for ng_name in (
                "NG_Brushed_Metal_Core",
                "NG_Cyber_Neon_Tube",
                "NG_Vertical_Rain_Drip",
                "NG_Triplanar_Ramp_Bump",
                "GN_Cyber_Rain_Field",
            ):
                self.assertIsNotNone(bpy.data.node_groups.get(ng_name), f"Missing node group {ng_name}")

            gn_rain = bpy.data.node_groups.get("GN_Cyber_Rain_Field")
            gn_types = [n.bl_idname for n in gn_rain.nodes]
            self.assertNotIn("GeometryNodeSetMaterial", gn_types)
            self.assertIn("GeometryNodeInputSceneTime", gn_types)
            self.assertIn("GeometryNodeTranslateInstances", gn_types)

            # Verify consolidated steam plumes (4 meshes instead of 12) and WAVE count = 29
            self.assertEqual(len(bpy.data.objects), 571)
            self.assertEqual(len(bpy.data.meshes), 372)
            wave_count = sum(1 for o in bpy.data.objects for m in o.modifiers if m.type == "WAVE")
            self.assertEqual(wave_count, 29)

            # Verify anti-firefly Cycles & Compositor settings
            scene = bpy.context.scene
            self.assertAlmostEqual(scene.cycles.sample_clamp_indirect, 3.5)
            self.assertAlmostEqual(scene.cycles.blur_glossy, 0.18)
            self.assertEqual(scene.cycles.volume_bounces, 0)
            self.assertEqual(scene.cycles.adaptive_min_samples, 128)
            comp_types = [n.bl_idname for n in scene.node_tree.nodes]
            self.assertIn("CompositorNodeDenoise", comp_types)
            self.assertIn("CompositorNodeDespeckle", comp_types)

            # Verify verify_and_lock_scene_for_batch_render locks all objects and F-Curves
            lock_fn = ns["verify_and_lock_scene_for_batch_render"]
            rep = lock_fn(scene)
            self.assertEqual(rep["status"], "LOCKED_READY")
            self.assertEqual(rep["objects"], 571)
            self.assertEqual(rep["cameras"], 3)
            self.assertEqual(rep["actions"], 62)
            self.assertEqual(rep["fcurves"], 85)
            self.assertEqual(rep["missing_textures"], 0)
            self.assertTrue(all(all(obj.lock_location) and all(obj.lock_rotation) and all(obj.lock_scale) for obj in bpy.data.objects))

            # Verify error raised when an external image texture is missing
            bad_img = bpy.data.images.new("Missing_Ext_Tex")
            bad_img.source = "FILE"
            bad_img.packed_file = None
            bad_img.filepath = "//nonexistent_texture_404.png"
            with self.assertRaisesRegex(RuntimeError, "Missing external textures"):
                lock_fn(scene)
        finally:
            for k, v in old_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
            temporary_scene.cleanup()


if __name__ == "__main__":
    unittest.main()
