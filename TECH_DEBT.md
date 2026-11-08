# Technical debt: cyberpunk floating ramen stall

Reviewed 2026-11-08 against local source. Lines refer to the unchanged production files. Each diff is a complete proposal and passes a non-mutating git apply check plus Python syntax parsing. No production fix or performance improvement is claimed as applied.

## 1. Validate RGBZ cache metadata and bound decompression

Location: `video_muxer.py:65-74`. The reader ignores the version, trusts record sizes and calls unbounded zlib.decompress. It never verifies the decoded byte count against width * height * 3. A one-pixel declared frame carrying four decoded bytes is accepted. Truncation also leaks low-level struct/zlib errors. This boundary can allocate much more decoded data than declared.

The proposal supports v1, limits aggregate decoded and compressed data to a documented 256 MiB policy, limits record count to 10,000, validates exact RGB24 length and refuses trailing records. The limits are engineering policy, not measured hardware capacity. They admit the normal 240-frame 320x192 pipeline. Larger export workflows would need an explicit higher budget or streaming reader.

```diff
diff --git a/video_muxer.py b/video_muxer.py
--- a/video_muxer.py
+++ b/video_muxer.py
@@ -63,14 +63,42 @@
 
 
 def read_frame_cache(path):
+    max_bytes = 256 * 1024 * 1024
+    file_bytes = os.path.getsize(path)
+    if file_bytes > max_bytes:
+        raise ValueError("RGBZ compressed file exceeds 256 MiB policy")
     with open(path, "rb") as fh:
         if fh.read(4) != FRAME_CACHE_MAGIC:
             raise ValueError(f"{path}: not an RGBZ frame cache")
-        _ver, width, height, count = struct.unpack("<IIII", fh.read(16))
+        header = fh.read(16)
+        if len(header) != 16:
+            raise ValueError("truncated RGBZ header")
+        version, width, height, count = struct.unpack("<IIII", header)
+        expected = width * height * 3
+        if version != 1 or not width or not height or not 1 <= count <= 10000:
+            raise ValueError("unsupported RGBZ version, dimensions or frame count")
+        if expected * count > max_bytes or count > (file_bytes - 20) // 4:
+            raise ValueError("RGBZ decoded size or record count exceeds available budget")
         frames = []
         for _ in range(count):
-            (n,) = struct.unpack("<I", fh.read(4))
-            frames.append(zlib.decompress(fh.read(n)))
+            size_raw = fh.read(4)
+            if len(size_raw) != 4:
+                raise ValueError("truncated RGBZ record header")
+            (size,) = struct.unpack("<I", size_raw)
+            if not size or size > file_bytes - fh.tell():
+                raise ValueError("invalid RGBZ compressed record length")
+            compressed = fh.read(size)
+            decoder = zlib.decompressobj()
+            try:
+                frame = decoder.decompress(compressed, expected + 1)
+            except zlib.error as exc:
+                raise ValueError("invalid RGBZ compressed frame") from exc
+            if (len(frame) != expected or not decoder.eof
+                    or decoder.unused_data or decoder.unconsumed_tail):
+                raise ValueError("RGBZ frame does not match exact RGB24 dimensions")
+            frames.append(frame)
+        if fh.read(1):
+            raise ValueError("trailing RGBZ data")
     return width, height, frames
 
 
```

Validation: Keep valid roundtrips. Reject unsupported version, zero dimensions/count, short headers, oversized compressed records, short/oversized decompression, concatenated zlib streams and trailing bytes. Check rejection before allocating declared large outputs.

## 2. Enforce CPKS section integrity at its parser

Location: `build_interactive_studio_html.py:1924-1935`; digests are written at `1902-1921`. The parser returns any version and slices sections without validating their extent or stored SHA-256. A flipped payload byte returns changed geometry while leaving the metadata digest untouched. Current callers must perform any further checks themselves.

The proposal validates version, metadata integer types, alignment, nonoverlap, extents and section digests. It verifies byte integrity, not semantic validity of every mesh/material field. Browser-side parsing is separate code in HTML_TEMPLATE and needs equivalent checks before treating externally supplied CPKS as trusted.

```diff
diff --git a/build_interactive_studio_html.py b/build_interactive_studio_html.py
--- a/build_interactive_studio_html.py
+++ b/build_interactive_studio_html.py
@@ -1922,16 +1922,38 @@
 
 
 def parse_container(blob):
-    """Inverse of assemble_container (used by the verifier for integrity checks)."""
-    if blob[:4] != PAYLOAD_MAGIC:
-        raise ValueError("bad payload magic")
+    """Read CPKS v1 byte sections with verified extents and stored digests."""
+    if len(blob) < 12 or blob[:4] != PAYLOAD_MAGIC:
+        raise ValueError("bad or truncated payload magic/header")
     version, jlen = struct.unpack_from("<II", blob, 4)
+    if version != PAYLOAD_VERSION or jlen > len(blob) - 12:
+        raise ValueError("unsupported payload version or JSON extent")
     header = json.loads(blob[12:12 + jlen].decode("ascii"))
+    if not isinstance(header, dict) or not isinstance(header.get("sections"), dict):
+        raise ValueError("payload requires an object section table")
     start = (12 + jlen + 3) & ~3
     sections = {}
+    intervals = []
     for name, meta in header["sections"].items():
-        raw = blob[start + meta["offset"]:start + meta["offset"] + meta["length"]]
-        sections[name] = byte_unshuffle(raw, meta["shuffle"])
+        if not isinstance(meta, dict):
+            raise ValueError("invalid section metadata")
+        offset, length, width = (meta.get("offset"), meta.get("length"), meta.get("shuffle"))
+        if (type(offset) is not int or type(length) is not int or type(width) is not int
+                or offset < 0 or length < 0 or width not in (1, 2, 4, 8)
+                or offset % 4 or length % width):
+            raise ValueError("invalid section extent or shuffle width")
+        end = start + offset + length
+        if start + offset > len(blob) or end > len(blob):
+            raise ValueError("section crosses payload boundary")
+        raw = byte_unshuffle(blob[start + offset:end], width)
+        if hashlib.sha256(raw).hexdigest() != meta.get("sha256"):
+            raise ValueError(f"section digest mismatch: {name}")
+        if length:
+            intervals.append((offset, offset + length))
+        sections[name] = raw
+    intervals.sort()
+    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
+        raise ValueError("overlapping payload sections")
     return version, header, sections
 
 
```

Validation: Preserve multiple aligned sections and shuffle roundtrips. Reject payload tampering, wrong versions, negative/fractional offsets, crossed extents, malformed metadata and overlaps. SHA-256 here detects corruption against the stored header; it does not authenticate an adversarially replaced header plus payload.

## 3. Use a conservative default process count for memory-heavy renders

Location: `render_diorama_animation.py:2143-2177,2196-2291`. The default starts one worker per logical CPU, clamped only by frame count. The spawn initializer receives a scene snapshot per process. Results also retain every RGB frame until muxing. Frame buffers alone cost 240 * width * height * 3 bytes, before geometry, G-buffers, bloom and encoded copies.

The proposal caps the automatic default at two workers while preserving an explicit --workers override. This reduces default concurrency pressure. It does not bound total frame memory or prove faster wall time. Streaming completed frames to a disk cache is a separate pipeline design change.

```diff
diff --git a/render_diorama_animation.py b/render_diorama_animation.py
--- a/render_diorama_animation.py
+++ b/render_diorama_animation.py
@@ -2142,7 +2142,7 @@
 
 def _parse_args(argv):
     opts = {"frames": list(range(FRAME_START, FRAME_END + 1)), "width": DEFAULT_W, "height": DEFAULT_H,
-            "workers": max(1, (os.cpu_count() or 2)), "mux": True, "snapshot_cache": None, "out_dir": SCRIPT_DIR}
+            "workers": max(1, min(2, os.cpu_count() or 2)), "mux": True, "snapshot_cache": None, "out_dir": SCRIPT_DIR}
     i = 0
     while i < len(argv):
         a = argv[i]
```

Validation: Mock os.cpu_count for 1, 2, 32 and None. Check the default and explicit override without starting workers. Measure peak resident memory and throughput on a representative bounded render before choosing a different default.

## 4. Bound the optional ffmpeg child process

Location: `video_muxer.py:530-547`. subprocess.run waits indefinitely for the optional transcoder. A stuck process prevents mux_all from returning even after the native AVI/MP4 outputs are complete. Add an explicit timeout and a failed optional-transcode result. The 1800-second value is a configurable policy candidate, not a measured encode deadline.

```diff
diff --git a/video_muxer.py b/video_muxer.py
--- a/video_muxer.py
+++ b/video_muxer.py
@@ -541,7 +541,11 @@
         "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off",
         "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path,
     ]
-    proc = subprocess.run(cmd, capture_output=True, text=True)
+    try:
+        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
+    except subprocess.TimeoutExpired:
+        return {"available": True, "ok": False, "path": None,
+                "bytes": 0, "stderr": "ffmpeg exceeded the 1800-second policy"}
     ok = proc.returncode == 0 and os.path.isfile(out_path)
     return {"available": True, "ok": ok, "path": out_path if ok else None,
             "bytes": os.path.getsize(out_path) if ok else 0, "stderr": proc.stderr[-400:]}
```

Validation: Mock TimeoutExpired and confirm the caller receives available=true, ok=false, path=null. Preserve missing-executable and normal nonzero-exit behavior. A timeout can leave a partial output file; the unsuccessful report must not present it as a completed artifact.

## 5. Stop claiming full-video verification from fabricated fallback metadata

Priority: high. Evidence: `verify_blender_script.py:2329-2374` validates a real AVI/MP4 pair only when both files exist. Otherwise it writes and validates a three-frame synthetic fixture, deletes that fixture, and assigns literal byte sizes and SHA-256 digests to the absent 240-frame outputs. The absent H.264/AAC branch also uses literal historical metadata. Lines 2382-2429 then write those values into the manifest and compare the copied values with themselves. This does not read or hash the missing original files. Line 2612 also labels mock API version profiles as verified Blender compatibility.

The current native builder also preserves previous phase-7 metadata when the corresponding artifact is absent (`cyberpunk_floating_ramen_stall.py:4779-4813`). Its `artifacts` map therefore mixes measured files and inherited records without a provenance field. Its alternative `renders/` lookup records a root-relative filename even when it read the nested path.

The proposal removes this inheritance and records the actual relative path for measured files. It keeps the existing three-frame muxer checks, then fails the full-artifact verification explicitly when required files are missing. This deliberately makes the full verifier report an incomplete artifact check instead of publishing invented metadata. The independent small-fixture unit suite remains runnable and does not call this full-verifier routine. A CI fixture-only mode would need a separate manifest schema and result status before replacing this fail-closed policy. No production change is applied here.

```diff
diff --git a/verify_blender_script.py b/verify_blender_script.py
--- a/verify_blender_script.py
+++ b/verify_blender_script.py
@@ -2345,33 +2345,18 @@
         _validate_isobmff_mp4_container(tmp_mp4, expected_frames=3, min_size=50000)
         os.remove(tmp_avi)
         os.remove(tmp_mp4)
-        avi_stats = {
-            "filepath": avi_path,
-            "byte_size": 46168654,
-            "width": w_st,
-            "height": h_st,
-            "video_chunks": 240,
-            "audio_chunks": 240,
-            "idx1_entries": 480,
-            "sha256_digest": "61847d9dd29a60716baed6dc4a88379575789e4207b17dcd871d386985639d10",
-        }
-        mp4_stats = {
-            "filepath": mp4_path,
-            "byte_size": 24160460,
-            "video_samples": 240,
-            "sha256_digest": "b480159fc8b7460eaa2469f0e878e286dc48660f8d3a7804b0a33c21e4b8635f",
-        }
+        raise FileNotFoundError(
+            "The three-frame muxer fixture passed, but full 240-frame AVI/MP4 "
+            "artifacts are absent. Fixture success cannot verify those outputs."
+        )
 
     h264_aac_path = os.path.join(script_dir, "cyberpunk_diorama_240f_h264_aac.mp4")
     if os.path.isfile(h264_aac_path):
         h264_aac_stats = _validate_isobmff_mp4_container(h264_aac_path, expected_frames=240, min_size=50000)
     else:
-        h264_aac_stats = {
-            "filepath": h264_aac_path,
-            "byte_size": 1806392,
-            "video_samples": 240,
-            "sha256_digest": "7a08bc7b1a67728977a360c5150890a546fdb64b4ada5a6837d9535245bee3fa",
-        }
+        raise FileNotFoundError(
+            "Missing H.264/AAC artifact: " + h264_aac_path
+        )
 
     # Track 28: Record & verify SHA-256 digests in cyberpunk_diorama_manifest.json
     manifest_path = os.path.join(script_dir, "cyberpunk_diorama_manifest.json")
@@ -2609,7 +2594,7 @@
     stp = st8["scene_payload"]
 
     rows = [
-        ("Target Compatibility", "Blender 4.0.2 & Blender 4.3.0 (VERIFIED -- ALL 32 TRACKS)"),
+        ("Target Compatibility", "Mock API profiles 4.0.2 & 4.3.0; native Blender not executed"),
         ("Render Engine / Preset", f"{m['render_engine']} @ {m['cycles_samples']} samples (AgX High Contrast)"),
         ("Collections Hierarchy", f"{m['collections']['total_collections']} total (1 Master + 9 Sub-Collections)"),
         ("Total Scene Objects", f"{m['total_objects']} (MESH={obt['MESH']}, CURVE={obt['CURVE']}, LIGHT={obt['LIGHT']}, CAM={obt['CAMERA']}, ARM={obt['ARMATURE']}, LAT={obt['LATTICE']}, EMPTY={obt['EMPTY']})"),
diff --git a/cyberpunk_floating_ramen_stall.py b/cyberpunk_floating_ramen_stall.py
--- a/cyberpunk_floating_ramen_stall.py
+++ b/cyberpunk_floating_ramen_stall.py
@@ -4777,11 +4777,6 @@
     topo_digest = hashlib.sha256("\n".join(topology_records).encode("utf-8")).hexdigest()
 
     manifest_dir = os.path.dirname(os.path.abspath(output_path))
-    prev_artifacts = {}
-    if os.path.isfile(output_path):
-        with open(output_path, "r", encoding="utf-8") as mf_prev_art:
-            prev_art_data = json.load(mf_prev_art)
-        prev_artifacts = prev_art_data.get("phase7_animation_and_video", {}).get("artifacts", {})
     phase7_artifacts = {}
     for artifact_name in (
         "cyberpunk_diorama_240f.avi",
@@ -4805,12 +4800,10 @@
                         break
                     h_art.update(chunk)
             phase7_artifacts[artifact_name] = {
-                "filepath": artifact_name,
+                "filepath": os.path.relpath(artifact_path, manifest_dir).replace(os.sep, "/"),
                 "byte_size": os.path.getsize(artifact_path),
                 "sha256_digest": h_art.hexdigest(),
             }
-        elif artifact_name in prev_artifacts:
-            phase7_artifacts[artifact_name] = prev_artifacts[artifact_name]
 
     frame_pixel_hashes = {}
     report_path = os.path.join(manifest_dir, "cyberpunk_diorama_render_report.json")
```

Validation: the complete proposed module parses as Python, and this patch passes `git apply --check --recount --ignore-space-change`. The heavy verifier was not executed. Before applying, check two workflows: absent full videos must preserve the successful fixture checks and exit with the explicit missing-artifact result, while a complete present artifact set must still reach independently measured container metadata. Neither workflow is claimed to have run in this review.
