#!/usr/bin/env python3
"""
video_muxer.py
==============

GOD MODE BLENDER PHASE 7B / TRACK 27 -- from-scratch binary RIFF/AVI 1.0 and
ISO-BMFF (.mp4) audio-video muxer. Pure Python 3.14 standard library
(struct, zlib, wave, hashlib; subprocess/shutil only for the optional ffmpeg
transcode).

Inputs : 240 RGB24 frame buffers (top-down, W*H*3 bytes) @ 24 fps and the
         10.0 s stereo 48 kHz 16-bit PCM ``cyberpunk_ramen_soundtrack.wav``.
Outputs: ``cyberpunk_diorama_240f.avi``
             RIFF 'AVI ' > LIST hdrl (avih, LIST strl[vids strh + BITMAPINFOHEADER
             24-bit BGR], LIST strl[auds strh + WAVEFORMATEX PCM]) > LIST movi
             (240 x '00db' + 240 x '01wb' interleaved, 8,000 audio bytes / frame)
             > idx1 keyframe index.
         ``cyberpunk_diorama_240f.mp4``
             ftyp(isom) > mdat > moov(mvhd, video trak[tkhd, mdia(mdhd, hdlr,
             minf(vmhd, dinf, stbl(stsd/avc1+avcC+colr(nclx), stts, stss, stsc, stsz, stco)))],
             audio trak[... stsd/sowt 48 kHz stereo 16-bit PCM ...]).
             Video is a valid H.264 Constrained-Baseline bitstream: one IDR slice
             per frame whose macroblocks are all I_PCM (lossless 4:2:0 BT.709
             limited-range samples), stored AVCC length-prefixed.
         ``cyberpunk_diorama_240f_h264_aac.mp4`` (optional, ffmpeg on PATH).

CLI::

    python video_muxer.py            # re-mux from cyberpunk_diorama_frames_240f.rgbz
"""

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import wave
import zlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AVI_NAME = "cyberpunk_diorama_240f.avi"
MP4_NAME = "cyberpunk_diorama_240f.mp4"
MP4_FFMPEG_NAME = "cyberpunk_diorama_240f_h264_aac.mp4"
WAV_NAME = "cyberpunk_ramen_soundtrack.wav"
FRAME_CACHE_NAME = "cyberpunk_diorama_frames_240f.rgbz"
FRAME_CACHE_MAGIC = b"RGBZ"

# ==============================================================================
# FRAME CACHE (zlib-compressed RGB24 frames)
# ==============================================================================


def write_frame_cache(path, width, height, frames):
    with open(path, "wb") as fh:
        fh.write(FRAME_CACHE_MAGIC + struct.pack("<IIII", 1, width, height, len(frames)))
        for rgb in frames:
            comp = zlib.compress(rgb, 6)
            fh.write(struct.pack("<I", len(comp)))
            fh.write(comp)
    return path


def read_frame_cache(path):
    with open(path, "rb") as fh:
        if fh.read(4) != FRAME_CACHE_MAGIC:
            raise ValueError(f"{path}: not an RGBZ frame cache")
        _ver, width, height, count = struct.unpack("<IIII", fh.read(16))
        frames = []
        for _ in range(count):
            (n,) = struct.unpack("<I", fh.read(4))
            frames.append(zlib.decompress(fh.read(n)))
    return width, height, frames


# ==============================================================================
# AUDIO
# ==============================================================================


def load_wav(path):
    with wave.open(path, "rb") as wf:
        ch, sw, rate, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        pcm = wf.readframes(n)
    if sw != 2:
        raise ValueError("expected 16-bit PCM")
    return pcm, rate, ch, sw


def _audio_slices(pcm, frame_count, fps, rate, block_align):
    """Exact per-video-frame audio slices (48000/24 = 2000 sample frames = 8000 bytes)."""
    total = len(pcm) // block_align
    out = []
    for i in range(frame_count):
        s0 = (i * rate) // fps
        s1 = ((i + 1) * rate) // fps
        s0, s1 = min(s0, total), min(s1, total)
        chunk = pcm[s0 * block_align:s1 * block_align]
        need = ((i + 1) * rate) // fps - (i * rate) // fps
        if len(chunk) < need * block_align:
            chunk = chunk + b"\x00" * (need * block_align - len(chunk))
        out.append(chunk)
    return out


# ==============================================================================
# RIFF / AVI 1.0
# ==============================================================================


def _riff_chunk(fourcc, payload):
    data = fourcc + struct.pack("<I", len(payload)) + payload
    if len(payload) & 1:
        data += b"\x00"
    return data


def _riff_list(list_type, payload):
    return b"LIST" + struct.pack("<I", len(payload) + 4) + list_type + payload


def rgb_to_bottom_up_bgr(rgb, width, height):
    stride = width * 3
    pad = (4 - stride % 4) % 4
    rows = []
    for y in range(height - 1, -1, -1):
        row = bytearray(rgb[y * stride:(y + 1) * stride])
        row[0::3], row[2::3] = row[2::3], row[0::3]
        rows.append(bytes(row) + b"\x00" * pad)
    return b"".join(rows)


def mux_avi(frames, width, height, fps, pcm, rate, channels, out_path):
    block_align = channels * 2
    byte_rate = rate * block_align
    n = len(frames)
    frame_bytes = ((width * 3 + 3) & ~3) * height
    audio = _audio_slices(pcm, n, fps, rate, block_align)
    max_audio = max(len(a) for a in audio)

    avih = struct.pack(
        "<IIIIIIIIII4I",
        int(round(1_000_000 / fps)),          # dwMicroSecPerFrame (41667)
        (frame_bytes + max_audio + 16) * fps,  # dwMaxBytesPerSec
        0,                                    # dwPaddingGranularity
        0x10 | 0x100,                         # AVIF_HASINDEX | AVIF_ISINTERLEAVED
        n,                                    # dwTotalFrames
        0,                                    # dwInitialFrames
        2,                                    # dwStreams
        frame_bytes + 8,                      # dwSuggestedBufferSize
        width, height, 0, 0, 0, 0,
    )
    strh_v = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"vids", b"DIB ", 0, 0, 0, 0,
        1, fps, 0, n,                         # dwScale, dwRate, dwStart, dwLength
        frame_bytes, 0xFFFFFFFF, frame_bytes,  # dwSuggestedBufferSize, dwQuality, dwSampleSize
        0, 0, width, height,                  # rcFrame
    )
    bih = struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, frame_bytes, 2835, 2835, 0, 0)
    strh_a = struct.pack(
        "<4s4sIHHIIIIIIIIhhhh",
        b"auds", b"\x00\x00\x00\x00", 0, 0, 0, 0,
        block_align, byte_rate, 0, len(pcm) // block_align,  # dwScale=nBlockAlign, dwRate=nAvgBytesPerSec
        max_audio, 0xFFFFFFFF, block_align,
        0, 0, 0, 0,
    )
    wfx = struct.pack("<HHIIHHH", 1, channels, rate, byte_rate, block_align, 16, 0)
    hdrl = _riff_list(b"hdrl", _riff_chunk(b"avih", avih)
                      + _riff_list(b"strl", _riff_chunk(b"strh", strh_v) + _riff_chunk(b"strf", bih))
                      + _riff_list(b"strl", _riff_chunk(b"strh", strh_a) + _riff_chunk(b"strf", wfx)))

    movi_parts = []
    index = []
    offset = 4  # idx1 offsets are relative to the 'movi' fourcc
    for i in range(n):
        vid = rgb_to_bottom_up_bgr(frames[i], width, height)
        for fourcc, payload in ((b"00db", vid), (b"01wb", audio[i])):
            ck = _riff_chunk(fourcc, payload)
            index.append(struct.pack("<4sIII", fourcc, 0x10, offset, len(payload)))
            movi_parts.append(ck)
            offset += len(ck)
    movi = _riff_list(b"movi", b"".join(movi_parts))
    idx1 = _riff_chunk(b"idx1", b"".join(index))
    body = b"AVI " + hdrl + movi + idx1
    with open(out_path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", len(body)) + body)
    return {"path": out_path, "bytes": os.path.getsize(out_path), "video_chunks": n, "audio_chunks": n,
            "audio_bytes_per_chunk": max_audio, "idx1_entries": len(index)}


# ==============================================================================
# H.264 (Annex-B / AVCC) I_PCM ENCODER
# ==============================================================================


class BitWriter:
    def __init__(self):
        self.bits = []

    def u(self, n, v):
        for i in range(n - 1, -1, -1):
            self.bits.append((v >> i) & 1)

    def ue(self, v):
        v += 1
        n = v.bit_length()
        self.u(n - 1, 0)
        self.u(n, v)

    def se(self, v):
        self.ue(2 * v - 1 if v > 0 else -2 * v)

    def align_zero(self):
        while len(self.bits) % 8:
            self.bits.append(0)

    def trailing(self):
        self.bits.append(1)
        self.align_zero()

    def tobytes(self):
        out = bytearray()
        for i in range(0, len(self.bits), 8):
            b = 0
            for bit in self.bits[i:i + 8]:
                b = (b << 1) | bit
            out.append(b)
        return bytes(out)


def emulation_prevent(rbsp):
    out = bytearray()
    i = 0
    n = len(rbsp)
    pos = 0
    while True:
        j = rbsp.find(b"\x00\x00", pos)
        if j < 0 or j + 2 >= n:
            break
        if rbsp[j + 2] <= 3:
            out += rbsp[i:j + 2]
            out.append(3)
            i = j + 2
            pos = j + 2
        else:
            pos = j + 1
    out += rbsp[i:]
    return bytes(out)


def nal(nal_ref_idc, nal_type, rbsp):
    return bytes([(nal_ref_idc << 5) | nal_type]) + emulation_prevent(rbsp)


def build_sps(width, height, fps, level_idc=42):
    bw = BitWriter()
    bw.u(8, 66)                 # profile_idc: Baseline
    bw.u(8, 0xC0)               # constraint_set0 | constraint_set1 (Constrained Baseline)
    bw.u(8, level_idc)
    bw.ue(0)                    # seq_parameter_set_id
    bw.ue(0)                    # log2_max_frame_num_minus4
    bw.ue(2)                    # pic_order_cnt_type = 2 (output order == decode order)
    bw.ue(1)                    # max_num_ref_frames
    bw.u(1, 0)                  # gaps_in_frame_num_value_allowed_flag
    bw.ue(width // 16 - 1)      # pic_width_in_mbs_minus1
    bw.ue(height // 16 - 1)     # pic_height_in_map_units_minus1
    bw.u(1, 1)                  # frame_mbs_only_flag
    bw.u(1, 1)                  # direct_8x8_inference_flag
    bw.u(1, 0)                  # frame_cropping_flag
    bw.u(1, 1)                  # vui_parameters_present_flag
    bw.u(1, 0)                  # aspect_ratio_info_present_flag
    bw.u(1, 0)                  # overscan_info_present_flag
    bw.u(1, 1)                  # video_signal_type_present_flag
    bw.u(3, 5)                  # video_format: unspecified
    bw.u(1, 0)                  # video_full_range_flag: limited (16-235)
    bw.u(1, 1)                  # colour_description_present_flag
    bw.u(8, 1)                  # colour_primaries BT.709
    bw.u(8, 1)                  # transfer_characteristics BT.709
    bw.u(8, 1)                  # matrix_coefficients BT.709 (ITU-R BT.709-6)
    bw.u(1, 0)                  # chroma_loc_info_present_flag
    bw.u(1, 1)                  # timing_info_present_flag
    bw.u(32, 1)                 # num_units_in_tick
    bw.u(32, 2 * fps)           # time_scale (frame rate = time_scale / (2 * num_units_in_tick))
    bw.u(1, 1)                  # fixed_frame_rate_flag
    bw.u(1, 0)                  # nal_hrd_parameters_present_flag
    bw.u(1, 0)                  # vcl_hrd_parameters_present_flag
    bw.u(1, 0)                  # pic_struct_present_flag
    bw.u(1, 0)                  # bitstream_restriction_flag
    bw.trailing()
    return nal(3, 7, bw.tobytes())


def build_pps():
    bw = BitWriter()
    bw.ue(0)        # pic_parameter_set_id
    bw.ue(0)        # seq_parameter_set_id
    bw.u(1, 0)      # entropy_coding_mode_flag (CAVLC)
    bw.u(1, 0)      # bottom_field_pic_order_in_frame_present_flag
    bw.ue(0)        # num_slice_groups_minus1
    bw.ue(0)        # num_ref_idx_l0_default_active_minus1
    bw.ue(0)        # num_ref_idx_l1_default_active_minus1
    bw.u(1, 0)      # weighted_pred_flag
    bw.u(2, 0)      # weighted_bipred_idc
    bw.se(0)        # pic_init_qp_minus26
    bw.se(0)        # pic_init_qs_minus26
    bw.se(0)        # chroma_qp_index_offset
    bw.u(1, 1)      # deblocking_filter_control_present_flag
    bw.u(1, 0)      # constrained_intra_pred_flag
    bw.u(1, 0)      # redundant_pic_cnt_present_flag
    bw.trailing()
    return nal(3, 8, bw.tobytes())


_Y_R = [47 * i for i in range(256)]
_Y_G = [157 * i for i in range(256)]
_Y_B = [16 * i + 128 for i in range(256)]


def rgb_to_yuv420(rgb, width, height):
    """BT.709 limited-range 8-bit Y'CbCr 4:2:0 planes (values stay in 16..240, never 0)."""
    r = rgb[0::3]
    g = rgb[1::3]
    b = rgb[2::3]
    yr, yg, yb = _Y_R, _Y_G, _Y_B
    Y = bytes(((yr[rv] + yg[gv] + yb[bv]) >> 8) + 16 for rv, gv, bv in zip(r, g, b))
    cw, ch = width // 2, height // 2
    U = bytearray(cw * ch)
    V = bytearray(cw * ch)
    for cy in range(ch):
        o0 = (2 * cy) * width
        o1 = o0 + width
        for cx in range(cw):
            i0 = o0 + 2 * cx
            i1 = o1 + 2 * cx
            rs = r[i0] + r[i0 + 1] + r[i1] + r[i1 + 1]
            gs = g[i0] + g[i0 + 1] + g[i1] + g[i1 + 1]
            bs = b[i0] + b[i0 + 1] + b[i1] + b[i1 + 1]
            j = cy * cw + cx
            U[j] = ((-26 * rs - 87 * gs + 112 * bs + 512) >> 10) + 128
            V[j] = ((112 * rs - 102 * gs - 10 * bs + 512) >> 10) + 128
    return Y, bytes(U), bytes(V)


def encode_ipcm_idr(rgb, width, height, idr_pic_id):
    """One IDR slice NAL (type 5) whose every macroblock is I_PCM."""
    Y, U, V = rgb_to_yuv420(rgb, width, height)
    mbw, mbh = width // 16, height // 16
    cw = width // 2
    bw = BitWriter()
    bw.ue(0)                # first_mb_in_slice
    bw.ue(7)                # slice_type: I (all slices)
    bw.ue(0)                # pic_parameter_set_id
    bw.u(4, 0)              # frame_num (IDR)
    bw.ue(idr_pic_id)       # idr_pic_id
    bw.u(1, 0)              # no_output_of_prior_pics_flag
    bw.u(1, 0)              # long_term_reference_flag
    bw.se(0)                # slice_qp_delta
    bw.ue(1)                # disable_deblocking_filter_idc = 1 (off; I_PCM is lossless)
    bw.ue(25)               # mb_type = I_PCM (first macroblock)
    bw.align_zero()         # pcm_alignment_zero_bits
    parts = [bw.tobytes()]
    mb_hdr = b"\x0d\x00"    # ue(25) = 000011010 + 7 alignment zero bits (byte aligned MBs)
    first = True
    for my in range(mbh):
        for mx in range(mbw):
            if not first:
                parts.append(mb_hdr)
            first = False
            y0 = my * 16 * width + mx * 16
            parts.append(b"".join(Y[y0 + r * width:y0 + r * width + 16] for r in range(16)))
            c0 = my * 8 * cw + mx * 8
            parts.append(b"".join(U[c0 + r * cw:c0 + r * cw + 8] for r in range(8)))
            parts.append(b"".join(V[c0 + r * cw:c0 + r * cw + 8] for r in range(8)))
    parts.append(b"\x80")   # rbsp_slice_trailing_bits
    return nal(3, 5, b"".join(parts))


# ==============================================================================
# ISO-BMFF / MP4
# ==============================================================================


def box(fourcc, payload):
    return struct.pack(">I", 8 + len(payload)) + fourcc + payload


def fullbox(fourcc, version, flags, payload):
    return box(fourcc, struct.pack(">I", (version << 24) | flags) + payload)


_MATRIX = struct.pack(">9I", 0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000)


def _mvhd(duration_ms, next_track):
    return fullbox(b"mvhd", 0, 0, struct.pack(">IIII", 0, 0, 1000, duration_ms) + struct.pack(">IH", 0x00010000, 0x0100)
                   + b"\x00" * 10 + _MATRIX + b"\x00" * 24 + struct.pack(">I", next_track))


def _tkhd(track_id, duration_ms, width, height, audio):
    return fullbox(b"tkhd", 0, 3, struct.pack(">IIIII", 0, 0, track_id, 0, duration_ms) + b"\x00" * 8
                   + struct.pack(">hhhH", 0, 0, 0x0100 if audio else 0, 0) + _MATRIX
                   + struct.pack(">II", (width << 16) if not audio else 0, (height << 16) if not audio else 0))


def _mdhd(timescale, duration):
    return fullbox(b"mdhd", 0, 0, struct.pack(">IIIIHH", 0, 0, timescale, duration, 0x55C4, 0))


def _hdlr(handler, name):
    return fullbox(b"hdlr", 0, 0, struct.pack(">I4sIII", 0, handler, 0, 0, 0) + name + b"\x00")


def _dinf():
    return box(b"dinf", fullbox(b"dref", 0, 0, struct.pack(">I", 1) + fullbox(b"url ", 0, 1, b"")))


def _avcc(sps, pps):
    return box(b"avcC", bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1]) + struct.pack(">H", len(sps)) + sps
               + bytes([1]) + struct.pack(">H", len(pps)) + pps)


def _colr_nclx(primaries=1, transfer=1, matrix=1, full_range=0):
    return box(
        b"colr",
        b"nclx" + struct.pack(">HHH", primaries, transfer, matrix) + bytes([(1 if full_range else 0) << 7]),
    )


def _avc1(width, height, sps, pps):
    name = b"GodMode I_PCM H.264"
    comp = bytes([len(name)]) + name + b"\x00" * (31 - len(name))
    payload = (b"\x00" * 6 + struct.pack(">H", 1) + struct.pack(">HH", 0, 0) + b"\x00" * 12
               + struct.pack(">HHIIIH", width, height, 0x00480000, 0x00480000, 0, 1) + comp
               + struct.pack(">Hh", 0x0018, -1) + _avcc(sps, pps) + _colr_nclx(1, 1, 1, 0))
    return box(b"avc1", payload)


def _sowt(channels, rate):
    payload = (b"\x00" * 6 + struct.pack(">H", 1) + struct.pack(">HHI", 0, 0, 0)
               + struct.pack(">HHHHI", channels, 16, 0, 0, rate << 16))
    return box(b"sowt", payload)


def _stbl(stsd_entry, stts, stsc, sizes_box, chunk_offsets, sync=None):
    parts = [fullbox(b"stsd", 0, 0, struct.pack(">I", 1) + stsd_entry),
             fullbox(b"stts", 0, 0, struct.pack(">I", len(stts)) + b"".join(struct.pack(">II", c, d) for c, d in stts))]
    if sync is not None:
        parts.append(fullbox(b"stss", 0, 0, struct.pack(">I", len(sync)) + b"".join(struct.pack(">I", s) for s in sync)))
    parts.append(fullbox(b"stsc", 0, 0, struct.pack(">I", len(stsc)) + b"".join(struct.pack(">III", *e) for e in stsc)))
    parts.append(sizes_box)
    parts.append(fullbox(b"stco", 0, 0, struct.pack(">I", len(chunk_offsets))
                         + b"".join(struct.pack(">I", o) for o in chunk_offsets)))
    return box(b"stbl", b"".join(parts))


def mux_mp4(frames, width, height, fps, pcm, rate, channels, out_path):
    if width % 16 or height % 16:
        raise ValueError("I_PCM H.264 muxer requires width/height multiples of 16")
    n = len(frames)
    block_align = channels * 2
    sps = build_sps(width, height, fps)
    pps = build_pps()
    audio = _audio_slices(pcm, n, fps, rate, block_align)
    ftyp = box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2avc1mp41")

    samples = []
    for i, rgb in enumerate(frames):
        idr = encode_ipcm_idr(rgb, width, height, i & 1)
        samples.append(struct.pack(">I", len(idr)) + idr)          # AVCC 4-byte length prefix

    mdat_payload_len = sum(len(s) for s in samples) + sum(len(a) for a in audio)
    mdat_header = struct.pack(">I", 8 + mdat_payload_len) + b"mdat"
    offset = len(ftyp) + len(mdat_header)
    v_offsets, a_offsets = [], []
    for i in range(n):
        v_offsets.append(offset)
        offset += len(samples[i])
        a_offsets.append(offset)
        offset += len(audio[i])

    v_timescale = fps * 512
    v_duration = n * 512
    dur_ms = int(round(n * 1000 / fps))
    total_audio = sum(len(a) for a in audio) // block_align

    v_stbl = _stbl(_avc1(width, height, sps, pps), [(n, 512)], [(1, 1, 1)],
                   fullbox(b"stsz", 0, 0, struct.pack(">II", 0, n) + b"".join(struct.pack(">I", len(s)) for s in samples)),
                   v_offsets, sync=list(range(1, n + 1)))
    v_trak = box(b"trak", _tkhd(1, dur_ms, width, height, False) + box(b"mdia", _mdhd(v_timescale, v_duration)
                 + _hdlr(b"vide", b"GodMode Diorama Video") + box(b"minf", fullbox(b"vmhd", 0, 1, b"\x00" * 8)
                 + _dinf() + v_stbl)))
    per_chunk = len(audio[0]) // block_align
    stsc = [(1, per_chunk, 1)]
    if any(len(a) // block_align != per_chunk for a in audio):
        stsc = [(i + 1, len(a) // block_align, 1) for i, a in enumerate(audio)]
    a_stbl = _stbl(_sowt(channels, rate), [(total_audio, 1)], stsc,
                   fullbox(b"stsz", 0, 0, struct.pack(">II", block_align, total_audio)), a_offsets)
    a_trak = box(b"trak", _tkhd(2, dur_ms, 0, 0, True) + box(b"mdia", _mdhd(rate, total_audio)
                 + _hdlr(b"soun", b"GodMode Diorama Soundtrack") + box(b"minf", fullbox(b"smhd", 0, 0, b"\x00" * 4)
                 + _dinf() + a_stbl)))
    moov = box(b"moov", _mvhd(dur_ms, 3) + v_trak + a_trak)

    with open(out_path, "wb") as fh:
        fh.write(ftyp)
        fh.write(mdat_header)
        for i in range(n):
            fh.write(samples[i])
            fh.write(audio[i])
        fh.write(moov)
    return {"path": out_path, "bytes": os.path.getsize(out_path), "video_samples": n,
            "audio_pcm_frames": total_audio, "codec": "avc1 (H.264 Constrained Baseline, I_PCM IDR)",
            "audio_codec": "sowt (PCM s16le 48 kHz stereo)", "sps_hex": sps.hex(), "pps_hex": pps.hex()}


# ==============================================================================
# DRIVER
# ==============================================================================


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def ffmpeg_transcode(src_path, wav_path, out_path):
    exe = shutil.which("ffmpeg")
    if exe is None:
        return {"available": False}
    cmd = [
        exe, "-y", "-v", "error", "-i", src_path,
        "-vf", "scale=out_color_matrix=bt709:out_range=tv",
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709",
        "-color_trc", "bt709", "-color_range", "tv",
        "-x264-params", "colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    ok = proc.returncode == 0 and os.path.isfile(out_path)
    return {"available": True, "ok": ok, "path": out_path if ok else None,
            "bytes": os.path.getsize(out_path) if ok else 0, "stderr": proc.stderr[-400:]}


def mux_all(frames, width, height, fps, wav_path, out_dir):
    pcm, rate, channels, _sw = load_wav(wav_path)
    avi_path = os.path.join(out_dir, AVI_NAME)
    mp4_path = os.path.join(out_dir, MP4_NAME)
    print(f"[mux] RIFF/AVI 1.0 -> {AVI_NAME}", flush=True)
    avi = mux_avi(frames, width, height, fps, pcm, rate, channels, avi_path)
    print(f"[mux] ISO-BMFF H.264 I_PCM + PCM -> {MP4_NAME}", flush=True)
    mp4 = mux_mp4(frames, width, height, fps, pcm, rate, channels, mp4_path)
    avi["sha256"] = _sha256_file(avi_path)
    mp4["sha256"] = _sha256_file(mp4_path)
    print("[mux] optional ffmpeg H.264+AAC transcode", flush=True)
    ff = ffmpeg_transcode(avi_path, wav_path, os.path.join(out_dir, MP4_FFMPEG_NAME))
    if ff.get("ok"):
        ff["sha256"] = _sha256_file(ff["path"])
    return {"avi": avi, "mp4": mp4, "ffmpeg_h264_aac": ff, "fps": fps, "frames": len(frames),
            "audio": {"rate": rate, "channels": channels, "bytes": len(pcm)}}


def main():
    cache = os.path.join(SCRIPT_DIR, FRAME_CACHE_NAME)
    width, height, frames = read_frame_cache(cache)
    rep = mux_all(frames, width, height, 24, os.path.join(SCRIPT_DIR, WAV_NAME), SCRIPT_DIR)
    print(json.dumps({k: v for k, v in rep.items() if k != "ffmpeg_h264_aac"}, indent=2)[:2000])


if __name__ == "__main__":
    main()
