import time
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import uuid

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from PIL import Image
import base64
import io as _io
import folder_paths
import av
import node_helpers

from .herrgotts_bridge.latent_math import (
    FRAME_RESCALE as _HG_FRAME_RESCALE,
    phase_aligned_extended_context_slice as _hg_phase_aligned_extended_context_slice,
    audio_slice_for_pixel_window as _hg_audio_slice_for_pixel_window,
)
from .herrgotts_bridge.patch_layout import (
    HC_INDEX as _HG_HC_INDEX,
    HC_AUDIO_END_FRAME as _HG_HC_AUDIO_END_FRAME,
    LEGACY_LAYOUT_MODE as _HG_LEGACY_LAYOUT_MODE,
    NATIVE_LAYOUT_MODE as _HG_NATIVE_LAYOUT_MODE,
)
from .herrgotts_bridge.runtime_patches import ensure_h3_runtime_patches as _hg_ensure_runtime_patches
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview
from comfy_api.latest import io
from . import minimax_plan as _ref_plan
from . import minimax_media as _ref_media
from .minimax_core import core as _ref_core



class _DHanUnconnected:
    """Empty socket sentinel for reference-style lazy H3 model resolution."""
    def __repr__(self):
        return "<unconnected>"


_DHan_UNCONNECTED = _DHanUnconnected()

log = logging.getLogger(__name__)

_CACHE = {}

# Keep only the most recent native FL2VA conditioning result.  The Qwen3-VL
# encoder is large and can take a long time on CPU; ComfyUI's native H3 node
# recomputes it every execution.  DHan frequently re-executes because timeline
# widget state changes, even when the compiled H3 conditioning did not.
_DHan_H3_TEXT_CACHE = {}
_DHan_H3_VAE_CACHE = {}

def _dhan_tensor_digest(t):
    if t is None:
        return "none"
    x = t.detach().to(device="cpu", dtype=torch.float16).contiguous()
    h = hashlib.sha1()
    h.update(str(tuple(x.shape)).encode("ascii"))
    h.update(x.numpy().tobytes())
    return h.hexdigest()

def _dhan_fl2va_native_timed(mm, clip, vae, prompt, width, height, length, first_frame=None, last_frame=None):
    """Native ComfyUI FL2VA conditioning with internal timing + content-keyed caches.

    This mirrors comfy_extras.nodes_minimax_h3.MiniMaxH3ImageToVideo.execute;
    it does not alter the conditioning math. Cache keys intentionally avoid
    transient ComfyUI CLIP/VAE wrapper object identities so identical prompt
    and keyframe content can be reused across Director re-executions.
    """
    t0 = time.perf_counter()
    def mark(name, prev):
        now = time.perf_counter()
        log.info("[Comfyui-DHan-H3 Cond] %-22s +%.3fs (total %.3fs)", name, now-prev, now-t0)
        return now

    latent, frame_count = mm._empty_av_latent(width, height, length)
    t = mark("empty AV latent", t0)

    images = []
    keyframes = []
    if first_frame is not None:
        img = mm._resize(first_frame[:1], width, height, "disabled")
        images.append(img)
        keyframes.append({"resolved_frame_index": 0, "image": img})
    if last_frame is not None:
        img = mm._resize(last_frame[:1], width, height, "center")
        images.append(img)
        keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})
    t = mark("keyframes resized", t)

    image_sig = tuple(_dhan_tensor_digest(x) for x in images)
    text_key = (str(prompt), image_sig)
    cond = _DHan_H3_TEXT_CACHE.get(text_key)
    if cond is None:
        tokens = clip.tokenize(prompt, images=images)
        t = mark("Qwen tokenize", t)
        cond = clip.encode_from_tokens_scheduled(tokens)
        t = mark("Qwen encode", t)
        _DHan_H3_TEXT_CACHE.clear()
        _DHan_H3_TEXT_CACHE[text_key] = cond
        log.info("[Comfyui-DHan-H3 Cond] Qwen conditioning cache MISS")
    else:
        t = mark("Qwen cache HIT", t)

    if keyframes:
        cooked = []
        for kf, sig in zip(keyframes, image_sig):
            vae_key = sig
            z = _DHan_H3_VAE_CACHE.get(vae_key)
            if z is None:
                z = vae.encode(kf["image"])
                _DHan_H3_VAE_CACHE.clear()
                _DHan_H3_VAE_CACHE[vae_key] = z
                log.info("[Comfyui-DHan-H3 Cond] keyframe VAE cache MISS")
            else:
                log.info("[Comfyui-DHan-H3 Cond] keyframe VAE cache HIT")
            cooked.append({"resolved_frame_index": kf["resolved_frame_index"], "latent": z})
        t = mark("keyframe VAE encode", t)
        cond = node_helpers.conditioning_set_values(cond, {"minimax_keyframes": cooked})
        t = mark("conditioning attach", t)

    return io.NodeOutput(cond, latent)


def _extra(module_name, probe_attr=None):
    if module_name in _CACHE:
        return _CACHE[module_name]
    suffix = "comfy_extras/" + module_name
    for name, mod in list(sys.modules.items()):
        if mod is not None and str(name).replace("\\", "/").endswith(suffix):
            if probe_attr is None or hasattr(mod, probe_attr):
                _CACHE[module_name] = mod
                return mod
    import nodes as comfy_nodes
    path = os.path.join(os.path.dirname(os.path.realpath(comfy_nodes.__file__)), "comfy_extras", module_name + ".py")
    if not os.path.exists(path):
        raise ImportError(f"Comfyui-DHan-Minimax H3 Director requires a current ComfyUI. Missing: {path}")
    spec = importlib.util.spec_from_file_location("_dhan_" + module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[module_name] = mod
    return mod


def _core():
    return _extra("nodes_minimax_h3", "MiniMaxH3ImageToVideo")


def _unpack(out):
    vals = getattr(out, "args", None) or getattr(out, "result", None)
    if vals is None:
        if isinstance(out, (tuple, list)):
            return tuple(out)
        return (out,)
    return tuple(vals)



def _h3_latent_streams(latent):
    """Return H3 video/audio tensors from a LATENT dict."""
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Comfyui-DHan-H3 continuation expects an H3 LATENT dict with 'samples'.")

    samples = latent["samples"]
    if hasattr(samples, "tensors"):
        parts = list(samples.tensors)
    elif hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(f"Comfyui-DHan-H3 continuation: unsupported latent container {type(samples)!r}.")

    if len(parts) < 2:
        raise ValueError("Comfyui-DHan-H3 continuation requires the joint H3 video+audio latent.")

    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)

    if video.ndim != 5 or audio.ndim != 4 or video.shape[1] != 24:
        raise ValueError(
            "Comfyui-DHan-H3 continuation: unexpected AV latent shapes "
            f"video={tuple(video.shape)}, audio={tuple(audio.shape)}."
        )
    return video, audio


def _h3_context_geometry(context_frames):
    """
    H3 guide clips live on the 17k+5 pixel-frame grid.
    5 / 22 / 39 / 56... frames correspond to 2 / 7 / 12 / 17... video latent steps.
    """
    requested = max(5, int(context_frames))
    valid = 5
    while valid + 17 <= requested:
        valid += 17
    video_steps = 2 if valid <= 5 else ((valid - 5) // 17) * 5 + 2
    audio_steps = max(1, round(valid / 24.0 * 40.0))
    return valid, video_steps, audio_steps


def _apply_h3_latent_continuity(
    positive, target_latent, previous_latent, context_frames=22,
    replace_frame0_keyframe=False
):
    """
    Inject the previous sampled H3 latent tail directly at frame 0 of the next window.

    When replace_frame0_keyframe=True, any existing frame-0 image LATENT keyframe is
    removed before the continuation tail is attached. The source image still remains
    inside Qwen's multimodal conditioning, so it acts as a fresh visual identity/style
    reference without fighting the motion-continuity latent at the same frame index.
    """
    prev_video, prev_audio = _h3_latent_streams(previous_latent)
    target_video, target_audio = _h3_latent_streams(target_latent)

    if tuple(prev_video.shape[-2:]) != tuple(target_video.shape[-2:]):
        raise ValueError(
            "Comfyui-DHan-H3 latent continuation requires identical render resolution. "
            f"Previous latent grid={tuple(prev_video.shape[-2:])}, "
            f"next grid={tuple(target_video.shape[-2:])}."
        )

    pixel_frames, video_steps, audio_steps = _h3_context_geometry(context_frames)
    if prev_video.shape[2] < video_steps:
        raise ValueError(
            f"Comfyui-DHan-H3 continuation requested {pixel_frames} context frames "
            f"({video_steps} latent steps), but the previous latent only has "
            f"{prev_video.shape[2]} video latent steps."
        )

    video_tail = prev_video[:1, :, -video_steps:].clone()
    audio_steps = min(int(audio_steps), int(prev_audio.shape[-1]), int(target_audio.shape[-1]))
    audio_tail = prev_audio[:1, ..., -audio_steps:].clone() if audio_steps > 0 else None

    keyframes = list(positive[0][1].get("minimax_keyframes", []))
    if replace_frame0_keyframe:
        keyframes = [
            kf for kf in keyframes
            if int(kf.get("resolved_frame_index", -1)) != 0
        ]

    continuity = {
        "resolved_frame_index": 0,
        "latent": video_tail,
    }
    if audio_tail is not None:
        continuity["audio_latent"] = audio_tail
    keyframes.append(continuity)

    positive = node_helpers.conditioning_set_values(
        positive,
        {"minimax_keyframes": keyframes}
    )
    return positive, pixel_frames



def _sample_h3_no_preview(noise, guider, sampler, sigmas, latent_image):
    """
    Auto Render sampler with native Comfy progress only.
    Keeps the thin green Director progress bar while sending no preview image.
    """
    latent = latent_image
    samples_in = latent["samples"]
    latent = latent.copy()

    samples_in = comfy.sample.fix_empty_latent_channels(
        guider.model_patcher,
        samples_in,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )
    latent["samples"] = samples_in

    noise_mask = latent.get("noise_mask", None)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    pbar = comfy.utils.ProgressBar(max(1, len(sigmas) - 1))

    def progress_only_callback(step, x0, x, total_steps):
        pbar.update_absolute(step + 1, total_steps, None)

    samples = guider.sample(
        noise.generate_noise(latent),
        samples_in,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=progress_only_callback,
        disable_pbar=disable_pbar,
        seed=noise.seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _concat_h3_sampled_latents(sampled_latents, trim_context_frames, target_length, mm):
    """
    Join sampled H3 AV latent windows. For windows after the first, remove the
    directly-carried continuity context, then crop the packed AV streams to the
    model-valid temporal shape for the requested full timeline.
    """
    if not sampled_latents:
        raise ValueError("Comfyui-DHan-H3 Auto Render produced no sampled windows.")

    videos = []
    audios = []
    for i, latent in enumerate(sampled_latents):
        v, a = _h3_latent_streams(latent)
        if i > 0:
            # H3 video/audio temporal grids do not divide evenly:
            # 22 frames / 24fps * 40Hz = 36.666... audio latent steps.
            # Carrying uses the rounded value (37) for continuity, but trimming
            # 37 at every seam loses ~1 audio step per continuation window.
            # Trim FLOOR timing instead; the final crop below removes the tiny
            # retained overlap without creating an audio gap.
            pixel_ctx, vtrim, _ = _h3_context_geometry(trim_context_frames)
            atrim = max(0, int(math.floor(float(pixel_ctx) / 24.0 * 40.0)))
            v = v[:, :, min(vtrim, v.shape[2]):]
            a = a[..., min(atrim, a.shape[-1]):]
        videos.append(v)
        audios.append(a)

    video = torch.cat(videos, dim=2)
    audio = torch.cat(audios, dim=-1)

    frame_count, target_video_t, target_audio_t = mm.temporal_shape(int(target_length))
    if video.shape[2] < target_video_t:
        raise ValueError(
            "Comfyui-DHan-H3 Auto Render: stitched video latent is shorter than the requested "
            f"timeline shape ({video.shape[2]}/{target_video_t})."
        )

    # Residual AV-grid rounding can leave audio short by a tiny number of
    # latent steps across several seams. Pad only that fractional residue by
    # repeating the final audio latent step. Larger shortages remain errors.
    audio_short = int(target_audio_t) - int(audio.shape[-1])
    if audio_short > 0:
        if audio_short > 4 or audio.shape[-1] < 1:
            raise ValueError(
                "Comfyui-DHan-H3 Auto Render: stitched audio latent is materially shorter than "
                f"the requested timeline shape ({audio.shape[-1]}/{target_audio_t})."
            )
        log.info(
            "[Comfyui-DHan-H3 Long Sampler] Padding %d audio latent step(s) for AV-grid rounding.",
            audio_short,
        )
        tail = audio[..., -1:].expand(*audio.shape[:-1], audio_short)
        audio = torch.cat((audio, tail), dim=-1)

    video = video[:, :, :target_video_t].contiguous()
    audio = audio[..., :target_audio_t].contiguous()
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio)),
        "dhan_h3_frame_count": int(frame_count),
    }


def _dhan_standard_sampler(name="res_multistep"):
    custom = _extra("nodes_custom_sampler", "KSamplerSelect")
    return _unpack(custom.KSamplerSelect.execute(str(name)))[0]


def _dhan_turbo_sampler():
    """
    Use Larryvrh/ComfyUI-MiniMax-H3-Turbo's registered sampler node.
    We intentionally resolve through ComfyUI NODE_CLASS_MAPPINGS so this package
    does not hard-import another custom-node folder by filesystem name.
    """
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("MiniMaxH3TurboSampler")
    if cls is None:
        raise ValueError(
            "Comfyui-DHan-Minimax H3 Settings: Turbo preset requires "
            "'MiniMax-H3 Turbo Sampler (4-step)' from "
            "Larryvrh/ComfyUI-MiniMax-H3-Turbo. Install/update that custom node "
            "or switch sampling_preset to Standard."
        )
    obj = cls()
    if hasattr(obj, "get_sampler"):
        result = obj.get_sampler()
    else:
        fn = getattr(obj, getattr(obj, "FUNCTION", "get_sampler"))
        result = fn()
    return _unpack(result)[0]


def _dhan_sampler_for_settings(settings):
    mode = str(settings.get("sampler_mode", "standard"))
    if mode == "turbo":
        return _dhan_turbo_sampler()
    return _dhan_standard_sampler(str(settings.get("sampler_name", "res_multistep")))

def _snap32(v):
    """Snap a positive dimension down to a multiple of 32. 0 is reserved for AUTO."""
    v = int(v)
    if v <= 0:
        return 0
    return max(32, (v // 32) * 32)


def _resolve_input_path(seg: dict):
    filename = seg.get("imageFile", "")
    if not filename:
        return None
    subfolder = seg.get("imageSubfolder", "") or ""
    p = os.path.join(folder_paths.get_input_directory(), subfolder, filename)
    if os.path.exists(p):
        return p
    p2 = os.path.join(folder_paths.get_input_directory(), filename)
    return p2 if os.path.exists(p2) else None


def _load_image_tensor(seg: dict) -> torch.Tensor:
    path = _resolve_input_path(seg)
    if path:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if b64_str and not b64_str.startswith("/view?"):
        try:
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            img = Image.open(_io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).unsqueeze(0)
        except Exception:
            pass
    raise FileNotFoundError("Timeline image could not be resolved.")


def _resize_image(tensor: torch.Tensor, target_w: int, target_h: int, method: str, divisible_by: int) -> torch.Tensor:
    def snap(v, d):
        return max(d, (int(v) // d) * d)

    tw = snap(target_w, divisible_by)
    th = snap(target_h, divisible_by)
    n, h, w, c = tensor.shape
    if h == th and w == tw:
        return tensor

    t = tensor.permute(0, 3, 1, 2)
    if method == "stretch to fit":
        out = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)
    elif method == "maintain aspect ratio":
        ratio = min(tw / w, th / h)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        out = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    elif method in ("pad", "pad green"):
        ratio = min(tw / w, th / h)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        inner = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        left, top = (tw - nw) // 2, (th - nh) // 2
        if method == "pad green":
            out = torch.zeros((n, c, th, tw), dtype=t.dtype, device=t.device)
            out[:, 0, :, :] = 102 / 255.0
            out[:, 1, :, :] = 1.0
            out[:, :, top:top+nh, left:left+nw] = inner
        else:
            out = F.pad(inner, (left, tw-nw-left, top, th-nh-top), mode="constant", value=0)
    else:  # crop
        ratio = max(tw / w, th / h)
        nw, nh = max(1, int(w * ratio)), max(1, int(h * ratio))
        inner = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        left, top = max(0, (nw - tw)//2), max(0, (nh - th)//2)
        out = inner[:, :, top:top+th, left:left+tw]
    return out.permute(0, 2, 3, 1)




def _first_timeline_image_size(tdata):
    """Return (width, height) of the first resolvable DHan timeline image."""
    for seg in sorted(tdata.get("segments", []), key=lambda s: float(s.get("start", 0))):
        if seg.get("type", "image") != "image":
            continue
        if not (seg.get("imageFile") or seg.get("imageB64")):
            continue
        try:
            image = _load_image_tensor(seg)
            if image is not None and image.ndim == 4:
                return int(image.shape[2]), int(image.shape[1])
        except Exception:
            continue
    return None


def _resolve_generation_size(tdata, requested_w, requested_h, divisible_by=32):
    """Mirror the tuned DHan Director dimension rules exactly."""
    rw = max(0, int(requested_w or 0))
    rh = max(0, int(requested_h or 0))
    src = _first_timeline_image_size(tdata)
    sw, sh = src if src else (1344, 768)

    def snap(val, div):
        return max(div, (int(val) // div) * div)

    if rw > 0 and rh > 0:
        return snap(rw, divisible_by), snap(rh, divisible_by)
    if rw > 0:
        tgt_w = snap(rw, divisible_by)
        tgt_h = snap(int(sh * tgt_w / sw), divisible_by)
        return tgt_w, tgt_h
    if rh > 0:
        tgt_h = snap(rh, divisible_by)
        tgt_w = snap(int(sw * tgt_h / sh), divisible_by)
        return tgt_w, tgt_h
    return snap(sw, divisible_by), snap(sh, divisible_by)



def _resolve_uploaded_file(value: str):
    if not value:
        return None
    value = str(value).replace("\\", "/").lstrip("/")
    p = os.path.join(folder_paths.get_input_directory(), *value.split("/"))
    if os.path.exists(p):
        return p
    p2 = os.path.join(folder_paths.get_input_directory(), os.path.basename(value))
    return p2 if os.path.exists(p2) else None


def _load_ref_video_tensor(seg: dict, target_fps: float = 24.0):
    """Decode the trimmed DHan REF VIDEO selection to [N,H,W,3] float frames."""
    path = _resolve_uploaded_file(seg.get("videoFile") or seg.get("imageFile") or "")
    if not path:
        raise FileNotFoundError("Reference video file could not be resolved.")

    trim_start_frames = max(0, int(round(float(seg.get("trimStart", 0) or 0))))
    wanted_frames = max(1, min(360, int(round(float(seg.get("length", 1) or 1)))))
    start_sec = trim_start_frames / max(1.0, float(target_fps))

    decoded = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base:
            try:
                container.seek(int(max(0.0, start_sec - 0.5) / float(stream.time_base)),
                               stream=stream, backward=True)
            except Exception:
                pass

        next_time = start_sec
        frame_step = 1.0 / max(1.0, float(target_fps))
        for frame in container.decode(stream):
            t = frame.time
            if t is None and frame.pts is not None and stream.time_base:
                t = float(frame.pts * stream.time_base)
            if t is None or t + 1e-4 < start_sec or t + 1e-4 < next_time:
                continue
            decoded.append(frame.to_ndarray(format="rgb24"))
            next_time += frame_step
            if len(decoded) >= wanted_frames:
                break

    if not decoded:
        raise RuntimeError("Reference video decoded zero frames.")
    return torch.from_numpy(np.stack(decoded).astype(np.float32) / 255.0)



def _load_ref_timeline_image(seg: dict):
    """Load a static image stored on the Ref2VA REF lane."""
    path = _resolve_uploaded_file(seg.get("videoFile") or seg.get("imageFile") or "")
    if path:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    b64_str = seg.get("imageB64", "")
    if b64_str and not b64_str.startswith("/view?"):
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img = Image.open(_io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    raise FileNotFoundError("Ref2VA timeline image could not be resolved.")


def _timeline_ref_images(tdata, max_images=9):
    """Static REF-lane images, ordered left-to-right."""
    refs = []
    motion = tdata.get("motionSegments", []) if isinstance(tdata, dict) else []
    for seg in sorted(motion, key=lambda s: float(s.get("start", 0))):
        if len(refs) >= max_images:
            break
        if not seg.get("isStaticImage"):
            continue
        try:
            refs.append(_load_ref_timeline_image(seg))
        except Exception as e:
            log.warning("[Comfyui-DHan-H3] Could not load Ref2VA REF image; skipping: %s", e)
    return refs


def _load_ref_audio(seg: dict):
    """Load a DHan audio timeline segment into ComfyUI's AUDIO dictionary format."""
    value = seg.get("audioFile") or seg.get("fileName") or ""
    path = _resolve_uploaded_file(value)
    if not path:
        raise FileNotFoundError(f"Reference audio file not found: {value}")

    waveform, sample_rate = torchaudio.load(path)
    # Comfy AUDIO is [B, C, samples].
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    return {
        "waveform": waveform,
        "sample_rate": int(sample_rate),
    }


def _timeline_ref_audios(tdata, max_audios=3):
    """
    Ref2VA standalone audio references.
    Left-to-right timeline order maps to <Audio 1>, <Audio 2>, <Audio 3>.
    Timeline position is organizational; native Ref2VA treats these as reference
    media rather than frame-anchored audio guides.
    """
    refs = {}
    segments = tdata.get("audioSegments", []) if isinstance(tdata, dict) else []
    for seg in sorted(segments, key=lambda s: float(s.get("start", 0))):
        if len(refs) >= max_audios:
            break
        if seg.get("type") not in ("audio", None):
            continue
        if not (seg.get("audioFile") or seg.get("fileName")):
            continue
        try:
            refs[f"ref_audio_{len(refs)+1}"] = _load_ref_audio(seg)
        except Exception as e:
            log.warning("[Comfyui-DHan-H3] Could not load Ref2VA REF AUDIO; skipping: %s", e)
    return refs or None

def _timeline_ref_videos(tdata, fps=24.0, max_videos=3):
    """Secondary DHan track -> native H3 ref_video_1..3 dictionary."""
    refs = {}
    motion = tdata.get("motionSegments", []) if isinstance(tdata, dict) else []
    for seg in sorted(motion, key=lambda s: float(s.get("start", 0))):
        if len(refs) >= max_videos:
            break
        if seg.get("type") != "motion_video" or seg.get("isStaticImage"):
            continue
        if not (seg.get("videoFile") or seg.get("imageFile")):
            continue
        try:
            clip = _load_ref_video_tensor(seg, fps)
            if clip.shape[0] < 5:
                log.warning("[Comfyui-DHan-H3] REF VIDEO has fewer than 5 decoded frames; skipping.")
                continue
            refs[f"ref_video_{len(refs)+1}"] = clip
        except Exception as e:
            log.warning("[Comfyui-DHan-H3] Could not load Ref2VA REF VIDEO; skipping: %s", e)
    return refs or None

def _fit_image(tensor, width, height, resize_method, divisible_by):
    if tensor is None:
        return None
    return _resize_image(tensor, width, height, resize_method, divisible_by)


def _compile_storyboard(tdata, global_prompt, start_frame, end_frame, fps):
    chunks = []
    gp = (global_prompt or tdata.get("global_prompt", "") or "").strip()
    if gp:
        chunks.append(gp)

    segs = sorted(tdata.get("segments", []), key=lambda s: float(s.get("start", 0)))
    for seg in segs:
        prompt = (seg.get("prompt") or "").strip()
        if not prompt:
            continue
        s0 = float(seg.get("start", 0))
        s1 = s0 + max(1.0, float(seg.get("length", 1)))
        if s1 <= start_frame or s0 >= end_frame:
            continue
        a = max(s0, start_frame)
        b = min(s1, end_frame)
        rel_a = max(0.0, (a - start_frame) / fps)
        rel_b = max(rel_a, (b - start_frame) / fps)
        chunks.append(f"[{rel_a:.2f}s-{rel_b:.2f}s] {prompt}")

    return "\n".join(chunks).strip()


def _timeline_images(tdata, start_frame, end_frame, fps, width, height, resize_method, divisible_by):
    events = []
    for seg in sorted(tdata.get("segments", []), key=lambda s: float(s.get("start", 0))):
        if seg.get("type", "image") != "image":
            continue
        if not (seg.get("imageFile") or seg.get("imageB64")):
            continue
        s0 = int(round(float(seg.get("start", 0))))
        s1 = s0 + max(1, int(round(float(seg.get("length", 1)))))
        if s1 <= start_frame or s0 >= end_frame:
            continue
        try:
            image = _load_image_tensor(seg)
            image = _fit_image(image, width, height, resize_method, divisible_by)
            events.append({
                "image": image,
                "start": s0,
                "end": s1,
                "is_end": bool(seg.get("isEndFrame", False)),
                "prompt": seg.get("prompt", ""),
            })
        except Exception as e:
            log.warning("[Comfyui-DHan-H3] Could not load timeline image; image guide will be skipped: %s", e)
    return events



H3Settings = io.Custom("DHan_H3_SETTINGS")
H3Refs = io.Custom("DHan_H3_REFS")
H3Window = io.Custom("DHan_H3_WINDOW")
H3Plan = io.Custom("DHan_H3_TIMELINE_PLAN")


class DHanMiniMaxH3SmartSampler(io.ComfyNode):
    """
    Drop-in H3 sampler for the Director workflow.

    Short timelines: behaves like ComfyUI SamplerCustomAdvanced.
    Long Auto Render timelines: Director already sampled/chained the full latent,
    so this node passes it through instead of sampling the finished latent again.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanMiniMaxH3SmartSampler",
            display_name="Comfyui-DHan-Minimax H3 Smart Sampler",
            category="Comfyui-DHan/Minimax H3 Director",
            description=(
                "Use in place of SamplerCustomAdvanced. <=15s H3 jobs sample normally. "
                "15s+ Director Auto Render results are detected and passed through so "
                "the completed long latent is never sampled twice."
            ),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Model.Input("model", tooltip="Patched H3 model output from Director/Preview Override."),
                H3Settings.Input("h3_settings"),
                io.Latent.Input("latent_image"),
                io.Sigmas.Input("sigmas", optional=True),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ],
        )

    @classmethod
    def execute(cls, noise, guider, sampler, model, h3_settings, latent_image, sigmas=None):
        if isinstance(latent_image, dict) and latent_image.get("dhan_h3_already_sampled", False):
            out = latent_image.copy()
            # The marker is useful for debugging but should not affect decode nodes.
            log.info(
                "[Comfyui-DHan-H3 Smart Sampler] Auto Render latent already sampled (%s frames); bypassing sampler.",
                out.get("dhan_h3_project_length", "?"),
            )
            return io.NodeOutput(out, out)

        log.info("[Comfyui-DHan-H3 Smart Sampler] Normal H3 latent; running one visible sampling pass.")

        if sigmas is None:
            settings = h3_settings if isinstance(h3_settings, dict) else {}
            scheduler = str(settings.get("scheduler", "beta"))
            steps = int(settings.get("steps", 20))
            sched = _extra("nodes_custom_sampler", "BasicScheduler")
            sigmas = _unpack(sched.BasicScheduler.execute(
                model,
                scheduler,
                steps,
                1.0,
            ))[0]

        custom = _extra("nodes_custom_sampler", "SamplerCustomAdvanced")
        result = custom.SamplerCustomAdvanced.execute(
            noise, guider, sampler, sigmas, latent_image
        )
        vals = _unpack(result)
        if len(vals) >= 2:
            return io.NodeOutput(vals[0], vals[1])
        return io.NodeOutput(vals[0], vals[0])



class DHanH3Guider(io.ComfyNode):
    """Automatic Basic/CFG guider for MiniMax H3."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanH3Guider",
            display_name="Comfyui-DHan-H3 Guider",
            category="Comfyui-DHan/Minimax H3 Director",
            description=(
                "Reads the Director's Negative Prompt switch automatically. "
                "OFF = BasicGuider. ON = CFGGuider using Director.negative."
            ),
            inputs=[
                io.Model.Input("model", display_name="model"),
                io.Conditioning.Input("positive", display_name="positive"),
                io.Conditioning.Input("negative", display_name="negative", optional=True),
                io.Float.Input(
                    "cfg", default=2.0, min=1.0, max=20.0, step=0.1,
                    tooltip=(
                        "Automatically used when the Director Negative Prompt switch is ON. "
                        "Ignored while OFF. Default: 2.0."
                    )
                ),
            ],
            outputs=[io.Guider.Output(display_name="guider")],
        )

    @classmethod
    def execute(cls, model, positive, negative=None, cfg=2.0):
        custom = _extra("nodes_custom_sampler")
        enabled = False
        has_text = False

        if negative is not None:
            try:
                meta = negative[0][1]
                enabled = bool(meta.get("dhan_negative_prompting", False))
                has_text = bool(meta.get("dhan_negative_has_text", False))
            except Exception:
                pass

        use_cfg = bool(enabled and has_text)
        if use_cfg:
            guider = _unpack(custom.CFGGuider.execute(
                model, positive, negative, float(cfg)
            ))[0]
            log.info(
                "[Comfyui-DHan-H3 Guider] Director Negative Prompt ON -> CFG mode, cfg=%.2f.",
                float(cfg)
            )
        else:
            guider = _unpack(custom.BasicGuider.execute(model, positive))[0]
            if enabled and not has_text:
                log.info("[Comfyui-DHan-H3 Guider] Negative Prompt enabled but empty -> BasicGuider.")
            else:
                log.info("[Comfyui-DHan-H3 Guider] Director Negative Prompt OFF -> BasicGuider.")

        try:
            setattr(guider, "_dhan_h3_guider_meta", {
                "negative_prompting": use_cfg,
                "negative": negative,
                "cfg": float(cfg),
            })
        except Exception:
            pass

        return io.NodeOutput(guider)


class DHanH3SamplingPreset(io.ComfyNode):
    """Downstream H3 sampler/sigma preset. Connect Director.model here."""

    PRESETS = {
        "Standard": {
            "sampler_name": "res_multistep",
            "schedule_mode": "basic",
            "scheduler": "beta",
            "steps": 20,
            "alpha": 0.79,
            "beta": 0.50,
            "extend_steps": 0,
            "extend_start_sigma": 0.8,
            "extend_end_sigma": 0.0,
            "extend_spacing": "linear",
        },
        "Turbo 4-Step": {
            "sampler_name": "euler",
            "schedule_mode": "beta",
            "scheduler": "beta",
            "steps": 4,
            "alpha": 0.79,
            "beta": 0.50,
            "extend_steps": 0,
            "extend_start_sigma": 0.8,
            "extend_end_sigma": 0.0,
            "extend_spacing": "linear",
        },
        "Turbo 6-Step": {
            "sampler_name": "euler",
            "schedule_mode": "beta",
            "scheduler": "beta",
            "steps": 6,
            "alpha": 0.79,
            "beta": 0.50,
            "extend_steps": 0,
            "extend_start_sigma": 0.8,
            "extend_end_sigma": 0.0,
            "extend_spacing": "linear",
        },
        "Turbo 8-Step": {
            "sampler_name": "euler",
            "schedule_mode": "beta",
            "scheduler": "beta",
            "steps": 8,
            "alpha": 0.79,
            "beta": 0.50,
            "extend_steps": 0,
            "extend_start_sigma": 0.8,
            "extend_end_sigma": 0.0,
            "extend_spacing": "linear",
        },
    }

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanH3SamplingPreset",
            display_name="Comfyui-DHan-H3 Sampling Preset",
            category="Comfyui-DHan/Minimax H3 Director",
            description=(
                "MiniMax H3 sampling preset. Turbo presets use their named number "
                "of steps with Euler and BetaSamplingScheduler. Custom can extend sigmas."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    display_name="model",
                    tooltip="Patched H3 model output from Comfyui-DHan-Minimax H3 Director."
                ),

                # IMPORTANT: Keep these first five widgets in the exact legacy order.
                # Existing workflows serialize widget values positionally.
                io.Combo.Input(
                    "preset",
                    options=["Standard", "Turbo 4-Step", "Turbo 6-Step", "Turbo 8-Step", "Custom"],
                    default="Standard",
                ),
                io.Combo.Input(
                    "custom_sampler",
                    options=["euler", "res_multistep", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"],
                    default="euler",
                    tooltip="Custom only. Working H3 workflow uses Euler."
                ),
                io.Combo.Input(
                    "custom_scheduler",
                    options=["beta", "normal", "simple", "sgm_uniform", "karras", "exponential"],
                    default="beta",
                    tooltip="Legacy Custom scheduler. Used when Schedule Mode = basic."
                ),
                io.Int.Input(
                    "custom_steps",
                    default=4,
                    min=1,
                    max=1000,
                    step=1,
                    tooltip="Base scheduler steps before optional sigma extension."
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Sampling denoise strength. 1.0 uses the full sigma schedule."
                ),

                # New controls are appended after the legacy widgets so old workflows
                # load with their existing values intact.
                io.Combo.Input(
                    "custom_schedule_mode",
                    options=["beta", "basic"],
                    default="beta",
                    tooltip="Custom only. 'beta' uses BetaSamplingScheduler; 'basic' uses BasicScheduler."
                ),
                io.Float.Input(
                    "beta_alpha",
                    default=0.79,
                    min=0.0,
                    max=50.0,
                    step=0.01,
                    tooltip="Custom only. BetaSamplingScheduler alpha."
                ),
                io.Float.Input(
                    "beta_beta",
                    default=0.50,
                    min=0.0,
                    max=50.0,
                    step=0.01,
                    tooltip="Custom only. BetaSamplingScheduler beta."
                ),
                io.Int.Input(
                    "extend_steps",
                    default=3,
                    min=0,
                    max=100,
                    step=1,
                    tooltip="Custom only. 0 disables ExtendIntermediateSigmas. Working workflow uses 3."
                ),
                io.Float.Input(
                    "extend_start_sigma",
                    default=0.8,
                    min=-1.0,
                    max=20000.0,
                    step=0.01,
                    tooltip="Custom only. Working workflow extends from sigma 0.8."
                ),
                io.Float.Input(
                    "extend_end_sigma",
                    default=0.0,
                    min=0.0,
                    max=20000.0,
                    step=0.01,
                    tooltip="Custom only. Working workflow extends down to sigma 0."
                ),
                io.Combo.Input(
                    "extend_spacing",
                    options=["linear", "cosine", "sine"],
                    default="linear",
                    tooltip="Custom only. Working workflow uses linear."
                ),
            ],
            outputs=[
                io.Sampler.Output(display_name="sampler"),
                io.Sigmas.Output(display_name="sigmas"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        preset="Standard",
        custom_sampler="euler",
        custom_scheduler="beta",
        custom_steps=4,
        denoise=1.0,
        custom_schedule_mode="beta",
        beta_alpha=0.79,
        beta_beta=0.50,
        extend_steps=3,
        extend_start_sigma=0.8,
        extend_end_sigma=0.0,
        extend_spacing="linear",
    ):
        preset = str(preset)

        if preset == "Custom":
            cfg = {
                "sampler_name": str(custom_sampler),
                "schedule_mode": str(custom_schedule_mode),
                "scheduler": str(custom_scheduler),
                "steps": int(custom_steps),
                "alpha": float(beta_alpha),
                "beta": float(beta_beta),
                "extend_steps": int(extend_steps),
                "extend_start_sigma": float(extend_start_sigma),
                "extend_end_sigma": float(extend_end_sigma),
                "extend_spacing": str(extend_spacing),
            }
        else:
            cfg = dict(cls.PRESETS.get(preset, cls.PRESETS["Standard"]))

        sampler = _dhan_standard_sampler(str(cfg["sampler_name"]))
        custom = _extra("nodes_custom_sampler")

        if str(cfg["schedule_mode"]) == "beta":
            sigmas = _unpack(custom.BetaSamplingScheduler.execute(
                model,
                int(cfg["steps"]),
                float(cfg["alpha"]),
                float(cfg["beta"]),
            ))[0]

            if float(denoise) < 1.0:
                if float(denoise) <= 0.0:
                    sigmas = sigmas[:0]
                else:
                    split = _unpack(custom.SplitSigmasDenoise.execute(
                        sigmas, float(denoise)
                    ))
                    sigmas = split[1]
        else:
            sigmas = _unpack(custom.BasicScheduler.execute(
                model,
                str(cfg["scheduler"]),
                int(cfg["steps"]),
                float(denoise),
            ))[0]

        if int(cfg["extend_steps"]) > 0 and len(sigmas) > 1:
            sigmas = _unpack(custom.ExtendIntermediateSigmas.execute(
                sigmas,
                int(cfg["extend_steps"]),
                float(cfg["extend_start_sigma"]),
                float(cfg["extend_end_sigma"]),
                str(cfg["extend_spacing"]),
            ))[0]

        log.info(
            "[Comfyui-DHan-H3 Sampling Preset] %s | sampler=%s schedule=%s steps=%s "
            "alpha=%.3f beta=%.3f extend=%s sigma %.3f->%.3f %s | final_sigmas=%s",
            preset,
            cfg["sampler_name"],
            cfg["schedule_mode"],
            cfg["steps"],
            cfg["alpha"],
            cfg["beta"],
            cfg["extend_steps"],
            cfg["extend_start_sigma"],
            cfg["extend_end_sigma"],
            cfg["extend_spacing"],
            len(sigmas),
        )

        return io.NodeOutput(sampler, sigmas)



def _dhan_long_original_reference(plan):
    """Return the Director's earliest resolvable FL2VA image as a fresh visual reference."""
    tdata = plan.get("tdata") or {}
    segs = sorted(
        tdata.get("segments", []) or [],
        key=lambda s: float(s.get("start", 0) or 0)
    )
    for seg in segs:
        if seg.get("type", "image") != "image":
            continue
        if not (seg.get("imageFile") or seg.get("imageB64")):
            continue
        try:
            image = _ref_media.load_image_tensor(seg)
            if image is not None:
                return image[:1]
        except Exception as e:
            log.warning(
                "[Comfyui-DHan-H3 Long Sampler] Could not reload original FL2VA reference '%s': %s",
                seg.get("imageFile", ""), e
            )
    return None



def _dhan_hg_keyframe(pixel_index, latent, runtime_mode):
    pixel_index = int(pixel_index)
    if runtime_mode == _HG_NATIVE_LAYOUT_MODE:
        return {"resolved_frame_index": pixel_index, "latent": latent}
    if runtime_mode == _HG_LEGACY_LAYOUT_MODE:
        return {
            "resolved_frame_index": 0,
            _HG_HC_INDEX: pixel_index,
            "latent": latent,
        }
    raise RuntimeError(f"Comfyui-DHan-H3 Herrgotts Core: unknown runtime mode {runtime_mode!r}")


def _dhan_hg_condition_values(keyframes, refs, frame_count, runtime_mode):
    values = {
        "minimax_keyframes": keyframes,
        "minimax_refs": refs,
    }
    if runtime_mode == _HG_LEGACY_LAYOUT_MODE:
        values["minimax_frame_count"] = int(frame_count)
    return values


def _dhan_build_herrgotts_continuation(
    plan,
    prompt,
    win_length,
    previous_latent,
    context_frames,
    last_src=None,
    qwen_reference=None,
):
    """
    DHan adaptation of Herrgotts' direct AV continuation core.

    - video context: separate native H3 keyframes at phase-aligned offsets
    - audio context: minimax_refs timeline block, NOT attached to a video keyframe
    - original Director image: Qwen-only guide
    - optional window Last Frame: native endpoint quality-reset anchor
    """
    mm = _ref_core()
    runtime_mode = _hg_ensure_runtime_patches()

    width = int(plan["width"])
    height = int(plan["height"])
    clip = plan["clip"]
    vae = plan["vae"]

    prev_video, prev_audio = _h3_latent_streams(previous_latent)

    # Herrgotts manual no-lock fallback path: use the latest valid source boundary.
    # Freeze-aware cutoff will be added as a subsequent layer; this core handoff does
    # not arbitrarily throw away the final 2 frames.
    sl = _hg_phase_aligned_extended_context_slice(
        int(prev_video.shape[2]),
        int(context_frames),
        desired_tail_frames=0,
    )

    # Build a clean target latent with no competing frame-0 image keyframe.
    base = mm.MiniMaxH3ImageToVideo.execute(
        clip=clip,
        vae=vae,
        prompt=str(prompt or ""),
        width=width,
        height=height,
        length=int(win_length),
        first_frame=None,
        last_frame=None,
    )
    _, target_latent = _unpack(base)[:2]
    target_video, _ = _h3_latent_streams(target_latent)

    if tuple(prev_video.shape[-2:]) != tuple(target_video.shape[-2:]):
        raise ValueError(
            "Comfyui-DHan-H3 Herrgotts Core requires identical continuation resolution. "
            f"Previous grid={tuple(prev_video.shape[-2:])}, "
            f"target grid={tuple(target_video.shape[-2:])}."
        )

    source = prev_video[:1, :, int(sl["start_t"]):int(sl["end_t"])].clone()
    if int(source.shape[2]) != int(sl["context_steps"]):
        raise RuntimeError("Comfyui-DHan-H3 Herrgotts Core: video context slice mismatch.")

    keyframes = []
    for k, pixel_offset in enumerate(sl["offsets"]):
        keyframes.append(
            _dhan_hg_keyframe(
                int(pixel_offset),
                source[:, :, k:k + 1],
                runtime_mode,
            )
        )

    # Optional end-of-window visual anchor. This is the repeated FL2VA quality reset
    # mechanism when the Director timeline actually supplies a Last Frame.
    frame_count = int(mm.temporal_shape(int(win_length))[0])
    keyframe_images = []
    if last_src is not None:
        last = last_src[:1]
        keyframe_images.append(last)
        keyframes.append(
            _dhan_hg_keyframe(
                frame_count - 1,
                vae.encode(last),
                runtime_mode,
            )
        )

    # Audio gets its own H3 reference block and timeline end marker.
    a0, a1, end_error_steps = _hg_audio_slice_for_pixel_window(
        int(prev_audio.shape[-1]),
        int(sl["source_start_frame"]),
        int(sl["source_end_frame"]),
    )
    audio_context = prev_audio[:1, ..., int(a0):int(a1)].clone()
    ref_audio_t = int(audio_context.shape[-1])
    actual_context_frames = int(sl.get("actual_context_frames", context_frames))
    audio_end_frame = (
        float(actual_context_frames)
        + float(end_error_steps) / float(_HG_FRAME_RESCALE)
    )
    refs = [{
        "kind": "audio",
        "ref_audio_t": ref_audio_t,
        "audio_latent": audio_context,
        _HG_HC_AUDIO_END_FRAME: audio_end_frame,
    }]

    # Keep the original Director reference Qwen-only. Do not VAE-encode it into
    # minimax_refs and do not create a competing frame-0 visual keyframe.
    if qwen_reference is not None:
        try:
            tokens = clip.tokenize(
                str(prompt or ""),
                minimax_ref_items=[{"type": "image", "data": qwen_reference[:1]}],
            )
        except Exception:
            # Newer stock path supports ordered images directly.
            pictures = [*keyframe_images, qwen_reference[:1]]
            tokens = clip.tokenize(str(prompt or ""), images=pictures)
    elif keyframe_images:
        tokens = clip.tokenize(str(prompt or ""), images=keyframe_images)
    else:
        tokens = clip.tokenize(str(prompt or ""))

    positive = clip.encode_from_tokens_scheduled(tokens)
    positive = node_helpers.conditioning_set_values(
        positive,
        _dhan_hg_condition_values(keyframes, refs, frame_count, runtime_mode),
    )

    meta = {
        "mode": "herrgotts_core",
        "runtime_mode": runtime_mode,
        "actual_context_frames": actual_context_frames,
        "context_steps": int(sl["context_steps"]),
        "audio_steps": ref_audio_t,
        "source_start_frame": int(sl["source_start_frame"]),
        "source_end_frame": int(sl["source_end_frame"]),
        "ignored_tail_frames": int(sl.get("ignored_tail_frames", 0)),
        "context_extension_frames": int(sl.get("context_extension_frames", 0)),
    }
    return positive, target_latent, meta


def _dhan_build_masked_continuation(plan, prompt, win_length, previous_latent, context_frames, last_src=None):
    mm = _ref_core()
    out = mm.MiniMaxH3ImageToVideo.execute(
        clip=plan["clip"], vae=plan["vae"], prompt=str(prompt or ""),
        width=int(plan["width"]), height=int(plan["height"]),
        length=int(win_length), first_frame=None, last_frame=last_src,
    )
    positive, target = _unpack(out)[:2]
    source_video, source_audio = _h3_latent_streams(previous_latent)
    target_video, target_audio = _h3_latent_streams(target)
    if source_video.shape[1:2] + source_video.shape[3:] != target_video.shape[1:2] + target_video.shape[3:]:
        raise ValueError("Comfyui-DHan-H3 masked continuation requires matching video latent geometry.")
    if source_audio.shape[1:3] != target_audio.shape[1:3]:
        raise ValueError("Comfyui-DHan-H3 masked continuation requires matching audio latent geometry.")

    _, video_steps, _ = _h3_context_geometry(context_frames)
    audio_steps = round(context_frames * 40 / 24)
    if video_steps >= min(source_video.shape[2], target_video.shape[2]):
        raise ValueError("Comfyui-DHan-H3 masked continuation window is too short for its video context.")
    if audio_steps >= min(source_audio.shape[-1], target_audio.shape[-1]):
        raise ValueError("Comfyui-DHan-H3 masked continuation window is too short for its audio context.")

    video = target_video.clone()
    audio = target_audio.clone()
    video[:, :, :video_steps] = source_video[:, :, -video_steps:].to(video)
    audio[..., :audio_steps] = source_audio[..., -audio_steps:].to(audio)
    video_mask = torch.ones((video.shape[0], 1, video.shape[2], 1, 1), device=video.device)
    audio_mask = torch.ones((audio.shape[0], 1, 1, audio.shape[-1]), device=audio.device)
    video_mask[:, :, :video_steps] = 0
    audio_mask[..., :audio_steps] = 0
    target = target.copy()
    target["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    target["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))
    return positive, target, {
        "mode": "masked_av", "actual_context_frames": context_frames,
        "context_steps": video_steps, "audio_steps": audio_steps,
    }


def _concat_h3_sampled_latents_exact(sampled_latents, trim_meta, target_length, mm):
    """Stitch windows using the exact video/audio context lengths injected per seam."""
    if not sampled_latents:
        raise ValueError("Comfyui-DHan-H3 Long Sampler produced no sampled windows.")

    videos, audios = [], []
    for i, latent in enumerate(sampled_latents):
        v, a = _h3_latent_streams(latent)
        if i > 0:
            meta = trim_meta[i] if i < len(trim_meta) else {}
            vtrim = max(0, int(meta.get("context_steps", 0)))
            atrim = max(0, int(meta.get("audio_steps", 0)))
            v = v[:, :, min(vtrim, v.shape[2]):]
            a = a[..., min(atrim, a.shape[-1]):]
        videos.append(v)
        audios.append(a)

    video = torch.cat(videos, dim=2)
    audio = torch.cat(audios, dim=-1)

    _, target_video_t, target_audio_t = mm.temporal_shape(int(target_length))
    if int(video.shape[2]) < int(target_video_t):
        raise ValueError(
            f"Comfyui-DHan-H3 Herrgotts Core: video stitch short "
            f"{video.shape[2]}/{target_video_t}."
        )

    audio_short = int(target_audio_t) - int(audio.shape[-1])
    if audio_short > 0:
        if audio_short > 4 or int(audio.shape[-1]) < 1:
            raise ValueError(
                f"Comfyui-DHan-H3 Herrgotts Core: audio stitch short "
                f"{audio.shape[-1]}/{target_audio_t}."
            )
        tail = audio[..., -1:].expand(*audio.shape[:-1], audio_short)
        audio = torch.cat((audio, tail), dim=-1)

    video = video[:, :, :int(target_video_t)].contiguous()
    audio = audio[..., :int(target_audio_t)].contiguous()

    out = sampled_latents[0].copy()
    out["samples"] = comfy.nested_tensor.NestedTensor((video, audio))
    return out


def _dhan_build_long_fl2va_window(
    plan, win_start, win_length, previous_latent=None, context_frames=22,
    refresh_reference=True
):
    """Build one <=15s FL2VA conditioning window from Director runtime metadata.

    Follow-up windows can refresh Qwen's visual conditioning from the original Director
    image while the actual frame-0 latent anchor comes from the previous sampled window.
    """
    mm = _ref_core()
    tdata = plan["tdata"]
    fps = float(plan.get("fps", 24.0) or 24.0)
    width = int(plan["width"])
    height = int(plan["height"])
    resize_method = str(plan.get("resize_method", "maintain aspect ratio"))
    img_compression = int(plan.get("img_compression", 0) or 0)
    div = 32

    p = _ref_plan.plan_timeline(
        tdata,
        int(win_start),
        int(win_length),
        fps,
        global_prompt=str(plan.get("global_prompt", "") or ""),
        use_custom_motion=bool(plan.get("use_custom_motion", False)),
        use_custom_audio=bool(plan.get("use_custom_audio", False)),
        override_audio=bool(plan.get("override_audio", False)),
        extra_ref_image_count=0,
        ref_image_notes="",
    )

    # Load only media used by this logical window.
    for ev in p["events"]:
        seg = ev["seg"]
        if ev["kind"] == "video":
            seg_start = float(seg.get("start", 0))
            trim = float(seg.get("trimStart", 0)) + max(0.0, float(win_start) - seg_start)
            ev["tensor"] = _ref_media.load_video_tensor(
                seg.get("imageFile", ""),
                trim / fps,
                float(seg.get("length", 1)) / fps,
            )
        else:
            ev["tensor"] = _ref_media.load_image_tensor(seg)

    first_src = last_src = None
    for ev in p["events"]:
        if ev["role"] == _ref_plan.ROLE_FIRST:
            first_src = ev["tensor"][:1]
        elif ev["role"] == _ref_plan.ROLE_LAST:
            last_src = ev["tensor"][-1:]

    # For continuation windows, the logical sub-window normally no longer overlaps the
    # original image at t=0. Re-introduce that source image to Qwen every window so visual
    # identity/material/lighting do not recursively drift from generated-to-generated data.
    #
    # If the sub-window itself has an intentional first-frame image, that local image wins.
    qwen_first_src = first_src
    refreshed_from_original = False
    if previous_latent is not None and refresh_reference and qwen_first_src is None:
        qwen_first_src = _dhan_long_original_reference(plan)
        refreshed_from_original = qwen_first_src is not None

    if qwen_first_src is not None:
        qwen_first_src = _ref_media.resize_image(
            qwen_first_src, width, height, resize_method, div
        )
        if img_compression > 0:
            qwen_first_src = _ref_media.compress_image(qwen_first_src, img_compression)
    if last_src is not None:
        last_src = _ref_media.resize_image(last_src, width, height, resize_method, div)
        if img_compression > 0:
            last_src = _ref_media.compress_image(last_src, img_compression)

    out = mm.MiniMaxH3ImageToVideo.execute(
        clip=plan["clip"],
        vae=plan["vae"],
        prompt=p["prompt"],
        width=width,
        height=height,
        length=int(p["length"]),
        first_frame=qwen_first_src,
        last_frame=last_src,
    )
    positive, latent = _unpack(out)[:2]

    carried_frames = 0
    if previous_latent is not None:
        positive, carried_frames = _apply_h3_latent_continuity(
            positive,
            latent,
            previous_latent,
            int(context_frames),
            replace_frame0_keyframe=(qwen_first_src is not None),
        )
        if refreshed_from_original:
            log.info(
                "[Comfyui-DHan-H3 Long Sampler] Visual reference refreshed from original Director image; "
                "previous latent tail remains the frame-0 continuity anchor."
            )

    return positive, latent, p, int(carried_frames)


class _DHanOverallProgressCallback:
    """One Comfy progress bar shared across every internal Long Sampler window."""

    def __init__(self, total_windows, steps_per_window):
        self.total_windows = max(1, int(total_windows))
        self.steps_per_window = max(1, int(steps_per_window))
        self.total_steps = self.total_windows * self.steps_per_window
        self.window_index = 0
        self.pbar = comfy.utils.ProgressBar(self.total_steps)

    def set_window(self, window_index):
        self.window_index = max(0, int(window_index))

    def callback_factory(self, model, steps, x0_output_dict=None):
        # Mirror ComfyUI latent_preview.prepare_callback(), except the ProgressBar
        # is shared and its absolute position includes all previously finished windows.
        preview_format = "JPEG"
        try:
            previewer = latent_preview.get_previewer(model.load_device, model.model.latent_format)
        except Exception:
            previewer = None

        base = self.window_index * self.steps_per_window
        overall_total = self.total_steps

        def callback(step, x0, x, total_steps):
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0

            preview_bytes = None
            if previewer:
                try:
                    preview_x0 = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                    preview_bytes = previewer.decode_latent_to_preview_image(
                        preview_format, preview_x0
                    )
                except Exception:
                    preview_bytes = None

            # Clamp to the configured per-window work count. This preserves a smooth
            # 0..100% bar even if a custom sampler reports a slightly different
            # total_steps value in its callback.
            local_done = min(self.steps_per_window, max(0, int(step) + 1))
            overall_done = min(overall_total, base + local_done)
            self.pbar.update_absolute(overall_done, overall_total, preview_bytes)

        return callback


class DHanMiniMaxH3LongSampler(io.ComfyNode):
    """Drop-in SamplerCustomAdvanced replacement with automatic H3 window chaining."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanMiniMaxH3LongSampler",
            display_name="Comfyui-DHan-Minimax H3 Long Sampler",
            category="Comfyui-DHan/Minimax H3 Director",
            description=(
                "Use in place of SamplerCustomAdvanced. At or below the safe duration it "
                "samples exactly once. Above it, FL2VA is rendered as chained H3-safe "
                "windows using direct AV-latent tail context, then stitched back into one "
                "joint video+audio latent."
            ),
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input(
                    "latent_image",
                    tooltip="Connect the DHan Director latent output. Long-render metadata is carried invisibly inside it."
                ),
                io.Float.Input(
                    "safe_window_seconds",
                    default=15.0,
                    min=5.0,
                    max=15.0,
                    step=0.5,
                    tooltip="Maximum logical duration per H3 sampling pass. <= this duration samples normally."
                ),
                io.Int.Input(
                    "context_frames",
                    default=22,
                    min=5,
                    max=107,
                    step=17,
                    tooltip=(
                        "Tail context carried directly from one sampled H3 AV latent into the next. "
                        "22 frames is about 0.9s at 24fps; valid H3 grid values are 5,22,39,56..."
                    )
                ),
                io.Int.Input(
                    "max_windows",
                    default=8,
                    min=2,
                    max=32,
                    step=1,
                    tooltip="Safety limit for chained windows."
                ),
                io.Boolean.Input(
                    "refresh_reference",
                    default=True,
                    tooltip=(
                        "Use the original Director image as a Qwen-only visual reference "
                        "on Herrgotts Core and Stable DHan continuation windows."
                    )
                ),
                io.Combo.Input(
                    "continuation_mode",
                    options=["Masked AV", "Herrgotts Core", "Stable DHan"],
                    default="Herrgotts Core",
                    tooltip=(
                        "Masked AV preserves the previous video and audio latent tail during denoising. "
                        "Herrgotts Core and Stable DHan remain available for existing workflows."
                    )
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ],
        )

    @classmethod
    def execute(
        cls, noise, guider, sampler, sigmas, latent_image,
        safe_window_seconds=15.0, context_frames=22, max_windows=8,
        refresh_reference=True, continuation_mode="Herrgotts Core"
    ):
        runtime = latent_image.get("dhan_h3_long_plan") if isinstance(latent_image, dict) else None
        custom = _extra("nodes_custom_sampler", "SamplerCustomAdvanced")

        # No DHan plan: behave as a normal SamplerCustomAdvanced.
        if not isinstance(runtime, dict):
            log.info("[Comfyui-DHan-H3 Long Sampler] No DHan long plan; normal single-pass sampling.")
            vals = _unpack(custom.SamplerCustomAdvanced.execute(
                noise, guider, sampler, sigmas, latent_image
            ))
            return io.NodeOutput(vals[0], vals[1] if len(vals) > 1 else vals[0])

        fps = float(runtime.get("fps", 24.0) or 24.0)
        project_length = max(1, int(runtime.get("duration_frames", 1)))
        project_start = max(0, int(runtime.get("start_frame", 0)))
        safe_frames = max(5, min(360, int(round(float(safe_window_seconds) * fps))))

        mode = str(runtime.get("model_type", "FL2VA")).lower()
        # Only FL2VA supports continuation; other modes use the supplied latent unchanged.
        if project_length <= safe_frames or mode != "fl2va":
            log.info(
                "[Comfyui-DHan-H3 Long Sampler] %s, %d frames (%.2fs); normal single-pass sampling.",
                mode, project_length, project_length / fps
            )
            clean_latent = latent_image.copy()
            clean_latent.pop("dhan_h3_long_plan", None)
            vals = _unpack(custom.SamplerCustomAdvanced.execute(
                noise, guider, sampler, sigmas, clean_latent
            ))
            out = vals[0]
            den = vals[1] if len(vals) > 1 else vals[0]
            if isinstance(out, dict):
                out = out.copy()
                out.pop("dhan_h3_long_plan", None)
            if isinstance(den, dict):
                den = den.copy()
                den.pop("dhan_h3_long_plan", None)
            return io.NodeOutput(out, den)

        context_frames, _, _ = _h3_context_geometry(int(context_frames))
        # Normal sampled H3 windows end on phase 2, so valid 5/22/39-frame
        # contexts already begin on phase 0. Reserving the theoretical +13-frame
        # extension here shifted every continuation plan backward even though the
        # handoff actually carried only context_frames, causing timing drift.
        planning_context = context_frames
        new_frames_per_followup = max(1, safe_frames - planning_context)
        needed = 1
        if project_length > safe_frames:
            remaining = project_length - safe_frames
            needed += (remaining + new_frames_per_followup - 1) // new_frames_per_followup
        if needed > int(max_windows):
            raise ValueError(
                f"Comfyui-DHan-H3 Long Sampler needs {needed} windows for {project_length/fps:.1f}s, "
                f"but max_windows is {int(max_windows)}."
            )

        log.info(
            "[Comfyui-DHan-H3 Long Sampler] Long render %.2fs: safe=%.2fs, context=%d frames, windows=%d, refresh_reference=%s, mode=%s.",
            project_length / fps, safe_frames / fps, context_frames, needed, bool(refresh_reference), str(continuation_mode)
        )

        # ComfyUI normally creates a fresh node ProgressBar for every
        # SamplerCustomAdvanced call. Long Sampler intentionally makes several calls,
        # so use one shared bar spanning all windows.
        steps_per_window = max(1, int(len(sigmas) - 1))
        overall_progress = _DHanOverallProgressCallback(needed, steps_per_window)
        original_prepare_callback = latent_preview.prepare_callback

        sampled = []
        denoised = []
        seam_meta = [{}]
        previous = None
        produced = 0
        window_index = 0
        basic = _extra("nodes_custom_sampler", "BasicGuider")

        # The Preview Override already knows how to accumulate Auto Render windows;
        # give every internal sample a shared session id + window metadata so it can
        # display: finished windows + the current in-progress window.
        preview_session = uuid.uuid4().hex
        preview_model_options = getattr(guider.model_patcher, "model_options", None)
        previous_preview_meta = None
        if isinstance(preview_model_options, dict):
            previous_preview_meta = preview_model_options.get("dhan_h3_auto_preview")

        while produced < project_length:
            if window_index == 0:
                logical_start = project_start
                logical_length = min(safe_frames, project_length)
                trim_context = False
            else:
                remaining = project_length - produced
                new_frames = min(new_frames_per_followup, remaining)
                planned_ctx = planning_context
                logical_start = max(project_start, project_start + produced - planned_ctx)
                logical_length = planned_ctx + new_frames
                if str(continuation_mode) == "Masked AV":
                    logical_length = max(logical_length, planned_ctx + 17)
                trim_context = True

            seam = {}
            if window_index > 0 and str(continuation_mode) in ("Herrgotts Core", "Masked AV"):
                # Build planner data for this logical timeline window, but construct
                # Plan the latent/media overlap separately from the new prompt interval.
                pwin = _ref_plan.plan_timeline(
                    runtime["tdata"],
                    int(logical_start),
                    int(logical_length),
                    fps,
                    global_prompt=str(runtime.get("global_prompt", "") or ""),
                    use_custom_motion=bool(runtime.get("use_custom_motion", False)),
                    use_custom_audio=bool(runtime.get("use_custom_audio", False)),
                    override_audio=bool(runtime.get("override_audio", False)),
                    extra_ref_image_count=0,
                    ref_image_notes="",
                )

                # Resolve any explicit Last Frame inside this sub-window.
                last_src = None
                for ev in pwin["events"]:
                    seg = ev["seg"]
                    if ev["kind"] == "video":
                        seg_start = float(seg.get("start", 0))
                        trim = float(seg.get("trimStart", 0)) + max(
                            0.0, float(logical_start) - seg_start
                        )
                        ev["tensor"] = _ref_media.load_video_tensor(
                            seg.get("imageFile", ""),
                            trim / fps,
                            float(seg.get("length", 1)) / fps,
                        )
                    else:
                        ev["tensor"] = _ref_media.load_image_tensor(seg)
                    if ev["role"] == _ref_plan.ROLE_LAST:
                        last_src = ev["tensor"][-1:]

                if last_src is not None:
                    last_src = _ref_media.resize_image(
                        last_src,
                        int(runtime["width"]),
                        int(runtime["height"]),
                        str(runtime.get("resize_method", "maintain aspect ratio")),
                        32,
                    )

                qwen_ref = None
                if bool(refresh_reference) and str(continuation_mode) == "Herrgotts Core":
                    qwen_ref = _dhan_long_original_reference(runtime)
                    if qwen_ref is not None:
                        qwen_ref = _ref_media.resize_image(
                            qwen_ref,
                            int(runtime["width"]),
                            int(runtime["height"]),
                            str(runtime.get("resize_method", "maintain aspect ratio")),
                            32,
                        )

                # IMPORTANT: latent/media continuity and prompt scheduling use different
                # timeline ranges. The continuation latent intentionally reaches backward
                # into the previous window, but already-consumed storyboard beats must not
                # be re-issued to Qwen/H3 or the action can restart/repeat near long-video
                # boundaries.
                #
                # Build a second prompt-only plan from the first NEW output frame onward.
                # Keep pwin for media/keyframe loading and native window geometry.
                prompt_start = project_start + produced
                prompt_length = max(1, min(new_frames, project_length - produced))
                pprompt = _ref_plan.plan_timeline(
                    runtime["tdata"],
                    int(prompt_start),
                    int(prompt_length),
                    fps,
                    global_prompt=str(runtime.get("global_prompt", "") or ""),
                    use_custom_motion=bool(runtime.get("use_custom_motion", False)),
                    use_custom_audio=bool(runtime.get("use_custom_audio", False)),
                    override_audio=bool(runtime.get("override_audio", False)),
                    extra_ref_image_count=0,
                    ref_image_notes="",
                )

                if str(continuation_mode) == "Masked AV":
                    positive, window_latent, seam = _dhan_build_masked_continuation(
                        runtime, pprompt["prompt"], int(pwin["length"]), previous,
                        int(context_frames), last_src=last_src,
                    )
                else:
                    positive, window_latent, seam = _dhan_build_herrgotts_continuation(
                        runtime, pprompt["prompt"], int(pwin["length"]), previous,
                        int(context_frames), last_src=last_src, qwen_reference=qwen_ref,
                    )
                plan = pwin
                plan["prompt_window_start"] = int(prompt_start)
                plan["prompt_window_length"] = int(prompt_length)
                plan["prompt"] = pprompt["prompt"]
                log.info(
                    "[Comfyui-DHan-H3 Long Sampler] Prompt window %d/%d: NEW timeline frames %d..%d; "
                    "latent/media window remains %d..%d for continuity.",
                    window_index + 1, needed,
                    int(prompt_start), int(prompt_start + prompt_length),
                    int(logical_start), int(logical_start + logical_length),
                )
                carried = int(seam["actual_context_frames"])
                log.info(
                    "[Comfyui-DHan-H3 Long Sampler] %s handoff: runtime=%s "
                    "ctx=%d frames / %d video steps / %d audio steps / tail=%d.",
                    seam.get("mode"),
                    seam.get("runtime_mode", "native"),
                    seam.get("actual_context_frames"),
                    seam.get("context_steps"),
                    seam.get("audio_steps"),
                    seam.get("ignored_tail_frames", 0),
                )
            else:
                positive, window_latent, plan, carried = _dhan_build_long_fl2va_window(
                    runtime,
                    logical_start,
                    logical_length,
                    previous_latent=previous if window_index > 0 else None,
                    context_frames=context_frames,
                    refresh_reference=bool(refresh_reference),
                )
                if window_index > 0:
                    _, vst, ast = _h3_context_geometry(int(carried))
                    seam = {
                        "mode": "stable_dhan",
                        "actual_context_frames": int(carried),
                        "context_steps": int(vst),
                        "audio_steps": int(ast),
                    }

            guider_meta = getattr(guider, "_dhan_h3_guider_meta", None)
            if isinstance(guider_meta, dict) and bool(guider_meta.get("negative_prompting", False)):
                negative_cond = guider_meta.get("negative")
                if negative_cond is None:
                    raise ValueError(
                        "Comfyui-DHan-H3 Long Sampler: CFG guider metadata is missing negative conditioning."
                    )
                window_guider = _unpack(
                    basic.CFGGuider.execute(
                        guider.model_patcher, positive, negative_cond,
                        float(guider_meta.get("cfg", 2.0))
                    )
                )[0]
            else:
                window_guider = _unpack(
                    basic.BasicGuider.execute(guider.model_patcher, positive)
                )[0]

            log.info(
                "[Comfyui-DHan-H3 Long Sampler] Window %d/%d: timeline frames %d..%d (%d logical; native %d), carried=%d.",
                window_index + 1, needed,
                logical_start, logical_start + logical_length,
                logical_length, int(plan["length"]), carried
            )

            if isinstance(preview_model_options, dict):
                preview_model_options["dhan_h3_auto_preview"] = {
                    "session": preview_session,
                    "window_index": int(window_index),
                    "total_windows": int(needed),
                    "context_frames": int(context_frames if window_index > 0 else 0),
                    "project_frames": int(project_length),
                    "fps": float(fps),
                }

            overall_progress.set_window(window_index)
            try:
                latent_preview.prepare_callback = overall_progress.callback_factory
                vals = _unpack(custom.SamplerCustomAdvanced.execute(
                    noise, window_guider, sampler, sigmas, window_latent
                ))
            finally:
                latent_preview.prepare_callback = original_prepare_callback

            window_sampled = vals[0]
            window_denoised = vals[1] if len(vals) > 1 else vals[0]
            sampled.append(window_sampled)
            denoised.append(window_denoised)
            previous = window_sampled

            if window_index == 0:
                produced += logical_length
            else:
                seam_meta.append(dict(seam))
                produced += logical_length - int(carried)

            window_index += 1
            if window_index > int(max_windows):
                raise RuntimeError("Comfyui-DHan-H3 Long Sampler exceeded max_windows unexpectedly.")

        mm = _ref_core()
        if str(continuation_mode) in ("Herrgotts Core", "Masked AV"):
            final = _concat_h3_sampled_latents_exact(
                sampled, seam_meta, project_length, mm
            )
            final_den = _concat_h3_sampled_latents_exact(
                denoised, seam_meta, project_length, mm
            )
        else:
            final = _concat_h3_sampled_latents(
                sampled, context_frames, project_length, mm
            )
            final_den = _concat_h3_sampled_latents(
                denoised, context_frames, project_length, mm
            )
        final["dhan_h3_windows"] = int(window_index)
        final["dhan_h3_project_length"] = int(project_length)
        final_den["dhan_h3_windows"] = int(window_index)
        final_den["dhan_h3_project_length"] = int(project_length)

        overall_progress.pbar.update_absolute(
            overall_progress.total_steps,
            overall_progress.total_steps,
            None,
        )

        if isinstance(preview_model_options, dict):
            if previous_preview_meta is None:
                preview_model_options.pop("dhan_h3_auto_preview", None)
            else:
                preview_model_options["dhan_h3_auto_preview"] = previous_preview_meta

        log.info(
            "[Comfyui-DHan-H3 Long Sampler] Complete: %d windows stitched to %d requested frames.",
            window_index, project_length
        )
        return io.NodeOutput(final, final_den)


class DHanMiniMaxH3RenderWindow(io.ComfyNode):
    """Select one H3-safe render chunk from a longer DHan timeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanMiniMaxH3RenderWindow",
            display_name="Comfyui-DHan-Minimax H3 Render Window",
            category="Comfyui-DHan/Minimax H3 Director",
            description=(
                "Lets the Director use timelines longer than one native H3 window. "
                "Select which <=15s chunk to render; optionally supply the previous "
                "render's last frame for continuity."
            ),
            inputs=[
                io.Int.Input(
                    "window_index",
                    default=0,
                    min=0,
                    max=999,
                    step=1,
                    tooltip="0 = first render window, 1 = second, etc."
                ),
                io.Int.Input(
                    "max_window_frames",
                    default=360,
                    min=24,
                    max=362,
                    step=1,
                    tooltip="H3 render-window size. 360 frames = exactly 15s at H3's 24fps."
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=0,
                    min=0,
                    max=120,
                    step=1,
                    tooltip="Optional overlap between neighboring render windows."
                ),
                io.Latent.Input(
                    "continuity_latent",
                    optional=True,
                    tooltip=(
                        "Previous window's sampled H3 AV latent. Recommended: carries motion + audio "
                        "directly without decoding/re-encoding."
                    )
                ),
                io.Int.Input(
                    "context_frames",
                    default=22,
                    min=5,
                    max=107,
                    step=17,
                    tooltip="Direct latent context length. H3-valid values are 5, 22, 39, 56... frames."
                ),
                io.Image.Input(
                    "continuity_frame",
                    optional=True,
                    tooltip="Fallback decoded-frame continuity. Ignored when continuity_latent is connected."
                ),
            ],
            outputs=[H3Window.Output(display_name="h3_window")],
        )

    @classmethod
    def execute(
        cls,
        window_index=0,
        max_window_frames=360,
        overlap_frames=0,
        continuity_latent=None,
        context_frames=22,
        continuity_frame=None,
    ):
        max_frames = max(24, min(362, int(max_window_frames)))
        overlap = max(0, min(max_frames - 1, int(overlap_frames)))
        return io.NodeOutput({
            "window_index": max(0, int(window_index)),
            "max_window_frames": max_frames,
            "overlap_frames": overlap,
            "continuity_latent": continuity_latent,
            "context_frames": int(context_frames),
            "continuity_frame": continuity_frame,
        })


class DHanMiniMaxH3ReferenceHub(io.ComfyNode):
    """Dynamic Ref2VA image stacker. Input order maps directly to <Picture 1> ... <Picture 9>."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanMiniMaxH3ReferenceHub",
            display_name="Comfyui-DHan-Minimax H3 Reference Hub",
            category="Comfyui-DHan/Minimax H3 Director",
            description=(
                "Dynamic Ref2VA reference-image stacker. Connect image_1 and a new image input grows automatically, "
                "up to 9 images. Input order maps to <Picture 1> ... <Picture 9>."
            ),
            inputs=[
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max", "diffusers"],
                    default="match",
                    tooltip="Ref2VA reference sizing. 'match' is recommended for the distilled Turbo models; 'diffusers' forces the original 2048px short-edge behavior."
                ),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image", tooltip="Ref2VA reference image"),
                        prefix="image_",
                        min=1,
                        max=9,
                    ),
                ),
            ],
            outputs=[
                H3Refs.Output(display_name="h3_refs"),
            ],
        )

    @classmethod
    def execute(cls, ref_image_size="match", images=None):
        keyed = {}
        if isinstance(images, dict):
            def _index(key):
                try:
                    return int(str(key).rsplit("_", 1)[-1])
                except Exception:
                    return 999
            for key in sorted(images.keys(), key=_index):
                image = images.get(key)
                if image is not None and len(keyed) < 9:
                    keyed[f"ref_image_{len(keyed)+1}"] = image
        elif images:
            for image in list(images)[:9]:
                if image is not None:
                    keyed[f"ref_image_{len(keyed)+1}"] = image
        return io.NodeOutput({"images": keyed, "ref_image_size": str(ref_image_size)})



class DHanMiniMaxH3Director(io.ComfyNode):
    """The tuned DHan Director frontend backed by native MiniMax H3 conditioning."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanMiniMaxH3Director",
            display_name="Comfyui-DHan-Minimax H3 Director",
            category="Comfyui-DHan/Minimax H3 Director",
            description="Tuned DHan timeline behavior with native MiniMax H3 FL2VA/Ref2VA conditioning. Sampling stays external.",
            inputs=[
                io.Model.Input(
                    "model",
                    display_name="model (FL2VA)",
                    optional=True,
                    lazy=True,
                    tooltip="FL2VA H3 model branch."
                ),
                io.Model.Input(
                    "model_ref2va",
                    display_name="model (Ref2VA)",
                    optional=True,
                    lazy=True,
                    tooltip="Ref2VA H3 model branch."
                ),
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True, tooltip="MiniMax H3 video VAE"),
                io.Vae.Input("audio_vae", optional=True, lazy=True, tooltip="MiniMax H3 audio VAE; required for Ref2VA"),

                io.Combo.Input("model_type", options=["FL2VA", "Ref2VA", "Retake"], default="FL2VA"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),

                # Exact widget/state contract expected by the tuned DHan frontend.
                io.Float.Input("start_second", default=0.0, min=0.0, max=1000.0, step=0.01),
                io.Float.Input("end_second", default=5.0, min=0.0, max=1000.0, step=0.01),
                io.Float.Input("duration_seconds", default=5.0, min=0.1, max=1000.0, step=0.01),
                io.Int.Input("start_frame", default=0, min=0, max=100000, step=1),
                io.Int.Input("end_frame", default=120, min=1, max=100000, step=1),
                io.Int.Input("duration_frames", default=120, min=1, max=100000, step=1),
                io.String.Input("timeline_data", default=""),
                io.Boolean.Input("use_custom_audio", default=False, optional=True),
                io.Boolean.Input("use_custom_motion", default=True, optional=True),
                io.Boolean.Input("inpaint_audio", default=True, optional=True),
                io.String.Input("local_prompts", multiline=True, default=""),
                io.String.Input("segment_lengths", default=""),
                io.Float.Input("epsilon", default=0.001, min=0.0001, max=0.99, step=0.0001),
                io.Float.Input("frame_rate", default=24.0, min=1.0, max=240.0, step=1.0, optional=True),
                io.Combo.Input("display_mode", options=["frames", "seconds"], default="seconds", optional=True),
                io.String.Input("guide_strength", default=""),
                io.Int.Input("custom_width", default=0, min=0, max=8192, step=1, optional=True, tooltip="Target output width. Set to 0 to use/derive from the source image."),
                io.Int.Input("custom_height", default=0, min=0, max=8192, step=1, optional=True, tooltip="Target output height. Set to 0 to use/derive from the source image."),
                io.Combo.Input(
                    "resize_method",
                    options=["maintain aspect ratio", "stretch to fit", "pad", "pad green", "crop"],
                    default="maintain aspect ratio",
                    optional=True,
                    tooltip="DHan resize behavior. Use width=0 or height=0 to preserve source aspect ratio automatically."
                ),
                io.Int.Input("img_compression", default=0, min=0, max=100, step=1, optional=True),
                io.Combo.Input("voice_reference_audio", options=["none"], default="none", optional=True),
                io.Combo.Input("voice_lora_name", options=["None"], default="None", optional=True),
                io.Float.Input("voice_lora_strength", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                io.Float.Input("voice_identity_guidance", default=3.0, min=0.0, max=100.0, step=0.01, optional=True),
                io.Float.Input("voice_reference_seconds", default=5.0, min=1.0, max=30.0, step=0.1, optional=True),
                io.Boolean.Input("override_audio", default=False, optional=True),
                io.String.Input(
                    "negative_prompt", multiline=True, default="", optional=True,
                    tooltip="Optional H3 negative prompt. Used by Comfyui-DHan-H3 Guider when Negative Prompting is enabled."
                ),
                io.Boolean.Input(
                    "negative_prompting", default=False, optional=True,
                    tooltip=(
                        "Enable negative prompting. Comfyui-DHan-H3 Guider automatically switches "
                        "from BasicGuider to CFG when this is ON."
                    )
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.Float.Output(display_name="fps"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def check_lazy_status(
        cls,
        model_type,
        timeline_data="",
        model=_DHan_UNCONNECTED,
        model_ref2va=_DHan_UNCONNECTED,
        audio_vae=_DHan_UNCONNECTED,
        override_audio=False,
        **_,
    ):
        """Resolve only the branches required by the active Comfyui-DHan-H3 mode.

        This follows the reference MiniMax Director's lazy checkpoint behavior,
        while also keeping DHan-only reference inputs out of the FL2VA execution
        path.  Lazy inputs are requested in stages so ComfyUI does not evaluate
        unused upstream branches before Director.execute().
        """
        ref_on = str(model_type).lower() == "ref2va"

        # 1) Resolve exactly the checkpoint required by the active mode.
        name = "model_ref2va" if ref_on else "model"
        have = {"model": model, "model_ref2va": model_ref2va}
        if have[name] is None:
            return [name]

        if not ref_on:
            return []

        # 2) The audio VAE is needed only when the Ref2VA timeline actually uses
        #    standalone audio, or Override Audio asks for reference-video audio.
        try:
            tdata = _ref_plan.parse_timeline(timeline_data)
            if not isinstance(tdata, dict):
                tdata = {}
        except Exception:
            tdata = {}
        has_ref_audio = bool(tdata.get("audioSegments")) or bool(override_audio)
        if has_ref_audio and audio_vae is not _DHan_UNCONNECTED and audio_vae is None:
            return ["audio_vae"]

        return []

    @classmethod
    def execute(
        cls,
        model_type, shift_video, shift_audio,
        start_second, end_second, duration_seconds,
        start_frame, end_frame, duration_frames, timeline_data,
        use_custom_audio, use_custom_motion, inpaint_audio,
        local_prompts, segment_lengths, epsilon, frame_rate,
        display_mode, guide_strength, custom_width, custom_height,
        resize_method, img_compression, voice_reference_audio,
        voice_lora_name, voice_lora_strength, voice_identity_guidance,
        voice_reference_seconds, override_audio, negative_prompt="", negative_prompting=False,
        model=None, model_ref2va=None, clip=None, vae=None,
        audio_vae=None, h3_window=None
    ):
        """
        DHan v0.4.51 UI/schema with MiniMax Director processing architecture.
        No frontend/timeline behavior is changed here.
        """
        global_prompt = ""
        _t0 = time.perf_counter()
        _last_t = _t0
        def _stage(name):
            nonlocal _last_t
            now = time.perf_counter()
            log.info("[Comfyui-DHan-H3 Timing] %-24s +%.3fs (total %.3fs)", name, now - _last_t, now - _t0)
            _last_t = now

        _stage("execute entered")

        if clip is None:
            raise ValueError("Comfyui-DHan-Minimax H3 Director: connect CLIP.")
        if vae is None:
            raise ValueError("Comfyui-DHan-Minimax H3 Director: connect the MiniMax H3 video VAE.")

        mm = _ref_core()

        # Parse the exact timeline JSON produced by our original DHan frontend.
        tdata = _ref_plan.parse_timeline(timeline_data)
        if not isinstance(tdata, dict):
            tdata = {}

        # Reference backend mode is derived from our existing DHan model switch.
        mode_name = str(model_type).lower()
        is_ref = mode_name == "ref2va"
        is_retake_mode = mode_name == "retake"
        if is_retake_mode:
            # Retake uses H3's FL2VA endpoint anchors around the selected section.
            # The visual editor stores the source video/range inside timeline_data.
            tdata["retakeMode"] = True
            tdata["reference_mode"] = "OFF"
        else:
            tdata["retakeMode"] = False
            tdata["reference_mode"] = "REF2VA" if is_ref else "OFF"

        fps = float(frame_rate) if frame_rate else 24.0
        retake_state = _ref_plan.retake_state(tdata) if is_retake_mode else None
        if is_retake_mode and retake_state is None:
            raise ValueError(
                "Comfyui-DHan-Minimax H3 Director: Retake mode needs a base video. "
                "Use Add Video in the Retake timeline first."
            )
        if retake_state:
            win_start = int(retake_state["start"])
            win_length = max(1, int(retake_state["length"]))
        else:
            win_start = int(start_frame)
            win_length = max(1, int(duration_frames))

        # Same reference planner used by the MiniMax Director.
        # Ref2VA subject images/descriptions now live inside timeline_data
        # (timeline.characters), so there is no external Reference Hub payload.
        extra_refs = 0

        p = _ref_plan.plan_timeline(
            tdata,
            win_start,
            win_length,
            fps,
            global_prompt=global_prompt,
            use_custom_motion=bool(use_custom_motion),
            use_custom_audio=bool(use_custom_audio),
            override_audio=bool(override_audio),
            extra_ref_image_count=extra_refs,
            ref_image_notes="",
        )

        _stage("timeline planned")
        length = int(p["length"])
        if length > _ref_plan.TRAINED_MAX_FRAMES:
            log.warning(
                "[Comfyui-DHan-H3] %d frames (%.1fs) is past H3's trained range of ~%d-%d frames (~4-15s).",
                length, p["actual_seconds"], _ref_plan.TRAINED_MIN_FRAMES, _ref_plan.TRAINED_MAX_FRAMES
            )

        # Load only media the reference plan calls for.
        for ev in p["events"]:
            seg = ev["seg"]
            if ev["kind"] == "video":
                seg_start = float(seg.get("start", 0))
                trim = float(seg.get("trimStart", 0)) + max(0.0, win_start - seg_start)
                ev["tensor"] = _ref_media.load_video_tensor(
                    seg.get("imageFile", ""), trim / fps, float(seg.get("length", 1)) / fps
                )
            else:
                ev["tensor"] = _ref_media.load_image_tensor(seg)

        _stage("timeline media loaded")
        first_src = last_src = None
        retake = p.get("retake")
        if retake:
            # Anchor the regenerated middle on the original video's neighboring frames.
            # This is the native MiniMax Director retake strategy: the frame immediately
            # before the marked range becomes first_frame and the first frame after the
            # range becomes last_frame.
            before_idx = int(retake["start"]) - 1
            if before_idx >= 0:
                frames = _ref_media.load_video_tensor(
                    retake["video"], before_idx / fps, 1.0 / 24.0
                )
                if frames is not None and frames.shape[0] > 0:
                    first_src = frames[:1]
            tail_idx = int(retake["start"]) + int(retake["length"])
            if not retake.get("base_frames") or tail_idx < int(retake.get("base_frames") or 0):
                frames = _ref_media.load_video_tensor(
                    retake["video"], tail_idx / fps, 1.0 / 24.0
                )
                if frames is not None and frames.shape[0] > 0:
                    last_src = frames[:1]
            if first_src is None and last_src is None:
                log.warning(
                    "[Comfyui-DHan-H3] Retake could not read surrounding anchor frames from '%s'; "
                    "falling back to prompt-only generation.", retake["video"]
                )
        else:
            for ev in p["events"]:
                if ev["role"] == _ref_plan.ROLE_FIRST:
                    first_src = ev["tensor"][:1]
                elif ev["role"] == _ref_plan.ROLE_LAST:
                    last_src = ev["tensor"][-1:]

        # Reference canvas policy.
        canvas_src = first_src
        if canvas_src is None and p["events"]:
            canvas_src = p["events"][0]["tensor"]
        canvas_size = None
        if canvas_src is not None:
            canvas_size = int(canvas_src.shape[2]), int(canvas_src.shape[1])
        elif is_ref and p.get("ref_video_segs"):
            ref_seg = p["ref_video_segs"][0]
            canvas_size = _ref_media.video_dimensions(
                ref_seg.get("videoFile") or ref_seg.get("imageFile") or ""
            )

        div = 32
        if int(custom_width or 0) > 0 and int(custom_height or 0) > 0:
            if canvas_src is not None:
                fitted = _ref_media.resize_image(
                    canvas_src[:1], int(custom_width), int(custom_height),
                    resize_method or "maintain aspect ratio", div
                )
                width, height = int(fitted.shape[2]), int(fitted.shape[1])
            elif canvas_size is not None and (resize_method or "maintain aspect ratio") == "maintain aspect ratio":
                src_w, src_h = canvas_size
                ratio = min(int(custom_width) / src_w, int(custom_height) / src_h)
                width = max(div, (int(src_w * ratio) // div) * div)
                height = max(div, (int(src_h * ratio) // div) * div)
            else:
                width = max(div, (int(custom_width) // div) * div)
                height = max(div, (int(custom_height) // div) * div)
        elif canvas_size is not None:
            src_w, src_h = canvas_size
            if int(custom_width or 0) > 0:
                width = max(div, (int(custom_width) // div) * div)
                height = max(div, (int(src_h * width / max(1, src_w)) // div) * div)
            elif int(custom_height or 0) > 0:
                height = max(div, (int(custom_height) // div) * div)
                width = max(div, (int(src_w * height / max(1, src_h)) // div) * div)
            else:
                width, height = mm.adapt_canvas(src_w, src_h)
        else:
            width, height = mm.adapt_canvas(1344, 768)

        # Fit anchor frames exactly as reference media path does.
        if first_src is not None:
            first_src = _ref_media.resize_image(
                first_src, width, height, resize_method or "maintain aspect ratio", div
            )
            if int(img_compression or 0) > 0:
                first_src = _ref_media.compress_image(first_src, int(img_compression))
        if last_src is not None:
            last_src = _ref_media.resize_image(
                last_src, width, height, resize_method or "maintain aspect ratio", div
            )
            if int(img_compression or 0) > 0:
                last_src = _ref_media.compress_image(last_src, int(img_compression))

        prompt = p["prompt"]
        log.info("[Comfyui-DHan-H3 Timing] compiled prompt: %d chars", len(prompt or ""))
        _stage("canvas/anchors ready")

        # Pick only the active lazy model, same architecture as reference Director.
        selected_model = model_ref2va if is_ref else model
        if selected_model is None:
            required = "model (Ref2VA)" if is_ref else "model (FL2VA)"
            raise ValueError(f"Comfyui-DHan-Minimax H3 Director: connect {required} for the selected mode.")

        if is_ref:
            _has_audio_refs = bool(p.get("ref_audio_segs"))
            if _has_audio_refs and audio_vae is None:
                raise ValueError(
                    "Comfyui-DHan-Minimax H3 Director: audio/voice references require the MiniMax H3 audio VAE."
                )

            ref_images = {}
            picture_idx = 1

            # Use the reference planner's canonical <Picture i> order. This includes:
            #   1) inline Ref2VA subject cards (timeline.characters)
            #   2) any timeline image references
            # The subject descriptions are compiled by minimax_plan into the prompt and
            # @charN references resolve to their corresponding <Picture i> entries.
            for slot in p.get("ref_image_slots", []) or []:
                src = slot.get("source")
                tensor = None

                if src == "char":
                    img = slot.get("image") or {}
                    tensor = _ref_media.load_image_source(
                        img.get("b64", ""), img.get("name", "")
                    )
                elif src == "input":
                    # External ref_images were intentionally removed from DHan; keep this
                    # branch harmless for compatibility with planner data.
                    continue
                else:
                    ev = slot.get("event") or {}
                    tensor = ev.get("tensor")
                    if tensor is None and ev.get("seg"):
                        tensor = _ref_media.load_image_tensor(ev["seg"])
                    if tensor is not None and slot.get("keyframe"):
                        # Timeline keyframes follow the output canvas, matching the
                        # reference Director's conditioning path.
                        if slot.get("keyframe") == _ref_plan.ROLE_LAST:
                            tensor = tensor[-1:]
                        else:
                            tensor = tensor[:1]
                        tensor = _ref_media.resize_image(
                            tensor, width, height,
                            resize_method or "maintain aspect ratio", div
                        )
                        if int(img_compression or 0) > 0:
                            tensor = _ref_media.compress_image(tensor, int(img_compression))

                if tensor is None:
                    continue
                if tensor.ndim == 3:
                    tensor = tensor.unsqueeze(0)
                ref_images[f"ref_image_{picture_idx}"] = tensor[:1]
                picture_idx += 1
                if picture_idx > 9:
                    break

            # Global Voice Reference slots and the timed REF AUDIO lane both become
            # native Ref2VA <Audio N> references. Voice refs are loaded whole; timeline
            # audio keeps its existing trim/length semantics. The planner caps the
            # combined set at H3's audio-reference limit.
            ref_audios = {}
            for seg in p.get("ref_audio_segs", []) or []:
                clip_audio = None
                if seg.get("_voice_ref"):
                    try:
                        clip_audio = _load_ref_audio(seg)
                    except Exception as e:
                        log.warning("[Comfyui-DHan-H3] Could not load Voice Ref; skipping: %s", e)
                else:
                    clip_audio = _ref_media.load_audio_segment(seg, fps)
                if clip_audio is not None:
                    ref_audios[f"ref_audio_{len(ref_audios)}"] = clip_audio

            ref_videos = {
                f"ref_video_{i + 1}": _load_ref_video_tensor(seg, fps)
                for i, seg in enumerate(p.get("ref_video_segs", []) or [])
            }

            _stage("before Ref2VA conditioning")
            out = mm.MiniMaxH3ReferenceToVideo.execute(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                ref_image_size="match",
                ref_images=ref_images or None,
                ref_videos=ref_videos or None,
                ref_video_audios=None,
                ref_audios=ref_audios or None,
            )
        else:
            _stage("before FL2VA conditioning")
            out = mm.MiniMaxH3ImageToVideo.execute(
                clip=clip, vae=vae, prompt=prompt,
                width=width, height=height, length=length,
                first_frame=first_src, last_frame=last_src,
            )

        _stage("conditioning complete")
        positive, latent = _unpack(out)[:2]

        # Optional text-only negative conditioning for CFG guidance.
        neg_text = str(negative_prompt or "").strip()
        neg_tokens = clip.tokenize(neg_text)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)
        negative = node_helpers.conditioning_set_values(
            negative,
            {
                "dhan_negative_prompting": bool(negative_prompting),
                "dhan_negative_has_text": bool(neg_text),
            }
        )
        if neg_text:
            log.info(
                "[Comfyui-DHan-H3] negative conditioning encoded: %d chars (enabled=%s)",
                len(neg_text), bool(negative_prompting)
            )

        # Same placement as reference Director: SigmaShift after conditioning.
        selected_model = _unpack(mm.MiniMaxH3SigmaShift.execute(
            model=selected_model,
            shift_video=float(shift_video),
            shift_audio=float(shift_audio),
        ))[0]

        _stage("SigmaShift complete")

        # Carry long-render planning data invisibly inside the existing LATENT dict.
        # Normal SamplerCustomAdvanced ignores extra dict keys; Comfyui-DHan-H3 Long Sampler
        # consumes this metadata only when the requested timeline exceeds its safe window.
        if isinstance(latent, dict):
            latent = latent.copy()
            latent["dhan_h3_long_plan"] = {
                "version": 1,
                "model_type": "Ref2VA" if is_ref else ("Retake" if is_retake_mode else "FL2VA"),
                "tdata": tdata,
                "timeline_data": timeline_data,
                "clip": clip,
                "vae": vae,
                "audio_vae": audio_vae,
                "fps": fps,
                "start_frame": int(win_start if retake_state else start_frame),
                "duration_frames": int(win_length if retake_state else duration_frames),
                "width": int(width),
                "height": int(height),
                "resize_method": resize_method or "maintain aspect ratio",
                "img_compression": int(img_compression or 0),
                "use_custom_motion": bool(use_custom_motion),
                "use_custom_audio": bool(use_custom_audio),
                "override_audio": bool(override_audio),
                "global_prompt": global_prompt,
            }

        # Retake intentionally returns only the regenerated replacement section.
        return io.NodeOutput(
            selected_model,
            positive,
            latent,
            24.0,
            negative,
        )
