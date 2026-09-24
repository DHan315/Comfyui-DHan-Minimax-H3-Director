
import base64
import io as _io
import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

import aiohttp
import asyncio
import torch
from PIL import Image

import server
import folder_paths
from comfy_api.latest import io

log = logging.getLogger(__name__)
EVENT = "dhan_h3_prompt_enhancer_result"

SYSTEM_PROMPT = """You are a storyboard writer specialized for MiniMax H3 video generation.

You will receive:
- a total video duration,
- an exact list of storyboard parts with fixed start/end/duration,
- an optional reference image for specific parts.

Produce TWO complementary outputs:

GLOBAL PROMPT
- Persistent context for the entire video.
- Establish subject identity, stable appearance, wardrobe, environment, visual style,
  lighting, atmosphere, broad camera language, and continuity constraints.
- Do NOT narrate the changing action sequence here.

STORYBOARD
- Exactly one prompt for every supplied timed part.
- Parts are consecutive action beats in one continuous video, not automatic camera cuts.
- In Part 1, establish the attached opening image briefly, then begin the intended
  action promptly. Describe its onset, visible development, and result or reaction.
- EACH PROMPT MUST FIT REALISTICALLY INSIDE THAT PART'S GIVEN DURATION.
- Keep motion at real-world speed unless the user explicitly requests slow motion.
- Do not stretch one brief gesture, expression, or static pose across several seconds.
  If a part outlasts its main action, add related follow-through, reactions, and
  natural secondary motion rather than slowing that action down or inventing a
  new major event. Respect explicitly requested stillness.
- 1-3 seconds: one simple, readable action or reaction.
- 3-6 seconds: one primary action with a clear onset and follow-through; include
  small related changes if the action itself is brief.
- 6-10 seconds: describe a short sequence of closely related actions that evolves
  throughout the part, without a long opening hold.
- Do not cram several sequential actions into a short part.
- If a part has an attached image, treat that image as the visual reference for THAT PART.
- Preserve physical and visual continuity between adjacent parts.
- Focus on what changes over time: action, interaction, expression, camera movement,
  and brief sound cues when useful.
- Keep camera motion continuous and describe its type, range, and speed when it
  matters; do not imply a cut unless the user asks for one.
- Do not repeatedly restate persistent identity/environment/style from the global prompt.

AI PACING RULES
- When Timing Mode is AI Pacing, decide start/end/duration for each part yourself.
- First part starts at 0. Final part ends exactly at the requested total duration.
- Parts are contiguous with no gaps or overlaps.
- Do not default to equal timing. Allocate more time to beats with more motion/action and less to simple holds/reactions.
- Give brief gestures and opening reference-image holds only the time they need;
  do not use them to fill a long first part.
- Return numeric start, end, and duration values for every storyboard item.

GENERAL RULES
- Preserve supplied image references.
- Do not invent new characters, wardrobe changes, props, locations, or major events unless asked.
- Use concrete visual language.
- No prompt-engineering commentary.
- Do not use <Picture>, <Video>, <Audio>, or <Subject> tags.
- Return exactly the requested number of storyboard parts.

Return VALID JSON ONLY:
{
  "global_prompt": "persistent MiniMax H3 global prompt",
  "storyboard": [
    {"part": 1, "start": 0.0, "end": 3.5, "duration": 3.5, "prompt": "timed action prompt"}
  ]
}
"""


def _normalize_url(value):
    value = (value or "").strip() or "http://127.0.0.1:11434"
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value.rstrip("/")


def _get_ollama_models_sync(url="http://127.0.0.1:11434", timeout=1.0):
    """Best-effort installed-model list for the initial Combo schema."""
    try:
        base = _normalize_url(url)
        req = urllib.request.Request(base + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        names = []
        for item in data.get("models", []) or []:
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
        return sorted(set(names), key=str.lower)
    except Exception:
        return []


@server.PromptServer.instance.routes.get("/dhan_h3/ollama_models")
async def dhan_h3_ollama_models(request):
    """Frontend refresh endpoint. Queries whichever Ollama URL the node currently uses."""
    raw_url = request.query.get("url", "http://127.0.0.1:11434")
    url = _normalize_url(raw_url)
    timeout = aiohttp.ClientTimeout(total=4)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url + "/api/tags") as resp:
                body = await resp.text()
                if resp.status >= 400:
                    return server.web.json_response(
                        {"ok": False, "models": [], "error": f"Ollama HTTP {resp.status}: {body[:300]}"},
                        status=resp.status,
                    )
                data = json.loads(body)
        names = []
        for item in data.get("models", []) or []:
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.append(name)
        return server.web.json_response(
            {"ok": True, "models": sorted(set(names), key=str.lower), "url": url}
        )
    except Exception as e:
        return server.web.json_response(
            {"ok": False, "models": [], "error": str(e), "url": url},
            status=500,
        )



def _input_file_to_b64(value, max_size=768):
    """Load a ComfyUI input-file path produced by /upload/image and encode for Ollama."""
    value = str(value or "").strip().replace("\\", "/")
    if not value:
        return None
    input_root = Path(folder_paths.get_input_directory()).resolve()
    candidate = (input_root / value).resolve()
    try:
        candidate.relative_to(input_root)
    except Exception:
        raise ValueError(f"Storyboard image path is outside ComfyUI input directory: {value}")
    if not candidate.is_file():
        raise ValueError(f"Storyboard image not found: {value}")
    im = Image.open(candidate).convert("RGB")
    w, h = im.size
    if max(w, h) > int(max_size):
        scale = float(max_size) / float(max(w, h))
        im = im.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.LANCZOS,
        )
    bio = _io.BytesIO()
    im.save(bio, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(bio.getvalue()).decode("ascii")



def _normalize_ai_timing(raw_items, parts, duration_seconds):
    parts = max(1, int(parts))
    total = max(0.1, float(duration_seconds))
    proposed = []
    for i in range(parts):
        raw = raw_items[i] if i < len(raw_items) and isinstance(raw_items[i], dict) else {}
        dur = None
        try:
            if raw.get("duration") is not None:
                dur = float(raw.get("duration"))
        except Exception:
            dur = None
        if dur is None:
            try:
                dur = max(0.0, float(raw.get("end")) - float(raw.get("start")))
            except Exception:
                dur = 1.0
        if not dur or dur <= 0:
            dur = 1.0
        proposed.append(float(dur))

    min_each = min(1.0, total / parts)
    remaining = max(0.0, total - min_each * parts)
    extras = [max(0.0, d - min_each) for d in proposed]
    extra_sum = sum(extras)
    durations = (
        [min_each + remaining * (x / extra_sum) for x in extras]
        if extra_sum > 1e-8 else
        [total / parts for _ in range(parts)]
    )

    out, cursor = [], 0.0
    for i, d in enumerate(durations):
        start = cursor
        end = total if i == parts - 1 else min(total, cursor + d)
        out.append({
            "part": i + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })
        cursor = end

    if out:
        out[-1]["end"] = round(total, 3)
        out[-1]["duration"] = round(out[-1]["end"] - out[-1]["start"], 3)
    return out


def _timed_parts(parts, duration_seconds):
    parts = max(1, int(parts))
    total = max(0.1, float(duration_seconds))
    # Equal fixed edit windows for the first authoring build.
    boundaries = [total * i / parts for i in range(parts + 1)]
    out = []
    for i in range(parts):
        start = boundaries[i]
        end = boundaries[i + 1]
        out.append({
            "part": i + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })
    return out


def _image_to_b64(image, max_size=768):
    if image is None or not isinstance(image, torch.Tensor):
        return None
    t = image[0] if image.ndim == 4 else image
    if t.ndim != 3:
        return None
    arr = (t.detach().float().cpu().clamp(0, 1).numpy() * 255.0 + 0.5).astype("uint8")
    im = Image.fromarray(arr)
    w, h = im.size
    if max(w, h) > int(max_size):
        scale = float(max_size) / float(max(w, h))
        im = im.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.LANCZOS,
        )
    bio = _io.BytesIO()
    im.save(bio, format="JPEG", quality=88, optimize=True)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _format_storyboard(items, parts):
    items = list(items or [])
    out = []
    for i, item in enumerate(items[:parts], 1):
        s = str(item or "").strip()
        s = re.sub(
            r"(?im)^\s*(?:part|shot|scene|segment)\s*#?\s*\d+\s*[:.)-]?\s*",
            "",
            s,
        )
        out.append(f"Part {i}\n{s}")

    while len(out) < parts:
        out.append(f"Part {len(out) + 1}\n")

    return "\n\n".join(out)


async def _force_unload(session, url, model):
    """
    Ollama officially supports unloading through an empty generate request
    with keep_alive=0. This is a backstop even though the chat request itself
    also uses keep_alive=0.
    """
    try:
        async with session.post(
            url + "/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        ) as resp:
            await resp.read()
            if resp.status >= 400:
                log.warning(
                    "[Comfyui-DHan-H3 Prompt Enhancer] Ollama unload returned HTTP %s.",
                    resp.status,
                )
            else:
                log.info(
                    "[Comfyui-DHan-H3 Prompt Enhancer] Ollama model '%s' unloaded from memory.",
                    model,
                )
    except Exception as e:
        log.warning(
            "[Comfyui-DHan-H3 Prompt Enhancer] Ollama unload request failed: %s",
            e,
        )


class DHanH3PromptEnhancer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DHanH3PromptEnhancer",
            display_name="DHan-H3 Storyboard Enhancer",
            category="Comfyui-DHan/Minimax H3 Director",
            description=(
                "Standalone Ollama storyboard authoring tool. Select the node and use "
                "ComfyUI's native Execute-to-selected-output button. The VLM is unloaded "
                "after generation so H3 can reclaim VRAM."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "idea",
                    multiline=True,
                    default="",
                    tooltip="Simple description of what should happen."
                ),
                io.Int.Input(
                    "parts",
                    default=3,
                    min=1,
                    max=10,
                    step=1,
                    tooltip="Exact number of storyboard action parts."
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=1.0,
                    max=120.0,
                    step=0.5,
                    tooltip="Total target duration used to pace the progression."
                ),
                io.Combo.Input(
                    "ollama_model",
                    options=(_get_ollama_models_sync() or ["<refresh models>"]),
                    default=(_get_ollama_models_sync()[0] if _get_ollama_models_sync() else "<refresh models>"),
                    tooltip="Installed Ollama model. Use Refresh Models after changing ollama_url."
                ),
                io.String.Input(
                    "ollama_url",
                    default="http://127.0.0.1:11434",
                    optional=True
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0x7fffffff,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Fixed until you change it. With identical inputs ComfyUI can reuse "
                        "the cached result instead of loading Ollama again."
                    )
                ),
                io.Combo.Input(
                    "timing_mode",
                    options=["Equal", "AI Pacing"],
                    default="AI Pacing",
                    tooltip="Equal splits runtime evenly. AI Pacing lets the model allocate time according to each story beat."
                ),
                io.Combo.Input(
                    "detail_level",
                    options=["Easy", "Medium", "Advanced"],
                    default="Medium",
                    tooltip="Controls prompt detail while respecting each part duration."
                ),
                io.Int.Input(
                    "max_image_size",
                    default=768,
                    min=256,
                    max=1536,
                    step=64,
                    optional=True
                ),
                io.String.Input(
                    "part_images_json",
                    multiline=True,
                    default="[]",
                    optional=True,
                    tooltip="Internal Storyboard UI image mapping. Managed by the custom card UI."
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    async def execute(
        cls,
        idea="",
        parts=3,
        duration_seconds=5.0,
        ollama_model="qwen2.5vl:7b",
        ollama_url="http://127.0.0.1:11434",
        seed=0,
        timing_mode="AI Pacing",
        detail_level="Medium",
        max_image_size=768,
        part_images_json="[]",
        **kwargs,
    ):
        idea = (idea or "").strip()
        if not idea:
            raise ValueError("Comfyui-DHan-H3 Storyboard Enhancer: enter an idea.")

        parts = max(1, min(10, int(parts)))
        model = (ollama_model or "").strip()
        if not model or model == "<refresh models>":
            raise ValueError(
                "Comfyui-DHan-H3 Storyboard Enhancer: choose an installed Ollama model. "
                "Click Refresh Models on the node if the dropdown is empty."
            )

        url = _normalize_url(ollama_url)
        timing_mode = str(timing_mode or "AI Pacing").strip()
        use_ai_pacing = timing_mode.lower() == "ai pacing"
        timing = None if use_ai_pacing else _timed_parts(parts, float(duration_seconds))

        try:
            part_image_refs = json.loads(part_images_json or "[]")
            if not isinstance(part_image_refs, list):
                part_image_refs = []
        except Exception:
            part_image_refs = []

        normalized_refs = []
        for ref in part_image_refs:
            if not isinstance(ref, dict):
                continue
            file_value = str(ref.get("file") or "").strip()
            try:
                part_value = int(ref.get("part") or 1)
            except Exception:
                part_value = 1
            if file_value and 1 <= part_value <= int(parts):
                normalized_refs.append({"file": file_value, "part": part_value})

        # Describe exact edit windows and independently assigned image references.
        timing_lines = []
        encoded_part_images = []
        image_ref_meta = []
        refs_by_part = {}
        for ref in normalized_refs:
            refs_by_part.setdefault(int(ref["part"]), []).append(ref["file"])

        if use_ai_pacing:
            for idx in range(1, int(parts) + 1):
                files_for_part = refs_by_part.get(idx, [])
                suffix = "" if not files_for_part else f" — {len(files_for_part)} attached image reference" + ("s" if len(files_for_part) != 1 else "")
                timing_lines.append(f"Part {idx}: timing to be decided by you{suffix}")
        else:
            for item in timing:
                idx = int(item["part"])
                files_for_part = refs_by_part.get(idx, [])
                suffix = "" if not files_for_part else f" — {len(files_for_part)} attached image reference" + ("s" if len(files_for_part) != 1 else "")
                timing_lines.append(
                    f'Part {idx}: {item["start"]:.3f}s to {item["end"]:.3f}s '
                    f'({item["duration"]:.3f}s)' + suffix
                )

        for ref in normalized_refs:
            b64 = _input_file_to_b64(ref["file"], int(max_image_size))
            if b64:
                encoded_part_images.append(b64)
                image_ref_meta.append({"file": ref["file"], "part": int(ref["part"])})

        detail = str(detail_level or "Medium").strip().lower()
        if detail == "easy":
            detail_rule = "Easy = concise action plus essential camera direction only."
        elif detail == "advanced":
            detail_rule = (
                "Advanced = rich but duration-appropriate choreography, expression, camera/lens/motion, "
                "environment response, and brief sound cues when useful."
            )
        else:
            detail_rule = "Medium = balanced visual/action/camera detail without overloading the beat."

        msg = {
            "role": "user",
            "content": (
                f"Create a MiniMax H3 global prompt and exactly {parts} storyboard prompts "
                f"for a fixed {float(duration_seconds):.3f}-second edit.\n"
                f"Prompt detail level: {detail_level}. {detail_rule}\n"
                f"Timing mode: {timing_mode}.\n\n"
                + (
                    "YOU MUST CHOOSE THE DURATION OF EACH PART based on story/action complexity. "
                    f"The sequence must begin at 0.000s and end exactly at {float(duration_seconds):.3f}s, "
                    "with contiguous parts and no gaps or overlaps. Do not split evenly unless equal pacing is genuinely appropriate.\n"
                    if use_ai_pacing else
                    "THESE TIMINGS ARE FIXED. Write each action so it can realistically complete inside its assigned duration.\n"
                )
                + "\n".join(timing_lines)
                + f"\n\nUser idea:\n{idea}"
            ),
        }

        # Storyboard image references are managed entirely by the built-in reference panel.
        ollama_images = []
        if encoded_part_images:
            ollama_images.extend(encoded_part_images)
            labels = ", ".join(
                f"Image {i + 1} -> Part {meta['part']}"
                for i, meta in enumerate(image_ref_meta)
            )
            msg["content"] += (
                "\nAdditional storyboard images are ordered exactly as follows: "
                + labels
                + ". Multiple images may belong to the same part. Use each only as a visual reference for its assigned part."
            )
        if ollama_images:
            msg["images"] = ollama_images

        payload = {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                msg,
            ],
            "options": {
                "seed": int(seed),
                "temperature": 0.4,
                "num_predict": 2400,
            },
            "format": "json",
            "keep_alive": 0,
        }

        request_timeout = 1800
        log.info(
            "[Comfyui-DHan-H3 Storyboard Enhancer] requesting %d parts from '%s'%s (timeout=%ds).",
            parts,
            model,
            " with multimodal context" if encoded_part_images else "",
            request_timeout,
        )

        log.info("[Comfyui-DHan-H3 Storyboard Enhancer] Ollama endpoint: %s (generation timeout=%ds)", url, request_timeout)

        # Keep connection timeout separate from inference timeout. A localhost
        # connection failure means Ollama is unavailable/wedged, not that Qwen
        # spent the full generation timeout thinking.
        connect_timeout = 15
        timeout = aiohttp.ClientTimeout(
            total=request_timeout,
            connect=connect_timeout,
            sock_connect=connect_timeout,
            sock_read=request_timeout,
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                health_error = None
                for attempt in range(3):
                    try:
                        health_timeout = aiohttp.ClientTimeout(
                            total=10, connect=5, sock_connect=5, sock_read=10
                        )
                        async with session.get(url + "/api/tags", timeout=health_timeout) as health_resp:
                            if health_resp.status < 500:
                                health_error = None
                                break
                            health_error = RuntimeError(
                                f"Ollama health check returned HTTP {health_resp.status}"
                            )
                    except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                        health_error = exc
                    if attempt < 2:
                        await asyncio.sleep(2.0)

                if health_error is not None:
                    raise ConnectionError(
                        "Comfyui-DHan-H3 Storyboard Enhancer: cannot connect to Ollama at "
                        f"{url}. Ollama is not accepting connections. This is not a "
                        f"{request_timeout}-second generation timeout; check that Ollama "
                        "is running and port 11434 is responsive."
                    ) from health_error

                try:
                    async with session.post(url + "/api/chat", json=payload) as resp:
                        body = await resp.text()
                        if resp.status >= 400:
                            raise ValueError(
                                f"Ollama HTTP {resp.status}: {body[:600]}"
                            )
                        data = json.loads(body)
                except aiohttp.ClientConnectorError as exc:
                    raise ConnectionError(
                        "Comfyui-DHan-H3 Storyboard Enhancer: lost connection to Ollama before "
                        f"the request could start ({url})."
                    ) from exc
                except aiohttp.ConnectionTimeoutError as exc:
                    raise ConnectionError(
                        "Comfyui-DHan-H3 Storyboard Enhancer: Ollama did not accept the connection "
                        f"within {connect_timeout} seconds ({url}). This is a connection "
                        "problem, not a model-generation timeout."
                    ) from exc
                except asyncio.TimeoutError as exc:
                    raise TimeoutError(
                        "Comfyui-DHan-H3 Storyboard Enhancer: Ollama connected successfully, but "
                        f"generation did not finish within {request_timeout} seconds. "
                        "Use a smaller/faster model quantization if this repeatedly times out."
                    ) from exc

                parsed = _extract_json(
                    ((data.get("message") or {}).get("content") or "").strip()
                )
                global_prompt = str(parsed.get("global_prompt") or "").strip()
                raw_items = parsed.get("storyboard") or []
                if not isinstance(raw_items, list):
                    raw_items = [raw_items]

                if not global_prompt:
                    raise ValueError("Comfyui-DHan-H3 Storyboard Enhancer: global prompt was empty.")

                prompts = []
                for raw in raw_items[:parts]:
                    if isinstance(raw, dict):
                        prompts.append(str(raw.get("prompt") or "").strip())
                    else:
                        prompts.append(str(raw or "").strip())
                while len(prompts) < parts:
                    prompts.append("")

                if use_ai_pacing:
                    timing = _normalize_ai_timing(raw_items, parts, float(duration_seconds))

                structured = []
                display_items = []
                for i, timing_item in enumerate(timing):
                    prompt_text = prompts[i]
                    item = dict(timing_item)
                    item["prompt"] = prompt_text
                    item["images"] = list(refs_by_part.get(i + 1, []))
                    item["imageFile"] = item["images"][0] if item["images"] else ""
                    structured.append(item)
                    display_items.append(
                        f'Part {i + 1}  |  {timing_item["start"]:.2f}-{timing_item["end"]:.2f}s  '
                        f'({timing_item["duration"]:.2f}s)\n{prompt_text}'
                    )

                storyboard = "\n\n".join(display_items)

                server.PromptServer.instance.send_sync(
                    EVENT,
                    {
                        "node_id": str(cls.hidden.unique_id),
                        "global_prompt": global_prompt,
                        "storyboard": storyboard,
                        "items": structured,
                        "duration_seconds": float(duration_seconds),
                        "parts": int(parts),
                        "timing_mode": timing_mode,
                    },
                )

                return io.NodeOutput()
            finally:
                # Explicit unload backstop so the VLM doesn't sit on the GPU
                # after the authoring request finishes or errors out.
                await _force_unload(session, url, model)
