## v0.4.80 — persistent compact resize row
- Fixed `resize_method` native row reappearing after browser/workflow refresh.
- Removed the legacy restore-time `showWidget(resize_method)` override.
- `resize_method` is now hard-collapsed with the other compact-panel duplicate rows and re-hidden after Vue restore passes.
- Split-panel Resize remains the only visible resize control.

## v0.4.79 — Retake filmstrip timeline

- Retake source video is shown as a compact 8–12-thumbnail filmstrip instead of a repeated midpoint frame.
- The playhead stays at the last scrubbed/retake-boundary position after mouse release.
- Retake lane label is preserved as `RETAKE` through sidebar refresh/resize updates.
- FL2VA/Ref2VA generation behavior is unchanged.

## v0.4.77 — Simplified Retake workflow

- Retake is now just: load source video → select range → enter prompt → generate the replacement clip.
- Removed the visible `retake_info` output.
- Removed the separate `Comfyui-DHan-Minimax H3 Retake Stitch` node.
- Sampling/decoding remains the normal external ComfyUI path.
- FL2VA/Ref2VA behavior and timeline UI are otherwise unchanged from v0.4.76.

# Comfyui-DHan-Minimax H3 Director

## v0.4.76 — Retake lane label

- Retake mode timeline lane is labeled `RETAKE` instead of inheriting `FL2VA`/`VIDEO`.
- No timeline behavior, conditioning, or layout logic changed.

## v0.4.74 — Retake mode + compact timing cleanup

- Adds `Retake` as a third Director mode alongside FL2VA and Ref2VA.
- Retake regenerates only the marked source-video range, anchored by the original frames immediately before/after it.
- Adds `retake_info` output and a `Comfyui-DHan-Minimax H3 Retake Stitch` node for splicing the regenerated range back into the source video.
- Removes the duplicate native Start / End / Duration widget rows; the split settings panel remains the visible UI.
- Starts from the v0.4.73 stable baseline; existing FL2VA/Ref2VA subject, voice-ref, and timeline behavior is retained.

## v0.4.73 — Separate Ref2VA voice-reference slots

- Moved the Ref2VA Subjects container below the full timeline/editor.
- Added a compact `Drop / Add Voice Ref` area to each `@charN` subject card.
- Voice refs are bound to their subject as native Ref2VA `<Audio N>` voice/timbre references.
- Up to three voice/audio references are allowed across the Ref2VA conditioning payload.
- Voice refs remain separate from the timed REF AUDIO lane.
- No Analyze/LLM controls were added.

## v0.4.67 — inline Ref2VA subjects

- Ref2VA now exposes three compact subject-reference cards directly inside the Director.
- Each subject accepts up to two images plus a manual description.
- Use `@char1`, `@char2`, or `@char3` in prompts; the MiniMax planner maps them to native `<Picture i>` / `<Subject i>` references.
- The subject panel is hidden in FL2VA mode and appears only in Ref2VA.
- The Director no longer requires the separate `h3_refs` / Reference Hub input.
- Existing timeline interaction behavior is otherwise unchanged.

## v0.4.65 — reference-native FL2VA conditioning

- Restored the exact MiniMax Director FL2VA execution architecture: `mm.MiniMaxH3ImageToVideo.execute(...)` is called directly.
- Removed the DHan custom timed/cache FL2VA wrapper from the Director execution path.
- Timeline frontend behavior and styling are unchanged.
- Sampling Preset denoise control remains restored.


## v0.4.64 — content-keyed H3 conditioning cache
- Removed transient `id(clip)` from the Qwen conditioning cache key.
- Removed transient `id(vae)` from the keyframe VAE cache key.
- Cache reuse now follows compiled prompt + keyframe image content across Director re-executions.
- FL2VA timing diagnostics remain enabled.
- Timeline/frontend JS is unchanged.
- Sampling Preset denoise control remains restored.


## v0.4.63 — FL2VA conditioning isolation + cache

- FL2VA now mirrors ComfyUI native H3 conditioning inline so Qwen tokenize, Qwen encode,
  and keyframe VAE encode are timed separately.
- Added a one-entry conditioning cache for identical compiled prompt + keyframe inputs.
  Re-runs caused only by DHan timeline/widget state no longer repeat the large Qwen3-VL encode.
- Added a one-entry keyframe VAE cache.
- Conditioning math remains the native ComfyUI MiniMax H3 path; timeline JS is unchanged.
- Added INFO-level timing markers around Director planning, media preparation, native H3 conditioning, and SigmaShift to pinpoint pre-sampler stalls.
- Logs compiled prompt character count to expose accidental prompt growth.
- Ref2VA now requires the audio VAE only when audio references are actually present, matching the reference MiniMax Director behavior.
- Sampling Preset denoise control from v0.4.61 remains intact.
- Timeline frontend JS is unchanged.
# Comfyui-DHan-Minimax H3 Director v0.4.0

This build transplants the user's tuned DHan/LTX Director frontend instead of recreating it.

- Exact DHan TimelineEditor behavior/UI base
- FL2VA / Ref2VA model switch
- grouped model sockets
- separate Comfyui-DHan-Minimax H3 Settings node
- Ref2VA 9-image reference strip
- native ComfyUI MiniMax H3 conditioning
- external sampling/Turbo LoRA compatible


## v0.4.1
- Added `Comfyui-DHan-Minimax H3 Reference Hub`.
- Dynamic Ref2VA image stack using ComfyUI Autogrow: starts with one image input and grows as inputs are connected, up to 9.
- Hub preserves image order as `<Picture 1>` through `<Picture 9>`.
- `ref_image_size` (`match` / `max`) lives on the Reference Hub.
- Director now accepts one optional `h3_refs` socket.
- Removed the in-Director reference-image strip / JSON reference bank.


## v0.4.2
- FL2VA and Ref2VA model inputs now use ComfyUI lazy evaluation.
- Only the model selected by `model_type` is requested/evaluated.
- `audio_vae` is lazy and only requested in Ref2VA mode.
- Restored DHan `0 = AUTO` resolution behavior.
- width=0 + fixed height derives width from the first timeline image aspect ratio.
- fixed width + height=0 derives height from source aspect ratio.
- 0/0 uses the first timeline image dimensions.
- Resolved dimensions are aligned to multiples of 32 for MiniMax H3.


## v0.4.3 — H3 frontend cleanup
- Keeps the transplanted tuned DHan MAIN timeline behavior.
- Hides Add Audio, Add Video, and Add IC Video.
- Hides/collapses AUDIO and IC-LoRA Input tracks.
- Hides the inherited VOICE REF panel.
- Ref2VA references remain non-temporal and are supplied by Comfyui-DHan-Minimax H3 Reference Hub.
- Adds a small Ref2VA References status row only when Ref2VA is selected.


## v0.4.4 — DHan sizing controls
- `resize_method` is visible again on the Director beside the dimension controls.
- Default resize method is `maintain aspect ratio`.
- `custom_width` supports true `0 = AUTO`.
- `custom_height` supports true `0 = AUTO`.
- The H3 frontend re-applies min=0 after old workflow restoration so stale DHan widget metadata cannot clamp AUTO back to 32.
- Examples:
  - width 0 / height 768 = derive landscape width from source aspect ratio
  - width 1344 / height 0 = derive height
  - width 0 / height 0 = use first source image dimensions


## v0.4.5 — exact tuned DHan resize behavior
- custom_width default 0, min 0, step 1.
- custom_height default 0, min 0, step 1.
- resize methods exactly match tuned DHan: maintain aspect ratio, stretch to fit, pad, pad green, crop.
- Both dimensions set: apply selected resize method.
- Width only: derive height from source aspect ratio.
- Height only: derive width from source aspect ratio.
- Both zero: keep source dimensions, snapped to H3 divisibility.
- resize_method is now owned by the Director rather than overridden by H3 Settings.

## v0.4.6
- Removed `divisible_by` from H3 Settings and Director.
- H3's required 32-pixel tensor alignment is now internal only.
- DHan-style width/height remain free UI values with `0 = AUTO`.
- Removed an accidental duplicate `resize_method` widget from the Director schema.

## v0.4.7 — timeline image execution fix
- Restored `_resolve_input_path`, `_load_image_tensor`, and `_resize_image` helpers accidentally lost in the sizing refactor.
- Timeline images now reach MiniMax H3 first-frame / timed-guide conditioning again.
- DHan 0=AUTO dimensions remain unchanged.


## v0.4.8 — Ref2VA REF VIDEO
- Repurposes the tuned DHan secondary video track as `REF VIDEO`.
- `Add Ref Video` and the REF VIDEO track appear only in Ref2VA mode.
- AUDIO/Inpaint remains removed from H3.
- REF VIDEO keeps DHan trim, thumbnail, drag/resize and serialization behavior.
- REF VIDEO order maps to `<Video 1>`, `<Video 2>`, `<Video 3>`.
- Reference Hub image output fixed to native keyed H3 Autogrow dictionaries for `<Picture 1>` ... `<Picture 9>`.
- Ref videos are decoded at 24 fps and passed to native `MiniMaxH3ReferenceToVideo`.


## v0.4.9 — clean mode-specific UI
- FL2VA shows only MAIN + Add Image + Add Text.
- Ref2VA renames MAIN to DIRECTING and shows Add Text + REF VIDEO + Add Ref Video.
- Add Image is hidden in Ref2VA because picture references come from Reference Hub.
- REF VIDEO lane is completely collapsed in FL2VA, removing the empty second-track area.
- `resize_method` is explicitly restored as a visible Director widget.


## v0.4.10 — definitive resize_method visibility fix
- Removed `resize_method` from the transplanted DHan global hidden-widget list.
- Explicitly restores it with DHan's native `showWidget()` after node creation.
- Re-restores it after workflow `onConfigure`.


## v0.4.11 — timeline cleanup
- Removed floating gap `+` buttons and their invisible click targets.
- Removed Retake/start/end/mark/help controls from the H3 toolbar.
- Kept snapping and settings controls.
- FL2VA remains MAIN + Add Image + Add Text.
- Ref2VA remains DIRECTING + Add Text + REF VIDEO + Add Ref Video.
- FL2VA image/video blocks are hidden from the DIRECTING lane while in Ref2VA, but preserved so switching back to FL2VA restores them.


## v0.4.12 — true timeline swap
- FL2VA and Ref2VA now use one visible timeline lane at a time.
- FL2VA: MAIN only.
- Ref2VA: REF VIDEO only; MAIN is collapsed to zero height.
- Restored DHan in-timeline `+` add buttons.
- FL2VA `+` menu: Image / Text / Paste Image.
- Ref2VA `+` menu: Ref Video only.
- Ref2VA toolbar hides Add Image/Add Text and shows Add Ref Video.


## v0.4.13 — mode-state isolation / UI cleanup
- FL2VA and Ref2VA now keep separate text/prompt segment banks.
- Switching modes saves the current mode's text blocks and loads the other mode's blocks.
- Clears selection, ghost, hover, and canvas backing-buffer state on mode switch.
- Prevents stale FL2VA imagery from flashing into Ref2VA after switching.
- Suppresses legacy/static IC-LoRA image segments from REF VIDEO.
- Text segments now have a permanent subtle outline when not selected.


## v0.4.14 — true FL2VA / Ref2VA prompt isolation
- Hooks the actual tuned DHan `commitChanges(skipRender=false)` method.
- FL2VA and Ref2VA use independent runtime segment banks.
- Banks are independently serialized into node properties for workflow persistence.
- Global Prompt is also independent per model mode.
- Switching modes saves the outgoing mode and loads only the incoming mode.
- Removed an old misplaced Ref2VA filter that had accidentally landed inside video-upload code.


## v0.4.15 — Ref2VA single-lane geometry fix
- Ref2VA gap/add logic now sees only the REF VIDEO lane.
- FL2VA gap/add logic sees only MAIN.
- Hidden FL2VA segments cannot render into Ref2VA.
- Hidden Ref2VA motion segments cannot render into FL2VA.
- Track geometry is normalized on every render to prevent inherited DHan callbacks from resurrecting hidden lanes.
- Disabled old multi-track divider/vertical-resize behavior for H3's single visible timeline lane.


## v0.4.16 — shared lane resize + Ref2VA images
- FL2VA and Ref2VA now start at the same timeline lane height.
- One shared lane-height property is used by both model modes.
- Restored vertical resizing with a dedicated H3 single-lane bottom-edge resize handler.
- Ref2VA REF lane accepts both static images and videos.
- Added `Add Ref Image` and `Add Ref Video` toolbar buttons in Ref2VA.
- Ref2VA in-timeline `+` menu offers Ref Image and Ref Video.
- Static REF-lane images are passed to native H3 as ordered `<Picture N>` references.
- Reference Hub pictures are ordered first; REF-lane pictures fill remaining slots up to 9.


## v0.4.17 — H3 lane naming/state cleanup
- MAIN lane renamed to FL2VA.
- REF lane renamed to REF2VA.
- Active lane is forced enabled according to model mode.
- Inherited DHan audio track remains disabled and Add Audio is hidden.
- Inherited track visibility toggles are hidden because H3 model mode now owns lane visibility.


## v0.4.18 — track-label reset fix
- Patched inherited DHan reset path that was forcing the FL2VA label back to `MAIN`.
- FL2VA visible lane is now always labeled `FL2VA`.
- Ref2VA visible lane is now always labeled `REF2VA`.
- Labels are re-applied during mode sync and sidebar-height updates so later DHan callbacks cannot revert them.


## v0.4.19 — Settings / Reference Hub cleanup
- Removed `ref_image_size` from Comfyui-DHan-Minimax H3 Settings.
- Reference Hub now owns reference-image sizing.
- Reference Hub modes: `match`, `max`, `diffusers`.
- Added shift presets:
  - Standard = 12 / 3
  - Turbo 544p / Ref2VA = 12 / 3
  - Turbo 768p FL2VA = 6 / 3
  - Custom = manual video/audio shift values
- Manual shift values are visually de-emphasized unless Custom is selected.
- Sampling steps and Turbo LoRA application remain external to the Director.


## v0.4.20 — long-timeline render windows
- Removed the hard failure when the project timeline exceeds 362 frames.
- Added `Comfyui-DHan-Minimax H3 Render Window`.
- Timelines can now be longer than 15 seconds while each H3 job remains <=362 frames.
- Default render-window size is 360 frames (15.0 seconds at H3's 24 fps).
- `window_index` selects which chunk of the long timeline is sent to H3.
- Optional overlap is supported between neighboring chunks.
- Optional `continuity_frame` can be supplied from the previous rendered chunk:
  - FL2VA uses it as the next chunk's first frame.
  - Ref2VA applies it as a frame-0 guide.
- If a long timeline has no Render Window node connected, the Director renders window 0 and logs a warning rather than throwing an error.


## v0.4.21 — experimental direct latent continuation
- Render Window can now take the previous sampled H3 `continuity_latent`.
- Carries a short video + audio latent tail directly into the next H3 window.
- Avoids decode -> resize -> VAE re-encode for the continuity context.
- Default direct context is 22 frames; valid H3 clip-grid sizes are 5, 22, 39, 56...
- Director outputs `trim_frames`; remove these reconstructed context frames before stitching the next clip.
- `continuity_frame` remains as a fallback when no latent is connected.
- Director now outputs `timeline_plan`, containing the full DHan timeline plus current window metadata. This is the contract for a later automatic chain executor.
- This is intentionally a testable continuation stage before moving sampling/stitching inside an automatic executor.


## v0.4.22 — one-queue Auto Render
- Long H3 sampling/chaining can now happen inside the DHan Director.
- `Comfyui-DHan-Minimax H3 Render Window` is removed from the normal node menu (old workflows still keep its class code for compatibility).
- Settings adds `render_mode`: `External Sampler` or `Auto Render`.
- Auto Render uses ComfyUI's stock H3-compatible sampling stack internally:
  BasicGuider + KSamplerSelect + BasicScheduler + SamplerCustomAdvanced.
- Defaults: `res_multistep` sampler + `beta` scheduler.
- The Director follows its own full timeline for every <=15s H3 window.
- Each following window receives direct previous AV-latent context.
- Sampled windows are latent-stitched and normalized to the full requested H3 temporal shape.
- New `rendered_latent` output is the completed sampled long AV latent; connect it directly to
  `VAEDecode` and `VAEDecodeAudio`.
- Existing model / positive / latent outputs are preserved for `External Sampler` mode.


## v0.4.23 — Ref2VA REF AUDIO timeline
- Ref2VA now shows two timeline lanes:
  - `REF2VA` for reference images/videos
  - `REF AUDIO` directly below for standalone reference audio
- `Add Ref Audio` is visible only in Ref2VA mode.
- REF AUDIO reuses the tuned DHan audio upload/waveform/trim/move behavior.
- Audio references are ordered left-to-right and passed to native H3 as:
  `ref_audio_1`, `ref_audio_2`, `ref_audio_3`.
- Prompt tags therefore map to `<Audio 1>`, `<Audio 2>`, `<Audio 3>`.
- FL2VA continues to hide/ignore the REF AUDIO lane.
- REF AUDIO is reference media, not frame-anchored guide audio; timeline position controls ordering/organization.


## v0.4.24 — suppress Auto Render node preview
- Auto Render no longer creates ComfyUI's live latent/KSampler preview on the bottom of the Director node.
- Uses the same current `SamplerCustomAdvanced` sampling path internally, but intentionally passes `callback=None`.
- Terminal/progress-bar sampling progress is preserved.
- Normal external KSampler/SamplerCustom previews elsewhere in the workflow are unaffected.


## v0.4.25 — Comfyui-DHan-H3 animated Preview Override
- Added `Comfyui-DHan-Minimax H3 Preview Override`.
- Based on the H3-specific packed AV latent preview approach from the supplied MiniMax preview node.
- Accepts both FL2VA and Ref2VA model branches and returns both patched models.
- Intended placement: model loaders -> Preview Override -> Comfyui-DHan-H3 Director.
- Works with both External Sampler and the Director's internal Auto Render.
- Auto Render accumulation:
  - first H3 window previews normally;
  - later windows display completed prior preview windows + the current denoising window;
  - carried continuation context is removed from the accumulated preview;
  - a 20-second Auto Render therefore grows into a full ~20-second preview rather than restarting at 0 for window 2.
- Default preview is `latent2rgb (fast)`, target `node`, playback `source fps`.
- `vae (quality)` is available when the MiniMax H3 video VAE is connected.
- ComfyUI's default single-frame preview can remain suppressed.


## v0.4.26 — Preview Override routing fix
- Added explicit preview status events:
  - `Preview wrapper attached`
  - `Sampling hook active`
- Preview routing now tolerates ComfyUI hidden `unique_id` differences during indirect Auto Render calls.
- When only one Comfyui-DHan-H3 Preview Override exists, preview/status packets safely fall back to that node.
- Added backend + browser-console diagnostics for the H3 OUTER_SAMPLE hook.
- Does not change sampling or generation behavior.


## v0.4.27 — single latent output
- Removed the separate `rendered_latent` Director output.
- The Director now exposes one `latent` socket for both workflows:
  - `External Sampler` -> normal unsampled H3 latent for your downstream sampler.
  - `Auto Render` -> internally sampled/chained full-duration H3 latent.
- Long-duration Auto Render no longer requires alternate latent wiring.
- Existing model / positive / fps / length / compiled_prompt / timeline_plan / trim_frames outputs remain unchanged.


## v0.4.28 — centralized Standard / Turbo sampling
- `Comfyui-DHan-Minimax H3 Settings` is now the single source of truth for sampling.
- New `sampling_preset`:
  - `Standard` = `res_multistep`, `beta`, 20 steps, shifts 12/3
  - `Turbo 4-Step` = Larryvrh `MiniMax-H3 Turbo Sampler`, `simple`, 4 steps, shifts 12/3
  - `Turbo 6-Step` = same Turbo sampler, `simple`, 6 steps
  - `Turbo 8-Step` = same Turbo sampler, `simple`, 8 steps
  - `Custom` = manual stock sampler / scheduler / steps / shifts
- Settings now outputs:
  - `h3_settings`
  - `sampler` (`SAMPLER`) for direct use with `SamplerCustomAdvanced`
  - `scheduler` (STRING) for your `BasicScheduler`
  - `steps` (INT) for your `BasicScheduler`
- Director Auto Render uses the exact same sampler factory/preset for every 15+ second internal window.
- Turbo preset requires Larryvrh/ComfyUI-MiniMax-H3-Turbo to be installed; the node resolves its registered `MiniMaxH3TurboSampler` dynamically.


## v0.4.29 — Settings outputs ready-to-use SIGMAS
- `Comfyui-DHan-Minimax H3 Settings` now accepts an optional active `model` input.
- Removed separate `scheduler` and `steps` outputs.
- Settings now outputs:
  - `h3_settings`
  - `sampler` (`SAMPLER`)
  - `sigmas` (`SIGMAS`)
- The SIGMAS output is built internally with ComfyUI's `BasicScheduler` using the selected preset:
  - Standard -> beta / 20
  - Turbo 4-Step -> simple / 4
  - Turbo 6-Step -> simple / 6
  - Turbo 8-Step -> simple / 8
  - Custom -> chosen scheduler / steps
- External workflow can now wire:
  - `Settings.sampler` -> `SamplerCustomAdvanced.sampler`
  - `Settings.sigmas` -> `SamplerCustomAdvanced.sigmas`
- This removes the need for visible `KSamplerSelect` and `BasicScheduler` nodes.
- Director Auto Render still uses the same `h3_settings` preset internally for every continuation pass.


## v0.4.30 — native Comfy SAMPLER / SIGMAS sockets
- Fixed `SamplerCustomAdvanced` prompt validation failure from v0.4.29.
- Replaced generic `io.Custom("SAMPLER")` and `io.Custom("SIGMAS")` outputs with
  current ComfyUI V3 native `io.Sampler.Output()` and `io.Sigmas.Output()`.
- Settings `model` input is now required when using this node, so its SIGMAS output
  is always a real BasicScheduler tensor rather than `None`.
- Standard/Turbo preset behavior is unchanged.
- The legacy `button.js` warning shown at startup is unrelated to this validation fix.


## v0.4.31 — simplified Director I/O
- Removed the visible `h3_window` input from `Comfyui-DHan-Minimax H3 Director`.
  Auto Render still uses private internal window controls for recursive continuation.
- Removed obsolete visible outputs:
  - `timeline_plan`
  - `trim_frames`
- Director outputs are now only:
  - `model`
  - `positive`
  - `latent`
  - `fps`
  - `length`
  - `compiled_prompt`
- Auto Render continues to return the full sampled/chained result through the same `latent` socket.
- `length` in Auto Render reports the full requested project frame count.
- In External Sampler mode, a >15s timeline now warns to use Auto Render instead of referring to the retired Render Window workflow.


## v0.4.32b — reverted Director, dual-model Settings passthrough
- Director is kept exactly on the v0.4.31 dual-model architecture:
  - `model (FL2VA)`
  - `model (Ref2VA)`
  - Director's own FL2VA / Ref2VA selector remains.
  - Timeline switching remains unchanged.
- Settings now accepts both model branches and passes both back out separately.
- Settings outputs:
  - `model (FL2VA)`
  - `model (Ref2VA)`
  - `h3_settings`
  - shared `sampler`
  - `sigmas (FL2VA)`
  - `sigmas (Ref2VA)`
- Both sigma schedules use the same Standard/Turbo preset, but are calculated from their own model branch.
- No merged/selected model output is used.


## v0.4.33 — one shared SIGMAS output
- `Comfyui-DHan-Minimax H3 Settings` still accepts and passes through both model branches separately.
- Director remains unchanged from the restored v0.4.31 architecture.
- Replaced `sigmas (FL2VA)` + `sigmas (Ref2VA)` with one shared `sigmas` output.
- The shared sigma schedule uses the same preset scheduler/steps for either H3 branch and is built from whichever connected model is available.
- Settings outputs are now:
  - `model (FL2VA)`
  - `model (Ref2VA)`
  - `h3_settings`
  - `sampler`
  - `sigmas`


## v0.4.34 — Ref2VA lane order
- UI-only change.
- Ref2VA mode now stacks lanes in this order:
  1. `REF2VA`
  2. `REF AUDIO`
- Audio is drawn directly underneath the Ref2VA media lane instead of above it.
- Backend reference conditioning, prompts, sampling, and Auto Render are unchanged.


## v0.4.35 — actual Ref2VA DOM lane order fix
- Fixed the underlying sidebar DOM order, not just canvas coordinates.
- Ref2VA now displays:
  1. `REF2VA`
  2. `REF AUDIO`
- Ref2VA toolbar now presents:
  `Add Ref Image` -> `Add Ref Video` -> `Add Ref Audio` -> `Delete`
- Backend behavior is unchanged.


## v0.4.36 — lazy single-model Settings + Director mismatch guard
- Settings now accepts lazy `model (FL2VA)` and `model (Ref2VA)` inputs.
- New Settings `model_select` chooses which heavy branch is actually evaluated.
- Settings outputs one selected `model`, plus `h3_settings`, `sampler`, and `sigmas`.
- Director now takes one model input from Settings.
- Director keeps its own FL2VA / Ref2VA selector so timeline switching behavior is unchanged.
- Director shows an inline warning when:
  - Director mode = FL2VA, Settings model_select = Ref2VA
  - or vice versa.
- Execution is blocked with a clear model-mismatch error until the two selectors match.
- This avoids evaluating both heavy H3 model branches just to run Settings.


## v0.4.37 — Director model socket cleanup
- Director now exposes exactly one model input, labeled simply `model`.
- Removed any lingering `model (FL2VA)` / `model (Ref2VA)` input labeling from the Director.
- Settings still selects the actual FL2VA or Ref2VA model branch.
- Director keeps its own FL2VA / Ref2VA mode selector solely for timeline/conditioning behavior.


## v0.4.38 — Director progress bar only
- Restores ComfyUI's native thin green progress bar during internal Auto Render sampling.
- Uses `ProgressBar.update_absolute(..., None)` so no image/latent preview is attached to the Director.
- Separate Comfyui-DHan-H3 Preview Override behavior is unchanged.


## v0.4.39 — no double sampling
- Director Auto Render now activates only when timeline length exceeds `max_window_frames`.
- <=15s/default-window jobs:
  - Director outputs an ordinary unsampled H3 latent.
  - `Comfyui-DHan-Minimax H3 Smart Sampler` performs exactly one sampling pass.
- >15s jobs:
  - Director internally samples/chains the H3 windows because continuation requires the
    previous sampled latent.
  - The completed latent is marked as already sampled.
  - `Comfyui-DHan-Minimax H3 Smart Sampler` detects that marker and passes the latent through unchanged.
- Added `Comfyui-DHan-Minimax H3 Smart Sampler`, a drop-in replacement for `SamplerCustomAdvanced`
  with the same five inputs and two latent outputs.
- This preserves one workflow for short and long generations without manual bypassing.


## v0.4.40 — Preview Override routing cleanup
- Preview Override now has one MODEL input and one MODEL output.
- Correct routing:
  FL2VA/Ref2VA loaders -> Comfyui-DHan-H3 Settings -> Comfyui-DHan-H3 Preview Override -> Comfyui-DHan-H3 Director
- Settings remains the only node selecting between the two heavy H3 model branches.
- This removes the dependency-cycle path created by the previous dual-model preview architecture.


## v0.4.41 — hard cleanup of Preview Override model sockets
- `Comfyui-DHan-Minimax H3 Preview Override` exposes exactly one `model` input and one `model` output.
- Removed lingering FL2VA/Ref2VA-specific preview socket labels/references.
- Intended routing:
  `Comfyui-DHan-H3 Settings.model -> Comfyui-DHan-H3 Preview Override.model -> Comfyui-DHan-H3 Director.model`
- If an existing workflow still shows the old dual sockets after updating, reload ComfyUI/frontend
  and recreate the Preview Override node once so the saved old node slot layout is discarded.


## v0.4.42 — single-model socket naming cleanup
- Comfyui-DHan-Minimax H3 Preview Override now explicitly labels its one model input as `model`.
- Its one model output is also explicitly labeled `model`.
- No `(FL2VA)` or `(Ref2VA)` suffix is used on single generic model sockets.
- Mode-specific labels remain reserved only for nodes that expose separate FL2VA / Ref2VA model sockets.

## v0.4.43 — Comfyui-DHan-H3 Prompt Enhancer
- Added a separate Ollama-powered `Comfyui-DHan-H3 Prompt Enhancer`.
- Optional start/reference image + simple idea + Parts (2-10) + duration.
- Produces two intentionally non-duplicative outputs:
  - `global_prompt`: persistent identity/environment/style/lighting/continuity context.
  - `storyboard`: exactly N numbered temporal action prompts for copy/paste into text keyframes.
- `global_prompt` connects directly to the Director `global_prompt` input.
- Includes copyable Global Prompt and Storyboard result boxes on the node.
- The hidden system prompt explicitly keeps global context and temporal actions separate.


## v0.4.44 — standalone Storyboard Enhancer
- Simplified `Comfyui-DHan-H3 Prompt Enhancer` to `Comfyui-DHan-H3 Storyboard Enhancer`.
- Removed the unused Global Prompt result/output.
- Marked the enhancer `is_output_node=True`, enabling ComfyUI's native selected-output run button.
- Seed is now fixed unless manually changed; no automatic randomize-after-generate behavior.
- Same unchanged inputs can use ComfyUI cache instead of reloading the VLM.
- Ollama is requested with `keep_alive=0` and receives an explicit unload request in `finally`,
  including after errors, so the model releases memory after a real generation.
- Replaced the overflowing two-box UI with one contained Storyboard result box.
- Storyboard result box is fixed inside the node frame with internal scrolling and Copy button.


## v0.4.45 — Prompt Enhancer output/UI correction
- Fixed duplicate `storyboard` output sockets.
- Enhancer now exposes exactly:
  - `global_prompt`
  - `storyboard`
- Restored a separate Global Prompt result container.
- Global Prompt and Storyboard containers are both constrained inside the node frame.
- Both result areas have independent Copy buttons and internal scrolling.
- Ollama unload-after-generation behavior remains unchanged.


## v0.4.46 — UI-only Storyboard Enhancer
- Removed all output sockets from `Comfyui-DHan-H3 Storyboard Enhancer`.
- The node remains `is_output_node=True`, so it can still be run with ComfyUI's native
  blue Execute-to-selected-output control.
- Generated `Global Prompt` and `Storyboard` exist only inside the node UI.
- Both remain copyable from their in-node containers.
- No downstream wiring is required or exposed.


## v0.4.47 — Ollama model dropdown
- `ollama_model` is now a dropdown instead of a typed string.
- Initial dropdown values are read from Ollama `/api/tags` on the default local server.
- Added `Refresh Models` inside the Storyboard Enhancer.
- Refresh uses the node's current `ollama_url`, so remote/custom Ollama servers are supported.
- The dropdown updates without restarting ComfyUI.
- If Ollama is unavailable, the node shows `<refresh models>` instead of assuming a model name.


## v0.4.48 — native H3 core alignment
- Aligned DHan FL2VA conditioning to the uploaded MiniMax Director/native Comfy H3 path.
- FL2VA now uses only first/last image anchors.
- Middle timeline images are no longer injected with `MiniMaxH3AddGuide` in FL2VA; they
  produce a warning directing the user to Ref2VA instead.
- Moved H3 SigmaShift ownership to Settings:
  - Settings selects the model.
  - Settings applies `MiniMaxH3SigmaShift`.
  - Settings builds SIGMAS from that exact patched model.
  - The same patched model is passed to Preview Override and Director.
- Director no longer applies SigmaShift a second time.
- This removes the previous possibility of sampling with SIGMAS built from a different
  ModelSamplingAV configuration than the model actually being sampled.
- Preview Override now defaults to `true speed`, spreading sampled preview frames over
  the real shot duration rather than playing them flat at source FPS.
- Added lightweight `[Comfyui-DHan-H3 PERF]` timing for native FL2VA/Ref2VA conditioning.


## v0.4.49 — reference-Director model loading architecture
- Settings no longer accepts or outputs MODEL.
- Director again owns lazy dual model inputs:
  - `model (FL2VA)`
  - `model (Ref2VA)`
- Director's existing FL2VA/Ref2VA switch selects the active heavy branch, matching the
  uploaded reference Director's lazy-loading architecture.
- Director applies native `MiniMaxH3SigmaShift` after conditioning and outputs the patched model.
- Preview Override now belongs after Director:
  `Director.model -> Preview Override.model -> Guider / Comfyui-DHan-H3 Smart Sampler`
- Settings now outputs only:
  - `h3_settings`
  - `sampler`
- Smart Sampler now accepts the patched model + h3_settings and builds BasicScheduler SIGMAS
  internally from the exact model being sampled.
- This removes the heavy MODEL dependency from Settings and should allow Comfy Dynamic VRAM
  to stage the H3 text encoder/model in the same order as the faster reference Director.
- Includes missing `import time` for the PERF timers.


## v0.4.50 — Director SIGMAS output
- Added `sigmas` output directly to the Director.
- SIGMAS are generated with ComfyUI `BasicScheduler` from the exact SigmaShift-patched H3 model.
- Scheduler + step count come from `h3_settings`.
- This restores the normal stock `SamplerCustomAdvanced` workflow:
  - Director `sigmas` -> SamplerCustomAdvanced `sigmas`
  - Settings `sampler` -> SamplerCustomAdvanced `sampler`
  - Director `latent` -> SamplerCustomAdvanced `latent_image`
- No need to use DHan Smart Sampler for normal external sampling.


## v0.4.51 — downstream Sampling Preset architecture
- Removed the old upstream `Comfyui-DHan-Minimax H3 Settings` node from registration.
- Added `Comfyui-DHan-H3 Sampling Preset` AFTER the Director.
- Sampling Preset takes `Director.model` and outputs:
  - `sampler`
  - `sigmas`
- Presets:
  - Standard
  - Turbo 4-Step
  - Turbo 6-Step
  - Turbo 8-Step
  - Custom
- Director now owns only H3/timeline/conditioning controls.
- `shift_video` and `shift_audio` moved directly onto Director.
- Director no longer builds sampler/scheduler/sigmas.
- Default Director behavior now matches the reference Director more closely:
  the full requested timeline is conditioned in one native H3 pass, including >15s,
  with a warning beyond the trained ~360-frame range.
- Recommended graph:
  `H3 model -> Director`
  `Director.model -> Preview Override -> Guider`
  `Director.model -> Comfyui-DHan-H3 Sampling Preset`
  `Sampling Preset.sampler -> SamplerCustomAdvanced.sampler`
  `Sampling Preset.sigmas -> SamplerCustomAdvanced.sigmas`
  `Director.latent -> SamplerCustomAdvanced.latent_image`


## v0.4.59 — exact v0.4.51 UI/schema + reference generation core
- Started from the user's restored v0.4.51 build.
- `define_schema()` for Comfyui-DHan-Minimax H3 Director is preserved exactly from v0.4.51.
- Frontend `js/minimax_h3_director.js` is preserved byte-for-byte from v0.4.51.
- Therefore Add buttons, cut/paste, Add Text, lane behavior, widget ordering, and UI styling
  remain exactly as in the restored baseline.
- Replaced only:
  - `check_lazy_status()`
  - Director `execute()`
- Processing uses reference MiniMax Director architecture/helpers for:
  - lazy active-model resolution
  - timeline planning
  - media loading
  - canvas policy
  - native FL2VA/Ref2VA conditioning
  - SigmaShift after conditioning


## v0.4.60 — simplified Director sockets
- Moved `model (FL2VA)` and `model (Ref2VA)` to the very top of the Director input list.
- Removed unused Director inputs:
  - global_prompt
  - ref_images
  - start
  - end
  - duration
- Director outputs simplified to:
  - model
  - positive
  - latent
  - fps
- Removed unused outputs:
  - height
  - length
  - prompt / compiled_prompt
  - retake_info
- Timeline UI/cut-paste/Add Text frontend JavaScript is unchanged.


## Comfyui-DHan-Minimax H3 Long Sampler (v0.5.28 experimental)

Use this node in place of `SamplerCustomAdvanced`.

- At or below the configured safe window (default 15s), it performs one normal sampling pass.
- Above the safe window, FL2VA is rebuilt into <=15s windows.
- Follow-up windows carry the previous sampled H3 video+audio latent tail directly as context.
- Context is trimmed from follow-up windows and the sampled AV latents are stitched back to the requested duration.
- The Director UI/sockets are unchanged; runtime planning metadata travels invisibly in the existing latent dictionary.
- Initial experimental long chaining is FL2VA-only. Ref2VA/Retake remain to be validated.


### v0.5.28
- Fixes multi-window audio stitching losing ~1 audio latent step per continuation seam from 24fps/40Hz grid rounding.
- Keeps video overlap trimming unchanged.
- Uses floor timing for stitched audio overlap and exact final AV cropping.
- Allows only a tiny <=4-step defensive audio pad for residual fractional-grid rounding.


### v0.5.28
- Long Sampler now sends Auto Render window/session metadata to Comfyui-DHan-H3 Preview Override.
- With `accumulate_auto_render` enabled, the preview shows completed prior windows plus the current in-progress window instead of resetting at each chunk.
- Long-render generation, overlap, AV stitching, and sampling math are unchanged from v0.4.95.


### v0.5.28
- Long Sampler now uses one continuous ComfyUI node progress bar across every internal render window.
- Example: a three-window job progresses 0→33→66→100 instead of resetting to 0 at each section.
- Internal SamplerCustomAdvanced calls, cumulative video preview, conditioning, and AV stitching are unchanged.


### v0.5.28
- Adds long-render visual reference refresh to reduce cumulative burn/drift across many continuation windows.
- Follow-up FL2VA windows re-feed the original Director opening image into Qwen multimodal conditioning.
- The previous sampled AV latent tail remains the actual frame-0 continuation anchor; conflicting image keyframes at frame 0 are removed.
- New `refresh_reference` toggle defaults ON and is appended after existing Long Sampler widgets for workflow compatibility.
- Cumulative preview, continuous progress, AV overlap/stitching, and <=15s passthrough behavior are unchanged.


### v0.5.28 — Negative Prompt / Comfyui-DHan-H3 Guider
- Director adds optional `negative_prompt` and a new `negative` CONDITIONING output.
- New `Comfyui-DHan-H3 Guider`: Negative Prompting OFF = BasicGuider; ON = CFGGuider.
- CFG defaults to 2.0 and is ignored while negative prompting is OFF.
- Long Sampler preserves Basic/CFG mode across continuation windows.
- New Director input/output are appended for workflow compatibility.


### v0.5.28 — Director Negative Prompt UI
- Negative Prompt ON/OFF switch moved to the Director directly below Global Prompt.
- Negative prompt editor is collapsed while OFF and expands while ON.
- Director switch is now the source of truth; Comfyui-DHan-H3 Guider automatically chooses BasicGuider/CFGGuider.
- CFG remains adjustable on Comfyui-DHan-H3 Guider, defaults to 2.0, and is ignored while Director Negative Prompt is OFF.
- Long Sampler preserves active CFG/negative guidance across continuation windows.


### v0.5.28 — Negative Prompt UI expansion fix
- Fixes the Director Negative Prompt textarea being clipped when the toggle is switched ON.
- The Director now explicitly grows/shrinks when the negative editor is shown/hidden.
- Forces a timeline DOM-widget re-measure after the visibility change and on initial load.
- Negative conditioning / CFG behavior is unchanged from v0.5.00.


### v0.5.28 — Negative textarea rendering fix
- Negative Prompt no longer reuses the Director's fixed-panel `.pr-prompt-area` class.
- Uses explicit standalone textarea styling with visible background, border, placeholder, padding, and fixed initial height.
- Keeps ON/OFF expansion behavior from v0.5.01.

### v0.5.28 — Multi-selection proportional retime
- Ctrl/Cmd multi-selected timeline blocks now show a group bounding box with left/right handles.
- Drag either group edge to proportionally scale selected block starts and durations around the opposite edge.
- Group edges snap to output duration, playhead, generation range, and non-selected segment boundaries.
- Existing single-block resize and multi-selection center drag remain unchanged.

### v0.5.28 — Phase-aligned H3 continuation experiment
- Long Sampler adds `phase_aligned_context` (default ON).
- Continuation no longer treats the previous multi-step tail as one frame-0 latent blob.
- It chooses a real H3 latent cutoff, extends backward to a phase-0 source start, and places each carried latent step at its native 1/4/4/4/4 pixel offset.
- Adds `safe_tail_bridge` (default 2 frames) to avoid using the very final tail as the handover.
- Actual aligned context can be longer than the requested 22 frames; window planning and latent stitching now use the actual per-window carried duration.
- Existing Qwen original-reference refresh remains enabled as a visual quality/identity reminder.
- This patch adapts the phase-aligned handover concept only; Herrgotts' separate freeze detector / Auto Handover and decoded video/audio crossfade stitcher are not yet integrated.


### v0.5.28 — Stable Long Handoff
- Removes the experimental v0.5.05 phase-aligned per-step handoff that caused immediate continuation flicker.
- Restores the last proven Long Sampler handoff: direct H3 video+audio latent tail context.
- Default context remains 22 frames.
- Original Director reference refresh remains available/default ON to reduce long-chain visual drift.
- Cumulative preview, continuous overall progress, AV-grid audio stitching fix, negative prompting/guider, and group timeline retiming remain intact.
- No CFG behavior is changed; CFG remains isolated to the optional negative-prompt guider.


### v0.5.28 — Herrgotts Core continuation
- Adds `continuation_mode`: `Herrgotts Core` (default) or `Stable DHan`.
- Herrgotts Core uses the upstream phase-aligned-extended H3 latent math.
- Carried VIDEO is injected as separate H3 keyframes at canonical offsets.
- Carried AUDIO is injected separately through `minimax_refs` with timeline end metadata.
- Original Director reference is Qwen-only during continuation; it is no longer a competing frame-0 latent anchor.
- Uses Herrgotts runtime compatibility detection: stock arbitrary keyframes on native ComfyUI H3, compatibility marker/layout wrappers on legacy APIs.
- Exact video/audio context-step counts are used when stitching each seam.
- `Stable DHan` keeps the previous known-good direct 22-frame blob handoff as a fallback.
- Freeze-aware Auto Handover and decoded crossfade stitching are NOT integrated yet; this release validates the corrected direct AV handoff core first.

### Masked AV long-video continuation
- `Masked AV` is an optional Long Sampler mode. Each extension copies the previous sampled video and audio latent tails into the target and protects that overlap with ComfyUI's native AV denoise masks. Start comparisons with a 39-frame context.
- New and saved Long Samplers retain `Herrgotts Core` as the default while Masked AV performance and output are evaluated.
- A short final extension generates one full H3 temporal step and trims excess frames during assembly.

Third-party note:
`herrgotts_bridge/` contains unmodified helper modules from
Herrgotts-H3-Infinite-Continuation-Suite and its GPL-3.0 license.
Upstream: https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite


### v0.5.28 — Storyboard Authoring Test
- Storyboard Enhancer now treats each part as an exact timed edit window and explicitly tells the VLM to keep actions realistic for that duration.
- Adds optional image attachment slots for each storyboard part; attached images are sent to the multimodal Ollama model with explicit part association.
- Enhancer results include structured start/end/duration/prompt/image metadata and a timed-card preview.
- Adds Target Director selector and `Apply Storyboard to Director` button.
- First test implementation applies to FL2VA only: it sets Director duration/global prompt and replaces the main timeline with timed text/image segments.
- Applied timeline remains fully editable with normal Director tools, including group retiming.


### v0.5.28 — Storyboard Authoring UX
- 1–10 parts; Easy/Medium/Advanced detail.
- Independent images assigned to any Part; multiple images per part; picker, drag/drop, paste.
- Idea resizes independently. Results flex/scroll with node height; Apply footer stays fixed.


### v0.5.28 — Storyboard UI cleanup
- Removes redundant image input socket; all storyboard image references use the built-in reference panel.
- Image paste inside the enhancer now stops propagation, preventing an extra ComfyUI Load Image node.
- Results area now flexes with node height; the Apply button remains fixed at the bottom.
- Removes duplicate raw Storyboard textarea; timed result cards are the primary storyboard output view.
- Idea textarea remains independently vertically resizable.
- Parts schema now allows 1–10 parts.


### v0.5.28 — Storyboard resize feedback-loop fix
- Fixes runaway node height growth caused by computeSize reading node.size and feeding that height back into ComfyUI layout.
- Storyboard DOM widget now reserves a stable layout height.
- Manual node resizing only changes the internal wrapper/results viewport height; it no longer changes the widget's requested layout height.
- Initial load no longer forces node height upward; only a sane minimum width is enforced.
- Apply footer remains fixed and Results remain internally scrollable.


### v0.5.28 — Storyboard one-way resize model
- Fixes gray dead space under the Storyboard UI when the node is taller than the fixed DOM reservation.
- Manual node resize now updates a stored `sbWidgetHeight` exactly once.
- `computeSize()` returns only the stored height and never derives a new value from node.size during layout.
- Adds a resize guard so ComfyUI's own relayout callback cannot create a height feedback loop.
- On workflow load, storyboard height is inferred once from the saved node height, then remains stable until the user resizes.
- Results remain elastic/scrollable and Apply remains pinned to the Storyboard UI footer.


### v0.5.28 — Storyboard free-resize model
- Removes the stored-height sizing model from v0.5.12.
- ComfyUI/manual dragging now owns node height completely.
- Storyboard DOM widget advertises only a modest fixed reservation so it no longer blocks shrinking.
- Visible Storyboard UI fills the actual remaining node interior height on resize/layout without ever calling setSize from the fit routine.
- DOM host height is explicitly matched to the available interior to eliminate gray dead space when enlarging the node.
- Results area minimum height reduced so the node can collapse substantially while keeping Apply visible.


### v0.5.28 — Storyboard Results flex fix
- Fixes Results clipping when enlarging the Storyboard Enhancer node.
- Removes max-height clipping from the DOM widget host and its parent wrapper.
- Results becomes the sole flexible region (`flex: 1 1 0`) while toolbar, Idea, Image References, and Apply footer remain fixed.
- Apply footer is explicitly non-shrinking and remains visible.
- Lowers the DOM reservation/minimums again so the node can still be resized smaller.


### v0.5.28 — InlineImage resize architecture
- Replaces the Storyboard Enhancer resize experiments with the same one-way pattern used by the working ComfyUI Inline Image node.
- `computeSize()` now reports only a stable minimum and never reads `node.size`.
- `onResize()` directly recalculates the Storyboard viewport from the current node dimensions.
- Only true minimum node dimensions are clamped; growing never becomes a new minimum, so the node remains shrinkable.
- No parent DOM host height/max-height manipulation and no stored height state.
- Results remains the elastic region and Apply remains fixed at the bottom.

### v0.5.28 — AI Pacing
- Adds Timing Mode: Equal / AI Pacing.
- Equal keeps the current equal-duration split.
- AI Pacing asks the model to allocate duration by story/action complexity rather than splitting evenly.
- AI timings are normalized to the exact requested total runtime, exact part count, contiguous boundaries, and a practical minimum beat duration.
- Apply Storyboard uses those normalized timings directly.

### v0.5.28 — Ollama timeout control
- Adds `timeout_seconds` to Storyboard Enhancer, default 1800 seconds (30 minutes).
- Replaces the previous hard-coded 900-second timeout.
- Timeout failures now report a clear DHan message.
- AI Pacing behavior is unchanged.


### v0.5.28 — Image persistence on browser refresh
- Director now restores image previews from persistent `imageFile` paths and rebuilds fresh ComfyUI `/view` URLs after browser refresh.
- Old transient blob URLs are no longer required for Director image restoration.
- H3 mode-bank switches also rehydrate saved image objects.
- Storyboard Enhancer image references now persist redundantly in both the hidden widget and node properties.
- Storyboard refs are restored again after ComfyUI workflow configuration/widget values finish loading, preventing refresh-time initialization from wiping them.

### v0.5.28 — Ollama connection handling
- Separates localhost/Ollama connection failures from model inference timeouts.
- Adds a fast `/api/tags` health check with short retries before `/api/chat`.
- Uses a 15-second connection timeout while retaining the 1800-second generation timeout.
- Only reports a generation timeout after Ollama actually accepted the connection.


### v0.5.28 — Storyboard widget migration fix
- Fixes existing workflows mapping saved `part_images_json` into the new `timeout_seconds` INT widget.
- `timeout_seconds` is now appended after the pre-existing serialized Storyboard widgets.
- Existing saved image-reference JSON remains mapped to `part_images_json`; timeout falls back to its 1800-second default.


### v0.5.28 — Timeout widget removal / workflow compatibility
- Removes `timeout_seconds` from the serialized Storyboard node widgets entirely.
- Restores the pre-timeout widget layout so existing saved `part_images_json` values map correctly again.
- Keeps the generation timeout internally fixed at 1800 seconds (30 minutes).
- Retains v0.5.19 connection-health diagnostics and separate connection/inference timeout reporting.


### v0.5.28 — Timeout cleanup fix
- Fixes the leftover `timeout_seconds` diagnostic reference that caused an immediate NameError in v0.5.21.
- Storyboard Enhancer now uses only the internal `request_timeout = 1800` value.
- No serialized timeout widget is reintroduced, preserving existing workflow/image-reference compatibility.

### v0.5.28 — Timeout initialization fix
- Moves the internal `request_timeout = 1800` assignment before every diagnostic/request use.
- Fixes the immediate UnboundLocalError in v0.5.22.
- Keeps timeout internal; no serialized timeout widget is added.

### v0.5.28 — Storyboard full state persistence
- Persists Idea text independently of live DOM state.
- Persists Parts, Duration, Timing Mode, Detail Level, Ollama model/url, seed, and max image size.
- Keeps image references/part assignments persistent.
- Persists the most recent Global Prompt and timed storyboard result cards.
- Restores state after ComfyUI onConfigure with a delayed second pass to handle browser-refresh initialization order.
- Adds an onSerialize backstop so current custom UI state is copied into workflow properties before save.


### v0.5.28 — Storyboard Idea backend sync fix
- Hidden backend `idea` widget now serializes directly from the visible custom Idea textarea.
- Refresh restore pushes saved Idea text back through the real ComfyUI widget callback.
- Workflow serialization explicitly synchronizes the custom Idea DOM value into the backend widget.

### v0.5.28 — Reference image refresh persistence
- Captures Storyboard node properties at prototype `onConfigure` time, before custom DOM initialization can race workflow restoration.
- Prevents the empty initial reference list from overwriting saved `dhanStoryboardImageRefs`.
- Reference rendering is now hydration-aware: saved refs are loaded first, then normal UI synchronization is enabled.
- Exposes a stable restore callback used by ComfyUI configuration passes after browser refresh.


### v0.5.28 — Storyboard manual-only execution
- Storyboard Enhancer is forced to LiteGraph NEVER mode during normal workflow operation, so standard Queue no longer runs Ollama.
- ComfyUI's native Execute-to-selected-output path is detected through `queuePrompt(..., {queueNodeIds})`; the selected Storyboard node is temporarily enabled only while that prompt is serialized.
- Adds a `Run Storyboard` button that uses the same selected-output queue path as a convenient fallback.
- Workflow reloads from regression builds are forced back to manual-only mode.
- All v0.5.26 reference/Idea persistence behavior is retained.


### v0.5.28 — Non-repeating continuation prompt windows
- Separates Herrgotts Core latent/media overlap from storyboard prompt scheduling.
- Continuation windows still carry the validated overlap/context latent for seamless motion.
- Text prompts now begin at the first genuinely new output frame instead of backing up into already-consumed storyboard beats.
- Prevents a prompt near a 15-second long-video boundary from being compiled and issued again in the next continuation window.
- Adds console diagnostics showing the new-only prompt range separately from the overlapping latent/media range.
