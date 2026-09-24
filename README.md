# Comfyui-DHan-Minimax H3 Director

Timeline-based MiniMax H3 workflows for ComfyUI. The Director supports FL2VA, Ref2VA, and Retake modes, with companion nodes for reference media, sampling, guidance, long-video continuation, animated preview, and Ollama-assisted storyboarding.

## Installation

Clone this repository into `ComfyUI/custom_nodes/Comfyui-DHan-Minimax-H3-Director`, then restart ComfyUI. This package does not include MiniMax H3 model weights. The Storyboard Enhancer's Ollama features require a separately running Ollama server.

## Nodes

- **DHan-Minimax H3 Director** — timeline editor and conditioning for FL2VA, Ref2VA, and Retake.
- **DHan-Minimax H3 Reference Hub** — reference-image and reference-video inputs.
- **DHan-H3 Sampling Preset** — H3 sampling settings.
- **DHan-H3 Guider** — positive/negative guidance routing.
- **DHan-Minimax H3 Long Sampler** — multi-window continuation.
- **DHan-Minimax H3 Preview Override** — animated sampling preview.
- **DHan-H3 Storyboard Enhancer** — optional Ollama-assisted prompt planning.

The registered node IDs have been renamed with the `DHan` prefix. Workflows saved with the older IDs will need the nodes replaced and reconnected.

See [CHANGELOG.md](CHANGELOG.md) for the development history. The `herrgotts_bridge` directory contains vendored code from [Herrgotts-H3-Infinite-Continuation-Suite](https://github.com/HerrgottMargott/Herrgotts-H3-Infinite-Continuation-Suite) under GPL-3.0; its license is included in that directory.
