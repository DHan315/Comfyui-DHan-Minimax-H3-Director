from .dhan_h3_prompt_enhancer import DHanH3PromptEnhancer
from .dhan_h3_preview import DHanMiniMaxH3PreviewOverride
from .minimax_h3_nodes import DHanH3SamplingPreset, DHanH3Guider, DHanMiniMaxH3Director, DHanMiniMaxH3ReferenceHub, DHanMiniMaxH3LongSampler
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

class DHanMiniMaxH3Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DHanMiniMaxH3Director, DHanH3Guider, DHanH3SamplingPreset, DHanMiniMaxH3LongSampler, DHanMiniMaxH3ReferenceHub, DHanMiniMaxH3PreviewOverride, DHanH3PromptEnhancer]

async def comfy_entrypoint() -> DHanMiniMaxH3Extension:
    return DHanMiniMaxH3Extension()

NODE_CLASS_MAPPINGS = {"DHanMiniMaxH3Director": DHanMiniMaxH3Director, "DHanH3SamplingPreset": DHanH3SamplingPreset, "DHanH3Guider": DHanH3Guider, "DHanMiniMaxH3LongSampler": DHanMiniMaxH3LongSampler, "DHanMiniMaxH3ReferenceHub": DHanMiniMaxH3ReferenceHub, "DHanMiniMaxH3PreviewOverride": DHanMiniMaxH3PreviewOverride, "DHanH3PromptEnhancer": DHanH3PromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"DHanMiniMaxH3Director": "Comfyui-DHan-Minimax H3 Director", "DHanH3SamplingPreset": "Comfyui-DHan-H3 Sampling Preset", "DHanH3Guider": "Comfyui-DHan-H3 Guider", "DHanMiniMaxH3LongSampler": "Comfyui-DHan-Minimax H3 Long Sampler", "DHanMiniMaxH3ReferenceHub": "Comfyui-DHan-Minimax H3 Reference Hub", "DHanMiniMaxH3PreviewOverride": "Comfyui-DHan-Minimax H3 Preview Override", "DHanH3PromptEnhancer": "Comfyui-DHan-H3 Storyboard Enhancer"}
WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
