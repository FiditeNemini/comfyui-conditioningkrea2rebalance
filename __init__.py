"""
Rebalance-Pack - merged ComfyUI custom node package.

    omni_nodes.py           -> OmniNode
    foundational.py         -> quickwork utility nodes
    utilities.py            -> LoadImagesBlaze, ListDisplay
    model_toolkit.py        -> LoraLoaderBlock
    resize_toolkit.py       -> ImageResolutionCap, ImageAspectRatioCrop,
                               MaskAspectRatioBBox
    mask_toolkit.py         -> UncropImage, UncropMask, BorderMaskDetector, Mask Opacity
    conditioning_rebalance.py -> RebalanceGuider, StepRebalance, RebalanceCFG,
                               ConditioningMerge, ConditioningMergeMulti
    krea2.py                -> ConditioningKrea2Rebalance, Krea2EditRebalance,
                               Krea2EncodeRebalance
    ideogram4.py            -> ConditioningIdeogram4Rebalance,
                               Ideogram4EditRebalance, Ideogram4EncodeRebalance

"""

from .omni_nodes import OmniNode, OmniNodeGuide, OmniNodeAPI
from . import foundational
from . import utilities
from . import model_toolkit
from . import resize_toolkit
from . import mask_toolkit
from . import conditioning_rebalance
from . import krea2
from . import ideogram4

NODE_CLASS_MAPPINGS = {
    "OmniNode": OmniNode,
    "OmniNodeGuide": OmniNodeGuide,
    "OmniNodeAPI": OmniNodeAPI,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniNode": "Omni Node Manual",
    "OmniNodeGuide": "Omni Node Guide",
    "OmniNodeAPI": "Omni Node",
}

for _module in (foundational, utilities, resize_toolkit, mask_toolkit,
                model_toolkit, conditioning_rebalance, krea2, ideogram4):
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"
