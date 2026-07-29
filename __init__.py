"""
Rebalance-Pack - merged ComfyUI custom node package.

    omni_nodes.py           -> OmniNode
    foundational.py         -> quickwork utility nodes
    resize_toolkit.py       -> ImageResolutionCap, ImageAspectRatioCrop,
                               MaskAspectRatioBBox
    mask_toolkit.py         -> UncropImage, UncropMask, BorderMaskDetector
    conditioning_rebalance.py -> RebalanceGuider, StepRebalance, RebalanceCFG,
                               ConditioningMerge, ConditioningMergeMulti
    krea2.py                -> ConditioningKrea2Rebalance, Krea2EditRebalance,
                               Krea2EncodeRebalance
    ideogram4.py            -> ConditioningIdeogram4Rebalance,
                               Ideogram4EditRebalance, Ideogram4EncodeRebalance

"""

from .omni_nodes import OmniNode
from . import foundational
from . import resize_toolkit
from . import mask_toolkit
from . import conditioning_rebalance
from . import krea2
from . import ideogram4

NODE_CLASS_MAPPINGS = {
    "OmniNode": OmniNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniNode": "Omni Node",
}

for _module in (foundational, resize_toolkit, mask_toolkit,
                conditioning_rebalance, krea2, ideogram4):
    NODE_CLASS_MAPPINGS.update(getattr(_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

WEB_DIRECTORY = "./web"
