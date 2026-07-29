from __future__ import annotations

import re
import torch

try:
    import comfy.utils
    _HAVE_COMFY = True
except Exception:
    _HAVE_COMFY = False



class ImageResolutionCap:

    downscale_methods = ["area", "lanczos", "nearest-exact", "bilinear", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution_cap": ("INT", {
                    "default": 2048,
                    "min": 16,
                    "max": 16384,
                    "step": 1,
                }),
                "downscale_method": (s.downscale_methods, {"default": "area"}),
                "cap_longest_side": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "cap_resolution"
    CATEGORY = "Rebalance-Pack/image"
    ESSENTIALS_CATEGORY = "Image Tools"
    SEARCH_ALIASES = ["resolution cap", "cap resolution", "downscale cap", "max resolution"]

    def cap_resolution(
        self,
        image,
        resolution_cap: int,
        downscale_method: str,
        cap_longest_side: bool,
    ):
        if image is None:
            return (image,)

        _, h, w, _ = image.shape

        if cap_longest_side:
            longest = max(w, h)
            if longest <= resolution_cap:
                return (image,)
            scale = resolution_cap / longest
        else:
            shortest = min(w, h)
            if shortest <= resolution_cap:
                return (image,)
            scale = resolution_cap / shortest

        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))

        method = downscale_method

        if _HAVE_COMFY:
            samples = image.movedim(-1, 1)
            out = comfy.utils.common_upscale(samples, new_w, new_h, method, "disabled")
            out = out.movedim(1, -1)
        else:
            samples = image.movedim(-1, 1)
            mode = method if method != "lanczos" else "bicubic"
            out = torch.nn.functional.interpolate(samples, size=(new_h, new_w), mode=mode)
            out = out.movedim(1, -1)

        return (out,)



def _parse_resolutions(text: str) -> list[tuple[int, int]]:

    pairs: list[tuple[int, int]] = []
    for token in re.split(r"[,\n]+", text or ""):
        token = token.strip()
        if not token:
            continue
        match = re.match(r"^(\d+)\s*[xX\*:]\s*(\d+)$", token)
        if not match:
            continue
        w, h = int(match.group(1)), int(match.group(2))
        if w > 0 and h > 0:
            pairs.append((w, h))
    return pairs



class ImageAspectRatioCrop:

    crop_positions = ["center", "top", "bottom", "left", "right"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "aspect_ratios": ("STRING", {
                    "default": "1000x1000\n768x1344\n1344x768\n832x1216\n1216x832",
                    "multiline": True,
                }),
                "crop_position": (s.crop_positions, {"default": "center"}),
                "resize_to_target": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "chosen_resolution")
    FUNCTION = "crop_aspect"
    CATEGORY = "Rebalance-Pack/image"
    ESSENTIALS_CATEGORY = "Image Tools"
    SEARCH_ALIASES = ["aspect ratio crop", "crop aspect", "aspect ratio", "ratio crop"]

    def crop_aspect(
        self,
        image,
        aspect_ratios: str,
        crop_position: str,
        resize_to_target: bool,
    ):
        if image is None:
            return (image, "0x0")

        resolutions = _parse_resolutions(aspect_ratios)
        if not resolutions:
            return (image, "0x0")

        _, h, w, _ = image.shape
        img_ratio = w / h

        target_w, target_h = min(
            resolutions,
            key=lambda r: abs((r[0] / r[1]) - img_ratio),
        )
        target_ratio = target_w / target_h

        if w / h > target_ratio:
            new_w = int(round(h * target_ratio))
            new_h = h
        else:
            new_h = int(round(w / target_ratio))
            new_w = w

        new_w = max(1, min(new_w, w))
        new_h = max(1, min(new_h, h))

        if crop_position == "left":
            x1 = 0
        elif crop_position == "right":
            x1 = w - new_w
        else:
            x1 = (w - new_w) // 2

        if crop_position == "top":
            y1 = 0
        elif crop_position == "bottom":
            y1 = h - new_h
        else:
            y1 = (h - new_h) // 2

        x2 = x1 + new_w
        y2 = y1 + new_h

        out = image[:, y1:y2, x1:x2, :]

        if resize_to_target:
            if _HAVE_COMFY:
                samples = out.movedim(-1, 1)
                out = comfy.utils.common_upscale(
                    samples, target_w, target_h, "lanczos", "disabled"
                )
                out = out.movedim(1, -1)
            else:
                samples = out.movedim(-1, 1)
                out = torch.nn.functional.interpolate(
                    samples, size=(target_h, target_w), mode="bicubic"
                )
                out = out.movedim(1, -1)

        return (out, f"{target_w}x{target_h}")



class MaskAspectRatioBBox:

    crop_positions = ["center", "top", "bottom", "left", "right"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "aspect_ratios": ("STRING", {
                    "default": "1000x1000\n768x1344\n1344x768\n832x1216\n1216x832",
                    "multiline": True,
                }),
                "crop_position": (s.crop_positions, {"default": "center"}),
                "enforce_min_pixels": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("MASK", "chosen_resolution")
    FUNCTION = "bbox_aspect"
    CATEGORY = "Rebalance-Pack/mask"
    ESSENTIALS_CATEGORY = "Mask Tools"
    SEARCH_ALIASES = ["mask aspect ratio", "mask bbox aspect", "bbox aspect", "mask ratio"]

    @staticmethod
    def _candidate_rect(bw, bh, target_ratio):

        if bw / bh > target_ratio:
            crop_w = bh * target_ratio
            crop_h = float(bh)
            pad_w = float(bw)
            pad_h = bw / target_ratio
        else:
            crop_w = float(bw)
            crop_h = bw / target_ratio
            pad_w = bh * target_ratio
            pad_h = float(bh)
        return crop_w, crop_h, pad_w, pad_h

    @staticmethod
    def _place(new_w, new_h, bx1, by1, bx2, by2, bcx, bcy,
               crop_position, w, h):

        if new_w > w or new_h > h:
            return None

        if crop_position == "left":
            x1 = bx1
        elif crop_position == "right":
            x1 = bx2 - new_w
        else:
            x1 = int(round(bcx - new_w / 2))

        if crop_position == "top":
            y1 = by1
        elif crop_position == "bottom":
            y1 = by2 - new_h
        else:
            y1 = int(round(bcy - new_h / 2))

        x1 = max(0, min(x1, w - new_w))
        y1 = max(0, min(y1, h - new_h))

        return x1, y1, x1 + new_w, y1 + new_h

    def bbox_aspect(
        self,
        mask,
        aspect_ratios: str,
        crop_position: str,
        enforce_min_pixels: bool,
    ):
        if mask is None:
            return (mask, "0x0")

        resolutions = _parse_resolutions(aspect_ratios)
        if not resolutions:
            return (mask, "0x0")

        squeezed = False
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
            squeezed = True
        b, h, w = mask.shape

        out = torch.zeros_like(mask)
        chosen = "0x0"

        for i in range(b):
            frame = mask[i]
            ys, xs = torch.where(frame > 0)
            if ys.numel() == 0:
                continue

            by1 = int(ys.min().item())
            by2 = int(ys.max().item()) + 1
            bx1 = int(xs.min().item())
            bx2 = int(xs.max().item()) + 1
            bw = bx2 - bx1
            bh = by2 - by1
            bcx = bx1 + bw / 2
            bcy = by1 + bh / 2
            bbox_ratio = bw / bh
            bbox_area = bw * bh

            ordered = sorted(
                resolutions,
                key=lambda r: abs((r[0] / r[1]) - bbox_ratio),
            )

            placed = None
            chosen = "0x0"

            for target_w, target_h in ordered:
                target_ratio = target_w / target_h
                crop_w, crop_h, pad_w, pad_h = self._candidate_rect(
                    bw, bh, target_ratio
                )
                crop_area = crop_w * crop_h
                pad_area = pad_w * pad_h

                if (bbox_area - crop_area) <= (pad_area - bbox_area):
                    new_w, new_h = crop_w, crop_h
                else:
                    new_w, new_h = pad_w, pad_h

                if enforce_min_pixels:
                    new_w = max(new_w, float(target_w))
                    new_h = max(new_h, float(target_h))

                new_w = max(1, int(round(new_w)))
                new_h = max(1, int(round(new_h)))

                rect = self._place(
                    new_w, new_h, bx1, by1, bx2, by2, bcx, bcy,
                    crop_position, w, h,
                )
                if rect is None:
                    continue

                placed = rect
                chosen = f"{target_w}x{target_h}"
                break

            if placed is not None:
                x1, y1, x2, y2 = placed
                out[i, y1:y2, x1:x2] = 1.0

        if squeezed:
            out = out.squeeze(0)

        return (out, chosen)



NODE_CLASS_MAPPINGS = {
    "ImageResolutionCap": ImageResolutionCap,
    "ImageAspectRatioCrop": ImageAspectRatioCrop,
    "MaskAspectRatioBBox": MaskAspectRatioBBox,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResolutionCap": "Image Resolution Cap",
    "ImageAspectRatioCrop": "Image Aspect Ratio Crop",
    "MaskAspectRatioBBox": "Mask Aspect Ratio BBox",
}
