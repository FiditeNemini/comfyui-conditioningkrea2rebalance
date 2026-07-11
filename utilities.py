"""Utility nodes for the Conditioning Rebalance custom node pack."""

import os
import hashlib

import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

try:
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


_VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.gif'}


class _AnyType(str):
    """Wildcard type that matches any ComfyUI socket type."""

    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


_ANY = _AnyType("*")


def _normalize_path(path):
    """Normalize a path: strip whitespace, unify separators..."""
    if not path:
        return path
    path = path.strip()
    path = path.replace('\\', '/')
    path = os.path.normpath(path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def _load_single_image(file_path):
    """
    Mirrors the behavior of ComfyUI's LoadImage but operates on an arbitrary
    path. Returns None for the image if the file could not be loaded.
    """
    try:
        img = node_helpers.pillow(Image.open, file_path)
    except Exception as e:
        print(f"Warning: Could not load image {file_path}: {e}")
        return None, None, None

    i = node_helpers.pillow(ImageOps.exif_transpose, img)

    # Handle 16-bit images (mode 'I')
    if i.mode == 'I':
        i = i.point(lambda x: x * (1 / 65535))

    image = i.convert("RGB")
    w, h = image.size

    image_np = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np)[None,]

    if 'A' in i.getbands():
        mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask_np)
        mask_inverted = 1.0 - mask
    elif i.mode == 'P' and 'transparency' in i.info:
        rgba = i.convert('RGBA')
        mask_np = np.array(rgba.getchannel('A')).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask_np)
        mask_inverted = 1.0 - mask
    else:
        mask = torch.zeros((h, w), dtype=torch.float32, device="cpu")
        mask_inverted = 1.0 - mask

    return image_tensor, mask.unsqueeze(0), mask_inverted.unsqueeze(0)


def _read_sibling_txt(image_path):

    base, _ = os.path.splitext(image_path)
    txt_path = base + ".txt"
    if not os.path.isfile(txt_path):
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not read txt file {txt_path}: {e}")
        return ""


class LoadImages:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "directory_path": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "path/to/directory..",
                "tooltip": "Path to the directory containing images.",
            }),
            "start_index": ("INT", {
                "default": 0,
                "min": 0,
                "step": 1,
                "tooltip": "Index of the first image to load (sorted alphabetically).",
            }),
            "cap": ("INT", {
                "default": 4,
                "min": 1,
                "max": 1024,
                "step": 1,
                "tooltip": "Number of images to load.",
            }),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "filename", "caption")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "main"
    CATEGORY = "conditioning/utils"

    def main(self, directory_path, start_index, cap):
        normalized_path = _normalize_path(directory_path)

        if not normalized_path or not os.path.isdir(normalized_path):
            raise ValueError(f"Invalid directory path: {directory_path}")

        # Gather valid image files
        files = []
        for f in os.listdir(normalized_path):
            ext = os.path.splitext(f)[1].lower()
            if ext in _VALID_EXTENSIONS:
                files.append(os.path.join(normalized_path, f))

        files.sort()

        end_index = start_index + cap
        selected_files = files[start_index:end_index]

        if not selected_files:
            raise ValueError(
                f"No images found in range [{start_index}:{end_index}] in directory: {directory_path}"
            )

        output_images = []
        output_masks = []
        output_filenames = []
        output_captions = []

        for file_path in selected_files:
            image_tensor, mask, mask_inverted = _load_single_image(file_path)
            if image_tensor is None:
                continue

            output_images.append(image_tensor)
            output_masks.append(mask)
            output_filenames.append(os.path.basename(file_path))
            output_captions.append(_read_sibling_txt(file_path))

        if not output_images:
            raise ValueError("No valid images loaded (checked dimensions and validity).")

        # Return as lists so images of differing resolutions are kept separate
        # (each item is its own [1, H, W, C] tensor) rather than stacked.
        return (output_images, output_masks, output_filenames, output_captions)

    @classmethod
    def IS_CHANGED(cls, directory_path, start_index, cap):
        normalized_path = _normalize_path(directory_path)
        if not normalized_path or not os.path.isdir(normalized_path):
            return ""

        files = []
        try:
            for f in os.listdir(normalized_path):
                ext = os.path.splitext(f)[1].lower()
                if ext in _VALID_EXTENSIONS:
                    files.append(os.path.join(normalized_path, f))
        except Exception:
            return float("NaN")

        files.sort()
        end_index = start_index + cap
        selected_files = files[start_index:end_index]

        m = hashlib.sha256()
        for p in selected_files:
            try:
                m.update(p.encode('utf-8'))
                m.update(str(os.path.getmtime(p)).encode('utf-8'))
            except Exception:
                pass
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, directory_path, start_index, cap):
        if directory_path is None:
            return True
        if not isinstance(directory_path, str):
            return True
        normalized_path = _normalize_path(directory_path)
        if not normalized_path:
            return True
        if not os.path.isdir(normalized_path):
            return True
        return True


class Any:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "any": (_ANY, {}),
        }}

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("any",)
    FUNCTION = "main"
    CATEGORY = "conditioning/utils"

    def main(self, any):
        return (any,)


class Input:

    SEARCH_ALIASES = [
        "string",
        "text",
        "text box",
        "prompt",
        "multiline",
        "input text",
        "float",
        "INT",
        "text string",
    ]
    ESSENTIALS_CATEGORY = "Basics"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "string": ("STRING", {
                "default": "",
                "multiline": True,
            }),
        }}

    RETURN_TYPES = (_ANY,)
    RETURN_NAMES = ("OUT",)
    FUNCTION = "main"
    CATEGORY = "conditioning/utils"

    def main(self, string):
        return (string,)


NODE_CLASS_MAPPINGS = {
    "LoadImages": LoadImages,
    "Any": Any,
    "Input": Input,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImages": "Load Images",
    "Any": "Any",
    "Input": "Input",
}

__all__ = [
    "LoadImages",
    "Any",
    "Input",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
