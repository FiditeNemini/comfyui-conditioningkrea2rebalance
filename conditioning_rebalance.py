import math

import torch

try:
    import comfy.utils
    import node_helpers
    _COMFY_AVAILABLE = True
except ImportError:
    _COMFY_AVAILABLE = False


# Encoder profile registry + automatic routing
_ENCODER_PROFILES = {}


class EncoderProfile:

    def __init__(self, name, n_taps, hidden_dim, tap_layers=None):
        self.name = name
        self.n_taps = int(n_taps)
        self.hidden_dim = int(hidden_dim)
        self.feature_dim = self.n_taps * self.hidden_dim
        self.tap_layers = list(tap_layers) if tap_layers is not None else None

    def __repr__(self):
        return "EncoderProfile(name={!r}, n_taps={}, hidden_dim={}, feature_dim={})".format(
            self.name, self.n_taps, self.hidden_dim, self.feature_dim)


def register_encoder_profile(name, n_taps, hidden_dim, tap_layers=None):

    profile = EncoderProfile(name, n_taps, hidden_dim, tap_layers)
    _ENCODER_PROFILES[name] = profile
    # Index by feature_dim for fast lookup
    _ENCODER_PROFILES.setdefault(("_dim", profile.feature_dim), []).append(profile)
    return profile


def get_encoder_profile(name):
    return _ENCODER_PROFILES.get(name)


def detect_encoder_profile(feature_dim):

    matches = _ENCODER_PROFILES.get(("_dim", int(feature_dim)), [])
    if matches:
        return matches[0]
    return None


def _resolve_n_bands(t, default=12):

    flat = t.shape[-1]
    profile = detect_encoder_profile(flat)
    if profile is not None and flat % profile.n_taps == 0:
        return profile.n_taps
    if default > 1 and flat % default == 0:
        return default
    return 1


def _unit_norm_dim(t, eps=1e-8):
    dtype = t.dtype
    t = t.float()
    norm = torch.sqrt(t.pow(2).sum(dim=-1, keepdim=True) + eps)
    return (t / norm).to(dtype)


def _split_bands(t, n_bands=None):
    flat = t.shape[-1]
    if n_bands is None:
        n_bands = _resolve_n_bands(t)
    if n_bands > 1 and flat % n_bands == 0:
        d = flat // n_bands
        return t.view(*t.shape[:-1], n_bands, d), d
    return None, None


def _merge_bands(t):
    n_bands = t.shape[-2]
    d = t.shape[-1]
    return t.reshape(*t.shape[:-2], n_bands * d)


def _extract_cond_tensor(item):
    if isinstance(item, (list, tuple)) and len(item) == 2 \
            and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
        return item[0]
    if isinstance(item, torch.Tensor):
        return item
    return None


def _match_batch(ref_dir, target_batch):
    if ref_dir.shape[0] == 1 and target_batch != 1:
        return ref_dir.expand(target_batch, *ref_dir.shape[1:])
    if ref_dir.shape[0] != target_batch:
        ref_dir = ref_dir.mean(dim=0, keepdim=True).expand(target_batch, *ref_dir.shape[1:])
    return ref_dir


def _align_prompt(text):
    if text is None:
        return ""
    return str(text).replace("{", "{").replace("}", "}")


def _parse_floats(s):
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        vals = [float(x) for x in s.replace(";", ",").split(",") if x.strip() != ""]
    except ValueError:
        return None
    if len(vals) < 2:
        return None
    return vals

# ignored
SYS_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, "
    "objects, background), then explain how the user's text instruction should "
    "alter or modify the image. Generate a new image that meets the user's "
    "requirements while maintaining consistency with the original input where "
    "appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


# Target longest-side resolution per tier. Each image is scaled to it selected resolution independently
RESOLUTIONS = {"low": 256, "normal": 512, "high": 1024, "max": 1280}


def _scale_to_resolution(samples, target):

    n, c, h, w = samples.shape
    if h == target and w == target:
        return samples
    scale = target / max(h, w)
    nh = max(1, round(h * scale))
    nw = max(1, round(w * scale))
    return comfy.utils.common_upscale(samples, nw, nh, "area", "disabled")


def compile_edit(clip, prompt, images_with_size=None, llama_template=None):
    """Encode an edit prompt with optional reference images.

    ``llama_template`` selects the encoder's chat template. When ``None``, the
    legacy ``SYS_TEMPLATE`` is used (Krea 2 default). Encoder-specific modules
    pass their own template so the same helper serves both Krea 2 and Ideogram 4.
    """
    if not _COMFY_AVAILABLE:
        raise RuntimeError("Edit encode requires ComfyUI (comfy.utils, node_helpers).")

    if llama_template is None:
        llama_template = SYS_TEMPLATE

    images_vl = []
    image_prompt = ""

    if images_with_size:
        for i, (image, tier) in enumerate(images_with_size):
            if image is None:
                continue
            target = RESOLUTIONS.get(tier, 256)
            samples = image.movedim(-1, 1)  # NHWC -> NCHW
            scaled = _scale_to_resolution(samples, target)
            images_vl.append(scaled.movedim(1, -1))  # back to NHWC for clip.tokenize
            image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(len(images_vl))

    full_prompt = image_prompt + prompt if image_prompt else prompt

    tokens = clip.tokenize(
        full_prompt,
        images=images_vl if images_vl else None,
        llama_template=llama_template,
    )
    conditioning = clip.encode_from_tokens_scheduled(tokens)

    return conditioning


def _scale_cond_tensor(t, scale, weights=None):
    if weights is None:
        return t * scale

    flat = t.shape[-1]
    n_layers = len(weights)
    if n_layers > 1 and flat % n_layers == 0:
        layer_dim = flat // n_layers
        orig_dtype = t.dtype
        t = t.float()
        t = t.view(*t.shape[:-1], n_layers, layer_dim)
        gains = torch.tensor(weights, dtype=t.dtype, device=t.device)
        t = t * gains.view(*([1] * (t.dim() - 2)), n_layers, 1)
        t = t.view(*t.shape[:-2], flat)
        return t.to(orig_dtype) * scale
    return t * scale


def scale_conditioning(structure, scale, weights=None):
    if isinstance(structure, list):
        out = []
        for item in structure:
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                new_cond = _scale_cond_tensor(cond_t, scale, weights)
                out.append([new_cond, dict(extras)])
            else:
                out.append(scale_conditioning(item, scale, weights))
        return out
    if isinstance(structure, torch.Tensor):
        return _scale_cond_tensor(structure, scale, weights)
    if isinstance(structure, dict):
        return {k: scale_conditioning(v, scale, weights)
                for k, v in structure.items()}
    return structure


def refocus(conditioning, scale, weights):
    plw = _parse_floats(weights) if weights else None
    return scale_conditioning(conditioning, scale, weights=plw)


def _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign):
    b = cond_bands.shape[0]
    cond_mean = cond_bands.float().mean(dim=1)
    ref_mean = ref_bands.float().mean(dim=1)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)

    if per_band_strengths is None:
        gains = [strength] * n_bands
    else:
        gains = list(per_band_strengths)
        if len(gains) < n_bands:
            gains = gains + [strength] * (n_bands - len(gains))
        elif len(gains) > n_bands:
            gains = gains[:n_bands]

    gains_t = torch.tensor(gains, dtype=cond_bands.float().dtype, device=cond_bands.device)
    gains_t = gains_t.view(1, 1, n_bands, 1)

    cond_f = cond_bands.float()
    dir_exp = direction.unsqueeze(1)
    proj = (cond_f * dir_exp).sum(dim=-1, keepdim=True)
    out = cond_f + sign * gains_t * proj * dir_exp
    return _merge_bands(out.to(cond_bands.dtype))


def _project_dissim_whole(cond_t, ref_t, strength, sign):
    b = cond_t.shape[0]
    cond_mean = cond_t.float().mean(dim=1, keepdim=True)
    ref_mean = ref_t.float().mean(dim=1, keepdim=True)
    ref_mean = _match_batch(ref_mean, b)
    direction = _unit_norm_dim(cond_mean - ref_mean)
    proj = (cond_t.float() * direction).sum(dim=-1, keepdim=True)
    out = cond_t.float() + sign * strength * proj * direction
    return out.to(cond_t.dtype)


def _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=None):
    if n_bands is None:
        n_bands = _resolve_n_bands(cond_t)
    cond_bands, d = _split_bands(cond_t, n_bands)
    ref_bands, d2 = _split_bands(ref_t, n_bands)
    if cond_bands is not None and ref_bands is not None and d == d2:
        return _project_dissim_per_band(cond_bands, ref_bands, d, n_bands, strength, per_band_strengths, sign=+1)
    return _project_dissim_whole(cond_t, ref_t, strength, sign=+1)


def guidance_conditioning(structure, ref_structure, strength, per_band_strengths=None, n_bands=None):
    if isinstance(structure, list):
        out = []
        ref_iter = iter(ref_structure) if isinstance(ref_structure, list) else None
        for item in structure:
            ref_item = next(ref_iter, None) if ref_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_t, extras = item
                ref_t = _extract_cond_tensor(ref_item) if ref_item is not None else None
                new_cond = _apply_dissim(cond_t, ref_t, strength, per_band_strengths, n_bands=n_bands) \
                    if ref_t is not None else cond_t
                out.append([new_cond, dict(extras)])
            else:
                out.append(guidance_conditioning(item, ref_item, strength, per_band_strengths, n_bands=n_bands))
        return out
    if isinstance(structure, torch.Tensor):
        ref_t = _extract_cond_tensor(ref_structure) if ref_structure is not None else None
        if ref_t is not None:
            return _apply_dissim(structure, ref_t, strength, per_band_strengths, n_bands=n_bands)
        return structure
    return structure


def guidance(conditioning, reference, strength, n_bands=None):
    return guidance_conditioning(conditioning, reference, strength, per_band_strengths=None, n_bands=n_bands)


def _top_percent_mask(t, percent, dim=-1):
    """Boolean mask selecting the top `percent` of elements by magnitude along `dim`."""
    if percent >= 1.0:
        return torch.ones_like(t, dtype=torch.bool)
    if percent <= 0.0:
        return torch.zeros_like(t, dtype=torch.bool)
    mag = t.float().abs()
    size = t.shape[dim]
    k = max(1, int(round(percent * size)))
    k = min(k, size)
    _, idx = torch.topk(mag, k, dim=dim)
    mask = torch.zeros_like(t, dtype=torch.bool)
    mask.scatter_(dim, idx, True)
    return mask


def _pad_seq(t, target_len):
    """Right-pad a conditioning tensor along the token dimension (dim 1)."""
    cur = t.shape[1]
    if cur >= target_len:
        return t
    pad_shape = list(t.shape)
    pad_shape[1] = target_len - cur
    pad = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
    return torch.cat([t, pad], dim=1)


def _merge_top_match_tensor(cond_a, cond_b, match_percent):
    """Equally merge two conditioning tensors.

    Elements that are in the top `match_percent` of *both* tensors by magnitude
    and share the same sign (i.e. they "match") are averaged for an equal blend.
    Elements that do not match fall back to the higher-magnitude source so no
    information is lost where the two conditionings disagree.
    """
    orig_dtype = cond_a.dtype
    a = cond_a.float()
    b = cond_b.float()
    b = _match_batch(b, a.shape[0])

    # Pad the shorter tensor to match the longer one along the token dimension.
    if a.dim() >= 2 and b.dim() >= 2 and a.shape[1] != b.shape[1]:
        target_len = max(a.shape[1], b.shape[1])
        a = _pad_seq(a, target_len)
        b = _pad_seq(b, target_len)

    mask_a = _top_percent_mask(a, match_percent)
    mask_b = _top_percent_mask(b, match_percent)
    sign_match = (a.sign() == b.sign())
    match = mask_a & mask_b & sign_match

    avg = (a + b) / 2.0
    a_dominant = a.abs() >= b.abs()
    fallback = torch.where(a_dominant, a, b)
    out = torch.where(match, avg, fallback)
    return out.to(orig_dtype)


def merge_conditioning(structure_a, structure_b, match_percent):
    if isinstance(structure_a, list):
        out = []
        b_iter = iter(structure_b) if isinstance(structure_b, list) else None
        for item in structure_a:
            b_item = next(b_iter, None) if b_iter is not None else None
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                cond_a, extras = item
                cond_b = _extract_cond_tensor(b_item) if b_item is not None else None
                new_cond = _merge_top_match_tensor(cond_a, cond_b, match_percent) \
                    if cond_b is not None else cond_a
                out.append([new_cond, dict(extras)])
            else:
                out.append(merge_conditioning(item, b_item, match_percent))
        return out
    if isinstance(structure_a, torch.Tensor):
        cond_b = _extract_cond_tensor(structure_b) if structure_b is not None else None
        if cond_b is not None:
            return _merge_top_match_tensor(structure_a, cond_b, match_percent)
        return structure_a
    return structure_a


def _merge_top_match_tensor_n(tensors, match_percent):
    """Equally merge N conditioning tensors.

    Elements that are in the top `match_percent` of *all* tensors by magnitude
    and share the same sign are averaged for an equal blend. Elements that do
    not match fall back to the highest-magnitude source so no information is
    lost where the conditionings disagree.
    """
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    orig_dtype = tensors[0].dtype
    floats = [t.float() for t in tensors]
    # Match batch size to the first tensor.
    b0 = floats[0].shape[0]
    floats = [_match_batch(t, b0) for t in floats]
    # Pad the shorter tensors to match the longest along the token dimension.
    if all(t.dim() >= 2 for t in floats):
        target_len = max(t.shape[1] for t in floats)
        floats = [_pad_seq(t, target_len) for t in floats]

    stacked = torch.stack(floats, dim=0)  # (N, B, T, D)
    masks = torch.stack([_top_percent_mask(t, match_percent) for t in floats], dim=0)
    signs = torch.stack([t.sign() for t in floats], dim=0)
    # Match: in top percent of all AND all signs agree.
    match_all = masks.all(dim=0)
    sign_all = (signs == signs[0:1]).all(dim=0)
    match = match_all & sign_all

    avg = stacked.mean(dim=0)
    # Fallback: highest-magnitude tensor per element.
    mag = stacked.abs()
    dominant_idx = mag.argmax(dim=0, keepdim=False)
    fallback = stacked.gather(0, dominant_idx.unsqueeze(0)).squeeze(0)
    out = torch.where(match, avg, fallback)
    return out.to(orig_dtype)


def merge_conditioning_multi(structures, match_percent):
    """Merge a list of conditioning structures (up to N) into one."""
    structures = [s for s in structures if s is not None]
    if not structures:
        return None
    if len(structures) == 1:
        return structures[0]

    base = structures[0]
    if isinstance(base, list):
        out = []
        iters = [iter(s) if isinstance(s, list) else None for s in structures[1:]]
        for item in base:
            items = [item]
            for it in iters:
                items.append(next(it, None) if it is not None else None)
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                extras = item[1]
                cond_tensors = []
                for it_item in items:
                    t = _extract_cond_tensor(it_item) if it_item is not None else None
                    if t is not None:
                        cond_tensors.append(t)
                new_cond = _merge_top_match_tensor_n(cond_tensors, match_percent) \
                    if cond_tensors else item[0]
                out.append([new_cond, dict(extras)])
            else:
                sub = [s for s in items if s is not None]
                out.append(merge_conditioning_multi(sub, match_percent))
        return out
    if isinstance(base, torch.Tensor):
        cond_tensors = []
        for s in structures:
            t = _extract_cond_tensor(s) if s is not None else None
            if t is not None:
                cond_tensors.append(t)
        if cond_tensors:
            return _merge_top_match_tensor_n(cond_tensors, match_percent)
        return base
    return base


def _merge_anchor_tensor(anchor, tensors, match_percent):
    """Merge N conditioning tensors against an anchor.

    The anchor defines which elements are relevant: an element is blended
    (averaged across all inputs) only where it is in the top `match_percent`
    of the anchor by magnitude AND every input agrees with the anchor's sign
    there. Elements that do not match the anchor fall back to the
    highest-magnitude input, so unmatched regions keep their strongest source.
    """
    tensors = [t for t in tensors if t is not None]
    if not tensors:
        return anchor
    if anchor is None:
        return _merge_top_match_tensor_n(tensors, match_percent)

    orig_dtype = anchor.dtype
    a = anchor.float()
    b0 = a.shape[0]
    floats = [_match_batch(t.float(), b0) for t in tensors]
    # Pad everything to the longest token dimension.
    if a.dim() >= 2 and all(t.dim() >= 2 for t in floats):
        target_len = max([a.shape[1]] + [t.shape[1] for t in floats])
        a = _pad_seq(a, target_len)
        floats = [_pad_seq(t, target_len) for t in floats]

    stacked = torch.stack(floats, dim=0)  # (N, B, T, D)
    anchor_mask = _top_percent_mask(a, match_percent)
    anchor_sign = a.sign()
    sign_match = torch.stack([(t.sign() == anchor_sign) for t in floats], dim=0).all(dim=0)
    match = anchor_mask & sign_match

    avg = torch.cat([a.unsqueeze(0), stacked], dim=0).mean(dim=0)
    mag = stacked.abs()
    dominant_idx = mag.argmax(dim=0, keepdim=False)
    fallback = stacked.gather(0, dominant_idx.unsqueeze(0)).squeeze(0)
    out = torch.where(match, avg, fallback)
    return out.to(orig_dtype)


def merge_conditioning_anchor(anchor_structure, structures, match_percent):
    """Merge a list of conditioning structures against an anchor structure."""
    structures = [s for s in structures if s is not None]
    if not structures:
        return anchor_structure

    base = anchor_structure
    if isinstance(base, list):
        out = []
        iters = [iter(s) if isinstance(s, list) else None for s in structures]
        for item in base:
            items = []
            for it in iters:
                items.append(next(it, None) if it is not None else None)
            if isinstance(item, (list, tuple)) and len(item) == 2 \
                    and isinstance(item[0], torch.Tensor) and isinstance(item[1], dict):
                anchor_t, extras = item
                cond_tensors = []
                for it_item in items:
                    t = _extract_cond_tensor(it_item) if it_item is not None else None
                    if t is not None:
                        cond_tensors.append(t)
                new_cond = _merge_anchor_tensor(anchor_t, cond_tensors, match_percent) \
                    if cond_tensors else anchor_t
                out.append([new_cond, dict(extras)])
            else:
                sub = [s for s in items if s is not None]
                out.append(merge_conditioning_anchor(item, sub, match_percent))
        return out
    if isinstance(base, torch.Tensor):
        anchor_t = _extract_cond_tensor(base) if base is not None else None
        cond_tensors = []
        for s in structures:
            t = _extract_cond_tensor(s) if s is not None else None
            if t is not None:
                cond_tensors.append(t)
        if cond_tensors:
            return _merge_anchor_tensor(anchor_t, cond_tensors, match_percent)
        return base
    return base


class ConditioningMergeMulti:
    """Merge up to 5 conditionings equally by blending the highest
    `match_percent` of each that agree (match) with the others."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "match_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }, "optional": {
            "conditioning_2": ("CONDITIONING",),
            "conditioning_3": ("CONDITIONING",),
            "conditioning_4": ("CONDITIONING",),
            "conditioning_5": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning_1, match_percent=0.5, conditioning_2=None,
             conditioning_3=None, conditioning_4=None, conditioning_5=None):
        match_percent = float(min(max(match_percent, 0.0), 1.0))
        structures = [conditioning_1, conditioning_2, conditioning_3,
                      conditioning_4, conditioning_5]
        structures = [s for s in structures if s is not None]
        if not structures:
            raise ValueError("ConditioningMergeMulti: at least one conditioning is required.")
        return (merge_conditioning_multi(structures, match_percent),)


class ConditioningMergeAnchor:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "anchor": ("CONDITIONING",),
            "conditioning_1": ("CONDITIONING",),
            "match_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }, "optional": {
            "conditioning_2": ("CONDITIONING",),
            "conditioning_3": ("CONDITIONING",),
            "conditioning_4": ("CONDITIONING",),
            "conditioning_5": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, anchor, conditioning_1, match_percent=0.5, conditioning_2=None,
             conditioning_3=None, conditioning_4=None, conditioning_5=None):
        match_percent = float(min(max(match_percent, 0.0), 1.0))
        structures = [conditioning_1, conditioning_2, conditioning_3,
                      conditioning_4, conditioning_5]
        structures = [s for s in structures if s is not None]
        if not structures:
            raise ValueError("ConditioningMergeAnchor: at least one conditioning is required.")
        return (merge_conditioning_anchor(anchor, structures, match_percent),)


class ConditioningMerge:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "match_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning_1, conditioning_2, match_percent=0.5):
        match_percent = float(min(max(match_percent, 0.0), 1.0))
        return (merge_conditioning(conditioning_1, conditioning_2, match_percent),)


class RebalanceGuider:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "guidance_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, positive, negative, guidance_strength=0.500):
        return (guidance(positive, negative, guidance_strength),)


class StepRebalance:
    """Split a conditioning schedule at a step threshold."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "step": ("FLOAT", {"default": 0.00, "min": 0.000, "max": 1.000, "step": 0.01}),
            "bound": ("FLOAT", {"default": 0.00, "min": 0.000, "max": 1.000, "step": 0.01}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning_1, conditioning_2, step=0.00, bound=0.00):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Step Rebalance requires ComfyUI (node_helpers).")

        step = float(min(max(step, 0.0), 1.0))
        bound = float(min(max(bound, 0.0), 1.0))

        end_1 = float(min(step + bound, 1.0))
        start_2 = float(max(step - bound, 0.0))

        cond_split_1 = node_helpers.conditioning_set_values(
            conditioning_1, {"start_percent": 0.000, "end_percent": end_1},
        )
        cond_split_2 = node_helpers.conditioning_set_values(
            conditioning_2, {"start_percent": start_2, "end_percent": 1.000},
        )
        return (cond_split_2 + cond_split_1,)


def _parse_schedule(s):

    if not s:
        return None
    s = s.replace("\n", " ").replace("\t", " ").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(";") if p.strip() != ""]
    points = []
    for p in parts:
        if ":" in p:
            seg, mult_s = p.rsplit(":", 1)
        else:
            seg, mult_s = p, "1.0"
        if "-" in seg:
            start_s, end_s = seg.split("-", 1)
        else:
            start_s, end_s = seg, seg
        try:
            start = float(start_s.strip())
            end = float(end_s.strip())
            mult = float(mult_s.strip())
        except ValueError:
            continue
        start = float(min(max(start, 0.0), 1.0))
        end = float(min(max(end, 0.0), 1.0))
        if end < start:
            start, end = end, start
        points.append((start, end, mult))
    if not points:
        return None
    points.sort(key=lambda x: x[0])
    return points


def _build_schedule_segments(conditioning, points, interpolation, sub_steps=8):

    if not _COMFY_AVAILABLE:
        raise RuntimeError("Rebalance CFG requires ComfyUI (node_helpers).")
    out = []
    n = len(points)
    for i, (t0, t1, m_i) in enumerate(points):
        m_next = points[i + 1][2] if i + 1 < n else m_i
        if t1 <= t0:
            t1 = max(t1, t0)
            seg = scale_conditioning(conditioning, m_i)
            seg = node_helpers.conditioning_set_values(
                seg, {"start_percent": t0, "end_percent": t1},
            )
            out.append(seg)
            continue

        if interpolation == "gradual" and sub_steps > 1 and m_next != m_i:
            for k in range(sub_steps):
                f0 = k / sub_steps
                f1 = (k + 1) / sub_steps
                ts = t0 + (t1 - t0) * f0
                te = t0 + (t1 - t0) * f1
                # use the sub-segment midpoint for a stable linear ramp
                m = m_i + (m_next - m_i) * ((f0 + f1) / 2.0)
                seg = scale_conditioning(conditioning, m)
                seg = node_helpers.conditioning_set_values(
                    seg, {"start_percent": ts, "end_percent": te},
                )
                out.append(seg)
        else:
            seg = scale_conditioning(conditioning, m_i)
            seg = node_helpers.conditioning_set_values(
                seg, {"start_percent": t0, "end_percent": t1},
            )
            out.append(seg)

    combined = []
    for seg in out:
        combined = combined + seg
    return combined


class RebalanceCFG:
    """CFG-style conditioning rebalance from editable point schedule strings."""

    DEFAULT_SCHEDULE_1 = (
        "0.000-0.125:1.50; 0.125-0.250:1.40; 0.250-0.375:1.20; 0.375-0.500:1.00;"
        " 0.500-0.625:0.80; 0.625-0.750:0.60; 0.750-0.875:0.40; 0.875-1.000:0.20"
    )
    DEFAULT_SCHEDULE_2 = (
        "0.000-0.125:0.20; 0.125-0.250:0.40; 0.250-0.375:0.60; 0.375-0.500:0.80;"
        " 0.500-0.625:1.00; 0.625-0.750:1.20; 0.750-0.875:1.40; 0.875-1.000:1.50"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning_1": ("CONDITIONING",),
            "conditioning_2": ("CONDITIONING",),
            "schedule_1": ("STRING", {"default": cls.DEFAULT_SCHEDULE_1, "multiline": True}),
            "schedule_2": ("STRING", {"default": cls.DEFAULT_SCHEDULE_2, "multiline": True}),
            "interpolation": (["constant", "gradual"], {"default": "gradual"}),
            "sub_steps": ("INT", {"default": 8, "min": 1, "max": 64}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "main"
    CATEGORY = "conditioning"

    def main(self, conditioning_1, conditioning_2, schedule_1, schedule_2,
             interpolation="gradual", sub_steps=8):
        if not _COMFY_AVAILABLE:
            raise RuntimeError("Rebalance CFG requires ComfyUI (node_helpers).")

        pts1 = _parse_schedule(schedule_1)
        pts2 = _parse_schedule(schedule_2)
        if pts1 is None or pts2 is None:
            raise ValueError(
                "Rebalance CFG: invalid schedule string. "
                "Use 'start-end:multiplier; ...' (e.g. 0.000-0.125:1.5; ...)."
            )

        sub_steps = int(min(max(sub_steps, 1), 64))
        seg1 = _build_schedule_segments(conditioning_1, pts1, interpolation, sub_steps)
        seg2 = _build_schedule_segments(conditioning_2, pts2, interpolation, sub_steps)
        return (seg1 + seg2,)


NODE_CLASS_MAPPINGS = {
    "RebalanceGuider": RebalanceGuider,
    "StepRebalance": StepRebalance,
    "RebalanceCFG": RebalanceCFG,
    "ConditioningMerge": ConditioningMerge,
    "ConditioningMergeMulti": ConditioningMergeMulti,
    "ConditioningMergeAnchor": ConditioningMergeAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebalanceGuider": "Rebalance Guider",
    "StepRebalance": "Step Rebalance",
    "RebalanceCFG": "Rebalance CFG Custom",
    "ConditioningMerge": "Conditioning Merge",
    "ConditioningMergeMulti": "Conditioning Merge (Multi)",
    "ConditioningMergeAnchor": "Conditioning Merge (Anchor)",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",


    # Core helpers re-exported
    "EncoderProfile",
    "register_encoder_profile",
    "get_encoder_profile",
    "detect_encoder_profile",
    "compile_edit",
    "scale_conditioning",
    "refocus",
    "guidance",
    "guidance_conditioning",
    "merge_conditioning",
    "merge_conditioning_multi",
    "merge_conditioning_anchor",
    "ConditioningMerge",
    "ConditioningMergeMulti",
    "ConditioningMergeAnchor",
    "RebalanceCFG",
    "RebalanceGuider",
    "StepRebalance",
    "SYS_TEMPLATE",
    "RESOLUTIONS",
]
