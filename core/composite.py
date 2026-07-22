"""Feathered region compositing helpers.

Used to paste a fixed garment crop back into the full photoshoot without a hard
seam. Pure PIL (no ML) so it's free and deterministic.
"""
from __future__ import annotations
from PIL import Image, ImageFilter, ImageDraw


def expand_box(box, w, h, pad=0.06):
    """Expand a normalized (x0,y0,x1,y1) box by `pad` fraction, clamped to image."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0.0, x0 - bw * pad); y0 = max(0.0, y0 - bh * pad)
    x1 = min(1.0, x1 + bw * pad); y1 = min(1.0, y1 + bh * pad)
    return (int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))


def crop_px(img: Image.Image, box_px) -> Image.Image:
    return img.crop(box_px)


def upscale_long_edge(img: Image.Image, target: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) >= target:
        return img
    s = target / max(w, h)
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def feather_paste(base: Image.Image, patch: Image.Image, box_px, feather: int = 12) -> Image.Image:
    """Paste `patch` into `base` at box_px (x0,y0,x1,y1) with a feathered alpha edge."""
    base = base.convert("RGB").copy()
    x0, y0, x1, y1 = box_px
    tw, th = x1 - x0, y1 - y0
    patch = patch.convert("RGB").resize((tw, th), Image.LANCZOS)
    mask = Image.new("L", (tw, th), 0)
    d = ImageDraw.Draw(mask)
    d.rectangle([feather, feather, tw - feather, th - feather], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather))
    base.paste(patch, (x0, y0), mask)
    return base
