"""Image helpers: load / resize / base64 data-URLs / crop / contact sheets."""
from __future__ import annotations
import base64, io
from pathlib import Path
from typing import Iterable
from PIL import Image, ImageDraw, ImageFont

from .config import MAX_INPUT_EDGE


def load(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def downscale(img: Image.Image, max_edge: int = MAX_INPUT_EDGE) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_edge:
        return img
    s = max_edge / max(w, h)
    return img.resize((round(w * s), round(h * s)), Image.LANCZOS)


def to_data_url(img: Image.Image, fmt: str = "PNG", max_edge: int = MAX_INPUT_EDGE) -> str:
    img = downscale(img, max_edge)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def from_data_url(url: str) -> Image.Image:
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def save(img: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def crop_box(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    """Crop using a normalized (x0,y0,x1,y1) box in 0..1."""
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def contact_sheet(images: list[Image.Image], labels: list[str] | None = None,
                  cols: int | None = None, cell: int = 360, pad: int = 8,
                  bg=(255, 255, 255)) -> Image.Image:
    """Lay images out in a labeled grid for before/after evidence."""
    n = len(images)
    cols = cols or n
    rows = (n + cols - 1) // cols
    lab_h = 22 if labels else 0
    cw, ch = cell, cell + lab_h
    sheet = Image.new("RGB", (cols * cw + pad * (cols + 1),
                              rows * ch + pad * (rows + 1)), bg)
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        thumb = im.copy(); thumb.thumbnail((cell, cell), Image.LANCZOS)
        x = pad + c * (cw + pad)
        y = pad + r * (ch + pad)
        ox = x + (cell - thumb.width) // 2
        oy = y + lab_h + (cell - thumb.height) // 2
        sheet.paste(thumb, (ox, oy))
        if labels and i < len(labels):
            draw.text((x + 2, y + 2), labels[i][:48], fill=(20, 20, 20), font=font)
    return sheet
