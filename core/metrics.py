"""Local, free consistency metrics.

Task 3 (model): ArcFace face-embedding cosine similarity on aligned face crops.
Task 1 (garment): CLIP/DINO embedding cosine on the garment crop + color-histogram
                  distance (a proxy for color drift).

Heavy models (insightface, open-clip) are lazy-loaded and cached so importing this
module is cheap and the pipeline still runs if a backend is missing.
"""
from __future__ import annotations
import functools
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Color histogram distance (no ML dependency) -- garment color drift proxy
# ---------------------------------------------------------------------------
def color_hist_similarity(a: Image.Image, b: Image.Image, bins: int = 32) -> float:
    """Cosine similarity of normalized HSV color histograms (1.0 == identical palette)."""
    import cv2
    def hist(im):
        arr = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([arr], [0, 1, 2], None, [bins, bins, bins],
                         [0, 180, 0, 256, 0, 256])
        h = h.flatten(); n = np.linalg.norm(h)
        return h / n if n else h
    ha, hb = hist(a), hist(b)
    return float(np.dot(ha, hb))


def delta_e(a: Image.Image, b: Image.Image) -> float:
    """Mean CIEDE2000 color difference over the (resized) images. Lower == closer."""
    from skimage.color import rgb2lab, deltaE_ciede2000
    size = (256, 256)
    la = rgb2lab(np.asarray(a.convert("RGB").resize(size)) / 255.0)
    lb = rgb2lab(np.asarray(b.convert("RGB").resize(size)) / 255.0)
    return float(np.mean(deltaE_ciede2000(la, lb)))


# ---------------------------------------------------------------------------
# ArcFace identity similarity (Task 3)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def face_embedding(img: Image.Image):
    """Return (embedding, face_bbox) for the largest detected face, or (None, None)."""
    import cv2
    app = _face_app()
    bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None, None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    f = faces[0]
    return f.normed_embedding, f.bbox


def face_similarity(ref: Image.Image, gen: Image.Image) -> float | None:
    """Cosine similarity of ArcFace embeddings. ~>0.5 is typically 'same person'.
    Returns None if a face can't be detected in either image."""
    ea, _ = face_embedding(ref)
    eb, _ = face_embedding(gen)
    if ea is None or eb is None:
        return None
    return float(np.dot(ea, eb))  # normed embeddings -> cosine


# ---------------------------------------------------------------------------
# CLIP embedding similarity (Task 1 garment; also a generic visual-similarity)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def _clip():
    import torch, open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model.eval()
    return model, preprocess, torch


def clip_embedding(img: Image.Image):
    model, preprocess, torch = _clip()
    with torch.no_grad():
        t = preprocess(img.convert("RGB")).unsqueeze(0)
        e = model.encode_image(t)
        e = e / e.norm(dim=-1, keepdim=True)
    return e[0].cpu().numpy()


def clip_similarity(a: Image.Image, b: Image.Image) -> float:
    ea, eb = clip_embedding(a), clip_embedding(b)
    return float(np.dot(ea, eb))
