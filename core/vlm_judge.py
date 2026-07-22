"""VLM-as-judge: holistic consistency scoring against a rubric, plus region
localization used to crop the garment/person for the embedding metrics.

The judge complements the embedding metrics (which are precise but narrow):
- ArcFace only sees the face; the judge also sees hair, skin tone, body.
- CLIP sees overall garment gestalt; the judge can call out a dropped logo.
Disagreements between judge and metric are themselves a reported finding.
"""
from __future__ import annotations
import json
from PIL import Image
from .openrouter import OpenRouter
from .imageio import crop_box

# ---------------------------------------------------------------------------
# Region localization (for garment / person crops)
# ---------------------------------------------------------------------------
def locate(client: OpenRouter, image: Image.Image, target: str) -> tuple[float, float, float, float] | None:
    """Return a normalized (x0,y0,x1,y1) bounding box of `target`, or None."""
    prompt = (
        f"Return the tight bounding box of {target} in this image as JSON "
        '{"x0":..,"y0":..,"x1":..,"y1":..} with values in 0..1 (fractions of width/height). '
        'If not present, return {"x0":0,"y0":0,"x1":0,"y1":0}. Only JSON.'
    )
    try:
        d = client.chat_json(prompt, images=[image], tag="locate")
        box = (float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"]))
        if box[2] > box[0] and box[3] > box[1]:
            return box
    except Exception:
        pass
    return None


def garment_crop(client: OpenRouter, image: Image.Image, garment_type: str = "garment") -> Image.Image:
    box = locate(client, image, f"the {garment_type} the person is wearing")
    return crop_box(image, box) if box else image


# ---------------------------------------------------------------------------
# Garment consistency judge (Task 1)
# ---------------------------------------------------------------------------
GARMENT_JUDGE = (
    "You are a strict QA reviewer for a fashion brand. The FIRST image is the ground-truth "
    "garment packshot. The SECOND image is a generated photoshoot. Judge how faithfully the "
    "generated garment matches the packshot. Score each 1-5 (5=perfect match, 1=wrong garment):\n"
    "- print_logo_fidelity: are prints/logos/text reproduced exactly (position, spelling, shape)?\n"
    "- color_fidelity: same hue/shade?\n"
    "- structure_fidelity: same collar, closures, pockets, length, silhouette?\n"
    "- texture_fidelity: same material/fabric look?\n"
    'Return JSON: {"print_logo_fidelity":n,"color_fidelity":n,"structure_fidelity":n,'
    '"texture_fidelity":n,"overall":n,"issues":["..."]}. Only JSON.'
)


def judge_garment(client: OpenRouter, packshot: Image.Image, generated: Image.Image) -> dict:
    return client.chat_json(GARMENT_JUDGE, images=[packshot, generated], tag="judge-garment")


# ---------------------------------------------------------------------------
# Identity consistency judge (Task 3)
# ---------------------------------------------------------------------------
IDENTITY_JUDGE = (
    "You are a strict identity reviewer. The FIRST image is a reference person. The SECOND "
    "is a generated photoshoot that should depict the SAME person. Judge identity preservation. "
    "Score each 1-5 (5=clearly the same individual, 1=a different person):\n"
    "- facial_identity: same face (bone structure, nose, eyes, lips, jaw)?\n"
    "- skin_tone: same skin tone?\n"
    "- hair: same hairstyle, length, color?\n"
    "- body: same build and proportions?\n"
    'Return JSON: {"facial_identity":n,"skin_tone":n,"hair":n,"body":n,"overall":n,'
    '"same_person_confidence_0_100":n,"issues":["..."]}. Only JSON.'
)


def judge_identity(client: OpenRouter, model_img: Image.Image, generated: Image.Image) -> dict:
    return client.chat_json(IDENTITY_JUDGE, images=[model_img, generated], tag="judge-identity")
