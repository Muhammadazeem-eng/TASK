"""Prompt construction + VLM-derived structured descriptions.

Strategy A for both tasks = "describe-then-generate": a VLM reads the reference
(garment packshot / model image) and emits a structured description that we inject
into the generation prompt, plus negative constraints against the known failure
modes (color/print drift for garments; face averaging/beautification for people).
"""
from __future__ import annotations
import json
from PIL import Image
from .openrouter import OpenRouter

# ---------------------------------------------------------------------------
# Baseline prompts (no structured description) -- our control condition
# ---------------------------------------------------------------------------
def baseline_photoshoot_prompt(garment_desc: str = "the garment in the reference",
                               env_desc: str = "the environment in the reference") -> str:
    return (
        "Create a photorealistic fashion photoshoot. Place the person from the first "
        f"reference image, wearing {garment_desc}, standing naturally in {env_desc}. "
        "Full-body shot, natural lighting, realistic proportions."
    )


# ---------------------------------------------------------------------------
# Strategy A: VLM-derived structured descriptions
# ---------------------------------------------------------------------------
GARMENT_SCHEMA_PROMPT = (
    "You are a fashion product cataloguer. Describe ONLY the garment in this packshot "
    "as precise JSON with keys: "
    "type, category(top/bottom/outerwear/footwear/dress), primary_color (plain-English + hex), "
    "secondary_colors (list), pattern_or_print (describe exactly, or 'none'), "
    "logo_or_text (exact text and location, or 'none'), material_texture, "
    "collar_neckline, closure (buttons/zip/laces/etc + count if visible), "
    "sleeve_or_leg_length, distinctive_details (list), fit. "
    "Be literal and specific. Return only JSON."
)

PERSON_SCHEMA_PROMPT = (
    "You are describing a person for identity-preserving image generation. "
    "Return precise JSON with keys: "
    "apparent_gender, apparent_age_range, skin_tone (plain-English + Fitzpatrick I-VI), "
    "face_shape, distinctive_facial_features (list: e.g. cheekbones, jawline, nose, lips, freckles), "
    "eye_color, eyebrows, hair (style, length, color, texture), facial_hair, "
    "body_build, distinctive_features (tattoos/jewelry/etc). "
    "Describe the actual person literally; do NOT idealize or beautify. Return only JSON."
)


def describe_garment(client: OpenRouter, packshot: Image.Image) -> dict:
    return client.chat_json(GARMENT_SCHEMA_PROMPT, images=[packshot], tag="describe-garment")


def describe_person(client: OpenRouter, model_img: Image.Image) -> dict:
    return client.chat_json(PERSON_SCHEMA_PROMPT, images=[model_img], tag="describe-person")


def _kv(d: dict) -> str:
    parts = []
    for k, v in d.items():
        if v in (None, "", "none", [], "None"):
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        parts.append(f"{k.replace('_', ' ')}: {v}")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Structured prompts (Strategy A applied)
# ---------------------------------------------------------------------------
GARMENT_NEG = (
    "Do NOT alter, restyle, or hallucinate the print/logo; reproduce any logo text "
    "exactly. Do not change the garment color, pattern, closures, collar, or length. "
    "Match the packshot faithfully."
)
PERSON_NEG = (
    "Do NOT beautify, average, slim, or change the face. Preserve the exact facial "
    "identity, skin tone, hair, and body proportions of the reference person. "
    "The person must be recognizably the same individual."
)


def structured_prompt(person_desc: dict | None, garment_desc: dict | None,
                      env_desc: str) -> str:
    lines = ["Create a photorealistic full-body fashion photoshoot."]
    if person_desc:
        lines.append(f"THE PERSON (reference image 1) — {_kv(person_desc)}.")
    else:
        lines.append("THE PERSON is the individual in reference image 1.")
    if garment_desc:
        lines.append(f"THE GARMENT the person must wear — {_kv(garment_desc)}.")
    else:
        lines.append("THE GARMENT is the item in the garment reference image.")
    lines.append(f"SETTING: {env_desc}.")
    lines.append("Natural lighting and realistic proportions, ground the person in the scene.")
    lines.append(f"CONSTRAINTS: {PERSON_NEG} {GARMENT_NEG}")
    return " ".join(lines)
