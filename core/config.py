"""Central configuration: models, paths, budget guardrails.

All image generation goes through OpenRouter. We treat every image model as a
black box (API only) per the task brief.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# ---- paths -----------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
# .env lives one level up (in the task/ folder) so it is shared and never committed.
load_dotenv(REPO.parent / ".env")
load_dotenv(REPO / ".env")  # fall back to a local .env inside the repo if present

DATA = REPO / "data"
MODELS_DIR = DATA / "models"
GARMENTS_DIR = DATA / "garments"
ENV_DIR = DATA / "environments"

# ---- API -------------------------------------------------------------------
# The env var in the provided .env is `OpenRouter_API_KEY`; accept common spellings.
API_KEY = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OpenRouter_API_KEY")
    or os.getenv("OPENROUTER_KEY")
)
BASE_URL = "https://openrouter.ai/api/v1"

# ---- model choices ---------------------------------------------------------
# Prototype cheap, run finals on the best editing model.
MODEL_PROTOTYPE = "google/gemini-2.5-flash-image"   # "Nano Banana"  ~$0.039/img
MODEL_BEST = "google/gemini-3-pro-image"            # "Nano Banana Pro" best logo/text+identity
MODEL_FLASH2 = "google/gemini-3.1-flash-image"      # "Nano Banana 2" ~$0.08/img

# The generator used by experiments unless overridden by --model.
GEN_MODEL = os.getenv("WEON_GEN_MODEL", MODEL_PROTOTYPE)

# Text-only VLM used for describing garments/people and as an LLM-judge.
# Gemini 2.5 Flash is cheap and strong at vision.
VLM_MODEL = os.getenv("WEON_VLM_MODEL", "google/gemini-2.5-flash")

# ---- guardrails ------------------------------------------------------------
MAX_INPUT_EDGE = 1280          # downscale reference images before upload
DEFAULT_N = 3                  # best-of-N candidates
HARD_BUDGET_USD = float(os.getenv("WEON_BUDGET_USD", "20"))
