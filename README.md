# weon — Reference Consistency for AI Photoshoots (Tasks 1 & 3)

Improving **garment consistency** (Task 1) and **human-model consistency** (Task 3) around a
black-box image generator, treated as **one framework with two quality gates**.

**➡️ Read [`report/report.md`](report/report.md) first** — it's the 10-minute deliverable with
results, before/after evidence, and honest negatives.

## What this does
A single composition pipeline puts a **model** in a **garment** in an **environment**, then:
1. **Generates** candidates with Nano Banana Pro (`google/gemini-3-pro-image`) via OpenRouter.
2. **Scores** each with local, free metrics — **ArcFace** (identity) + **CLIP/color** (garment)
   — and a **VLM-as-judge**.
3. **Selects** best-of-N by the relevant gate, with an optional **gated face-refinement** pass.

## Setup
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use source .venv/bin/activate on *nix
pip install -r requirements.txt
# API key: put OPENROUTER_API_KEY=sk-or-... in ../.env or ./.env
```
First run downloads the ArcFace (`buffalo_l`) and CLIP weights automatically.

## Reproduce
```bash
python generate_all.py --model google/gemini-3-pro-image --n 3   # generate (spends money; resumable)
python generate_flash_baseline.py                                 # cheap-model baselines (model-choice study)
python score_all.py                                               # ArcFace + CLIP + color + VLM judge -> outputs/results.{json,csv}
python refine_face.py --cases c4_woman_shorts_street c5_woman_sneaker_forest c2_woman_jacket_forest
python make_report_assets.py                                      # grids + charts + summary table (offline, free)
```
Every generation step is **resumable** — existing images are skipped so re-runs don't re-spend.
Total cost of the full study as submitted: **$4.58**.

## Layout
```
core/            openrouter client (+budget tracker), metrics, vlm_judge, prompts, pipeline, cases
data/            9 test images (3 models, 3 garments, 3 environments)
outputs/         generated images, results.json/csv, manifest.json
report/          report.md + assets/ (charts, before/after grids, summary table)
generate_all.py  score_all.py  refine_face.py  make_report_assets.py  generate_flash_baseline.py
```

## Key design choices
- **Black box only** — no fine-tuning; all leverage is prompt / input-conditioning / pipeline /
  verify-and-retry, per the brief.
- **One candidate pool, two gates** — a shot has a person *and* a garment, so it's scored for
  both tasks at once (why pairing the tasks is efficient).
- **Metric-gated everything** — the finding is that prompting, model choice, and refinement are
  all high-variance individually; the automatic gate is what makes the pipeline reliable.

## Conditions
`C0` naive baseline · `C1` structured VLM prompt + negatives · `C2` best-of-N (per-gate) ·
`C3` gated face-refinement (weak cases).
