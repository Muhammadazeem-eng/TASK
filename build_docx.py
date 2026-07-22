"""Build the Word (.docx) deliverable with results tables and embedded images."""
from __future__ import annotations
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

REPO = Path(__file__).resolve().parent
A = REPO / "report" / "assets"
final = json.loads((REPO / "outputs" / "final_gated.json").read_text())

doc = Document()

# ---- base styles ----
normal = doc.styles["Normal"]
normal.font.name = "Calibri"; normal.font.size = Pt(10.5)

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREEN = RGBColor(0x1E, 0x7A, 0x3C)


def h(text, level=1):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = NAVY
    return p


def para(text, bold=False, italic=False, size=10.5, color=None, align=None):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = bold; r.italic = italic; r.font.size = Pt(size)
    if color: r.font.color.rgb = color
    if align: p.alignment = align
    return p


def bullets(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        # simple **bold** handling
        parts = it.split("**")
        for i, seg in enumerate(parts):
            run = p.add_run(seg); run.bold = (i % 2 == 1); run.font.size = Pt(10.5)


def img(path, width=6.3, caption=None):
    if not Path(path).exists():
        return
    doc.add_picture(str(path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        c = doc.add_paragraph(); r = c.add_run(caption)
        r.italic = True; r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def table(headers, rows, bold_last_row=False):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, hd in enumerate(headers):
        cell = t.rows[0].cells[j]; cell.text = ""
        run = cell.paragraphs[0].add_run(hd); run.bold = True; run.font.size = Pt(9.5)
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            run = cells[j].paragraphs[0].add_run(str(val)); run.font.size = Pt(9.5)
            if bold_last_row and ri == len(rows) - 1:
                run.bold = True
    return t


# ============================ TITLE ============================
title = doc.add_heading("Reference Consistency for AI Photoshoots", level=0)
for r in title.runs: r.font.color.rgb = NAVY
para("Garment Consistency (Task 1) + Human-Model Consistency (Task 3)", bold=True, size=13)
meta = [
    "Author: Muhammad Azeem  —  Role: AI Workflow & Prompt Engineer (take-home)",
    "Generator: Google Nano Banana Pro (google/gemini-3-pro-image) via OpenRouter — treated strictly as a black box.",
    "Spend: $6.28 / $20.   Test set: 6 photoshoots = 3 models × 3 garments × 3 environments (light / medium / deep skin tones, both genders; plain / logo / footwear garments).",
]
for m in meta:
    para(m, size=9.5, color=RGBColor(0x40,0x40,0x40))

# ============================ FRAMING ============================
h("Why these two tasks, framed as one problem", 1)
para("Garment fidelity and model identity are the two dimensions the brief flags as “extremely "
     "important!!!”, and they are the same underlying problem: preserve a reference through a "
     "black-box generator you cannot fine-tune. I built one framework with two quality gates — a "
     "face gate (ArcFace) and a garment gate (CLIP + colour) — and reused ~80% of the code across "
     "both tasks. A single generated photoshoot contains a person and a garment, so every image is "
     "scored by both gates at once.")

# ============================ TL;DR ============================
h("TL;DR — what worked", 1)
table(["Condition", "ArcFace (identity)", "CLIP (garment)", "Judge print/logo"],
      [["C0 — baseline (naive prompt)", "0.630", "0.472", "4.33"],
       ["C1 — + structured VLM prompt & negatives", "0.684", "0.507", "4.33"],
       ["C2 — + best-of-N auto-selected by the gate", "0.753", "0.537", "4.83"],
       ["Gated pipeline (all levers, metric-gated)", "0.778", "0.550", "1 → 4 (hero framing)"]],
      bold_last_row=True)
para("ArcFace cosine: ≳0.5 ≈ “same person”. CLIP on garment crop vs packshot. Judge = VLM 1–5. Means over 6 cases.",
     italic=True, size=9, color=RGBColor(0x60,0x60,0x60))
bullets([
    "Identity: 0.630 → 0.778 ArcFace (+24%).  Garment: 0.472 → 0.550 CLIP (+16%). The gated pipeline is monotonic — every case ends ≥ its baseline; the worst case (c4) went 0.363 → 0.789.",
    "Logo fidelity — the hard part — cracked by framing: a garment-hero shot lifts print/logo fidelity from 1 → 4 (judge). The binding constraint on logos is pixels-on-target, not the prompt.",
    "Headline finding: every enhancement lever is high-variance on its own; gating each lever by its metric makes the pipeline reliable and monotonic. Choosing a “better” model is not a reliable lever — the workflow around the black box is.",
])
img(A/"summary_identity.png", 3.2)
img(A/"summary_garment.png", 3.2)
doc.add_page_break()

# ============================ 1. TAXONOMY ============================
h("1. Failure-mode taxonomy (from our own generations)", 1)
table(["Dimension", "Failure mode", "Where it bites", "Evidence"],
      [["Identity", "Small-face-in-frame drift (dominant)", "full-body shots — face is a few hundred px, model reinvents toward a generic average", "c4 baseline ArcFace 0.363 (below same-person line)"],
       ["Identity", "Skin-tone / feature averaging", "darker skin tones & distinctive features drift toward a 'prettier' mean", "c4, c5 weakest at baseline"],
       ["Identity", "Already-robust large faces", "¾/close framing with a large face is largely fine", "c3, c6 baseline ≈ 0.81 / 0.83"],
       ["Garment", "Small-garment-in-frame (dominant)", "footwear/accessories occupy few px in a full-body shot", "sneakers c5/c6 CLIP ≈ 0.18–0.20"],
       ["Garment", "Logo / small-text loss", "fine printed text is dropped or garbled", "'Axel Arigato' on the sneaker never legible at full-body"],
       ["Garment", "Colour / structure drift", "mild with a frontier model; collars, closures mostly hold", "judge garment ≈ 4.3/5 at baseline"]])
para("Both tasks fail for the same reason — the reference detail (a face, a logo, a shoe) is too "
     "small in the final frame for the black box to reconstruct faithfully.", italic=True)

# ============================ 2. STRATEGIES ============================
h("2. Strategies tested (6 levers — the brief asks for ≥2 per task)", 1)
strat = [
    ("A. Structured prompting (prompt-engineering lever).", "A VLM reads the packshot and model image and emits a structured description (garment: colour-hex, print, collar, closures; person: face shape, skin tone Fitzpatrick, hair, distinctive features), injected into the prompt with negative constraints (‘do not beautify/average the face’, ‘do not alter the print’). Effect: small but consistent — +0.05 ArcFace, +0.035 CLIP."),
    ("B. Best-of-N with an automatic gate (pipeline lever).", "Generate N=3 candidates from the structured prompt and auto-select the winner with the relevant gate (ArcFace for identity, CLIP+colour for garment). Effect: the main win — converts per-sample variance into a reliable output."),
    ("C. Face-targeted refinement pass (verify-and-retry lever).", "For weak-identity cases, edit only the face of the best-of-N image to match the reference, best-of-2 by ArcFace. Effect: high-variance — see §5."),
    ("D. Crop-refine-composite for the garment (region-targeted lever).", "Locate the garment, crop and upscale it, edit the crop against the packshot, then feather-composite it back. Effect: the edited crop's fidelity improves (crop CLIP 0.74 → 0.80), but pasting a rectangular crop back leaves a visible seam and scale mismatch — NOT recommended without proper segmentation; superseded by Strategy F."),
    ("E. Literal logo pixel-compositing (guaranteed-fidelity lever).", "Copy the actual logo pixels from the packshot onto the located logo region (luminance-blended for shading), so the logo is pixel-exact by construction. Effect: at full-body framing the footwear logo is ~15 px — pixel-exact but imperceptible; a resolution limit, not a method failure."),
    ("F. Garment-hero framing (root-cause lever — the winner for logos).", "Generate a closer, garment-forward shot so the print gets enough native pixels. Effect: shorts print/logo fidelity 1 → 4 (judge). The reliable way to get a legible logo is to give it pixels at generation time."),
]
for head, body in strat:
    p = doc.add_paragraph(); r = p.add_run(head); r.bold = True; r.font.size = Pt(10.5)
    r2 = p.add_run(" " + body); r2.font.size = Pt(10.5)
img(A/"c4_woman_shorts_street_hero.jpg", 6.5, "Strategy F (the garment winner) — full-body (logo ~1/5) vs garment-hero framing (logo ~4/5). Giving the print pixels at generation time reproduces the two-tone panels and thigh logo cleanly, with no compositing seam.")
doc.add_page_break()

# ============================ 3. HERO ============================
h("3. Hero result — the pipeline rescues the hardest case", 1)
para("c4 (dark-skinned model, shaved head, gold neck-ring, olive utility shorts): the face is tiny "
     "at baseline and identity collapses to 0.36. Best-of-N finds a framing with a clearer face "
     "(0.70), and a gated face-refinement pass pushes it to 0.79 — while the gold neck-ring, "
     "two-tone shorts and gold heels are preserved throughout.")
img(A/"c4_woman_shorts_street_refine.jpg", 6.5, "Identity: baseline (0.36, fail) → best-of-N (0.70) → gated face-refine (0.79).")
para("Garment-hero framing recovers the logo (Strategy F): the same shorts generated as a closer "
     "garment-forward shot reproduce the two-tone panels and thigh logo faithfully — judge print/logo 1 → 4.")
img(A/"c4_woman_shorts_street_hero.jpg", 6.5, "Garment fidelity: full-body (logo ~1/5) → garment-hero framing (logo ~4/5).")

para("Per-case identity lift (baseline → best-of-N) — the safety-net pattern:", bold=True)
rows = [[cid.replace("_"," "), f"{v['identity_baseline']:.3f}", f"{v['identity_gated']:.3f}",
         f"{v['identity_gated']-v['identity_baseline']:+.3f}"] for cid, v in final["per_case"].items()]
table(["Case", "baseline ArcFace", "gated ArcFace", "Δ"], rows)
para("best-of-N helps most exactly where the baseline is weakest, and does no harm where it is "
     "already good — the ideal shape for a production safety net.", italic=True)
doc.add_page_break()

# ============================ 4. EVAL ============================
h("4. Evaluation methodology", 1)
bullets([
    "Identity — ArcFace/InsightFace cosine on aligned face crops. The single most trustworthy number; it discriminates where the VLM judge saturates (judge says 5/5 even when ArcFace says 0.36).",
    "Garment — CLIP (ViT-B/32) cosine between the packshot and a VLM-localized garment crop, plus HSV colour-histogram / CIEDE2000 as secondary colour signals.",
    "VLM-as-judge (Gemini 2.5 Flash) with a strict rubric (print/logo, colour, structure, texture / face, skin, hair, body) — catches semantics embeddings miss, e.g. a dropped logo.",
    "Human sanity check: I eyeballed all 6 cases; my ranking agreed with ArcFace and disagreed with the saturating judge.",
])
para("Where metrics disagree with the eye (reported honestly):", bold=True)
bullets([
    "The VLM judge saturates (≈5/5 & 90%+ 'same person') on cases ArcFace flags as failures — do not use a VLM judge alone to gate identity.",
    "CLIP/colour are background-sensitive: a worn-garment crop scores lower against a white-background packshot than true fidelity warrants, so CLIP is reliable relatively (across conditions), not as an absolute score. No single metric captures logo fidelity.",
])

# ============================ 5. NEGATIVES ============================
h("5. Honest negatives", 1)
bullets([
    "‘Use a better model’ is not an identity lever. Same naive prompt, cheap Nano Banana (2.5 Flash) vs Nano Banana Pro: mean ArcFace 0.648 vs 0.630 — statistically tied, and wildly case-dependent. The per-sample variance within a model dwarfs the difference between models — the argument for best-of-N.",
    "Ungated face-refinement is a coin flip: c4 +0.087 (win) but c5 −0.110 and c2 −0.043 (losses) — the editor re-renders the face and can drift further. It only becomes safe when gated (accept only if ArcFace improves).",
    "Crop-refine-composite is also high-variance: clear win on mid/large garments but not on a second shorts case or the footwear, and can leave a mild seam.",
    "Logos are resolution-bound, not prompt-bound. Literal pixel-compositing makes a logo pixel-exact but imperceptible at full-body (~15 px). The fix that works is garment-hero framing — route logo-critical shots to a closer framing.",
])
img(A/"model_choice.jpg", 6.3, "Model-choice study: cheap Flash (left of each pair) vs Pro baseline — no reliable identity advantage.")
para("The unifying lesson (the headline finding): every lever — prompting, model choice, refinement, "
     "compositing — is individually high-variance. The one reliable design is to gate every lever by "
     "the metric: accept an enhancement only if the gate score improves, else keep the previous best. "
     "Gated, the pipeline is monotonic — it never regresses and rescues the hard cases.", bold=True)
doc.add_page_break()

# ============================ 6. ROBUSTNESS & COST ============================
h("6. Robustness & cost", 1)
bullets([
    "Robustness: the gate self-adjusts — it spends its improvement budget on the hard cases (small faces, dark skin tones, small garments) and no-ops on easy ones. Degraded gracefully across 3 skin tones, both genders, and plain/logo/footwear items.",
    "Cost (measured, Nano Banana Pro ≈ $0.12/image): baseline ≈ $0.12; structured + best-of-3 ≈ $0.38; + gated refine ≈ $0.62. The whole study cost $6.28.",
    "Latency: N and the refine pass are serial extra calls; in production run candidates in parallel and reserve refinement for cases the gate flags as failing (≈1/3 here), keeping average cost near best-of-N.",
])

# ============================ 7. NEXT ============================
h("7. What I’d do next", 1)
bullets([
    "Tiered pipeline: best-of-N by default; trigger the gated refinement pass only when the gate < threshold — near-best-of-N cost with the tail cases rescued.",
    "Segmentation-based logo compositing: garment-hero framing (Strategy F) already recovers mid/large logos; for tiny brand marks, segment the garment (not a box), warp the packshot's actual logo pixels onto the fabric folds (thin-plate spline) and blend — exact small text without the seams that sank the naive box-composite (Strategy D).",
    "DINOv2 instead of CLIP for the garment gate (more texture/structure sensitive) and a background-masked colour metric.",
    "Calibrate the ArcFace threshold against a human ‘same person?’ blind test to set a production accept/retry bar; scale the eval to thousands/day with async gates.",
])

# ============================ APPENDIX: per-case final grids ============================
doc.add_page_break()
h("Appendix — per-case before/after (baseline → gated best)", 1)
for cid in final["per_case"]:
    img(A/f"{cid}_final.jpg", 6.6, cid.replace("_", " "))

out = REPO / "report" / "Weon_Consistency_Report_Muhammad_Azeem.docx"
doc.save(str(out))
print("saved:", out)
