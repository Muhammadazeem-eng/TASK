"""Stage 3 - ANALYSIS & ASSETS (free, offline).

From outputs/results.json + generated images produce:
  - report/assets/<case>_grid.jpg          before/after strips per case
  - report/assets/summary_identity.png     ArcFace by condition (bar)
  - report/assets/summary_garment.png      CLIP + judge by condition (bar)
  - report/assets/summary_table.md         mean metrics per condition
  - report/assets/model_choice.jpg         flash-baseline vs pro-baseline strip
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image
from core import config, cases
from core.imageio import load, contact_sheet, save

OUT = config.REPO / "outputs"
ASSETS = config.REPO / "report" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
R = json.loads((OUT / "results.json").read_text())

CONDS = ["C0_baseline", "C1_structured", "C2_bestN_garment", "C2_bestN_face"]


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else float("nan")


# ---- summary table ---------------------------------------------------------
def summary_table():
    lines = ["| Condition | ArcFace (identity) | CLIP (garment) | Judge garment | Judge print/logo | Judge identity | Same-person % |",
             "|---|---|---|---|---|---|---|"]
    for cond in CONDS:
        af = mean([R[c]["conditions"][cond].get("arcface") for c in R])
        cl = mean([R[c]["conditions"][cond].get("clip_garment") for c in R])
        jg = mean([(R[c]["conditions"][cond].get("judge_garment") or {}).get("overall") for c in R])
        jl = mean([(R[c]["conditions"][cond].get("judge_garment") or {}).get("print_logo_fidelity") for c in R])
        ji = mean([(R[c]["conditions"][cond].get("judge_identity") or {}).get("overall") for c in R])
        sp = mean([(R[c]["conditions"][cond].get("judge_identity") or {}).get("same_person_confidence_0_100") for c in R])
        lines.append(f"| {cond} | {af:.3f} | {cl:.3f} | {jg:.2f} | {jl:.2f} | {ji:.2f} | {sp:.0f} |")
    (ASSETS / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


# ---- bar charts (matplotlib optional) --------------------------------------
def bar_charts():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib unavailable, skipping charts:", e); return
    labels = ["baseline", "structured", "best-N\n(garment)", "best-N\n(face)"]

    af = [mean([R[c]["conditions"][cond].get("arcface") for c in R]) for cond in CONDS]
    plt.figure(figsize=(5, 3.2))
    bars = plt.bar(labels, af, color=["#c0392b", "#e67e22", "#f1c40f", "#27ae60"])
    plt.axhline(0.5, ls="--", c="gray", lw=1); plt.ylabel("ArcFace cosine (↑)")
    plt.title("Identity preservation (Task 3)"); plt.ylim(0, 1)
    for b, v in zip(bars, af): plt.text(b.get_x()+b.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(ASSETS / "summary_identity.png", dpi=130); plt.close()

    cl = [mean([R[c]["conditions"][cond].get("clip_garment") for c in R]) for cond in CONDS]
    jl = [mean([(R[c]["conditions"][cond].get("judge_garment") or {}).get("print_logo_fidelity") for c in R]) for cond in CONDS]
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.bar(labels, cl, color="#2980b9", alpha=.85); ax1.set_ylabel("CLIP garment sim (↑)", color="#2980b9"); ax1.set_ylim(0, 1)
    ax2 = ax1.twinx(); ax2.plot(labels, jl, "o-", color="#c0392b"); ax2.set_ylabel("Judge print/logo 1-5 (↑)", color="#c0392b"); ax2.set_ylim(0, 5)
    plt.title("Garment fidelity (Task 1)"); plt.tight_layout()
    plt.savefig(ASSETS / "summary_garment.png", dpi=130); plt.close()
    print("charts written")


# ---- per-case before/after grids -------------------------------------------
def case_grids():
    for cid, e in R.items():
        case = cases.BY_ID[cid]
        cdir = OUT / cid
        imgs = [case.model_img(), case.garment_img(),
                load(cdir / "baseline.png")]
        labels = ["REF person", "REF garment", "C0 baseline"]
        # structured[0] and best-N face pick
        imgs.append(load(cdir / "structured_0.png")); labels.append("C1 structured")
        fpath = e["conditions"]["C2_bestN_face"].get("_path")
        if fpath:
            imgs.append(load(config.REPO / fpath))
            af = e["conditions"]["C2_bestN_face"].get("arcface")
            labels.append(f"C2 best-N face ({af:.2f})")
        sheet = contact_sheet(imgs, labels, cols=len(imgs), cell=320)
        save(sheet, ASSETS / f"{cid}_grid.jpg")
    print("case grids written")


# ---- model-choice comparison (flash vs pro baseline) -----------------------
def model_choice_grid():
    imgs, labels = [], []
    for cid in R:
        cdir = OUT / cid
        fb = cdir / "baseline_flash.png"
        pb = cdir / "baseline.png"
        if fb.exists() and pb.exists():
            imgs += [load(fb), load(pb)]
            labels += [f"{cid[:12]} flash", f"{cid[:12]} PRO"]
    if imgs:
        sheet = contact_sheet(imgs, labels, cols=4, cell=260)
        save(sheet, ASSETS / "model_choice.jpg")
        print("model-choice grid written")
    else:
        print("flash baselines not found; skip model-choice grid")


if __name__ == "__main__":
    summary_table()
    bar_charts()
    case_grids()
    model_choice_grid()
    print("\nAssets in", ASSETS)
