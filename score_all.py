"""Stage 2 - SCORING (free except VLM-judge calls).

Reads outputs/manifest.json, scores every image with:
  - Task 1 garment gate: CLIP(packshot, garment-crop) + color-hist + deltaE + VLM judge
  - Task 3 face gate:     ArcFace(model, generated) + VLM identity judge

Derives the three conditions per case and writes outputs/results.json + a CSV.
Best-of-N selection uses ONLY the automatic gate (no judge), mirroring production.
"""
from __future__ import annotations
import json, csv, argparse
from pathlib import Path
from PIL import Image
from core import config
from core.openrouter import OpenRouter, BUDGET
from core.imageio import load
from core import metrics as M
from core import vlm_judge as J
from core import cases

OUT = config.REPO / "outputs"


def garment_gate(client, packshot, gen_img, gtype):
    crop = J.garment_crop(client, gen_img, gtype)
    return {
        "clip_garment": M.clip_similarity(packshot, crop),
        "color_hist": M.color_hist_similarity(packshot, crop),
        "delta_e": M.delta_e(packshot, crop),
    }, crop


def face_gate(model_img, gen_img):
    fs = M.face_similarity(model_img, gen_img)
    return {"arcface": fs}


def score_image(client, case, img, do_judge=True):
    packshot = case.garment_img()
    model_img = case.model_img()
    gtype = "garment"
    g, crop = garment_gate(client, packshot, img, gtype)
    f = face_gate(model_img, img)
    rec = {**g, **f}
    if do_judge:
        try:
            rec["judge_garment"] = J.judge_garment(client, packshot, img)
        except Exception as e:
            rec["judge_garment"] = {"error": str(e)}
        try:
            rec["judge_identity"] = J.judge_identity(client, model_img, img)
        except Exception as e:
            rec["judge_identity"] = {"error": str(e)}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="skip VLM-judge (metrics only)")
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()

    client = OpenRouter()
    manifest = json.loads((OUT / "manifest.json").read_text())
    case_ids = args.cases or list(manifest.keys())
    results = {}

    for cid in case_ids:
        m = manifest[cid]
        case = cases.BY_ID[cid]
        print(f"\n=== {cid} ===")
        entry = {"conditions": {}, "candidates": []}

        # ---- baseline (control) ------------------------------------------
        base_img = load(config.REPO / m["baseline"])
        entry["conditions"]["C0_baseline"] = score_image(client, case, base_img, not args.no_judge)
        print("  C0 baseline:", {k: round(v, 3) for k, v in entry["conditions"]["C0_baseline"].items() if isinstance(v, (int, float))})

        # ---- structured candidates ---------------------------------------
        cand_scores = []
        for i, rel in enumerate(m["structured"]):
            img = load(config.REPO / rel)
            s = score_image(client, case, img, not args.no_judge)
            s["_path"] = rel
            cand_scores.append(s)
            print(f"  structured_{i}:", {k: round(v, 3) for k, v in s.items() if isinstance(v, (int, float))})
        entry["candidates"] = cand_scores

        # C1 = structured[0] (isolates prompt lever)
        entry["conditions"]["C1_structured"] = cand_scores[0]
        # C2 garment = best structured by garment gate (clip+color)
        gsel = max(cand_scores, key=lambda s: (s.get("clip_garment", 0) + s.get("color_hist", 0)))
        entry["conditions"]["C2_bestN_garment"] = gsel
        # C2 face = best structured by ArcFace
        fsel = max(cand_scores, key=lambda s: (s.get("arcface") or -1))
        entry["conditions"]["C2_bestN_face"] = fsel

        results[cid] = entry

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))

    # flat CSV for quick reading
    rows = []
    for cid, e in results.items():
        for cond, s in e["conditions"].items():
            rows.append({
                "case": cid, "condition": cond,
                "arcface": s.get("arcface"),
                "clip_garment": s.get("clip_garment"),
                "color_hist": s.get("color_hist"),
                "delta_e": s.get("delta_e"),
                "judge_garment_overall": (s.get("judge_garment") or {}).get("overall"),
                "judge_identity_overall": (s.get("judge_identity") or {}).get("overall"),
                "judge_print_logo": (s.get("judge_garment") or {}).get("print_logo_fidelity"),
                "judge_same_person": (s.get("judge_identity") or {}).get("same_person_confidence_0_100"),
            })
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\nWROTE", OUT / "results.json", "and results.csv")
    print(BUDGET.summary())


if __name__ == "__main__":
    main()
