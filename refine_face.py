"""Strategy C (pipeline): face-targeted refinement pass.

Attacks the dominant identity failure mode (small face in full-body shots).
Takes the best-of-N composed image and instructs the editing model to refine
ONLY the face to match the reference person, keeping pose/clothing/background
identical. Best-of-2, selected by ArcFace. Run on the weak-identity cases.

Run: python refine_face.py --cases c4_woman_shorts_street c5_woman_sneaker_forest c2_woman_jacket_forest
"""
from __future__ import annotations
import argparse, json
from core.openrouter import OpenRouter, BUDGET
from core import config, cases, metrics as M
from core.imageio import load, save

REFINE_PROMPT = (
    "Edit the FIRST image. Refine ONLY the person's face and head so the individual is "
    "unmistakably the same person as in the SECOND reference image: match the exact bone "
    "structure, eye shape and spacing, nose, lips, jawline, skin tone, and hairstyle. "
    "Do NOT beautify, slim, or average the face. Keep the pose, body, clothing, lighting, "
    "and background EXACTLY the same. Photorealistic, seamless."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", required=True)
    ap.add_argument("--model", default=config.MODEL_BEST)
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()

    client = OpenRouter()
    R = json.loads((config.REPO / "outputs" / "results.json").read_text())
    out = {}
    for cid in args.cases:
        case = cases.BY_ID[cid]
        ref = case.model_img()
        base_path = R[cid]["conditions"]["C2_bestN_face"].get("_path")
        src = load(config.REPO / base_path)
        af_before = M.face_similarity(ref, src)
        cands = []
        for i in range(args.n):
            res = client.generate(REFINE_PROMPT, refs=[src, ref], model=args.model,
                                  tag=f"{cid}:refine#{i}")
            if res.images:
                img = res.images[0]
                af = M.face_similarity(ref, img)
                cands.append((af if af is not None else -1, img))
                save(img, config.REPO / "outputs" / cid / f"refined_{i}.png")
        cands.sort(key=lambda x: x[0], reverse=True)
        best_af, best_img = cands[0]
        save(best_img, config.REPO / "outputs" / cid / "refined_best.png")
        out[cid] = {"arcface_before_refine": af_before, "arcface_after_refine": best_af}
        print(f"[{cid}] ArcFace  bestN={af_before:.3f} -> refined={best_af:.3f}  (cum ${BUDGET.spent_usd:.3f})")

    (config.REPO / "outputs" / "refine_results.json").write_text(json.dumps(out, indent=2))
    print("\nDONE.", BUDGET.summary())


if __name__ == "__main__":
    main()
