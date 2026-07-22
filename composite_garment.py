"""Strategy D (pipeline): crop-refine-composite for logo/print fidelity.

Root cause (from our taxonomy): the garment is too small in a full-body frame for
the black box to reconstruct a logo faithfully. Fix:
  1. locate the garment region in the best-of-N output (VLM bbox),
  2. crop it and upscale so the logo is LARGE,
  3. edit the crop against the packshot -> "reproduce the exact print/logo/colors,
     keep the pose/folds/lighting of this crop",
  4. best-of-2 by CLIP(packshot, edited-crop),
  5. feather-composite the fixed crop back into the full image.

Run: python composite_garment.py --cases c3_man_shorts_meadow c4_woman_shorts_street \
        c5_woman_sneaker_forest c6_man_sneaker_street
"""
from __future__ import annotations
import utils.session
import argparse, json
from core.openrouter import OpenRouter, BUDGET
from core import config, cases, metrics as M
from core.imageio import load, save
from core.vlm_judge import locate, judge_garment
from core.composite import expand_box, crop_px, upscale_long_edge, feather_paste

EDIT_PROMPT = (
    "The FIRST image is a close-up crop of a garment a person is wearing. The SECOND image is "
    "the ground-truth product packshot of that exact garment. Edit the FIRST image so the "
    "garment's print, logo, text, colors, seams and details EXACTLY match the packshot — "
    "reproduce any logo/text faithfully and legibly. Keep the pose, fabric folds, wrinkles, "
    "body and lighting of the first image unchanged. Photorealistic, seamless."
)

GARMENT_TARGET = {  # what to ask the localizer for, per case
    "shorts": "the shorts the person is wearing",
    "sneaker": "the shoes/sneakers the person is wearing",
    "jacket": "the jacket the person is wearing",
}


def target_for(cid):
    for k, v in GARMENT_TARGET.items():
        if k in cid:
            return v
    return "the main garment the person is wearing"


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
        packshot = case.garment_img()
        base_path = R[cid]["conditions"]["C2_bestN_garment"].get("_path")
        full = load(config.REPO / base_path)
        W, H = full.size

        # 1-2) locate + crop + upscale
        box = locate(client, full, target_for(cid))
        if not box:
            print(f"[{cid}] garment not localized; skipping"); continue
        box_px = expand_box(box, W, H, pad=0.10)
        crop = upscale_long_edge(crop_px(full, box_px), 1024)
        clip_before = M.clip_similarity(packshot, crop_px(full, box_px))

        # 3-4) edit the crop, best-of-N by CLIP against packshot
        cands = []
        for i in range(args.n):
            res = client.generate(EDIT_PROMPT, refs=[crop, packshot], model=args.model,
                                  tag=f"{cid}:composite#{i}")
            if res.images:
                ed = res.images[0]
                cl = M.clip_similarity(packshot, ed)
                cands.append((cl, ed))
        if not cands:
            print(f"[{cid}] no edit produced; skipping"); continue
        cands.sort(key=lambda x: x[0], reverse=True)
        clip_after, best_crop = cands[0]

        # 5) feather-composite back
        composed = feather_paste(full, best_crop, box_px, feather=max(8, (box_px[2]-box_px[0])//40))
        save(composed, config.REPO / "outputs" / cid / "composited.png")
        save(best_crop, config.REPO / "outputs" / cid / "composited_crop.png")

        # re-score full image on the garment gate (crop region)
        clip_full_before = M.clip_similarity(packshot, crop_px(full, box_px))
        clip_full_after = M.clip_similarity(packshot, crop_px(composed, box_px))
        jg_before = judge_garment(client, packshot, full)
        jg_after = judge_garment(client, packshot, composed)
        out[cid] = {
            "clip_crop_before": clip_before, "clip_crop_after": clip_after,
            "clip_region_before": clip_full_before, "clip_region_after": clip_full_after,
            "judge_print_before": jg_before.get("print_logo_fidelity"),
            "judge_print_after": jg_after.get("print_logo_fidelity"),
            "judge_overall_before": jg_before.get("overall"),
            "judge_overall_after": jg_after.get("overall"),
        }
        print(f"[{cid}] CLIP crop {clip_before:.3f}->{clip_after:.3f} | "
              f"judge print/logo {jg_before.get('print_logo_fidelity')}->{jg_after.get('print_logo_fidelity')} "
              f"(cum ${BUDGET.spent_usd:.3f})")

    (config.REPO / "outputs" / "composite_results.json").write_text(json.dumps(out, indent=2))
    print("\nDONE.", BUDGET.summary())


if __name__ == "__main__":
    main()
