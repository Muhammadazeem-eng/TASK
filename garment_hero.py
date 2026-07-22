"""Strategy F: garment-hero framing (root-cause fix for logo fidelity).

Small logos fail because the garment is tiny in a full-body frame. Instead of
fighting that post-hoc, generate a closer, garment-forward framing so the print
gets enough native pixels. Best-of-N by garment-crop CLIP.

Run: python garment_hero.py --cases c4_woman_shorts_street c6_man_sneaker_street
"""
from __future__ import annotations
import argparse, json
from core.openrouter import OpenRouter, BUDGET
from core import config, cases, metrics as M
from core.imageio import load, save
from core.vlm_judge import garment_crop, judge_garment
from core.prompts import describe_garment, describe_person, structured_prompt

FRAMING = {
    "shorts": "Tight three-quarter crop from the waist to the knees, the shorts filling most of the frame, "
              "the brand logo on the thigh clearly visible and in sharp focus.",
    "sneaker": "Close-up product-style shot of the lower legs and feet on the ground, the shoes large and "
               "in sharp focus, the brand logo on the shoe clearly legible.",
    "jacket": "Waist-up three-quarter shot, the jacket filling most of the frame, details in sharp focus.",
}


def gkey(cid):
    for k in FRAMING:
        if k in cid:
            return k
    return "jacket"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", required=True)
    ap.add_argument("--model", default=config.MODEL_BEST)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    client = OpenRouter()
    manifest = json.loads((config.REPO / "outputs" / "manifest.json").read_text())
    out = {}
    for cid in args.cases:
        case = cases.BY_ID[cid]
        k = gkey(cid)
        m = manifest[cid]
        refs = [case.model_img(), case.garment_img(), case.env_img()]
        prompt = structured_prompt(m["person_desc"], m["garment_desc"], case.env_desc) + " " + FRAMING[k]
        best = None
        for i in range(args.n):
            res = client.generate(prompt, refs=refs, model=args.model, tag=f"{cid}:hero#{i}")
            if not res.images:
                continue
            img = res.images[0]
            crop = garment_crop(client, img, k)
            cl = M.clip_similarity(case.garment_img(), crop)
            save(img, config.REPO / "outputs" / cid / f"hero_{i}.png")
            if best is None or cl > best[0]:
                best = (cl, img)
        if best is None:
            print(f"[{cid}] no hero image"); continue
        save(best[1], config.REPO / "outputs" / cid / "hero_best.png")
        # compare garment fidelity: full-body best-of-N vs hero
        full = load(config.REPO / json.loads((config.REPO/"outputs"/"results.json").read_text())[cid]["conditions"]["C2_bestN_garment"]["_path"])
        jg_full = judge_garment(client, case.garment_img(), full)
        jg_hero = judge_garment(client, case.garment_img(), best[1])
        out[cid] = {
            "clip_garment_crop_hero": best[0],
            "judge_print_fullbody": jg_full.get("print_logo_fidelity"),
            "judge_print_hero": jg_hero.get("print_logo_fidelity"),
            "judge_overall_fullbody": jg_full.get("overall"),
            "judge_overall_hero": jg_hero.get("overall"),
        }
        print(f"[{cid}] judge print/logo  full-body={jg_full.get('print_logo_fidelity')} -> hero={jg_hero.get('print_logo_fidelity')} "
              f"| overall {jg_full.get('overall')}->{jg_hero.get('overall')}  (cum ${BUDGET.spent_usd:.3f})")
    (config.REPO / "outputs" / "hero_results.json").write_text(json.dumps(out, indent=2))
    print("\nDONE.", BUDGET.summary())


if __name__ == "__main__":
    main()
