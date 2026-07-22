"""Strategy E: LITERAL logo pixel-compositing (guaranteed logo fidelity).

Model-based editing *regenerates* a logo, so it can never guarantee an unseen brand
wordmark. This copies the REAL logo pixels from the packshot onto the worn garment:
  1. VLM locates the logo in the packshot  -> crop the true logo patch.
  2. VLM locates where that logo sits on the worn garment in the output -> target box.
  3. Warp/scale the patch to the target and blend with the target's luminance so it
     inherits fabric shading/folds (multiply-style), with a feathered edge.
The logo is then pixel-exact by construction.

Run: python logo_composite.py --cases c6_man_sneaker_street c4_woman_shorts_street
"""
from __future__ import annotations
import argparse, json
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from core.openrouter import OpenRouter, BUDGET
from core import config, cases, metrics as M
from core.imageio import load, save
from core.vlm_judge import locate, judge_garment
from core.composite import expand_box

# where the logo is, per garment, phrased for the localizer
PACK_LOGO = {
    "shorts": "the small embroidered brand logo/text on the front-left thigh of the shorts",
    "sneaker": "the small brand text/logo on the side of the shoe",
    "jacket": "the brand logo or label on the garment",
}
WORN_LOGO = {
    "shorts": "the small brand logo/text area on the front-left thigh of the shorts the person wears",
    "sneaker": "the side panel of the shoe where a small brand logo would be",
    "jacket": "the brand logo area on the jacket",
}


def gkey(cid):
    for k in PACK_LOGO:
        if k in cid:
            return k
    return "jacket"


def luminance_blend(base_rgb: np.ndarray, patch_rgb: np.ndarray) -> np.ndarray:
    """Modulate the patch by the base region's relative luminance to inherit shading."""
    base_lum = (0.299*base_rgb[..., 0] + 0.587*base_rgb[..., 1] + 0.114*base_rgb[..., 2])
    m = float(np.clip(base_lum.mean(), 1, 255))
    factor = np.clip(base_lum / m, 0.6, 1.4)[..., None]
    out = np.clip(patch_rgb.astype(np.float32) * factor, 0, 255)
    return out.astype(np.uint8)


def paste_logo(base: Image.Image, logo_patch: Image.Image, box_px, feather_frac=0.18) -> Image.Image:
    base = base.convert("RGB").copy()
    x0, y0, x1, y1 = box_px
    tw, th = x1 - x0, y1 - y0
    if tw < 4 or th < 4:
        return base
    patch = logo_patch.convert("RGB").resize((tw, th), Image.LANCZOS)
    region = base.crop((x0, y0, x1, y1))
    blended = Image.fromarray(luminance_blend(np.asarray(region), np.asarray(patch)))
    # feathered elliptical mask so the patch melts into the fabric
    mask = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(mask).ellipse([tw*0.04, th*0.04, tw*0.96, th*0.96], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(2, int(min(tw, th)*feather_frac))))
    base.paste(blended, (x0, y0), mask)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", required=True)
    ap.add_argument("--source", default="composited",
                    help="which output to paste onto: composited|C2_bestN_garment")
    args = ap.parse_args()
    client = OpenRouter()
    R = json.loads((config.REPO / "outputs" / "results.json").read_text())
    out = {}
    for cid in args.cases:
        case = cases.BY_ID[cid]
        packshot = case.garment_img()
        k = gkey(cid)
        # source image to paste onto
        src_path = config.REPO / "outputs" / cid / f"{args.source}.png"
        if not src_path.exists():
            src_path = config.REPO / R[cid]["conditions"]["C2_bestN_garment"]["_path"]
        full = load(src_path); W, H = full.size

        # 1) logo patch from packshot
        pbox = locate(client, packshot, PACK_LOGO[k])
        if not pbox:
            print(f"[{cid}] packshot logo not found; skip"); continue
        pw, ph = packshot.size
        pbox_px = expand_box(pbox, pw, ph, pad=0.15)
        logo_patch = packshot.crop(pbox_px)

        # 2) target location on worn garment
        tbox = locate(client, full, WORN_LOGO[k])
        if not tbox:
            print(f"[{cid}] worn logo location not found; skip"); continue
        tbox_px = expand_box(tbox, W, H, pad=0.05)

        # 3) paste + blend
        composed = paste_logo(full, logo_patch, tbox_px)
        save(composed, config.REPO / "outputs" / cid / "logo_composited.png")

        jg_b = judge_garment(client, packshot, full)
        jg_a = judge_garment(client, packshot, composed)
        out[cid] = {"judge_print_before": jg_b.get("print_logo_fidelity"),
                    "judge_print_after": jg_a.get("print_logo_fidelity"),
                    "judge_overall_before": jg_b.get("overall"),
                    "judge_overall_after": jg_a.get("overall"),
                    "target_box": tbox_px}
        print(f"[{cid}] judge print/logo {jg_b.get('print_logo_fidelity')}->{jg_a.get('print_logo_fidelity')} "
              f"overall {jg_b.get('overall')}->{jg_a.get('overall')} (cum ${BUDGET.spent_usd:.3f})")

    (config.REPO / "outputs" / "logo_results.json").write_text(json.dumps(out, indent=2))
    print("\nDONE.", BUDGET.summary())


if __name__ == "__main__":
    main()
