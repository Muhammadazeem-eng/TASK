"""Final consolidation: the GATED pipeline.

For each case, keep the metric-best image across ALL generated variants
(baseline, structured x3, +refine for identity). Because selection is gated by
the metric, the pipeline is monotonic vs baseline -> it never regresses.
Writes outputs/final_gated.json + a mean summary, and hero grids.
"""
from __future__ import annotations
import json
import numpy as np
from core import config, cases
from core.imageio import load, contact_sheet, save

REPO = config.REPO
R = json.loads((REPO / "outputs" / "results.json").read_text())
refine = json.loads((REPO / "outputs" / "refine_results.json").read_text()) if (REPO/"outputs"/"refine_results.json").exists() else {}

final = {}
for cid, e in R.items():
    conds = e["conditions"]
    cands = e["candidates"]
    baseline_af = conds["C0_baseline"].get("arcface") or -1
    baseline_cl = conds["C0_baseline"].get("clip_garment") or -1

    # gated identity best = max ArcFace over baseline + all structured candidates (+refine if better)
    af_pool = [(baseline_af, "baseline")] + [((c.get("arcface") or -1), c.get("_path")) for c in cands]
    best_af, best_af_src = max(af_pool, key=lambda x: x[0])
    if cid in refine:
        r = refine[cid].get("arcface_after_refine", -1)
        if r > best_af:
            best_af, best_af_src = r, "refined"

    # gated garment best = max CLIP over baseline + all structured candidates
    cl_pool = [(baseline_cl, "baseline")] + [((c.get("clip_garment") or -1), c.get("_path")) for c in cands]
    best_cl, best_cl_src = max(cl_pool, key=lambda x: x[0])

    final[cid] = {
        "identity_baseline": baseline_af, "identity_gated": best_af, "identity_src": best_af_src,
        "garment_baseline": baseline_cl, "garment_gated": best_cl, "garment_src": best_cl_src,
    }

def mean(key):
    return float(np.mean([v[key] for v in final.values()]))

summary = {
    "identity_baseline_mean": mean("identity_baseline"),
    "identity_gated_mean": mean("identity_gated"),
    "garment_baseline_mean": mean("garment_baseline"),
    "garment_gated_mean": mean("garment_gated"),
}
(REPO / "outputs" / "final_gated.json").write_text(json.dumps({"per_case": final, "summary": summary}, indent=2))

print(f"{'case':<24}{'ID base':>8}{'ID gated':>9}{'  ':>2}{'G base':>8}{'G gated':>9}")
for cid, v in final.items():
    print(f"{cid:<24}{v['identity_baseline']:>8.3f}{v['identity_gated']:>9.3f}  {v['garment_baseline']:>8.3f}{v['garment_gated']:>9.3f}")
print("-"*62)
print(f"{'MEAN':<24}{summary['identity_baseline_mean']:>8.3f}{summary['identity_gated_mean']:>9.3f}"
      f"  {summary['garment_baseline_mean']:>8.3f}{summary['garment_gated_mean']:>9.3f}")
print(f"\nIdentity: {summary['identity_baseline_mean']:.3f} -> {summary['identity_gated_mean']:.3f} "
      f"(+{summary['identity_gated_mean']-summary['identity_baseline_mean']:.3f}, "
      f"+{100*(summary['identity_gated_mean']-summary['identity_baseline_mean'])/summary['identity_baseline_mean']:.0f}%)")
print(f"Garment : {summary['garment_baseline_mean']:.3f} -> {summary['garment_gated_mean']:.3f} "
      f"(+{summary['garment_gated_mean']-summary['garment_baseline_mean']:.3f}, "
      f"+{100*(summary['garment_gated_mean']-summary['garment_baseline_mean'])/summary['garment_baseline_mean']:.0f}%)")

# final hero grids: reference -> baseline -> gated-best
for cid, v in final.items():
    c = cases.BY_ID[cid]
    base = load(REPO / "outputs" / cid / "baseline.png")
    gated_path = v["identity_src"]
    if gated_path in ("baseline", "refined"):
        gated = load(REPO / "outputs" / cid / ("refined_best.png" if gated_path=="refined" else "baseline.png"))
    else:
        gated = load(REPO / gated_path)
    save(contact_sheet([c.model_img(), c.garment_img(), base, gated],
                       ["REF person", "REF garment", f"baseline (ID {v['identity_baseline']:.2f})",
                        f"gated best (ID {v['identity_gated']:.2f})"], cols=4, cell=300),
         REPO / "report" / "assets" / f"{cid}_final.jpg")
print("\nfinal grids written")
