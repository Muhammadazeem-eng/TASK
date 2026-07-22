"""Stage 1 - GENERATION (the only step that spends money).

Per case we produce one shared candidate pool that serves both tasks:
  - baseline (N=1): naive prompt, our control condition
  - structured (N=K): VLM-derived person+garment description + negative constraints

Conditions derived later from the pool:
  C0 baseline            = baseline image
  C1 structured prompt   = structured[0]           (isolates the prompt lever)
  C2 structured + best-N = argmax(gate) over structured  (isolates the best-of-N lever)

Resumable: existing images are skipped so re-runs don't re-spend.
Run:  python generate_all.py --model google/gemini-3-pro-image --n 3
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from core.openrouter import OpenRouter, BUDGET
from core.prompts import (baseline_photoshoot_prompt, describe_garment,
                          describe_person, structured_prompt)
from core import cases, config
from core.imageio import save, load

OUT = config.REPO / "outputs"
DESC = OUT / "descriptions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.GEN_MODEL)
    ap.add_argument("--n", type=int, default=3, help="structured candidates per case")
    ap.add_argument("--cases", nargs="*", default=None, help="case ids (default: all)")
    args = ap.parse_args()

    client = OpenRouter()
    case_list = [cases.BY_ID[c] for c in args.cases] if args.cases else cases.CASES
    DESC.mkdir(parents=True, exist_ok=True)
    manifest = {}
    print(f"model={args.model}  n={args.n}  cases={[c.id for c in case_list]}")
    print("budget left: $%.2f\n" % BUDGET.remaining())

    for c in case_list:
        cdir = OUT / c.id
        cdir.mkdir(parents=True, exist_ok=True)
        refs = [c.model_img(), c.garment_img(), c.env_img()]

        # --- VLM descriptions (cached) -------------------------------------
        gpath = DESC / f"{c.id}.garment.json"
        ppath = DESC / f"{c.id}.person.json"
        if gpath.exists():
            gdesc = json.loads(gpath.read_text())
        else:
            gdesc = describe_garment(client, c.garment_img()); gpath.write_text(json.dumps(gdesc, indent=2))
        if ppath.exists():
            pdesc = json.loads(ppath.read_text())
        else:
            pdesc = describe_person(client, c.model_img()); ppath.write_text(json.dumps(pdesc, indent=2))

        # --- baseline (control) --------------------------------------------
        b_path = cdir / "baseline.png"
        if not b_path.exists():
            prompt = baseline_photoshoot_prompt(c.garment_hint, c.env_desc)
            res = client.generate(prompt, refs=refs, model=args.model, tag=f"{c.id}:baseline")
            if res.images:
                save(res.images[0], b_path)
            print(f"[{c.id}] baseline  $%.4f  (cum {BUDGET.spent_usd:.3f})" % res.cost)
        else:
            print(f"[{c.id}] baseline  (cached)")

        # --- structured candidates -----------------------------------------
        sprompt = structured_prompt(pdesc, gdesc, c.env_desc)
        (cdir / "structured_prompt.txt").write_text(sprompt, encoding="utf-8")
        for i in range(args.n):
            s_path = cdir / f"structured_{i}.png"
            if s_path.exists():
                print(f"[{c.id}] structured_{i} (cached)"); continue
            res = client.generate(sprompt, refs=refs, model=args.model, tag=f"{c.id}:struct#{i}")
            if res.images:
                save(res.images[0], s_path)
            print(f"[{c.id}] structured_{i}  $%.4f  (cum {BUDGET.spent_usd:.3f})" % res.cost)

        manifest[c.id] = {
            "baseline": str(b_path.relative_to(config.REPO)),
            "structured": [str((cdir / f"structured_{i}.png").relative_to(config.REPO))
                           for i in range(args.n)],
            "garment_desc": gdesc, "person_desc": pdesc,
            "model_ref": str(c.model_path.relative_to(config.REPO)),
            "garment_ref": str(c.garment_path.relative_to(config.REPO)),
        }

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nDONE.", BUDGET.summary())
    print("manifest:", OUT / "manifest.json")


if __name__ == "__main__":
    main()
