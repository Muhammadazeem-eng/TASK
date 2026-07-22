"""Generate a CHEAP-model baseline (Nano Banana 2.5 Flash) per case, same naive
prompt as the Pro baseline. Used to (a) surface failure modes more clearly and
(b) quantify 'model choice' as a lever (flash-baseline vs pro-baseline).
Saved as outputs/<case>/baseline_flash.png. Resumable.
"""
from __future__ import annotations
from core.openrouter import OpenRouter, BUDGET
from core.prompts import baseline_photoshoot_prompt
from core import cases, config
from core.imageio import save

client = OpenRouter()
for c in cases.CASES:
    out = config.REPO / "outputs" / c.id / "baseline_flash.png"
    if out.exists():
        print(f"[{c.id}] baseline_flash (cached)"); continue
    refs = [c.model_img(), c.garment_img(), c.env_img()]
    prompt = baseline_photoshoot_prompt(c.garment_hint, c.env_desc)
    res = client.generate(prompt, refs=refs, model=config.MODEL_PROTOTYPE, tag=f"{c.id}:flash")
    if res.images:
        save(res.images[0], out)
    print(f"[{c.id}] baseline_flash  $%.4f  (cum {BUDGET.spent_usd:.3f})" % res.cost)
print("\nDONE.", BUDGET.summary())
