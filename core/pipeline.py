"""Shared composition pipeline: generate -> score -> best-of-N / retry.

Both tasks reuse this. The only difference is the scorer:
  Task 1 -> garment gate (CLIP on garment crop + color histogram)
  Task 3 -> face gate   (ArcFace cosine)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from PIL import Image
from .openrouter import OpenRouter, GenResult


@dataclass
class Candidate:
    image: Image.Image
    scores: dict = field(default_factory=dict)
    combined: float = 0.0
    cost: float = 0.0


def generate_candidates(client: OpenRouter, prompt: str, refs: list[Image.Image],
                        n: int, model: str | None = None, tag: str = "gen") -> list[Candidate]:
    cands: list[Candidate] = []
    for i in range(n):
        res: GenResult = client.generate(prompt, refs=refs, model=model, tag=f"{tag}#{i}")
        for im in res.images:
            cands.append(Candidate(image=im, cost=res.cost))
    return cands


def score_candidates(cands: list[Candidate], scorer: Callable[[Image.Image], dict],
                     combine: Callable[[dict], float]) -> list[Candidate]:
    for c in cands:
        c.scores = scorer(c.image)
        c.combined = combine(c.scores)
    return cands


def best_of_n(client: OpenRouter, prompt: str, refs: list[Image.Image], n: int,
              scorer: Callable[[Image.Image], dict], combine: Callable[[dict], float],
              model: str | None = None, tag: str = "bon") -> tuple[Candidate, list[Candidate]]:
    """Generate N candidates, score each, return (best, all). Auto-selection is the
    'pipeline' half of our strategy set: the model is a black box, so we sample and
    let the metric pick the winner."""
    cands = generate_candidates(client, prompt, refs, n, model=model, tag=tag)
    if not cands:
        raise RuntimeError("No candidates generated.")
    cands = score_candidates(cands, scorer, combine)
    cands.sort(key=lambda c: c.combined, reverse=True)
    return cands[0], cands
