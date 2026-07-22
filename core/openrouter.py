"""Thin OpenRouter client for image generation/editing and VLM calls.

Every image model is treated as a black box (API only). A global BudgetTracker
accumulates the `usage.cost` OpenRouter reports so experiments stay under $20.
"""
from __future__ import annotations
import json, time, threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import requests
from PIL import Image

from . import config
from .imageio import to_data_url, from_data_url


class BudgetError(RuntimeError):
    pass


@dataclass
class BudgetTracker:
    limit_usd: float = config.HARD_BUDGET_USD
    spent_usd: float = 0.0
    calls: int = 0
    log: list[dict] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, cost: float, tag: str = ""):
        with self._lock:
            self.spent_usd += float(cost or 0.0)
            self.calls += 1
            self.log.append({"tag": tag, "cost": cost, "cumulative": self.spent_usd})
            if self.spent_usd > self.limit_usd:
                raise BudgetError(
                    f"Budget exceeded: ${self.spent_usd:.4f} > ${self.limit_usd:.2f}")

    def remaining(self) -> float:
        return self.limit_usd - self.spent_usd

    def summary(self) -> str:
        return f"${self.spent_usd:.4f} spent over {self.calls} calls (${self.remaining():.4f} left)"


BUDGET = BudgetTracker()


@dataclass
class GenResult:
    images: list[Image.Image]
    text: str
    cost: float
    raw: dict


class OpenRouter:
    def __init__(self, api_key: str | None = None, budget: BudgetTracker = BUDGET):
        self.api_key = api_key or config.API_KEY
        if not self.api_key:
            raise RuntimeError("No OpenRouter API key found (set OPENROUTER_API_KEY).")
        self.budget = budget
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://weon.ai",
            "X-Title": "weon-consistency-task",
        })

    # -- low level -----------------------------------------------------------
    def _post(self, payload: dict, tag: str, retries: int = 3, timeout: int = 180) -> dict:
        last = None
        for attempt in range(retries):
            try:
                r = self.session.post(f"{config.BASE_URL}/chat/completions",
                                      data=json.dumps(payload), timeout=timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"{r.status_code}: {r.text[:200]}"
                    time.sleep(2 * (attempt + 1)); continue
                data = r.json()
                if "error" in data:
                    last = data["error"]
                    # content policy / bad request: don't retry blindly
                    if r.status_code == 400:
                        raise RuntimeError(f"API error: {last}")
                    time.sleep(2 * (attempt + 1)); continue
                # OpenRouter only bills successful generations.
                cost = (data.get("usage") or {}).get("cost", 0.0)
                self.budget.add(cost, tag)
                return data
            except requests.RequestException as e:
                last = str(e); time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Request failed after {retries} tries: {last}")

    @staticmethod
    def _content_with_images(text: str, images: list[Image.Image] | None) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": text}]
        for im in (images or []):
            content.append({"type": "image_url",
                            "image_url": {"url": to_data_url(im)}})
        return content

    # -- image generation / editing -----------------------------------------
    def generate(self, prompt: str, refs: list[Image.Image] | None = None,
                 model: str | None = None, tag: str = "gen") -> GenResult:
        """Generate/edit an image. `refs` are reference images (model/garment/env
        for composition, or the source image for editing)."""
        model = model or config.GEN_MODEL
        payload = {
            "model": model,
            "modalities": ["image", "text"],
            "messages": [{"role": "user",
                          "content": self._content_with_images(prompt, refs)}],
        }
        data = self._post(payload, tag=f"{tag}:{model}")
        msg = data["choices"][0]["message"]
        imgs = [from_data_url(i["image_url"]["url"]) for i in (msg.get("images") or [])]
        cost = (data.get("usage") or {}).get("cost", 0.0)
        return GenResult(images=imgs, text=msg.get("content") or "", cost=cost, raw=data)

    # -- text/vision (VLM) ---------------------------------------------------
    def chat(self, prompt: str, images: list[Image.Image] | None = None,
             model: str | None = None, json_mode: bool = False, tag: str = "vlm") -> str:
        model = model or config.VLM_MODEL
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user",
                          "content": self._content_with_images(prompt, images)}],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = self._post(payload, tag=f"{tag}:{model}")
        return data["choices"][0]["message"]["content"] or ""

    def chat_json(self, prompt: str, images: list[Image.Image] | None = None,
                  model: str | None = None, tag: str = "vlm-json") -> dict:
        txt = self.chat(prompt, images, model=model, json_mode=True, tag=tag)
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # strip code fences if the model wrapped the JSON, else salvage the braces
            t = (txt or "").strip().strip("`")
            if t.startswith("json"):
                t = t[4:]
            start, end = t.find("{"), t.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(t[start:end + 1])
                except json.JSONDecodeError:
                    pass
            # graceful degradation: empty/unparseable response shouldn't crash a run
            return {}
