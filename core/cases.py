"""Test-case registry: model x garment x environment combinations.

A single generated photoshoot serves BOTH tasks (it contains a person and a
garment), which is why pairing the two tasks is efficient.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from . import config
from .imageio import load


@dataclass
class Case:
    id: str
    model_path: Path
    garment_path: Path
    env_path: Path
    env_desc: str
    garment_hint: str          # short human hint, only used by the baseline prompt

    def model_img(self) -> Image.Image: return load(self.model_path)
    def garment_img(self) -> Image.Image: return load(self.garment_path)
    def env_img(self) -> Image.Image: return load(self.env_path)


M = config.MODELS_DIR
G = config.GARMENTS_DIR
E = config.ENV_DIR

ENV_DESC = {
    "env_cobblestone_street.png": "an old European cobblestone street with weathered buildings, overcast daylight",
    "env_birch_forest.png": "a misty birch forest with soft morning light filtering through the trees",
    "env_flower_meadow.png": "a sunlit summer flower meadow at golden hour with soft haze",
}

CASES: list[Case] = [
    Case("c1_man_jacket_street", M/"m3_man_bearded.png", G/"g3_jacket_daysmarch.png",
         E/"env_cobblestone_street.png", ENV_DESC["env_cobblestone_street.png"],
         "a dark green cotton field jacket with a corduroy collar"),
    Case("c2_woman_jacket_forest", M/"m2_woman_lightskin.png", G/"g3_jacket_daysmarch.png",
         E/"env_birch_forest.png", ENV_DESC["env_birch_forest.png"],
         "a dark green cotton field jacket with a corduroy collar"),
    Case("c3_man_shorts_meadow", M/"m3_man_bearded.png", G/"g1_shorts_beyondnordic.png",
         E/"env_flower_meadow.png", ENV_DESC["env_flower_meadow.png"],
         "olive-green two-tone utility shorts with cargo pockets"),
    Case("c4_woman_shorts_street", M/"m1_woman_darkskin.png", G/"g1_shorts_beyondnordic.png",
         E/"env_cobblestone_street.png", ENV_DESC["env_cobblestone_street.png"],
         "olive-green two-tone utility shorts with cargo pockets"),
    Case("c5_woman_sneaker_forest", M/"m2_woman_lightskin.png", G/"g2_sneaker_axelarigato.jpg",
         E/"env_birch_forest.png", ENV_DESC["env_birch_forest.png"],
         "brown and tan low-top leather sneakers with a gum sole"),
    Case("c6_man_sneaker_street", M/"m3_man_bearded.png", G/"g2_sneaker_axelarigato.jpg",
         E/"env_cobblestone_street.png", ENV_DESC["env_cobblestone_street.png"],
         "brown and tan low-top leather sneakers with a gum sole"),
]

BY_ID = {c.id: c for c in CASES}
