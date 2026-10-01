"""Concept vocabulary for concept-style SAM 3 terrain segmentation.

Each concept is a text prompt. `hazard` is a prior cost in [0, 1] (0 = safe to
traverse, 1 = never step on). Edit freely: comparing prompt phrasings is one of
the experiments (see README).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Concept:
    prompt: str
    hazard: float
    group: str  # "ground" | "hazard" | "obstacle"


CONCEPTS = [
    # traversable surfaces
    Concept("road", 0.0, "ground"),
    Concept("floor", 0.0, "ground"),
    Concept("gravel", 0.15, "ground"),
    Concept("grass", 0.25, "ground"),
    # semantically unsafe even if geometrically flat
    Concept("water puddle", 1.0, "hazard"),
    Concept("mud", 0.8, "hazard"),
    Concept("cable", 0.7, "hazard"),
    # obstacles / things not to step on
    Concept("person", 1.0, "obstacle"),
    Concept("cup", 0.9, "obstacle"),
    Concept("rock", 0.6, "obstacle"),
]

PROMPT_VARIANTS = {
    # used by scripts/prompt_sweep.py to test phrasing sensitivity
    "water": ["water", "water puddle", "puddle", "wet ground", "standing water"],
    "traversable": ["road", "path", "walkable ground", "drivable surface", "floor"],
    "unsafe": ["unsafe to step on", "slippery surface", "fragile object", "hazard"],
}
