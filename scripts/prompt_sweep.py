"""How sensitive is SAM 3 to prompt phrasing? Prints max score + mask area per variant."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import argparse, glob

from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3_terrain.concepts import PROMPT_VARIANTS

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/images/*.jpg")
a = ap.parse_args()

proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=0.3)
for p in sorted(glob.glob(a.images)):
    state = proc.set_image(Image.open(p).convert("RGB"))
    print(f"\n== {p}")
    for group, prompts in PROMPT_VARIANTS.items():
        for q in prompts:
            proc.reset_all_prompts(state)
            r = proc.set_text_prompt(state=state, prompt=q)
            s = r["scores"]
            area = float(r["masks"].float().mean()) * 100 if len(s) else 0.0
            print(f"{group:12s} {q:24s} n={len(s)} "
                  f"max={float(s.max()) if len(s) else 0:.2f} area={area:.1f}%")
