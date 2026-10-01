"""Print raw SAM 3 outputs per prompt at a low threshold, to tell 'nothing found' from a bug."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import argparse, glob, os

import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3_terrain.concepts import CONCEPTS

ap = argparse.ArgumentParser()
ap.add_argument("--image", nargs="*", help="file name(s) in data/images, extension optional")
ap.add_argument("--images", default="data/images/*.jpg")
ap.add_argument("--threshold", type=float, default=0.05)
ap.add_argument("--extra", nargs="*", default=["person", "tree", "sky", "car", "building"],
                help="sanity-check prompts that should exist in most photos")
a = ap.parse_args()

def resolve(names):
    out = []
    for n in names:
        c = [n, f"data/images/{n}"] + [f"data/images/{n}{e}" for e in (".jpg", ".jpeg", ".png", ".JPG")]
        hit = next((x for x in c if os.path.isfile(x)), None)
        if hit is None:
            raise SystemExit(f"image not found: {n}")
        out.append(hit)
    return out


proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=a.threshold)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for p in (resolve(a.image) if a.image else sorted(glob.glob(a.images))):
        img = Image.open(p).convert("RGB")
        print(f"\n== {p}  size={img.size}")
        state = proc.set_image(img)
        for q in [c.prompt for c in CONCEPTS] + a.extra:
            proc.reset_all_prompts(state)
            r = proc.set_text_prompt(state=state, prompt=q)
            s = r["scores"].float().cpu().sort(descending=True).values
            mk = r["masks"]
            px = float(mk.float().mean()) * 100 if mk.numel() else 0.0
            print(f"{q:16s} n={len(s)} top={[round(float(x),2) for x in s[:5]]} "
                  f"masks={tuple(mk.shape)} mask_px={px:.1f}%")
