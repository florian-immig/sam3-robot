"""Segment one image with text prompts and save the segmentation overlaid on the image.

  python scripts/segment.py --image test0 --prompts gravel grass tree sky --threshold 0.3

Output: outputs/segment/<image>_overlay.png (colored masks + contours + legend),
        outputs/segment/<image>_masks.npz  (per-prompt instance masks and scores)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import argparse, os

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True, help="file name in data/images (extension optional) or a path")
ap.add_argument("--prompts", nargs="+", required=True)
ap.add_argument("--threshold", type=float, default=0.3)
ap.add_argument("--alpha", type=float, default=0.5, help="mask opacity")
ap.add_argument("--out", default="outputs/segment")
a = ap.parse_args()

cands = [a.image, f"data/images/{a.image}"] + [f"data/images/{a.image}{e}" for e in (".jpg", ".jpeg", ".png", ".JPG")]
path = next((c for c in cands if os.path.isfile(c)), None)
if path is None:
    raise SystemExit(f"image not found: {a.image}")
os.makedirs(a.out, exist_ok=True)
name = os.path.splitext(os.path.basename(path))[0]

img = Image.open(path).convert("RGB")
rgb = np.array(img)
h, w = rgb.shape[:2]

proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=a.threshold)
masks, scores = [], []
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = proc.set_image(img)
    for q in a.prompts:
        proc.reset_all_prompts(state)
        r = proc.set_text_prompt(state=state, prompt=q)
        s = r["scores"].float().cpu().numpy().reshape(-1)
        m = (r["masks"].float().cpu().numpy().reshape(-1, h, w) > 0.5) if len(s) else np.zeros((0, h, w), bool)
        masks.append(m); scores.append(s)
        union = m.any(0).mean() * 100 if len(s) else 0.0
        print(f"{q:20s} n={len(s)} max={s.max() if len(s) else 0:.2f} area={union:.1f}%")

# each pixel takes the prompt of its highest-scoring instance
best = np.zeros((h, w), np.float32)
lab = np.full((h, w), -1, np.int16)
for i, (m, s) in enumerate(zip(masks, scores)):
    for inst, sv in zip(m, s):
        upd = inst & (sv > best)
        best[upd] = sv
        lab[upd] = i

colors = np.array([plt for plt in [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]])
colors = np.array([colors[i % len(colors)] for i in range(len(a.prompts))])

out = rgb.astype(np.float32)
for i in range(len(a.prompts)):
    mk = lab == i
    out[mk] = (1 - a.alpha) * out[mk] + a.alpha * colors[i]
out = np.ascontiguousarray(out.astype(np.uint8))
for i in range(len(a.prompts)):
    cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, tuple(int(c) for c in colors[i]), 1)

# legend
for row, q in enumerate(a.prompts):
    y = 8 + 24 * row
    pct = (lab == row).mean() * 100
    cv2.rectangle(out, (8, y), (26, y + 16), tuple(int(c) for c in colors[row]), -1)
    txt = f"{q} ({pct:.0f}%)" if len(scores[row]) else f"{q} (none)"
    cv2.putText(out, txt, (32, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(out, txt, (32, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

Image.fromarray(out).save(f"{a.out}/{name}_overlay.png")
np.savez_compressed(f"{a.out}/{name}_masks.npz", prompts=np.array(a.prompts),
                    labels=lab, **{f"masks_{i}": m for i, m in enumerate(masks)},
                    **{f"scores_{i}": s for i, s in enumerate(scores)})
print(f"saved {a.out}/{name}_overlay.png and {name}_masks.npz")
