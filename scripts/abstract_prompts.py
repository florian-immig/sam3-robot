"""Usage: python scripts/abstract_prompts.py --image test --prompts "traversable" "do not step on"

Do affordance-style prompts ("traversable", "do not step on") work in SAM 3?
Saves one grid per image: each tile = overlay of all instances for one prompt,
titled with prompt / #instances / max score / % of image covered."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import argparse, glob, os

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT = [
    # affordance / abstract
    "traversable", "walkable ground", "drivable surface", "safe to step on",
    "do not step on", "unsafe to step on", "obstacle", "hazard",
    "dangerous area", "slippery surface", "fragile object", "something to avoid",
    # concrete baselines for comparison
    "ground", "path", "water", "person",
]

ap = argparse.ArgumentParser()
ap.add_argument("--image", nargs="*", help="file name(s) inside data/images, extension optional")
ap.add_argument("--images", default="data/images/*.jpg", help="glob, used when --image is not given")
ap.add_argument("--out", default="outputs/abstract")
ap.add_argument("--threshold", type=float, default=0.2)
ap.add_argument("--prompts", nargs="*", default=DEFAULT)
ap.add_argument("--cols", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def resolve(names):
    paths = []
    for n in names:
        cands = [n, f"data/images/{n}"] + [f"data/images/{n}{e}" for e in (".jpg", ".jpeg", ".png", ".JPG")]
        hit = next((c for c in cands if os.path.isfile(c)), None)
        if hit is None:
            raise SystemExit(f"image not found: {n} (looked in data/images/)")
        paths.append(hit)
    return paths


proc = Sam3Processor(build_sam3_image_model(), confidence_threshold=a.threshold)
rng = np.random.RandomState(1)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    paths = resolve(a.image) if a.image else sorted(glob.glob(a.images))
    for p in paths:
        rgb = np.array(Image.open(p).convert("RGB"))
        h, w = rgb.shape[:2]
        scale = 480 / w
        tiles, all_masks, all_scores = [], [], []
        state = proc.set_image(Image.fromarray(rgb))
        print(f"\n== {p}")
        for q in a.prompts:
            proc.reset_all_prompts(state)
            r = proc.set_text_prompt(state=state, prompt=q)
            s = r["scores"].float().cpu().numpy().reshape(-1)
            m = r["masks"].float().cpu().numpy().reshape(-1, h, w) > 0.5 if len(s) else np.zeros((0, h, w), bool)
            all_masks.append(m); all_scores.append(s)
            out = rgb.astype(np.float32)
            for mk in m:
                out[mk] = 0.5 * out[mk] + 0.5 * rng.randint(60, 255, 3)
            area = float(m.any(0).mean()) * 100 if len(s) else 0.0
            title = f"{q} | n={len(s)} max={s.max() if len(s) else 0:.2f} area={area:.0f}%"
            print(title)
            tile = cv2.resize(out.astype(np.uint8), None, fx=scale, fy=scale)
            tile = cv2.copyMakeBorder(tile, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(tile, title, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % a.cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + a.cols]) for i in range(0, len(tiles), a.cols)]
        name = os.path.splitext(os.path.basename(p))[0]
        Image.fromarray(np.vstack(rows)).save(f"{a.out}/{name}.png")

        # ---- segmentation outputs ----
        name = os.path.splitext(os.path.basename(p))[0]
        # 1) per-prompt instance masks + scores (load with np.load(..., allow_pickle=True))
        np.savez_compressed(
            f"{a.out}/{name}_masks.npz", prompts=np.array(a.prompts),
            **{f"masks_{i}": mk for i, mk in enumerate(all_masks)},
            **{f"scores_{i}": sc for i, sc in enumerate(all_scores)})
        # 2) combined label map: each pixel -> prompt with highest-scoring instance (-1 = none)
        best = np.zeros((h, w), np.float32)
        lab = np.full((h, w), -1, np.int16)
        for i, (mk, sc) in enumerate(zip(all_masks, all_scores)):
            for inst, sv in zip(mk, sc):
                upd = inst & (sv > best)
                best[upd] = sv
                lab[upd] = i
        np.save(f"{a.out}/{name}_labels.npy", lab)
        colors = np.random.RandomState(7).randint(60, 255, (len(a.prompts), 3))
        seg = rgb.astype(np.float32)
        for i in range(len(a.prompts)):
            seg[lab == i] = 0.4 * seg[lab == i] + 0.6 * colors[i]
        seg = np.ascontiguousarray(seg.astype(np.uint8))
        used = [i for i in range(len(a.prompts)) if (lab == i).any()]
        for row, i in enumerate(used):
            cv2.rectangle(seg, (6, 6 + 22 * row), (22, 22 + 22 * row), tuple(int(c) for c in colors[i]), -1)
            cv2.putText(seg, f"{a.prompts[i]} ({(lab == i).mean() * 100:.0f}%)", (28, 20 + 22 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        Image.fromarray(seg).save(f"{a.out}/{name}_segmentation.png")
        print(f"saved: {a.out}/{name}.png (grid), {name}_segmentation.png, {name}_labels.npy, {name}_masks.npz")
