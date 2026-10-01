"""Concept-style SAM 3 on still images -> overlay, cost heatmap, per-concept stats."""
import argparse, glob, json, os, time

import numpy as np
from PIL import Image

from sam3_terrain.fusion import label_map, semantic_cost
from sam3_terrain.segmenter import ConceptSegmenter
from sam3_terrain.viz import cost_heatmap, overlay_labels

ap = argparse.ArgumentParser()
ap.add_argument("--images", default="data/images/*.jpg")
ap.add_argument("--out", default="outputs/images")
ap.add_argument("--threshold", type=float, default=0.5)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

seg = ConceptSegmenter(threshold=a.threshold)
for p in sorted(glob.glob(a.images)):
    img = Image.open(p).convert("RGB")
    t = time.time()
    dets = seg.segment_image(img)
    dt = time.time() - t
    rgb = np.array(img)
    cost, covered = semantic_cost(dets, rgb.shape[:2])
    idx, _ = label_map(dets, rgb.shape[:2])
    name = os.path.splitext(os.path.basename(p))[0]
    Image.fromarray(np.hstack([overlay_labels(rgb, dets, idx), cost_heatmap(cost)])
                    ).save(f"{a.out}/{name}.png")
    stats = {"seconds": round(dt, 3), "coverage": float(covered.mean()),
             "concepts": {d.concept.prompt: {"n": len(d.scores),
                          "max_score": float(d.scores.max()) if len(d.scores) else 0.0}
                          for d in dets}}
    json.dump(stats, open(f"{a.out}/{name}.json", "w"), indent=1)
    print(name, f"{dt:.2f}s", f"coverage={covered.mean():.2f}")
