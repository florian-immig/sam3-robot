"""Concept-style SAM 3 on video -> overlay mp4 + track stats (ID switches, lifetimes)."""
import argparse, collections, json, os

import cv2
import numpy as np

from sam3_terrain.fusion import label_map, semantic_cost
from sam3_terrain.segmenter import segment_video
from sam3_terrain.viz import cost_heatmap, overlay_labels

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--out", default="outputs/video")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

cap = cv2.VideoCapture(a.video)
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
fps = cap.get(cv2.CAP_PROP_FPS) or 15

per_frame = segment_video(a.video)
h, w = frames[0].shape[:2]
vw = cv2.VideoWriter(f"{a.out}/overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
lifetimes = collections.defaultdict(lambda: collections.Counter())  # concept -> id -> frames
for i, rgb in enumerate(frames):
    dets = per_frame.get(i, [])
    for d in dets:
        for tid in (d.ids if d.ids is not None else []):
            lifetimes[d.concept.prompt][int(tid)] += 1
    cost, _ = semantic_cost(dets, (h, w))
    idx, _ = label_map(dets, (h, w))
    vw.write(cv2.cvtColor(np.hstack([overlay_labels(rgb, dets, idx), cost_heatmap(cost)]),
                          cv2.COLOR_RGB2BGR))
vw.release()
json.dump({c: dict(v) for c, v in lifetimes.items()},
          open(f"{a.out}/track_lifetimes.json", "w"), indent=1)
print("frames:", len(frames), "-> ", a.out)
