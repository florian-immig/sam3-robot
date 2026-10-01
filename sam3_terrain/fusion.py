"""Turn concept detections into per-pixel semantic maps / a hazard costmap C_sem."""
import numpy as np

from .segmenter import Detections


def semantic_cost(dets: list[Detections], shape, unknown_cost=0.5):
    """C_sem in [0,1] per pixel.

    Hazard/obstacle concepts take the max of (hazard * score) over instances, so
    anything dangerous wins. Ground concepts only lower the cost where no hazard
    fired. Pixels no concept claims get `unknown_cost` (conservative default).
    Also returns a coverage map (fraction of pixels explained by any concept).
    """
    h, w = shape
    hazard = np.zeros((h, w), np.float32)
    ground = np.full((h, w), np.inf, np.float32)
    covered = np.zeros((h, w), bool)
    for d in dets:
        for m, s in zip(d.masks, d.scores):
            covered |= m
            if d.concept.group == "ground":
                ground[m] = np.minimum(ground[m], d.concept.hazard)
            else:
                hazard[m] = np.maximum(hazard[m], d.concept.hazard * float(s))
    cost = np.where(np.isfinite(ground), ground, unknown_cost).astype(np.float32)
    cost = np.maximum(cost, hazard)
    return cost, covered


def label_map(dets: list[Detections], shape):
    """Per-pixel index of the highest-score concept (-1 = none) and its score."""
    h, w = shape
    best = np.zeros((h, w), np.float32)
    idx = np.full((h, w), -1, np.int16)
    for k, d in enumerate(dets):
        for m, s in zip(d.masks, d.scores):
            upd = m & (s > best)
            best[upd] = s
            idx[upd] = k
    return idx, best
