import cv2
import numpy as np


def overlay_labels(image_rgb, dets, idx):
    rng = np.random.RandomState(3)
    colors = rng.randint(40, 255, (len(dets), 3))
    out = image_rgb.copy().astype(np.float32)
    for k, d in enumerate(dets):
        m = idx == k
        if m.any():
            out[m] = 0.45 * out[m] + 0.55 * colors[k]
            ys, xs = np.nonzero(m)
            cv2.putText(out, d.concept.prompt, (int(xs.mean()), int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out.astype(np.uint8)


def cost_heatmap(cost):
    return cv2.cvtColor(cv2.applyColorMap((cost * 255).astype(np.uint8),
                                          cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
