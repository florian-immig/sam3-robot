"""Try SAM3-I (instruction-following SAM 3) on one image with free-form instructions.

SAM3-I ships its own modified `sam3` package, so run this in a SEPARATE environment from the
Meta SAM 3 one (see README_sam3i.md). Adapted from SAM3-I's scripts/inference.py.

  python scripts/sam3i_segment.py --sam3i-root ~/SAM3-I --checkpoint ~/ckpt/stage3.pt \
      --image data/images/test0.png --category complex \
      --prompts "ground a four-legged robot can walk on" "something to avoid stepping on"
"""
import argparse, os, sys

import cv2
import numpy as np
import torch
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--sam3i-root", required=True, help="path of the cloned SAM3-I repo")
ap.add_argument("--checkpoint", required=True, help="SAM3-I stage-3 checkpoint (.pt)")
ap.add_argument("--image", required=True, help="path or name in data/images")
ap.add_argument("--prompts", nargs="+", required=True)
ap.add_argument("--category", choices=["concept", "simple", "complex"], default="complex")
ap.add_argument("--threshold", type=float, default=0.3)
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--out", default="outputs/sam3i")
a = ap.parse_args()

root = os.path.abspath(os.path.expanduser(a.sam3i_root))
sys.path.insert(0, os.path.join(root, "sam3"))      # their modified sam3 package, as in inference.py
sys.path.insert(0, os.path.join(root, "scripts"))
import inference as si                                # their helpers (module-level code only defines things)
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api as collate
from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata

cands = [a.image, f"data/images/{a.image}"] + [f"data/images/{a.image}{e}" for e in (".jpg", ".jpeg", ".png")]
path = next((c for c in cands if os.path.isfile(c)), None)
if path is None:
    raise SystemExit(f"image not found: {a.image}")
os.makedirs(a.out, exist_ok=True)
name = os.path.splitext(os.path.basename(path))[0]

stage = si.CATEGORY_TO_STAGE[a.category]
si.setup_torch()
bpe = si.DEFAULT_BPE if os.path.exists(si.DEFAULT_BPE) else None
model = build_sam3_image_model(
    bpe_path=bpe, checkpoint_path=os.path.expanduser(a.checkpoint), eval_mode=True,
    enable_segmentation=True, device="cuda", load_from_HF=False,
    inst_stage=stage, adapter_config=si.DEFAULT_ADAPTER).to("cuda").eval()
transform = si.build_transform()
post = PostProcessImage(max_dets_per_img=-1, iou_type="segm", use_original_sizes_box=True,
                        use_original_sizes_mask=True, convert_mask_to_rle=False,
                        detection_threshold=a.threshold, to_cpu=False)


def make_prompt(text):
    """Prompt format used by inference.py: plain string for concept, dict for simple/complex."""
    if a.category == "concept":
        return text
    d = {"concept": [], "simple_query": [], "complex_query": []}
    d["simple_query" if a.category == "simple" else "complex_query"] = [text, text]
    return d


pil = Image.open(path).convert("RGB")
rgb = np.array(pil)
h, w = rgb.shape[:2]
results = []
with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
    for q in a.prompts:
        dp = Datapoint(find_queries=[], images=[SAMImage(data=pil, objects=[], size=[h, w])])
        dp.find_queries.append(FindQueryLoaded(
            query_text=make_prompt(q), image_id=0, object_ids_output=[], is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(coco_image_id=0, original_image_id=0,
                                                 original_category_id=1, original_size=[h, w],
                                                 object_id=0, frame_index=0)))
        dp = transform(dp)
        batch = collate([dp], dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
        out = model(batch, stage)
        proc = post.process_results(out, batch.find_metadatas)
        det = proc[next(iter(proc))] if len(proc) else {"masks": [], "scores": []}
        if len(det["masks"]):
            m = det["masks"].cpu().numpy() if torch.is_tensor(det["masks"]) else np.asarray(det["masks"])
            m = (m > 0.5) if m.dtype != np.bool_ else m
            m = m.reshape(-1, h, w)
            s = det["scores"].float().cpu().numpy().reshape(-1)
        else:
            m, s = np.zeros((0, h, w), bool), np.zeros(0, np.float32)
        results.append((q, m, s))
        print(f"{q[:60]:60s} n={len(s)} max={s.max() if len(s) else 0:.2f} "
              f"area={(m.any(0).mean() * 100 if len(s) else 0):.1f}%")

palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180)]
best = np.zeros((h, w), np.float32)
lab = np.full((h, w), -1, np.int16)
for i, (_, m, s) in enumerate(results):
    for inst, sv in zip(m, s):
        upd = inst & (sv > best)
        best[upd] = sv
        lab[upd] = i
out = rgb.astype(np.float32)
for i in range(len(results)):
    mk = lab == i
    out[mk] = (1 - a.alpha) * out[mk] + a.alpha * np.array(palette[i % len(palette)])
out = np.ascontiguousarray(out.astype(np.uint8))
for i, (q, m, s) in enumerate(results):
    y = 8 + 24 * i
    cv2.rectangle(out, (8, y), (26, y + 16), palette[i % len(palette)], -1)
    txt = f"{q[:50]} ({(lab == i).mean() * 100:.0f}%)" if len(s) else f"{q[:50]} (none)"
    cv2.putText(out, txt, (32, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, txt, (32, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
Image.fromarray(out).save(f"{a.out}/{name}_{a.category}_overlay.png")
print(f"saved {a.out}/{name}_{a.category}_overlay.png")
