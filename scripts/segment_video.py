"""Segment a video with text prompts and save the segmentation overlaid on the video.

  python scripts/segment_video.py --video clip --prompts gravel grass tree --threshold 0.3
  python scripts/segment_video.py --video clip --prompts gravel grass tree --model sam3.1

Output: outputs/segment_video/<video>_<model>_overlay.mp4   colored masks + legend (+ track ids)
        outputs/segment_video/<video>_tracks.json   per prompt: track id -> number of frames seen
        outputs/segment_video/<video>_labels.npz    (only with --save-masks) per-frame label map
SAM 3 allows one text prompt per video session, so each prompt is propagated separately
and the results are merged; run time grows with the number of prompts.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))

import argparse, collections, json, os, time

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="file name in data/videos (extension optional) or a path")
ap.add_argument("--prompts", nargs="+", required=True)
ap.add_argument("--threshold", type=float, default=0.3, help="min per-instance score per frame")
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--prompt-frame", type=int, default=0, help="frame where the text prompt is added")
ap.add_argument("--no-ids", action="store_true", help="do not draw track ids")
ap.add_argument("--save-masks", action="store_true")
ap.add_argument("--seconds", type=float, default=0, help="only use this many seconds of video (0 = all)")
ap.add_argument("--start", type=float, default=0, help="start time in seconds (used with --seconds)")
ap.add_argument("--model", choices=["sam3", "sam3.1"], default="sam3",
                help="sam3 = facebook/sam3 (Nov 2025); sam3.1 = facebook/sam3.1 with Object Multiplex (Mar 2026)")
ap.add_argument("--max-objects", type=int, default=16,
                help="sam3.1 only: max tracked objects per prompt (16 = SAM 3.1 default)")
ap.add_argument("--fa3", action="store_true",
                help="sam3.1 only: use FlashAttention-3 (needs flash-attn-3 installed; off by default)")
ap.add_argument("--max-frames", type=int, default=0, help="use only the first N frames (0 = all)")
ap.add_argument("--stride", type=int, default=1, help="use every k-th frame (fps is divided accordingly)")
ap.add_argument("--interval-ms", type=float, default=0,
                help="keep one frame every N ms of video time, e.g. 100 (overrides --stride)")
ap.add_argument("--max-side", type=int, default=0, help="downscale so the longer side is at most this (0 = keep)")
ap.add_argument("--no-offload", action="store_true",
                help="keep frames and tracking state on the GPU (faster, but uses much more GPU memory)")
ap.add_argument("--out", default="outputs/segment_video")
a = ap.parse_args()

cands = [a.video, f"data/videos/{a.video}"] + [f"data/videos/{a.video}{e}" for e in (".mp4", ".mov", ".avi", ".mkv", ".MP4")]
path = next((c for c in cands if os.path.isfile(c)), None)
if path is None:
    raise SystemExit(f"video not found: {a.video}")
os.makedirs(a.out, exist_ok=True)
name = os.path.splitext(os.path.basename(path))[0]

# --- read frames (for the overlay only; the model reads the file itself) ---
cap = cv2.VideoCapture(path)
fps = cap.get(cv2.CAP_PROP_FPS) or 15
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
cap.release()
n_file = len(frames)                                        # frames in the file on disk
lo = int(round(a.start * fps))
hi = lo + int(round(a.seconds * fps)) if a.seconds else n_file
frames = frames[lo:hi]
if not frames:
    raise SystemExit(f"no frames left: clip has {n_file} frames ({n_file / fps:.1f}s), start={a.start}s")
n_orig = len(frames)
if a.interval_ms:
    step = a.interval_ms / 1000 * fps                       # source frames per kept frame
    idx = sorted({min(int(round(k * step)), n_orig - 1) for k in range(int(n_orig / step) + 1)})
    frames = [frames[i] for i in idx]
    fps = 1000 / a.interval_ms                              # play back in real time
else:
    frames = frames[::max(a.stride, 1)]
    fps = fps / max(a.stride, 1)
if a.max_frames:
    frames = frames[:a.max_frames]
if a.max_side and max(frames[0].shape[:2]) > a.max_side:
    k = a.max_side / max(frames[0].shape[:2])
    frames = [cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) for f in frames]
T, (H, W) = len(frames), frames[0].shape[:2]
print(f"{path}: using {T}/{n_file} frames, {W}x{H}, {fps:.1f} fps")
if T != n_file or (W, H) != (int(cv2.VideoCapture(path).get(3)), int(cv2.VideoCapture(path).get(4))):
    # the model reads a file, so write the trimmed/downscaled clip to a temp file
    path = f"{a.out}/_tmp_{name}.mp4"
    tw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in frames:
        tw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    tw.release()
    print(f"wrote temporary clip {path}")

# running per-pixel winner across prompts: label index + score (uint8 = score*255)
lab = np.full((T, H, W), -1, np.int8)
best = np.zeros((T, H, W), np.uint8)
ids_per_frame = [collections.defaultdict(list) for _ in range(T)]   # frame -> prompt idx -> [(id, cx, cy)]
lifetimes = {q: collections.Counter() for q in a.prompts}

t_start = time.time()
if a.model == "sam3.1":
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    # checkpoint=None -> downloads facebook/sam3.1 (needs HF access to that repo)
    predictor = build_sam3_multiplex_video_predictor(max_num_objects=a.max_objects, use_fa3=a.fa3)

    # Workaround for a bug in the SAM 3 repo: Sam3BasePredictor.start_session always passes
    # offload_state_to_cpu (and maybe video_loader_type) to model.init_state, but the 3.1
    # multiplex model's init_state does not accept them. Drop any kwargs it can't take.
    import functools, inspect
    _orig_init = predictor.model.init_state
    _ok = set(inspect.signature(_orig_init).parameters)

    @functools.wraps(_orig_init)
    def _init_state(*args, **kw):
        dropped = [k for k in kw if k not in _ok]
        if dropped:
            print(f"[sam3.1] init_state does not support {dropped}; ignoring")
        return _orig_init(*args, **{k: v for k, v in kw.items() if k in _ok})

    predictor.model.init_state = _init_state
else:
    predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])
print(f"model: {a.model}, loaded in {time.time() - t_start:.1f}s")
t_run = time.time()
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    sid = predictor.handle_request(dict(
        type="start_session", resource_path=path,
        offload_video_to_cpu=not a.no_offload,
        offload_state_to_cpu=not a.no_offload))["session_id"]
    for pi, q in enumerate(a.prompts):
        t0 = time.time()
        predictor.handle_request(dict(type="reset_session", session_id=sid))
        predictor.handle_request(dict(type="add_prompt", session_id=sid,
                                      frame_index=a.prompt_frame, text=q))
        n_obj = set()
        for r in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
            fi, o = r["frame_index"], r["outputs"]
            if fi >= T:
                continue
            masks = np.asarray(o["out_binary_masks"]).astype(bool)
            probs = np.asarray(o["out_probs"], dtype=np.float32).reshape(-1)
            oids = np.asarray(o["out_obj_ids"]).reshape(-1)
            for m, p, oid in zip(masks, probs, oids):
                if p < a.threshold or not m.any():
                    continue
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                sc = np.uint8(min(p, 1.0) * 255)
                upd = m & (sc > best[fi])
                best[fi][upd] = sc
                lab[fi][upd] = pi
                ys, xs = np.nonzero(m)
                ids_per_frame[fi][pi].append((int(oid), int(xs.mean()), int(ys.mean())))
                lifetimes[q][int(oid)] += 1
                n_obj.add(int(oid))
        print(f"{q:20s} tracks={len(n_obj)} time={time.time() - t0:.1f}s")
    predictor.handle_request(dict(type="close_session", session_id=sid))
t_infer = time.time() - t_run

# --- render ---
palette = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]
colors = np.array([palette[i % len(palette)] for i in range(len(a.prompts))])

fourcc_ok = False
for cc in ("avc1", "mp4v"):          # avc1 plays in browsers; mp4v is the fallback
    vw = cv2.VideoWriter(f"{a.out}/{name}_{a.model}_overlay.mp4", cv2.VideoWriter_fourcc(*cc), fps, (W, H))
    if vw.isOpened():
        fourcc_ok = True
        break
if not fourcc_ok:
    raise SystemExit("could not open a video writer")

for t in range(T):
    out = frames[t].astype(np.float32)
    for i in range(len(a.prompts)):
        mk = lab[t] == i
        if mk.any():
            out[mk] = (1 - a.alpha) * out[mk] + a.alpha * colors[i]
    out = np.ascontiguousarray(out.astype(np.uint8))
    if not a.no_ids:
        for pi, items in ids_per_frame[t].items():
            for oid, cx, cy in items:
                cv2.putText(out, str(oid), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                cv2.putText(out, str(oid), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    for row, q in enumerate(a.prompts):
        y = 8 + 24 * row
        pct = (lab[t] == row).mean() * 100
        cv2.rectangle(out, (8, y), (26, y + 16), tuple(int(c) for c in colors[row]), -1)
        txt = f"{q} ({pct:.0f}%)"
        cv2.putText(out, txt, (32, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(out, txt, (32, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    vw.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
vw.release()

json.dump({q: {str(k): v for k, v in c.items()} for q, c in lifetimes.items()},
          open(f"{a.out}/{name}_{a.model}_tracks.json", "w"), indent=1)
if a.save_masks:
    np.savez_compressed(f"{a.out}/{name}_{a.model}_labels.npz", prompts=np.array(a.prompts), labels=lab, scores=best)
print(f"inference time (all prompts): {t_infer:.1f}s for {T} frames ({t_infer / T:.2f}s/frame)")
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")
print(f"saved {a.out}/{name}_{a.model}_overlay.mp4, {name}_{a.model}_tracks.json" + (f", {name}_{a.model}_labels.npz" if a.save_masks else ""))
