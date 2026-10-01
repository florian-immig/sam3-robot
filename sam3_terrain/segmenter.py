"""Thin wrapper around SAM 3 for concept-style terrain segmentation.

Image path: the image is encoded once, then every concept is a cheap text
prompt on the cached features. Video path: SAM 3 allows one text prompt per
session, so we run one session per concept and merge the per-frame outputs.
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .concepts import CONCEPTS, Concept


@dataclass
class Detections:
    """All instances found for one concept in one frame."""
    concept: Concept
    masks: np.ndarray            # [N, H, W] bool
    scores: np.ndarray           # [N] float
    boxes: np.ndarray            # [N, 4] xyxy px (empty for video output)
    ids: np.ndarray = field(default=None)  # [N] persistent track ids (video only)


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


class ConceptSegmenter:
    def __init__(self, concepts=CONCEPTS, threshold=0.5, device="cuda"):
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self.concepts = list(concepts)
        self.processor = Sam3Processor(
            build_sam3_image_model(), device=device, confidence_threshold=threshold
        )

    def segment_image(self, image: Image.Image) -> list[Detections]:
        state = self.processor.set_image(image)
        w, h = image.size
        out = []
        for c in self.concepts:
            self.processor.reset_all_prompts(state)
            res = self.processor.set_text_prompt(state=state, prompt=c.prompt)
            masks = _np(res["masks"]).astype(bool)
            masks = masks.reshape(-1, h, w) if masks.size else np.zeros((0, h, w), bool)
            out.append(Detections(c, masks, _np(res["scores"]).reshape(-1),
                                  _np(res["boxes"]).reshape(-1, 4)))
        return out


def segment_video(video_path: str, concepts=CONCEPTS):
    """Returns {frame_idx: [Detections, ...]} using one SAM 3 session per concept."""
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor()
    per_frame: dict[int, list[Detections]] = {}
    sid = predictor.handle_request(
        dict(type="start_session", resource_path=video_path))["session_id"]
    for c in concepts:
        predictor.handle_request(dict(type="reset_session", session_id=sid))
        predictor.handle_request(dict(
            type="add_prompt", session_id=sid, frame_index=0, text=c.prompt))
        for r in predictor.handle_stream_request(
                dict(type="propagate_in_video", session_id=sid)):
            o = r["outputs"]
            masks = _np(o["out_binary_masks"]).astype(bool)
            h_w = masks.shape[-2:] if masks.size else (0, 0)
            per_frame.setdefault(r["frame_index"], []).append(Detections(
                c, masks.reshape(-1, *h_w), _np(o["out_probs"]).reshape(-1),
                np.zeros((0, 4)), _np(o["out_obj_ids"]).reshape(-1)))
    predictor.handle_request(dict(type="close_session", session_id=sid))
    return per_frame
