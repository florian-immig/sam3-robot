# SAM 3 concept-style terrain segmentation

Test whether SAM 3 text prompts ("water puddle", "mud", "person", ...) can give
semantic terrain labels + masks + tracks in one step, reducing the need for the VLM.

## Setup (needs an NVIDIA GPU)
1. Request access: https://huggingface.co/facebook/sam3, then `hf auth login`
2. `./setup.sh && source .venv/bin/activate && export PYTHONPATH=.`

## Run
- `python scripts/run_image.py --images "data/images/*.jpg"` -> overlay + C_sem heatmap + json stats
- `python scripts/prompt_sweep.py` -> sensitivity to phrasing
- `python scripts/run_video.py data/videos/clip.mp4` -> overlay mp4 + track lifetimes

Edit `sam3_terrain/concepts.py` to change the vocabulary and prior hazard costs.

## Experiments
1. Per-concept precision/recall on ~30 hand-labelled frames (water, mud, person, cup).
2. Prompt phrasing sweep, incl. abstract prompts ("unsafe to step on") — expect failures.
3. Negative prompts: concept absent from the scene -> false-positive rate vs threshold.
4. Coverage: how much of the ground is claimed by any concept (`coverage` in json).
5. Video: ID switches, lifetimes, occlusion recovery, split/merge on moving-camera clips.
6. Baseline: SAM 3 concepts vs SAM2 masks + VLM, on accuracy and latency.
7. Latency: per-frame time vs number of concepts (video cost scales with #concepts: one session each).
