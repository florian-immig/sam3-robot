# Trying SAM3-I (separate environment!)

SAM3-I bundles its own modified `sam3` package, so do NOT install it into the `.venv` that
runs Meta's SAM 3. Use a second venv.

```
git clone https://github.com/debby-0527/SAM3-I.git ~/SAM3-I
cd ~/SAM3-I && bash run.sh install          # follow its README; use a NEW venv
pip install pycocotools tqdm opencv-python-headless
```
Download the **Stage 3** checkpoint (Google Drive link in the SAM3-I README, "pretrained
checkpoints"), e.g. with `pip install gdown`. SAM3-I starts from the SAM 3 base weights on
Hugging Face, so `hf auth login` is needed here too.

Run (from this repo, in the SAM3-I venv):
```
python scripts/sam3i_segment.py --sam3i-root ~/SAM3-I --checkpoint ~/ckpt/stage3.pt \
  --image test0 --category complex \
  --prompts "ground a four-legged robot can walk on" "something to avoid stepping on"
```
`--category` picks the instruction level: concept (noun phrase) / simple (attributes, relations)
/ complex (function, affordance). Compare all three on the same image.
