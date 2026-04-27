# Codex Handoff: Semantic Drone YOLO26x + SAM3 Study

## Goal

Finish the drone-like overhead urban experiment on the virtual machine with 4x RTX A6000 GPUs.

The primary dataset is **Semantic Drone Dataset**, not iSAID. iSAID is kept only as a backup because it is too high-altitude for the requested visual setting.

The experiment must compare:

- `SAM3 text-only`
- `YOLO26x + SAM3`
- `GT bbox + SAM3`

and report results across the 2x2 strata:

- bbox overlap vs no bbox overlap
- high target-mask area vs low target-mask area

## What Is Already Implemented

Primary config:

- `configs/semantic_drone_car_yolo26x.yaml`

Dataset preparation:

- `scripts/prepare_semantic_drone_dataset.py`
- `src/sam3_bbox_study/data/semantic_drone.py`

Training:

- `scripts/train_yolo.py`
- Supports overrides such as `--epochs`, `--batch`, `--imgsz`, `--device`, `--workers`, `--name`.
- Main config is set for 4x A6000: `device: "0,1,2,3"`, `batch: 32`, `workers: 8`.

Evaluation and export:

- `scripts/evaluate_stratified_triplet.py`
- `scripts/export_stratified_presentation_assets.py`
- `scripts/estimate_yolo_training_time.py`

Backup iSAID pipeline:

- `configs/isaid_vehicle_yolo26x.yaml`
- `scripts/prepare_isaid_vehicle_dataset.py`
- `src/sam3_bbox_study/data/isaid.py`

Use it only if Semantic Drone is unusable.

## Expected Dataset Layout

Place Semantic Drone data under:

```text
data/semantic_drone_raw/
```

The converter auto-detects common layouts:

```text
data/semantic_drone_raw/aerial_semantic_drone/images/
data/semantic_drone_raw/aerial_semantic_drone/labels/png/
```

or:

```text
data/semantic_drone_raw/training_set/images/
data/semantic_drone_raw/training_set/gt/semantic/label_images/
```

or:

```text
data/semantic_drone_raw/training_set/gt/semantic/label_images_semantic/
```

The preferred masks are indexed/grayscale semantic masks. The default mapping uses `car = 17`.

If the downloaded labels are RGB color masks only, inspect the car RGB value and set this in `configs/semantic_drone_car_yolo26x.yaml`:

```yaml
dataset:
  target_rgb_colors:
    - [R, G, B]
```

Leave `target_category_ids` as `null` unless the local indexed label mapping differs. If class mapping files exist (`class_to_idx.json`, `idx_to_class.json`, `classes.csv`, or `class_dict.csv`), the converter tries to read them automatically.

## Environment Setup

Use Linux on the VM if possible. From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Check CUDA:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

SAM3 requires Hugging Face access:

```bash
export HF_TOKEN=...
python scripts/download_models.py --config configs/semantic_drone_car_yolo26x.yaml
```

## Full Execution Plan

1. Prepare Semantic Drone car dataset:

```bash
python scripts/prepare_semantic_drone_dataset.py --config configs/semantic_drone_car_yolo26x.yaml
```

Expected outputs:

```text
data/semantic_drone_car/train/images/
data/semantic_drone_car/train/labels/
data/semantic_drone_car/train/_annotations.coco.json
data/semantic_drone_car/train/metadata.csv
data/semantic_drone_car/val/...
data/semantic_drone_car/eval/...
data/semantic_drone_car/data.yaml
```

The `eval` split is balanced across the 4 strata. If it errors about missing strata, lower one or more of these in the config:

```yaml
evaluation:
  min_objects_per_image: 2
  max_objects_per_image: 120
  max_per_stratum: 20
```

2. Estimate training time:

```bash
python scripts/estimate_yolo_training_time.py --config configs/semantic_drone_car_yolo26x.yaml
```

Current fallback estimate for 4x A6000:

- about `3840` train tiles
- about `638.9` sec/epoch
- about `17.7` hours for `100` epochs

Once the prepared dataset exists, rerun the estimator so it uses the actual train tile count.

3. Run DDP smoke training:

```bash
python scripts/train_yolo.py --config configs/semantic_drone_car_yolo26x.yaml --epochs 3 --name smoke_ddp
```

Verify:

```text
runs/yolo26x_semantic_drone_car_s1536_i1024_4xa6000/smoke_ddp/results.csv
```

If OOM occurs, reduce batch:

```bash
python scripts/train_yolo.py --config configs/semantic_drone_car_yolo26x.yaml --epochs 3 --batch 24 --name smoke_ddp_b24
```

Then try `--batch 16` if needed.

4. Run full YOLO26x training:

```bash
python scripts/train_yolo.py --config configs/semantic_drone_car_yolo26x.yaml
```

Expected final detector:

```text
runs/yolo26x_semantic_drone_car_s1536_i1024_4xa6000/train/weights/best.pt
```

If the full run was done with a different `--name`, update `yolo.trained_weights` in `configs/semantic_drone_car_yolo26x.yaml`.

5. Run the three SAM3 pipelines on the balanced eval split:

```bash
python scripts/run_sam3_text.py --config configs/semantic_drone_car_yolo26x.yaml
python scripts/run_yolo_sam3.py --config configs/semantic_drone_car_yolo26x.yaml
python scripts/run_gt_box_sam3.py --config configs/semantic_drone_car_yolo26x.yaml
```

Expected outputs:

```text
results/semantic_drone_car_sam3_text/masks/
results/semantic_drone_car_yolo26x_sam3/masks/
results/semantic_drone_car_gt_box_sam3/masks/
```

6. Evaluate stratified metrics:

```bash
python scripts/evaluate_stratified_triplet.py --config configs/semantic_drone_car_yolo26x.yaml
```

Expected tables:

```text
results/semantic_drone_car_metrics_sam3_triplet/per_image_stratified_metrics.csv
results/semantic_drone_car_metrics_sam3_triplet/summary_overall_stratified.csv
results/semantic_drone_car_metrics_sam3_triplet/summary_by_stratum.csv
results/semantic_drone_car_metrics_sam3_triplet/pairwise_iou_by_stratum.csv
```

7. Export presentation assets:

```bash
python scripts/export_stratified_presentation_assets.py --config configs/semantic_drone_car_yolo26x.yaml
```

Expected output folder:

```text
../presentation_semantic_drone_bbox_study/
```

## Final Report Requirements

When writing the final summary, include:

- dataset: Semantic Drone Dataset, class `car`, semantic masks converted to connected-component boxes
- model: `YOLO26x`, `imgsz=1024`, `batch=32`, `device=0,1,2,3`
- detector metrics from YOLO `results.csv`: best mAP50, mAP50-95, precision, recall
- overall SAM3 triplet metrics: IoU, Dice, precision, recall, predicted/GT area ratio
- 2x2 stratified metrics:
  - `overlap__high_mask_area`
  - `overlap__low_mask_area`
  - `no_overlap__high_mask_area`
  - `no_overlap__low_mask_area`
- qualitative examples from each stratum

## Important Interpretation Caveat

Semantic Drone labels are semantic segmentation, not native instance segmentation. The implementation derives GT boxes from connected components of the `car` mask. This is acceptable for this experiment, but mention it explicitly. If adjacent cars touch in the semantic mask, they may become a single connected component.

## Do Not Do

- Do not switch back to Cityscapes; it is ground-level.
- Do not use iSAID as the primary dataset; it is visually too high-altitude for the requested drone-like city view.
- Do not report only a single mean IoU. The professor specifically wants the 2x2 strata.
- Do not overwrite unrelated local changes in `scripts/export_presentation_assets.py`; it had pre-existing edits.
