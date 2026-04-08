# SAM3 Text vs Box-Guided SAM3 on LandCover.ai Buildings

This repository contains a single focused study on top-down aerial imagery:

1. `SAM3 text-only`
2. `YOLO + SAM3`
3. `GT bbox + SAM3`

The goal is to test how much explicit geometry helps when SAM3 already has text-conditioned localization ability.

## Final Finding

On the LandCover.ai `building` validation split, evaluated with positive-only mask IoU:

| Pipeline | Mean IoU |
| --- | ---: |
| `SAM3 text-only` | `0.7034` |
| `YOLO + SAM3` | `0.7123` |
| `GT bbox + SAM3` | `0.7920` |

Interpretation:

- `SAM3 text-only` is already strong.
- `YOLO + SAM3` gives a small but real improvement.
- `GT bbox + SAM3` is much better, which shows that **box quality still matters**.

## Why This Dataset

The repo uses `LandCover.ai v1` and keeps only the `building` class.

- official and directly downloadable
- true aerial / orthophoto imagery
- pixel-level masks available
- easier and cleaner than tiny-object aerial categories for a controlled prompt study

Because LandCover.ai is semantic segmentation, YOLO detection labels are derived from connected components of the binary building masks.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill:

- `HF_TOKEN` for local `facebook/sam3`
- `ROBOFLOW_API_KEY` only if you intentionally switch to hosted SAM3

## Reproducible Workflow

### 1. Download LandCover.ai

```powershell
python scripts/download_dataset.py
```

### 2. Prepare the single-class dataset

```powershell
python scripts/prepare_landcover_dataset.py
```

This step:

- tiles the orthophotos into `512x512`
- keeps only `building`
- creates YOLO labels from connected components
- balances `train` negatives with `train_negative_ratio`
- writes COCO masks for evaluation

### 3. Download local SAM3 weights

```powershell
python scripts/download_models.py --download-sam3-local
```

### 4. Train the detector

```powershell
python scripts/train_yolo.py
```

### 5. Run the three pipelines

```powershell
python scripts/run_sam3_text.py
python scripts/run_yolo_sam3.py
python scripts/run_gt_box_sam3.py
```

### 6. Evaluate the triplet

```powershell
python scripts/evaluate_triplet.py
```

### 7. Export slide material

```powershell
python scripts/export_presentation_assets.py
```

By default this creates a sibling folder:

```text
../presentation_sam3_bbox_study/
```

with slide-ready tables, charts, copied qualitative examples, and speaking notes.

## Repository Layout

```text
project_yolo-sam/
├── configs/
│   └── experiment.yaml
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_landcover_dataset.py
│   ├── download_models.py
│   ├── train_yolo.py
│   ├── run_sam3_text.py
│   ├── run_yolo_sam3.py
│   ├── run_gt_box_sam3.py
│   ├── evaluate_triplet.py
│   └── export_presentation_assets.py
├── src/
│   └── pool_segmentation_compare/
├── data/
├── models/
├── runs/
└── results/
```

## Important Outputs

- detector weights:
  - `runs/yolo_building_s640/train/weights/best.pt`
- final triplet metrics:
  - `results/landcover_metrics_sam3_triplet/summary_sam3_triplet.csv`
  - `results/landcover_metrics_sam3_triplet/per_image_iou_sam3_triplet.csv`
- qualitative overlays:
  - `results/landcover_visualizations_sam3_triplet/`

## Notes

- Evaluation is configured as `positive_only: true`, because the main research question is mask quality on images that actually contain buildings.
- `GT bbox + SAM3` is an upper-bound style experiment: it isolates the value of accurate geometry by removing detector error.
- The repo intentionally centers the final SAM3-only study; older SAM2 / pool experiments are not part of the present workflow.
