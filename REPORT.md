# Experiment Summary

## Research Question

How much does explicit bounding-box guidance still help when SAM3 can already segment from a text prompt?

## Task

- Dataset: `LandCover.ai v1`
- Class: `building`
- Viewpoint: top-down aerial / orthophoto
- Metric: mask IoU on positive validation images

## Pipelines

1. `SAM3 text-only`
2. `YOLO + SAM3`
3. `GT bbox + SAM3`

## Main Result

| Pipeline | Mean IoU |
| --- | ---: |
| `SAM3 text-only` | `0.7034` |
| `YOLO + SAM3` | `0.7123` |
| `GT bbox + SAM3` | `0.7920` |

## Detector Quality

The stronger detector used in the final study was `YOLOv8s` at `640x640` for `100` epochs.

Best detector-side scores:

- `best mAP50 = 0.8647`
- `best mAP50-95 = 0.6725`
- `best recall = 0.8048`

This matters because the final gap is not well explained by a weak detector. `YOLO + SAM3` still trails `GT bbox + SAM3`, so the remaining room is largely about localization quality.

## Interpretation

- `SAM3 text-only` is strong enough to be a serious baseline.
- `YOLO + SAM3` improves mean IoU only slightly, so text-only SAM3 already solves much of the localization problem on this simple class.
- `GT bbox + SAM3` improves sharply, which directly supports the claim that **high-quality boxes are still valuable prompts for SAM3**.

## Practical Message

For regular, visually salient aerial objects such as buildings:

- text-only SAM3 can work well,
- detector-guided SAM3 can work slightly better,
- but accurate geometry remains the clearest path to higher segmentation quality.

## Reproducibility

Main entry points:

- `python scripts/train_yolo.py`
- `python scripts/run_sam3_text.py`
- `python scripts/run_yolo_sam3.py`
- `python scripts/run_gt_box_sam3.py`
- `python scripts/evaluate_triplet.py`
- `python scripts/export_presentation_assets.py`
