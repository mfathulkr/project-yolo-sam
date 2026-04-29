# iSAID Vehicle YOLO/SAM Study

This repository now centers the final iSAID overhead-vehicle experiment: text/referring segmentation versus detector-guided segmentation for small urban vehicles in aerial imagery.

The active study compares:

- `SAM3 text-only`
- `RemoteSAM text`
- `SegEarth-OV3 + SAM3`
- `YOLO26x + SAM3`
- `GT bbox + SAM3`
- `YOLO26x + SAM2`
- `GroundingDINO + SAM2`
- `YOLO26x + RingMo-SAM`
- `GT bbox + RingMo-SAM`

The target class is one merged `vehicle` class from iSAID `Small_Vehicle` and `Large_Vehicle` instance annotations.

## Final Outputs

Main local artifacts:

```text
results/isaid_vehicle_final_report/REPORT.md
results/isaid_vehicle_final_report/QA_MANIFEST.md
results/isaid_vehicle_final_report/ARTIFACT_MANIFEST.csv
../presentation_isaid_vehicle_sam3_sam2_study/isaid_vehicle_sam3_sam2_summary.pptx
../presentation_isaid_vehicle_sam3_sam2_study/isaid_vehicle_sam3_sam2_summary.pdf
```

Long-form walkthrough:

```text
docs/ISAID_VEHICLE_STUDY_WALKTHROUGH.md
```

## Key Result

Overall mean metrics over the 128-image stratified eval split:

| Pipeline | IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| GT bbox + SAM3 | 0.5230 | 0.6510 | 0.5826 | 0.8072 |
| YOLO + SAM2 | 0.4336 | 0.5615 | 0.5494 | 0.6549 |
| YOLO + SAM3 | 0.4076 | 0.5328 | 0.5130 | 0.6485 |
| RemoteSAM text | 0.3850 | 0.5132 | 0.5244 | 0.5993 |
| SegEarth-OV3 + SAM3 | 0.2952 | 0.4027 | 0.3599 | 0.6449 |
| SAM3 text-only | 0.2739 | 0.3782 | 0.3589 | 0.5850 |
| GT bbox + RingMo-SAM | 0.2625 | 0.3592 | 0.7247 | 0.2815 |
| YOLO + RingMo-SAM | 0.2349 | 0.3266 | 0.6292 | 0.2630 |
| GroundingDINO + SAM2 | 0.0713 | 0.1196 | 0.1059 | 0.3734 |

Short interpretation:

- `RemoteSAM text` is the strongest box-free text/referring baseline.
- `YOLO + SAM2` is the best trained-detector pipeline in this run.
- `GT bbox + SAM3` is the best IoU/Dice upper-bound case.
- `RingMo-SAM` is precise but misses too many vehicles.
- `GroundingDINO + SAM2` is useful as a zero-shot text-to-box control, but weak for this tiny overhead vehicle task.

## Active Config

```text
configs/isaid_vehicle_yolo26x_cpu_eval.yaml
```

This config points to the final local outputs and sets model devices to CPU for inference.

## Important Local Folders

These are intentionally ignored by Git but valuable on this VM:

```text
data/isaid_raw/
data/isaid_raw_downloads/
data/isaid_vehicle/
models/sam3_hf/
models/remotesam_hf/
models/ringmo_sam_hf/
runs/yolo26x_isaid_vehicle_s1024/train/
results/isaid_vehicle_*/
../presentation_isaid_vehicle_sam3_sam2_study/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_external_models.py
```

`external_models/` is not committed. The setup script clones and patches the RemoteSAM and SegEarth-OV3 repos for this CPU-oriented evaluation workflow.

SAM3 may require `HF_TOKEN` for gated model access. Do not commit `.env`.

## Re-run Evaluation

Use CPU-only inference when the GPUs are busy:

```bash
export CUDA_VISIBLE_DEVICES=''
```

Main final workflow:

```bash
python scripts/run_sam3_text.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_remotesam_text.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_segearth_ov3.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_yolo_sam3.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_gt_box_sam3.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_yolo_sam2.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_grounded_sam2.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_ringmo_sam.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/evaluate_stratified_triplet.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/export_curated_qualitative_examples.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/write_isaid_experiment_report.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/export_isaid_presentation_pdf.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/validate_isaid_experiment_outputs.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
```

## Notes

- Final iSAID GT masks and boxes come from iSAID instance polygons, not Semantic Drone semantic connected components.
- The eval split is balanced into four strata: overlap/no-overlap x low/high target-mask area.
- The old LandCover and Semantic Drone material is not the active final study.
