# iSAID Vehicle SAM3/SAM2 Study

> **Historical invalidation:** `SAM3 YOLO bbox`, `SAM3 GT bbox` ve SAM3
> text+bbox hybrid çıktıları gerçek belirli-instance bbox istemi yerine PCS
> visual-exemplar arayüzünü kullanmıştır. Bu satırlar bilimsel karşılaştırma
> için geçersizdir. Rapor yalnız deney geçmişi ve eski düzen referansı olarak
> korunur.

This folder contains the final iSAID vehicle segmentation comparison artifacts.

## Main Artifacts

- `isaid_vehicle_full_metric_document_colored.pdf`: invalidation uyarısı içeren tarihsel PDF.
- `isaid_vehicle_full_metric_document_colored.docx`: editable DOCX version of the same report.
- `isaid_vehicle_full_metric_document.md`: Markdown source generated from the metric tables.
- `tables/full_metric_document/`: CSV tables used by the full metric document.
- `figures/sample_cases/`: curated qualitative examples with GT and YOLO bbox overlays.
- `isaid_vehicle_sam3_sam2_summary.pptx`: older short presentation deck.
- `isaid_vehicle_sam3_sam2_summary.pdf`: PDF export of the older short deck.

## Evaluation Scope

- Dataset split: iSAID vehicle evaluation subset.
- Classes: `Small_Vehicle` and `Large_Vehicle` merged into one binary `vehicle` mask.
- Evaluation images: 128 positive `1024 x 1024` tiles.
- Groups: `Overall`, `No Overlap / Low Mask Area`, `No Overlap / High Mask Area`, `Overlap / Low Mask Area`, and `Overlap / High Mask Area`.
- Segment metrics are pixel-level binary mask metrics, averaged per image.
- Detector metrics are separate YOLO bounding-box metrics; detector IoU is BBox IoU, not mask IoU.

## Pipelines in the Full Report

- `SAM3 text only`
- `SAM3 YOLO bbox`
- `SAM3 GT bbox`
- `SAM3 hybrid YOLO bbox`
- `RemoteSAM text only`
- `RingMo-SAM GT bbox`
- `RingMo-SAM YOLO bbox`
- `SAM2 GT bbox`
- `SAM2 YOLO bbox`

Excluded from the final full report: `GroundingDINO + SAM2`, `SegEarth-OV3 + SAM3`, and `SAM3 hybrid GT bbox`.

## Main Takeaways

- `SAM2 GT bbox` is the strongest upper-bound prompt setting in this run.
- `SAM2 YOLO bbox` is the strongest practical detector-guided setting and outperforms `RemoteSAM text only` in the overall table.
- `SAM3 hybrid YOLO bbox` is not a guaranteed improvement over `SAM3 YOLO bbox`; the `vehicle` text prompt increases over-segmentation in this setup.
- The most useful next experiment is likely YOLO bbox crop + magnification before SAM2, rather than changing the whole segmentation model.

## Regeneration

From the repository root:

```bash
.venv/bin/python scripts/write_isaid_metric_document.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
.venv/bin/python scripts/export_isaid_metric_document_docx_pdf.py
```
