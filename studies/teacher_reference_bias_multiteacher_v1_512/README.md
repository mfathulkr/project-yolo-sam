# Multi-Teacher Reference Bias Study

Bu çalışma, aynı iSAID görüntüleri ve aynı SAM1/SAM2/SAM3 tahminleri sabit tutulurken değerlendirme referansının insan maskesinden SAM1, SAM2 veya SAM3 pseudo maskesine çevrilmesinin sonuçları nasıl değiştirdiğini ölçer.

## Kapsam

- `iSAID Plane`: 512 görüntü, 44 kaynak sahne, 5.447 instance.
- `iSAID Small Vehicle`: 512 görüntü, 31 kaynak sahne, 12.051 instance.
- Her hedef için dört eşit tabaka: `no_overlap/overlap × low/high mask area`, tabaka başına 128 görüntü.
- Aday modeller: SAM1, SAM2 ve SAM3.
- İstemler: insan GT bbox ve seed 42 ile eğitilmiş YOLO bbox.
- Referanslar: insan maskesi ile SAM1, SAM2 ve SAM3'ün aynı insan GT bbox isteminden ürettiği pseudo maskeler.
- Ana metrik: instance-macro Avg IoU. Dice, precision, recall ve IoU eşik başarı oranları full-metric raporlarda ayrıca verilir.

## Ana Çıktılar

### Yeni full-metric raporlar

- [Plane / SAM2 pseudo](reports/full_metrics/isaid_plane_pseudo_sam2/isaid_plane_pseudo_sam2_full_metric_document_colored.pdf)
- [Plane / SAM3 pseudo](reports/full_metrics/isaid_plane_pseudo_sam3/isaid_plane_pseudo_sam3_full_metric_document_colored.pdf)
- [Small Vehicle / SAM2 pseudo](reports/full_metrics/isaid_small_vehicle_pseudo_sam2/isaid_small_vehicle_pseudo_sam2_full_metric_document_colored.pdf)
- [Small Vehicle / SAM3 pseudo](reports/full_metrics/isaid_small_vehicle_pseudo_sam3/isaid_small_vehicle_pseudo_sam3_full_metric_document_colored.pdf)

### Öğretmen karşılaştırması

- [SAM1/SAM2/SAM3 karşılaştırma raporu](reports/teacher_comparison/sam_teacher_pseudo_reference_comparison_colored.pdf)
- [Aynı raporun Markdown kaynağı](reports/teacher_comparison/sam_teacher_pseudo_reference_comparison.md)

### Bildiri paketi

- [Literatür incelemesi](docs/LITERATURE_REVIEW.md)
- [Bildiri yapısı ve yazım planı](docs/PAPER_STRUCTURE.md)
- [Tablo ve figür kullanım planı](docs/PAPER_ASSET_PLAN.md)
- [QA raporu](docs/QA_REPORT.md)
- [Overleaf ana dosyası](paper/overleaf/main.tex)
- [BibTeX kaynakçası](paper/overleaf/ref.bib)
- [Yayın tabloları](paper/assets/tables)
- [Yayın figürleri](paper/assets/figures)

İnsan ve SAM1 pseudo referanslı önceki canonical full-metric raporlar, sırasıyla `teacher_reference_bias_v2_512` ve `teacher_reference_bias_small_vehicle_v1_512` çalışmalarında korunur. Bu uzantı yalnız SAM2/SAM3 referanslarını ve dört referanslı ortak analizi ekler.

## Tekrar Üretim

```bash
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/build_pseudo_references.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/evaluate_pseudo_references.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/compile_analysis.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/generate_figures.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/write_full_metric_reports.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/write_teacher_comparison_report.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/generate_paper_assets.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/validate_study.py
```

Ortak test paketi repository'nin `src` layout'u nedeniyle
`PYTHONPATH=src .venv/bin/python -m pytest tests studies/teacher_reference_bias_multiteacher_v1_512/tests -q`
komutuyla çalıştırılır.

Ham tahminler ve büyük instance küpü `.gitignore` altında tutulur. Raporlar,
yöntem kodu, doğrulanmış küçük özet tablolar, nitel görseller ve yayın paketi
Git ile izlenir. Instance düzeyinde sıfırdan yeniden hesaplama için iki
canonical çalışmanın frozen prediction dosyaları yerel makinede bulunmalıdır.
