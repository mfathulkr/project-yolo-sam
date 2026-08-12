# Current Study Report

## Otoritatif Çalışma

```text
studies/teacher_reference_bias_paper/
```

Bu çalışma dört kanonik deneyi tek protokolde toplar:

- iSAID Plane: human + SAM1/SAM2/SAM3 pseudo referans.
- iSAID Small Vehicle: human + SAM1/SAM2/SAM3 pseudo referans.
- SAMRS Plane: published + reproduced SAM1 + SAM2/SAM3 pseudo referans.
- SAMRS Small Vehicle: published + reproduced SAM1 + SAM2/SAM3 pseudo
  referans.

Her deney 512 görüntü, 4×128 tabaka, frozen SAM1/SAM2/SAM3, GT bbox ve seed
42 YOLO bbox koşullarını kapsar.

## Ana Çıktılar

```text
studies/teacher_reference_bias_paper/
├── analysis/main_cross_analysis_colored.pdf
├── experiments/<id>/reports/full_metrics/<reference>/
├── experiments/<id>/reports/cross_analysis/
├── paper_writing/assets/
├── paper_writing/overleaf/
├── literature_review/
└── docs/QA_REPORT.md
```

Toplam 16 full-metric MD/DOCX/PDF, 4 deney çapraz analiz MD/DOCX/PDF ve bir
dört-deney ana analiz MD/DOCX/PDF vardır.

## Ana Bulgular

iSAID YOLO-bbox koşulunda aynı modelin kendi pseudo referansına geçişi Plane
için SAM1/SAM2/SAM3'te `+0,276 / +0,279 / +0,224`, Small Vehicle için
`+0,176 / +0,163 / +0,142` eşlenmiş instance IoU artışı oluşturdu. Altı %95
güven aralığının tamamı sıfırın üzerindedir.

SAMRS published/reproduced-SAM1 referans uyumu Plane'de `0,990633`, Small
Vehicle'da `0,998338`dir. SAMRS referansı insan ground truth olmadığı için bu
bulgu bağımsız doğruluk değil, SAM1 kökeni ve model-reference affinity kanıtı
olarak yorumlanır.

## Doğrulama

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
.venv/bin/pytest -q studies/teacher_reference_bias_paper/tests
```

Yöntem, tekrar üretim ve devir ayrıntıları:

- `studies/teacher_reference_bias_paper/docs/SCIENTIFIC_PROTOCOL.md`
- `studies/teacher_reference_bias_paper/docs/REPRODUCIBILITY.md`
- `studies/teacher_reference_bias_paper/docs/HANDOFF.md`
- `docs/summary/TEACHER_REFERENCE_BIAS_HANDOFF.md`
