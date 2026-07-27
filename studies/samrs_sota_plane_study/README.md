# SAMRS SOTA Plane Study

## Durum

`invalid_for_paper_evidence`

Bu çalışma SAMRS SOTA-RBB içindeki `plane` sınıfında YOLO ve SAM1/SAM2/SAM3
pipeline'larını karşılaştıran tarihsel deneydir.

## Neden Korunuyor?

- SAMRS pseudo-maskelerinin SAM1 ile çok yüksek uyumunu ilk kez görünür kıldı.
- iSAID ve SAMRS sonuçları arasındaki büyük fark, teacher-reference affinity
  hipotezinin kurulmasına yol açtı.
- `Overall` ile `Overlap/No Overlap x Low/High Mask Area` tablolarını içerir.

Eski çalışma; iSAID ile aynı split, bbox, sınıf ve örnekleme protokolünü
kullanmadığı için yeni bildiride nicel kanıt olarak kullanılmaz. Geçerli kanıt
`teacher_reference_bias_v1` çalışmasındaki eşlenmiş deneydir.

## Dizinler

| Dizin | İçerik |
|---|---|
| `configs/` | Eski SAMRS SOTA plane configi |
| `data/prepared/` | Hazırlanmış train/val/eval veri |
| `results/detector_training/` | Eğitilmiş YOLO run'ı ve ağırlıkları |
| `results/pipelines/` | Eski pipeline maskeleri, metrikleri ve görselleri |
| `reports/` | Markdown, DOCX, PDF ve QA manifestleri |

## Ana Rapor

```text
reports/samrs_sota_plane_full_metric_document_colored.pdf
```

## Tarihsel Doğrulama

```bash
.venv/bin/python \
  studies/samrs_sota_plane_study/scripts/validate_samrs_sota_plane_experiment_outputs.py
```

Bu komut yalnız eski study çıktısını kontrol eder. Güncel eşlenmiş SAMRS
kanıtı `studies/teacher_reference_bias_v1/` altındadır.
