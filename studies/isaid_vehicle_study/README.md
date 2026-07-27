# iSAID Vehicle Study

## Durum

`historical_context_only`

Bu çalışma iSAID üzerindeki `Small_Vehicle` ve `Large_Vehicle` sınıflarını
tek bir `vehicle` hedefinde birleştirerek YOLO bbox ve farklı segmentation
pipeline'larını karşılaştıran ilk deneydir.

## Neden Korunuyor?

- İlk YOLO + SAM2/SAM3, RemoteSAM ve RingMoSAM karşılaştırmalarını içerir.
- `Overall` ile `Overlap/No Overlap x Low/High Mask Area` rapor formatının
  kaynağıdır.
- Teacher-reference-bias araştırma sorusuna giden deney geçmişini gösterir.

Bu çalışma eşlenmiş iSAID/SAMRS bildirisi için doğrudan kanıt değildir.
Hedef sınıfı, split yapısı ve bazı eski evaluator kararları güncel frozen
protokolden farklıdır.

## Dizinler

| Dizin | İçerik |
|---|---|
| `configs/` | Eski iSAID vehicle deney configleri |
| `data/prepared/` | Hazırlanmış train/val/eval tile ve anotasyonları |
| `results/detector_training/` | Eğitilmiş YOLO run'ı ve ağırlıkları |
| `results/pipelines/` | Eski pipeline maskeleri, metrikleri ve görselleri |
| `reports/` | Markdown, DOCX, PDF ve tarihsel PPTX |
| `docs/WALKTHROUGH.md` | Deneyin ayrıntılı teknik açıklaması |

## Ana Rapor

```text
reports/isaid_vehicle_full_metric_document_colored.pdf
```

## Tarihsel Doğrulama

```bash
.venv/bin/python \
  studies/isaid_vehicle_study/scripts/validate_isaid_experiment_outputs.py
```

Bu komut yalnız bu tarihsel çalışmanın kendi çıktısını denetler; güncel
teacher-reference-bias çalışmasının finalizer'ı değildir.
