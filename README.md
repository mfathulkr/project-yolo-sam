# Remote Sensing YOLO-SAM Studies

Bu depo, detection ile promptlanan SAM tabanlı uzaktan algılama
segmentasyon çalışmalarını içerir. Her deney kendi config, hazırlanmış veri,
script, sonuç, rapor ve testleriyle `studies/` altında bağımsız tutulur.
Kökte yalnız birden fazla çalışmanın gerçekten paylaştığı kod ve kaynaklar
bulunur.

## Çalışmalar

| Study | Durum | Amaç |
|---|---|---|
| [`teacher_reference_bias_v2_512`](studies/teacher_reference_bias_v2_512/README.md) | Tamamlandı, canonical | Sabit seed 42 detector ile dört alt grupta 128'er, Overall'da 512 plane görüntüsünü iSAID insan/SAM1-pseudo ve SAMRS SOTA SAM1-pseudo referanslarında ölçer. |
| [`teacher_reference_bias_small_vehicle_v1_512`](studies/teacher_reference_bias_small_vehicle_v1_512/README.md) | Devam ediyor, canonical eşlenmiş protokol | Plane deneyinin yalnız hedef sınıfı small-vehicle olacak şekilde birebir eşlenmiş tekrarıdır; veri, model, seed ve rapor sözleşmesi aynıdır. |
| [`teacher_reference_bias_v1`](studies/teacher_reference_bias_v1/README.md) | Tarihsel | İlk 4×32 teacher-reference-bias deneyini ve altı sayfalık taslağı değiştirilemez öncül olarak korur. |
| [`isaid_vehicle_study`](studies/isaid_vehicle_study/README.md) | Tarihsel | iSAID small/large vehicle birleşik maskelerinde eski pipeline karşılaştırmasını korur. |
| [`samrs_sota_plane_study`](studies/samrs_sota_plane_study/README.md) | Tarihsel | İlk SAMRS SOTA plane deneyini ve eski sunum çıktısını korur. |
| [`semantic_drone_car_study`](studies/semantic_drone_car_study/README.md) | Planlandı | Semantic Drone car deneyi için config ve handoff içeriğini tutar. |
| [`landcover_building_study`](studies/landcover_building_study/README.md) | Eksik legacy | Eski landcover.ai building hazırlığını tarihsel olarak tutar. |

Tarihsel iSAID ve SAMRS çalışmaları silinmemiştir. Ancak eşlenmemiş
protokolleri nedeniyle teacher-reference-bias bildirisine kanıt olarak
karıştırılmazlar.

## Tarihsel V1 Sonuçları

İlk teacher-reference-bias çalışmasının ana bulgusu, aynı tahminlerin bağımsız
iSAID insan maskesi yerine SAM1 üretimli pseudo maskeye karşı ölçülmesinin
skorları belirgin biçimde yükseltmesidir:

| Model | İnsan IoU | SAM1 pseudo IoU | IoU enflasyonu |
|---|---:|---:|---:|
| SAM1 | 0,648 | 0,998 | +0,350 |
| SAM2 | 0,580 | 0,806 | +0,225 |
| SAM3 | 0,540 | 0,723 | +0,184 |

Bu sonuç pseudo-maskelerin eğitim için değersiz olduğunu söylemez. Sonuç,
teacher üretimli maskelerin bağımsız test ground truth'u gibi
yorumlanmaması gerektiğini gösterir.

V1 tarihsel çıktıları:

- [Altı sayfalık bildiri](studies/teacher_reference_bias_v1/reports/paper/teacher_reference_bias_paper_6pages.pdf)
- [iSAID tam metrik PDF](studies/teacher_reference_bias_v1/reports/full_metrics/isaid_plane/isaid_plane_full_metric_document_colored.pdf)
- [SAMRS SOTA tam metrik PDF](studies/teacher_reference_bias_v1/reports/full_metrics/samrs_sota_plane/samrs_sota_plane_full_metric_document_colored.pdf)
- [Canonical analiz](studies/teacher_reference_bias_v1/results/analysis/)

## Canonical V2 Sonuçları

512 görüntülük canonical v2 çalışması tamamlandı. GT-bbox koşulundaki Overall
instance IoU:

| Referans | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan | 0,653 | 0,629 | 0,655 |
| iSAID SAM1 pseudo | 1,000 | 0,827 | 0,795 |
| SAMRS SAM1 pseudo | 0,991 | 0,781 | 0,611 |

Aynı iSAID tahminlerinde yalnız referans insan maskesinden SAM1 pseudo
maskesine değiştirildiğinde IoU artışı `+0,347 / +0,198 / +0,140` oldu.
İnsan referansındaki `SAM3 > SAM1 > SAM2` sırası pseudo referansta
`SAM1 > SAM2 > SAM3` olarak değişti. SAM1 pseudo satırındaki `1,000`,
kontrollü kimlik testidir; bağımsız segmentasyon başarısı değildir.

Final raporlar:

- [iSAID insan referansı](studies/teacher_reference_bias_v2_512/reports/full_metrics/isaid_plane_human/isaid_plane_human_full_metric_document_colored.pdf)
- [iSAID SAM1 pseudo referansı](studies/teacher_reference_bias_v2_512/reports/full_metrics/isaid_plane_pseudo_sam1/isaid_plane_pseudo_sam1_full_metric_document_colored.pdf)
- [SAMRS SOTA plane](studies/teacher_reference_bias_v2_512/reports/full_metrics/samrs_sota_plane/samrs_sota_plane_full_metric_document_colored.pdf)

## Dizin Yapısı

```text
.
├── datasets/                 # Paylaşılan değiştirilemez ham veri
├── docs/                     # Depo mimarisi, refactor ve worklog
├── external_models/          # Harici model kaynak kodları
├── models/                   # Paylaşılan checkpoint ve başlangıç ağırlıkları
├── src/yolo_sam/             # Çalışmadan bağımsız ortak kütüphane
├── studies/                  # Her deney için bağımsız çalışma alanı
├── tests/                    # Ortak kod testleri
└── tools/                    # Ortak bakım ve pipeline araçları
```

Sahiplik kuralı:

> Bir dosya yalnız bir deneyde kullanılıyorsa ilgili study altında; birden
> fazla deneyde aynı davranışla kullanılıyorsa ortak kök katmanda bulunur.

Ayrıntılı sözleşme:
[docs/REPOSITORY_ARCHITECTURE.md](docs/REPOSITORY_ARCHITECTURE.md)

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python tools/models/setup_external_models.py
```

Harici/gated SAM checkpoint dosyaları Git'e eklenmez. Canonical v2
ve small-vehicle çalışmalarında eğitilmiş YOLO ağırlıkları ile yeniden üretim
bundle'ları Git LFS ile tutulur. Yerel inference kurulumu:
[LOCAL_INFERENCE.md](studies/teacher_reference_bias_v2_512/docs/LOCAL_INFERENCE.md).

## Testler

Ortak kütüphane:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

Canonical teacher-reference-bias v2 study:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
  .venv/bin/python -m unittest discover \
  -s studies/teacher_reference_bias_v2_512/tests -p 'test_*.py' -v
```

Small-vehicle eşlenmiş study:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_small_vehicle_v1_512/src \
  .venv/bin/python -m unittest discover \
  -s studies/teacher_reference_bias_small_vehicle_v1_512/tests \
  -p 'test_*.py' -v
```

## Canonical Study Kullanımı

Ana CLI ve bütün tekrar üretim komutları study README’sindedir:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/study.py --help
```

Tam metrik belgeleri:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/write_full_metric_reports.py
```

Final bütünlük kontrolü:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/validate_full_metric_reports.py
```

## Dokümantasyon

- [Repository Architecture](docs/REPOSITORY_ARCHITECTURE.md)
- [Refactor Plan](docs/REFACTOR_PLAN.md)
- [Canonical Deney Planı](studies/teacher_reference_bias_v2_512/docs/EXPERIMENT_PLAN.md)
- [Legacy Status](docs/LEGACY_STATUS.md)
- [Worklog](docs/WORKLOG.md)
- [Migration Manifests](docs/migration/)
