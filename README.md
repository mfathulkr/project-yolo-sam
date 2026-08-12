# Remote Sensing YOLO-SAM Studies

Bu depo, detection ile promptlanan SAM tabanlı uzaktan algılama
segmentasyon çalışmalarını içerir. Ortak YOLO/SAM kodu kökte, araştırma
sorusuna özgü config, veri türevi, sonuç ve raporlar `studies/` altındadır.

## Kanonik Bildiri Çalışması

Güncel ve otoritatif çalışma:

[`studies/teacher_reference_bias_paper/`](studies/teacher_reference_bias_paper/README.md)

Bu çalışma aynı protokol altında dört deneyi birleştirir:

| Deney | Veri seti | Hedef | Referanslar |
|---|---|---|---|
| `isaid_plane` | iSAID | Plane | Human, SAM1, SAM2, SAM3 pseudo |
| `isaid_small_vehicle` | iSAID | Small Vehicle | Human, SAM1, SAM2, SAM3 pseudo |
| `samrs_plane` | SAMRS SOTA | Plane | Published SAMRS, reproduced SAM1, SAM2, SAM3 pseudo |
| `samrs_small_vehicle` | SAMRS SOTA | Small Vehicle | Published SAMRS, reproduced SAM1, SAM2, SAM3 pseudo |

Her deney 512 test görüntüsü kullanır. Dört
`overlap/no-overlap × low/high mask area` tabakasının her birinde 128 görüntü
vardır. Frozen SAM1, SAM2 ve SAM3 tahminleri hem GT bbox hem seed 42 ile
eğitilmiş YOLO bbox istemleriyle değerlendirilir.

Çalışmanın ana sorusu, bir model tarafından üretilen pseudo maskelerin
bağımsız test referansı gibi kullanılmasının aynı modele ölçüm avantajı verip
vermediğidir. Ana kanıt iSAID'in insan anotasyonlarıdır. SAMRS yayımlanmış
maskeleri SAM1 ile üretildiği için bağımsız doğruluk kanıtı değil, referans
kökeni ve model-referans yakınlığı için destekleyici kontroldür.

## Ana Bulgular

iSAID YOLO-bbox koşulunda, model kendi pseudo referansıyla ölçüldüğünde aynı
instance üzerindeki ortalama IoU artışları şöyledir:

| Hedef | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| Plane | +0,276 | +0,279 | +0,224 |
| Small Vehicle | +0,176 | +0,163 | +0,142 |

Altı artışın kaynak-sahne kümeli %95 bootstrap güven aralığı da sıfırın
üzerindedir. GT-bbox diagonalındaki `1,0` değerleri başarı sonucu değil, aynı
maskenin kendisiyle karşılaştırıldığı kimlik kontrolleridir.

SAMRS yayımlanmış referansı ile aynı protokolle yeniden üretilen SAM1
referansının ortalama instance IoU'su Plane için `0,990633`, Small Vehicle için
`0,998338`dir. Bu sonuç yayımlanmış SAMRS referansının SAM1 kökeniyle
uyumludur; insan ground truth yerine geçmez.

Ana karşılaştırma:
[main_cross_analysis_colored.pdf](studies/teacher_reference_bias_paper/analysis/main_cross_analysis_colored.pdf)

## Raporlar

Her deneyde dört referans için ayrı, eski okunaklı biçimle üretilmiş tam metrik
MD/DOCX/PDF ve bir çapraz analiz MD/DOCX/PDF bulunur:

```text
studies/teacher_reference_bias_paper/experiments/<experiment_id>/reports/
├── full_metrics/<reference>/
└── cross_analysis/
```

Toplam 16 tam metrik PDF ve 4 deney çapraz analiz PDF'si vardır. Tam metrik
raporlar `Overall` ile dört overlap × mask-area tablosunu, gerçek YOLO bbox
metriklerini ve seçilen görüntüdeki bütün hedef instance'ları gösteren nitel
örnekleri içerir.

Bildiri yazım varlıkları:

- [Bildiri yapısı](studies/teacher_reference_bias_paper/paper_writing/PAPER_STRUCTURE.md)
- [Overleaf iskeleti](studies/teacher_reference_bias_paper/paper_writing/overleaf/main.tex)
- [Figür ve tablolar](studies/teacher_reference_bias_paper/paper_writing/assets/)
- [Literatür incelemesi](studies/teacher_reference_bias_paper/literature_review/LITERATURE_REVIEW.md)
- [Arama denetimi](studies/teacher_reference_bias_paper/literature_review/SEARCH_AUDIT.md)

## Diğer Çalışmalar

| Study | Durum | Amaç |
|---|---|---|
| [`teacher_reference_bias_v1`](studies/teacher_reference_bias_v1/README.md) | Tarihsel | İlk 4×32 öncül deney; güncel bildirinin kanonik kanıtı değildir. |
| [`isaid_vehicle_study`](studies/isaid_vehicle_study/README.md) | Tarihsel | iSAID birleşik small/large vehicle ve eski model/prompt karşılaştırmaları. |
| [`samrs_sota_plane_study`](studies/samrs_sota_plane_study/README.md) | Tarihsel | İlk SAMRS SOTA plane çalışması ve eski sunum raporu. |
| [`semantic_drone_car_study`](studies/semantic_drone_car_study/README.md) | Planlandı | Semantic Drone car hazırlığı. |
| [`landcover_building_study`](studies/landcover_building_study/README.md) | Eksik legacy | Landcover.ai building hazırlığı. |

Önceki üç parçalı teacher-bias çalışma ağacı silinmemiştir; yalnız aktif
kökten kaldırılıp kanonik çalışma içindeki `archives/pre_unification/` altında
korunmuştur. Taşıma dosya boyutu, inode ve SHA-256 ile doğrulanmıştır.

## Dizin Yapısı

```text
.
├── datasets/                 # Paylaşılan değiştirilemez ham veri
├── docs/                     # Depo mimarisi ve ortak çalışma günlüğü
├── external_models/          # Harici model kaynak kodları
├── models/                   # Paylaşılan başlangıç ağırlıkları/cache
├── src/yolo_sam/             # Çalışmadan bağımsız ortak kütüphane
├── studies/                  # Araştırma sorusu bazında çalışma alanları
├── tests/                    # Ortak kod testleri
└── tools/                    # Ortak bakım araçları
```

Bir dosya yalnız bir deneyde kullanılıyorsa ilgili çalışma altında; birden
fazla deneyde aynı davranışla kullanılıyorsa ortak kök katmanda bulunur.

## Kurulum ve Doğrulama

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python tools/models/setup_external_models.py
```

Kanonik çalışma komutları:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py --help
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
.venv/bin/pytest -q studies/teacher_reference_bias_paper/tests
```

Tam sıra ve model/checkpoint ayrıntıları:
[REPRODUCIBILITY.md](studies/teacher_reference_bias_paper/docs/REPRODUCIBILITY.md)

## Ortak Dokümantasyon

- [Repository Architecture](docs/REPOSITORY_ARCHITECTURE.md)
- [Legacy Status](docs/LEGACY_STATUS.md)
- [Worklog](docs/WORKLOG.md)
- [Güncel devir özeti](docs/summary/TEACHER_REFERENCE_BIAS_HANDOFF.md)
