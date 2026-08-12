# Remote Sensing YOLO-SAM Teacher-Reference Bias Study

> **Tarihsel ve superseded çalışma:** Bu 4x32 sürümdeki mevcut SAM3 tahmin ve
> raporları, sonradan yanlış olduğu saptanan PCS visual-exemplar inference
> yolundan üretilmiştir. Güncel bilimsel sonuç veya SAM3 karşılaştırması olarak
> kullanılmamalıdır. Kanonik 4x128 dört-deney çalışması
> `studies/teacher_reference_bias_paper/` altındadır.

Bu repo, uzaktan algılama görüntülerinde bbox-prompted instance segmentation
modellerini karşılaştıran yeniden üretilebilir bir araştırma çalışmasıdır.
Aktif çalışma, SAM tarafından otomatik üretilmiş maskelerin bağımsız ground
truth gibi kullanılmasının model skorlarını ne ölçüde yükselttiğini inceler.

## Araştırma Sorusu

SAMRS SOTA maskeleri SAM1 ViT-H ve detection bbox prompt'larıyla üretilmiştir.
SAM1, SAM2 ve SAM3 bu maskelere karşı değerlendirildiğinde SAM1'in aynı
generator ailesinden gelen referansa yapısal olarak daha yakın olması beklenir.

Çalışma iki etkiyi birbirinden ayırır:

1. Veri seti ve görüntü zorluğu.
2. Referans maskeyi üreten model ile değerlendirilen model arasındaki yakınlık.

Bunun için aynı tahminler hem insan maskesine hem SAM1 pseudo-mask'e
karşı ölçülür. Ana terim **öğretmen-referans yakınlığı
(teacher-reference affinity)** veya **pseudo-referans değerlendirme
yanlılığı**dır. Pseudo-label'ların ön eğitim için
yararlı olması ile bağımsız benchmark ground truth olması aynı iddia değildir.

## Frozen Protocol

| Bileşen | Değer |
|---|---|
| Hedef sınıf | `plane` |
| Veri setleri | iSAID ve doğrulanmış SAMRS SOTA-RBB |
| Görüntü boyutu | `1024 x 1024` |
| Test görüntüsü | Veri seti başına `128` |
| Strata | `overlap/no_overlap x low/high_mask_area`, her biri `32` görüntü |
| Segmenter'lar | SAM1 ViT-H, SAM2.1 Hiera Large, SAM3 |
| Bbox kaynakları | iSAID resmi insan bbox'ı, SAMRS özgün DOTA RHBox'ı ve YOLO tahmini |
| YOLO seed'leri | `42`, `123`, `2026` |
| Ana değerlendirme | Instance-level mask metrics |
| İstatistik | Kaynak-sahne kümeli `%95` bootstrap güven aralığı |

Ana protokol:

```text
studies/teacher_reference_bias_v1/configs/protocol.yaml
```

Veri seti tanımları:

```text
studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml
studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml
```

## Bilimsel Kontroller

- Train, validation ve test ayrımı tile değil, parent source scene düzeyinde
  yapılır.
- İki veri setinde aynı hedef sınıf, görüntü boyutu, test sayısı ve strata
  protokolü kullanılır.
- iSAID GT prompt'u resmi insan instance annotation'ındaki bbox'tır; SAMRS GT
  prompt'u özgün DOTA detection RHBox'ıdır. Hiçbir GT bbox SAM1
  pseudo-maskeden yeniden türetilmez.
- SAM1, SAM2 ve SAM3 aynı görüntü ve aynı bbox üzerinde çalışır.
- Detector AP metrikleri mask metriklerinden ayrı tutulur.
- Ana mask sonuçları instance-level hesaplanır; büyük nesneler küçük nesneleri
  pixel ağırlığıyla perdeleyemez.
- İnsan ve pseudo referans aynı tahmin üzerinde değiştirilerek
  `referans enflasyonu` ölçülür.
- İnsan referans maskeleri kayıpsız COCO RLE olarak saklanır; boş decode veya
  declared area ile pixel area uyuşmazlığı preflight'ta engellenir.
- SAMRS arşiv kimliği, numeric class mapping ve RBox/RHBox geometrisi
  exhaustive audit ile doğrulanmıştır.
- Bütün final run'lar resolved config, input ve output hash'leriyle manifest
  üretir.
- Hazırlanmış train/validation/test görüntüleri, YOLO label'ları, COCO
  anotasyonları ve metadata dosyaları veri seti başına tek bir deterministic
  `content_manifest.json` içinde SHA-256 ile dondurulur.
- Detector eğitimi ayrıca yalnız gerçekten kullandığı train/validation
  görüntüleri, YOLO bbox label'ları ve `data.yaml` için
  `detector_training_content_manifest.json` kullanır. Maske RLE'si ve test
  split'i detector eğitim girdisi gibi gösterilmez.
- Giriş hash'leri run başlangıcında ve bitişinde ayrı kaydedilir. Süreç
  sırasında tek bir giriş değişirse final kalite kapısı run'ı reddeder.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python tools/models/setup_external_models.py
```

Model erişim bilgilerini `.env` veya shell environment içinde tutun. Token ve
checkpoint dosyalarını Git'e eklemeyin.

Test paketi:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s tests -p 'test_*.py' -v
```

## Tek CLI

Aktif çalışmanın ana giriş noktası:

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py --help
```

### 1. Preflight

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py preflight \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml
```

### 2. Matched veri hazırlama ve doğrulama

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py prepare \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py prepare \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py validate-prepared \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py validate-prepared \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py model-provenance

# Mevcut GT-bbox maskelerini pinli revision tekrarından önce dondur
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py prediction-parity --mode snapshot
```

Tamamlanmış ve doğrulanmış bir stage varsayılan olarak tekrar çalıştırılmaz.
Yalnız bilinçli yeniden üretim için `--force` kullanılır.

`model-provenance`, SAM1 ve SAM2 Hugging Face revision'larını ve SAM1/SAM2/SAM3
checkpoint SHA-256 değerlerini protokoldeki pinlerle karşılaştırır.

GT-bbox inference pinli revision'larla bilinçli olarak yeniden üretildikten
sonra maske içeriğinin birebir değişmediği şu komutla doğrulanır:

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py prediction-parity --mode verify
```

Bu denetim runtime ve sürüm etiketi gibi metadata alanlarını dışarıda bırakır;
`instance_id`, durum ve maske RLE içeriğini karşılaştırır. Final kalite kapısı
altı koşulun tamamında birebir parite arar.

### 3. Detector eğitimi ve değerlendirmesi

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py train-detector \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --device 0

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py detect \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --seed 42 \
  --device 0
```

`train-detector`, `--seed` verilmezse frozen protokoldeki üç seed'i sırayla
çalıştırır. Bir seed seçmek için `--seed 42` kullanılabilir.

`detect` komutu testten önce aynı seed'in validation tahminlerini üretir.
YOLO confidence eşiği validation setinde bbox IoU 0.50 için F1'i en yüksek
yapan noktada seçilip dondurulur; test seti eşik seçiminde kullanılmaz. COCO AP
hesabı ise confidence sıralamasının tamamını kullanır.

### 4. GT-bbox segmentation

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py infer \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --model sam1 \
  --bbox-source gt_bbox \
  --device 0
```

Aynı komut `sam2` ve `sam3` için çalıştırılır.

### 5. Kontrollü pseudo-reference

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py build-pseudo-reference \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml
```

Bu komut iSAID'deki SAM1 GT-bbox tahminlerini ikinci referans olarak
dondurur. iSAID insan maskeleri birincil referans olarak korunur.

### 6. YOLO-bbox segmentation

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py infer \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --model sam2 \
  --bbox-source yolo_bbox \
  --seed 42 \
  --device 0
```

Her veri seti, model ve detector seed kombinasyonu ayrı manifest üretir.

### 7. Değerlendirme

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py evaluate \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --model sam1 \
  --bbox-source gt_bbox

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py evaluate \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --model sam1 \
  --bbox-source yolo_bbox \
  --seed 42
```

iSAID kontrollü pseudo-reference mevcutsa evaluator aynı tahmin üzerinde
insan ve pseudo referans metriklerini birlikte üretir. SAMRS referans türü
manifestte açıkça `pseudo_sam1` olarak belirtilir.

### 8. Analiz, görseller ve bağımsız insan denetimi

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py shared-reference-audit --force
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py analyze
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py figures
```

`shared-reference-audit`, SAMRS test tile'larını aynı DOTA görüntüleri üzerinde
bağımsız etiketlenmiş iSAID insan maskelerine geri eşler. Böylece SAMRS
pseudo-mask kalitesi ve model skor enflasyonu aynı görüntülerde ölçülür.
Örtüşen SAMRS tile'larında aynı uçağın tekrarlanmasını gizlememek için hem
tile-instance hem benzersiz insan nesnesi sayısı kaydedilir; ayrı duyarlılık
tablosu her benzersiz uçağı eşit ağırlıklandırır.

### 9. Bildiri çıktıları

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py paper
```

Final mod, iki veri setindeki üç detector seed'i ve bütün YOLO-bbox sonuçları
tamamlanmadan belge üretmez. Yalnız çalışma sırasında açıkça işaretli taslak
gerekiyorsa:

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py paper --allow-partial
```

Çıktılar `studies/teacher_reference_bias_v1/reports/paper/` altında aynı canonical analiz
kaynağından Markdown, DOCX ve tam altı A4 sayfalık PDF olarak oluşturulur.

### 10. Tam metrik DOCX/PDF çıktıları

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v1/scripts/write_full_metric_reports.py
```

Komut iSAID ve SAMRS SOTA için ayrı landscape Markdown, renkli DOCX ve
renkli PDF üretir. Her raporda detector bbox AP tablosu, Overall ve dört
`overlap × mask-area` tablosu, nitel örnekler ve referans duyarlılığı
bulunur:

```text
studies/teacher_reference_bias_v1/reports/full_metrics/
├── isaid_plane/
└── samrs_sota_plane/
```

### 11. Final bütünlük ve yeniden üretilebilirlik kontrolü

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/repair_legacy_detector_manifests.py
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py finalize
```

Bu kapı; altı detector koşusunu, 6 GT-bbox ve 18 YOLO-bbox segmentasyon
koşulunu, 10.000 tekrarlı bootstrap analizini, ortak insan referansı
denetimini, bütün figürleri, tam altı sayfalık PDF'i ve DOCX bütünlüğünü
birlikte doğrular. Her tamamlanmış run manifestindeki giriş ve çıkış
SHA-256 değerleri mevcut dosyalarla yeniden karşılaştırılır. Bir çıktı eksik
veya değişmişse çalışma tamamlanmış sayılmaz.

`repair_legacy_detector_manifests.py` yalnız bu çalışma sürerken başlatılmış
ve yeni start/finish fingerprint alanlarından önce çalışan pahalı
`train_detector` koşularına uygulanır. Özgün manifestleri byte düzeyinde
arşivler ve scoped detector veri ağacını bağlar. Başka bir pipeline aşamasını
onarmaya çalışmaz; bu aşamalar yeni manifest şemasıyla yeniden çalıştırılır.

Ayrıca kanonik analizde iki veri seti, üç model, GT/YOLO bbox, üç detector
seed'i, uygun human/pseudo referansları ve beş stratumdan oluşan tam koşul
matrisi içerik düzeyinde doğrulanır. Eksik instance satırı, yinelenen veya
beklenmeyen koşul, sonlu olmayan metrik ve `[0, 1]` dışındaki skor final
kalite kapısını geçemez.

Başarılı kontrolden sonra şu iki izlenebilirlik çıktısı üretilir:

```text
studies/teacher_reference_bias_v1/results/reproducibility_manifest.json
studies/teacher_reference_bias_v1/docs/REPRODUCIBILITY_APPENDIX.md
```

## Metrikler

### Segmentation

- IoU
- Dice
- Pixel precision
- Pixel recall
- Boundary IoU
- Success@IoU 0.50
- Success@IoU 0.75
- Success@IoU 0.90

Bu metrikler instance-level ortalanır. Image-level union sonuçları ikincil
tanılayıcı sonuç olarak ayrıca üretilir.

YOLO-bbox koşulunda eşleşmeyen GT instance boş maskeyle sıfır skor alır.
Eşleşmeyen detector prediction'ı detector AP hesabında false positive ve
image-level union maskesinde tahmin olarak korunur. Bu nedenle instance mask
tablosu COCO mask AP değildir.

### Detection

- COCO bbox AP50
- COCO bbox AP75
- COCO bbox AP90
- COCO bbox AP50-95
- Precision ve recall operating point'leri

Detection bbox IoU ile mask IoU aynı ölçüm değildir. Detector tablosu yalnız
kutuların ground-truth kutularla eşleşmesini değerlendirir.

## Sonuç Yapısı

```text
studies/teacher_reference_bias_v1/results/
├── analysis/
├── audits/
├── dataset_audits/
├── detectors/
├── evaluation/
├── figures/
├── literature/
├── predictions/
├── references/
└── smoke/
```

Hazırlanmış veri içerik kimlikleri:

```text
studies/teacher_reference_bias_v1/data/prepared/isaid_plane/content_manifest.json
studies/teacher_reference_bias_v1/data/prepared/samrs_sota_plane/content_manifest.json
studies/teacher_reference_bias_v1/data/prepared/isaid_plane/detector_training_content_manifest.json
studies/teacher_reference_bias_v1/data/prepared/samrs_sota_plane/detector_training_content_manifest.json
```

Canonical analiz kaynakları:

```text
studies/teacher_reference_bias_v1/results/analysis/
├── canonical_instance_metrics.csv
├── aggregate_metrics.csv
├── paired_model_comparisons.csv
├── reference_inflation.csv
├── ranking_comparisons.csv
├── detector_seed_summary.csv
├── segmentation_seed_summary.csv
├── training_health_audit.csv
├── prediction_status_audit.csv
└── manifest.json
```

`training_health_audit.csv`, altı YOLO eğitiminin epok sayısını, son
precision/recall/AP değerlerini ve eğitim boyunca görülen sonlu olmayan
hücreleri ayrı ayrı kaydeder. Geçici validation-loss `NaN` değerleri
gizlenmez; final detector metriklerinin sonlu ve `[0, 1]` aralığında olması
zorunludur.

## Dokümantasyon

```text
studies/teacher_reference_bias_v1/docs/EXPERIMENT_PLAN.md
studies/teacher_reference_bias_v1/docs/LITERATURE_REVIEW.md
studies/teacher_reference_bias_v1/docs/REPRODUCIBILITY_APPENDIX.md
docs/REFACTOR_PLAN.md
docs/LEGACY_STATUS.md
docs/WORKLOG.md
```

`WORKLOG.md`, tamamlanan aşamaları ve açık sorunları basit Türkçe ile izler.
`REPRODUCIBILITY_APPENDIX.md` yalnız final kalite kapısı geçildikten sonra
otomatik üretilir. Teknik ayrıntılar run manifestlerinde ve plan
dokümanlarında tutulur.

## Legacy Çalışma

Önceki iSAID vehicle prompt study silinmemiştir. Eski config, script, sonuç ve
sunum dosyaları araştırma geçmişi olarak korunur; aktif matched plane
çalışmasının canonical sonuçlarına karıştırılmaz.

```text
studies/isaid_vehicle_study/
studies/samrs_sota_plane_study/
```

GroundingDINO, RemoteSAM, RingMo-SAM, SegEarth-OV3 ve text/hybrid prompt
pipeline'ları mevcut legacy çalışmaya aittir. Aktif bildirinin ana
karşılaştırması yalnız SAM1, SAM2 ve SAM3 bbox-prompted segmentation
pipeline'larıdır.
