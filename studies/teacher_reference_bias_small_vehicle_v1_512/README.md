# Teacher Reference Bias - Small Vehicle - 512 Görüntü

Bu çalışma, önceki uçak deneyinin hedef sınıfı küçük araç olacak şekilde birebir
eşlenmiş devamıdır. Amaç, SAM1 tarafından üretilmiş pseudo maskeler bağımsız
test referansı gibi kullanıldığında ölçülen başarının ve model sıralamasının
nasıl değiştiğini ikinci bir nesne sınıfında sınamaktır.

## Sabit Protokol

- Veri setleri: iSAID ve SAMRS SOTA-RBB
- Hedefler: iSAID `Small_Vehicle`, SAMRS `small-vehicle`
- Giriş çözünürlüğü: `1024×1024`
- Test: veri seti başına `512` görüntü
- Alt gruplar: `No Overlap/Overlap × Low/High Mask Area`, her biri `128`
- Detector: YOLO26x, sabit seed `42`, 100 epok üst sınırı, train/validation
  batch 12
- Segmenterler: SAM1 ViT-H, SAM2.1 Hiera Large, SAM3
- İstemler: GT bbox ve seed 42 YOLO bbox
- Ana değerlendirme: nesne örneği düzeyinde IoU, Dice, Precision, Recall ve
  `IoU ≥ 0.50/0.75/0.90`
- Detector değerlendirmesi: gerçek COCO bbox mAP50, mAP75, mAP90, mAP50-95
  ve sabit confidence noktasındaki Precision/Recall

Bir görüntüde birden fazla küçük araç varsa hepsi ayrı instance olarak modele
verilir ve ayrı değerlendirilir. Nitel rapor görsellerinde aynı sahnedeki bütün
GT kutular ve bütün instance maskeleri birlikte gösterilir.

Doğrulanan final splitlerde iSAID test kümesi 31 kaynak sahneden 12.051,
SAMRS test kümesi 17 kaynak sahneden 7.659 küçük araç instance'ı içerir.
Train/validation/test kaynak sahne kesişimleri iki veri setinde de sıfırdır.

## Üç Referans Raporu

1. `isaid_small_vehicle_human`: resmi iSAID insan maskeleri
2. `isaid_small_vehicle_pseudo_sam1`: aynı iSAID tahminleri, SAM1 GT-bbox
   çıktısından dondurulan kontrollü pseudo referans
3. `samrs_sota_small_vehicle`: yayımlanan SAM1 kaynaklı SAMRS pseudo maskeleri

Her rapor MD, renkli DOCX ve renkli PDF olarak üretilir. Her belgede bir
detector tablosu, Overall ve dört alt grup tablosu, dört nitel örnek ve dinamik
Discussion bölümü bulunur. RemoteSAM, RingMoSAM, proxy mAP ve tanımlanmamış
başka metrikler deney matrisine girmez.

## Dizinler

```text
configs/   Dondurulmuş protokol ve veri seti tanımları
data/      Çalışmaya özel master ve final prepared splitler
docs/      Deney planı, yöntem, yerel çalıştırma ve QA
reports/   Üç full-metric MD/DOCX/PDF belge
results/   Detector, tahmin, değerlendirme, analiz ve audit çıktıları
scripts/   Çalıştırma ve raporlama giriş noktaları
src/       Çalışmaya özel Python paketi
tests/     Birim ve entegrasyon testleri
```

## Çalıştırma

Ana giriş noktası:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_small_vehicle_v1_512/src \
.venv/bin/python \
  studies/teacher_reference_bias_small_vehicle_v1_512/scripts/study.py --help
```

Akış sırası `preflight`, `prepare`, `validate-prepared`, `train-detector`,
`detect`, `infer`, `build-pseudo-reference`, `evaluate`, `analyze`, `figures`
ve full-metric rapor üretimidir.

Uzun detector eğitimlerinin terminal kapanmasından etkilenmemesi için iki
yeniden başlatılabilir tamamlayıcı bulunur:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_small_vehicle_v1_512/scripts/complete_yolo_condition_after_training.py \
  --dataset studies/teacher_reference_bias_small_vehicle_v1_512/configs/datasets/isaid_small_vehicle.yaml \
  --device 2

.venv/bin/python \
  studies/teacher_reference_bias_small_vehicle_v1_512/scripts/finalize_after_post_training.py
```

İlk araç eğitim manifesti tamamlanana kadar bekler; ardından validation
confidence seçimi, test bbox AP, üç YOLO-bbox SAM koşulu ve değerlendirmeleri
çalıştırır. İkinci araç iki veri setini bekleyip canonical analiz, figür,
MD/DOCX/PDF, validator ve görüntüsüz aktarım paketini üretir. Her ikisi de
manifest tabanlıdır; tamamlanmış bilimsel aşamaları tekrar çalıştırmaz.

Ayrıntılı sözleşme [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md), yöntem
[docs/METHOD.md](docs/METHOD.md), tamamlanma kapıları ise
[docs/QA_CHECKLIST.md](docs/QA_CHECKLIST.md) içindedir.

## Yorum Sınırı

SAMRS maskeleri SAM1 ile üretilmiştir; bu veri setindeki yüksek SAM1 uyumu
insan çizimli bağımsız ground truth başarısı değildir. Nedensel teacher-reference
bias karşılaştırması, aynı iSAID görüntüleri, bbox istemleri ve tahminleri sabit
tutup yalnız insan referansını SAM1 pseudo referansıyla değiştiren eşlenmiş
deneydir.
