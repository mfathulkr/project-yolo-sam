# Teacher Reference Bias v2 - 512 Görüntü

Bu çalışma, iSAID insan maskeleri ile SAM1 kaynaklı pseudo maskelerin aynı
SAM ailesi modellerini değerlendirirken oluşturduğu farkı kontrollü biçimde
ölçer. Eski `teacher_reference_bias_v1` çalışması değiştirilmez; v2 daha büyük
ve dengeli test protokolünü ayrı bir çalışma olarak tutar.

## Deney Sorusu

Bir segmentasyon modeli veya aynı model ailesi tarafından otomatik üretilen
maskeler bağımsız test ground truth'u gibi kullanılırsa sonuçlar insan
etiketlerine kıyasla ne kadar değişir?

## Sabit Protokol

- Hedef sınıf: `plane`
- Giriş çözünürlüğü: `1024×1024`
- Segmenterler: SAM1 ViT-H, SAM2.1 Hiera Large, SAM3
- Bbox istemleri: GT bbox ve YOLO bbox
- Detector: YOLO26x
- Detector tekrarı: sabit seed `42`
- Test kümesi: veri seti başına `512` görüntü
- Alt gruplar: her biri tam `128` görüntü
- Ana maske granülaritesi: uçak örneği düzeyi
- Detector metriği: gerçek COCO bbox mAP
- Maske metrikleri: IoU, Dice, Precision, Recall ve IoU eşik başarı oranları

Alt gruplar birbirini dışlar:

1. `No Overlap × Low Mask Area`
2. `No Overlap × High Mask Area`
3. `Overlap × Low Mask Area`
4. `Overlap × High Mask Area`

`Overall`, bu dört grubun birleşimi olan 512 görüntüdür.

## Veri ve Referanslar

### iSAID Plane

- Birincil referans resmi insan çizimli iSAID instance maskesidir.
- Aynı görüntü, bbox ve tahminler SAM1 GT-bbox çıktısından dondurulan pseudo
  maskelere karşı da ölçülür.
- Geniş aday havuzundaki maskeler resmi iSAID poligonlarından tile
  koordinatlarında yeniden rasterize edilerek kanonik COCO RLE biçimine
  dönüştürülür.
- Veri seti: https://captain-whu.github.io/iSAID/
- Makale: https://arxiv.org/abs/1905.12886

### SAMRS SOTA Plane

- Resmi SAMRS SOTA-RBB maskeleri SAM1 ViT-H ve detection istemlerinden
  üretilmiş pseudo maskelerdir.
- Bu veri setindeki sonuçlar bağımsız insan ground truth başarısı olarak
  yorumlanmaz.
- Veri seti ve kod: https://github.com/ViTAE-Transformer/SAMRS
- Makale: https://arxiv.org/abs/2305.02034

GT-bbox istemi iSAID için resmi insan instance anotasyonundaki kutudan,
SAMRS için yayımlanan özgün detection anotasyonundan gelir. Ana deneyde
pseudo maskeden bbox türetilmez.

Her iki veri setinde split kaynak sahne düzeyindedir. Aynı kaynak sahnenin
tile'ları train, validation ve test arasında paylaşılmaz.

V2 hazırlayıcı, v1 çalışmasının doğrulanmış tam tile havuzunu yalnız başlangıç
kaynağı olarak kullanır. Kaynak `content_manifest.json` SHA-256 ile
sabitlenmiştir; detector ağırlığı, tahmin veya metrik sonucu yeniden
kullanılmaz. V2 kendi splitlerini ve COCO kayıtlarını üretir. iSAID maskeleri
resmi ham insan poligonlarından yeniden kurulduğu için v1 maskesi körlemesine
kopyalanmaz.

## Dizinler

```text
configs/   Dondurulmuş protokol ve veri seti tanımları
data/      Bu çalışmaya özel hazırlanmış splitler
docs/      Çalışmaya özel yöntem ve QA notları
reports/   Okunabilir full metric MD/DOCX/PDF belgeleri
results/   Detector, tahmin, değerlendirme ve analiz çıktıları
scripts/   Çalıştırma ve raporlama giriş noktaları
src/       Çalışmaya özel Python paketi
tests/     Çalışmaya özel testler
```

## Ana Komutlar

Ayrıntılı deney sözleşmesi:
[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md). Kısa yöntem özeti ve
kalite durumu sırasıyla [docs/METHOD.md](docs/METHOD.md) ve
[docs/QA_CHECKLIST.md](docs/QA_CHECKLIST.md) içindedir.

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
.venv/bin/python studies/teacher_reference_bias_v2_512/scripts/study.py \
  preflight \
  --dataset studies/teacher_reference_bias_v2_512/configs/datasets/isaid_plane.yaml \
  --dataset studies/teacher_reference_bias_v2_512/configs/datasets/samrs_sota_plane.yaml
```

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
.venv/bin/python studies/teacher_reference_bias_v2_512/scripts/study.py \
  prepare \
  --dataset studies/teacher_reference_bias_v2_512/configs/datasets/isaid_plane.yaml
```

`prepare`, `validate-prepared`, `train-detector`, `detect`, `infer`,
`build-pseudo-reference`, `evaluate`, `analyze` ve `figures` alt komutları
aynı giriş noktasından çalıştırılır.

## Full Metric Raporları

Üç belge ayrı üretilir:

```text
reports/full_metrics/isaid_plane_human/
reports/full_metrics/isaid_plane_pseudo_sam1/
reports/full_metrics/samrs_sota_plane/
```

İlk belge yalnız resmi insan referanslı iSAID sonuçlarını, ikinci belge aynı
iSAID tahminlerinin kontrollü SAM1 pseudo referanslı sonuçlarını, üçüncü belge
ise resmi SAMRS pseudo referanslı sonuçları verir.

Her belgede:

- bir YOLO bbox detector tablosu,
- Overall ve dört overlap × mask-area segmentasyon tablosu,
- SAM1/SAM2/SAM3 için GT bbox ve YOLO bbox satırları,
- kaynaklı nitel örnekler,
- sonuçlardan dinamik üretilen discussion bölümü bulunur.

Doğrudan raporlar:

- [iSAID insan referansı PDF](reports/full_metrics/isaid_plane_human/isaid_plane_human_full_metric_document_colored.pdf)
- [iSAID SAM1 pseudo referansı PDF](reports/full_metrics/isaid_plane_pseudo_sam1/isaid_plane_pseudo_sam1_full_metric_document_colored.pdf)
- [SAMRS SOTA plane PDF](reports/full_metrics/samrs_sota_plane/samrs_sota_plane_full_metric_document_colored.pdf)

## Ana Sonuçlar

GT-bbox koşulundaki Overall instance IoU:

| Referans | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan | 0,653 | 0,629 | 0,700 |
| Aynı iSAID görüntülerinde SAM1 pseudo | 1,000 | 0,827 | 0,820 |
| SAMRS resmi SAM1 pseudo | 0,991 | 0,781 | 0,808 |

YOLO-bbox koşulundaki sabit seed 42 Overall instance IoU:

| Referans | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan | 0,597 | 0,574 | 0,638 |
| Aynı iSAID görüntülerinde SAM1 pseudo | 0,873 | 0,750 | 0,741 |
| SAMRS resmi SAM1 pseudo | 0,813 | 0,679 | 0,691 |

Aynı iSAID tahminleri insan yerine kontrollü SAM1 pseudo referansla
ölçüldüğünde GT-bbox IoU artışı SAM1/SAM2/SAM3 için sırasıyla
`+0,347 / +0,198 / +0,120` olmuştur. İnsan referansında model sırası
`SAM3 > SAM1 > SAM2`, pseudo referansta `SAM1 > SAM2 > SAM3` olmuştur.
Kontrollü pseudo referans doğrudan SAM1 GT-bbox çıktısından üretildiği için
SAM1'in `1,000` satırı bağımsız başarı değil, kimlik kontrolüdür.

Sabit seed 42 detector sonuçları:

| Veri seti | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 |
|---|---:|---:|---:|---:|
| iSAID plane | 0,920 | 0,847 | 0,545 | 0,762 |
| SAMRS SOTA plane | 0,913 | 0,797 | 0,209 | 0,665 |

Rapor üretimi:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/write_full_metric_reports.py
```

Tablo değerleri çalıştırılmış detector ve segmentasyon sonuçlarından okunur.
Eksik koşul varsa rapor üretimi hata verir; varsayılan veya elle yazılmış
metrik kullanılmaz.

## Yerel Inference ve Büyük Dosyalar

Kanonik raporlar yalnız seed 42 ağırlıklarını kullanır. Önceki seed 123 ve 2026
ağırlıkları tarihsel yeniden üretilebilirlik için Git LFS'de tutulur, ancak bu
raporların hiçbir ortalamasına girmez. RTX 4060 8 GB VRAM için detector batch
`1` ve SAM `float16` kullanan ayrı çalışma profili de vardır.

Kurulum, model boyutları, SAM indirme, private test görüntüsü aktarımı ve tam
inference komutları:
[docs/LOCAL_INFERENCE.md](docs/LOCAL_INFERENCE.md).
