# Teacher Reference Bias Çalışması - Yerel Codex Devir Özeti

Bu belge, projeyi başka bir bilgisayardaki Codex oturumunun yalnız repository
üzerinden anlayıp sürdürebilmesi için hazırlanmıştır. Bilimsel amaç, deney
tasarımı, bugüne kadar yapılanlar, mevcut hesaplama durumu ve tamamlanma
koşulları birlikte açıklanır.

Durum anlık görüntüsü: **4 Ağustos 2026**. Ana dal: `main`.

## 1. En Kısa Özet

Bu çalışmanın ana sorusu şudur:

> Bir yapay zekâ modeliyle üretilmiş segmentasyon maskelerini bağımsız test
> ground truth'u gibi kullanırsak, aynı model veya aynı model ailesi yapay
> olarak daha başarılı görünür mü?

İlk gözlem iki önceki deneyden doğdu:

- iSAID'in insan çizimli maskelerinde SAM tabanlı sonuçlar orta düzeydeydi.
- SAMRS SOTA-RBB'nin SAM1 ile üretilmiş maskelerinde SAM1 sonuçları çok
  yüksekti.

Bu farkı yalnız iki farklı veri setinin doğal zorluk farkına bağlamak yeterli
değildi. Bu nedenle aynı iSAID görüntülerini, aynı bbox istemlerini ve aynı SAM
tahminlerini sabit tutup yalnız değerlendirme referansını değiştiren kontrollü
bir deney kuruldu:

1. Resmi insan çizimli iSAID maskesi
2. Aynı instance için SAM1 GT-bbox çıktısından dondurulmuş pseudo maske

Plane deneyinde referans değişince SAM1'in GT-bbox IoU'su `0,653`ten `1,000`a
çıktı ve model sıralaması değişti. `1,000` bağımsız başarı değildir; tahminin
kendisinin referans yapılmasından doğan kimlik kontrolüdür. Bulgumuz, model
üretimli referansların test ground truth'u gibi kullanılmasının ölçümü
üretici modele doğru kaydırabildiğidir.

Plane çalışması ve aynı protokolün `small-vehicle` sınıfındaki birebir tekrarı
tamamlandı. İki çalışmada da veri hazırlama, seed-42 YOLO26x eğitimi, GT/YOLO
bbox ile SAM1/SAM2/SAM3 çıkarımı, instance değerlendirmesi, raporlama ve
taşınabilirlik QA kapıları geçti.

## 2. Çalışmanın Ortaya Çıkışı

### 2.1 Eski iSAID vehicle deneyi

İlk geniş çalışma iSAID üzerindeki birleşik araç maskelerini ve farklı
detector-segmenter pipeline'larını karşılaştırdı. Bu çalışma yararlı bir keşif
çalışmasıydı; ancak RemoteSAM, RingMoSAM ve farklı prompt düzenleri gibi
bildirinin ana sorusuna doğrudan hizmet etmeyen koşullar içeriyordu.

Tarihsel çalışma:

- `studies/isaid_vehicle_study/`

Bu sonuçlar silinmedi, fakat güncel bildirinin kanonik deney matrisi değildir.

### 2.2 SAMRS SOTA plane deneyi

Hocanın "boxy olmayan bir nesne seç" yönlendirmesiyle SAMRS SOTA-RBB içinde
`plane` sınıfı seçildi. Uçak şekli yatay bbox'a tam oturmadığı için bbox
istemli segmentasyon açısından araçtan daha zor ve anlamlı bir hedefti.

Tarihsel çalışma:

- `studies/samrs_sota_plane_study/`

SAMRS SOTA-RBB maskeleri SAM1 ViT-H ve detection anotasyonlarından üretilmiş
pseudo maskelerdir. SAM1'in bu maskelere çok yüksek uyum göstermesi, bağımsız
insan ground truth başarısı olarak yorumlanamaz. Öğretmen-referans yanlılığı
fikri bu gözlemden çıktı.

### 2.3 Bildiri hipotezi

Hocanın önerdiği bildiri konusu, iki ayrı veri setinde farklı skor görmekten
daha güçlü ve kontrollü bir iddiaya dönüştürüldü:

> Test referansını üreten model ile değerlendirilen model aynı veya yakınsa,
> ölçülen segmentasyon başarısı şişebilir ve model sıralaması değişebilir.

Bu etkiye projede **teacher-reference bias** deniyor. Türkçe karşılığı
"öğretmen-referans yanlılığı" olarak düşünülebilir.

## 3. Kavramlar ve Doğru Terminoloji

### 3.1 Pseudo etiket

İnsan tarafından çizilmek yerine bir model tarafından otomatik üretilen
etikettir. Pseudo etiket eğitim verisini büyütmek, ön etiketleme yapmak veya
insan düzeltmesini hızlandırmak için yararlı olabilir.

### 3.2 Teacher-reference bias

Pseudo etiketi üreten modelin geometrik ve semantik tercihleri etikete geçer.
Aynı model bu etikete karşı ölçülürse, insan maskesine göre daha yüksek skor
alabilir. Başka model aileleri daha doğru bir insan maskesi üretse bile
öğretmenin stilinden ayrıldıkları için cezalandırılabilir.

### 3.3 Leakage

Buradaki risk, klasik anlamda train görüntüsünün test split'ine kopyalanması
olmak zorunda değildir. Değerlendirilen modelin çıktısının veya model
ailesinin tercihinin test referansına gömülmesi bir **evaluation leakage** ya
da **reference leakage** biçimidir.

### 3.4 "Overfit gibi" ifadesinin sınırı

SAM1'in SAM1 pseudo maskelerine çok iyi uyması davranış olarak overfit'e
benzeyebilir; fakat model bu test görüntüleri üzerinde yeniden eğitilmediyse
teknik olarak klasik eğitim overfitting'i değildir. Daha doğru ifade,
"değerlendirme referansının üretici modele bağımsız olmaması"dır.

## 4. Araştırma Soruları ve Hipotezler

Ana sorular:

1. Aynı tahminler insan yerine SAM1 pseudo referansla ölçülünce skor ne kadar
   değişiyor?
2. En büyük artışı pseudo etiketi üreten SAM1 mi alıyor?
3. Referans kaynağı model sıralamasını değiştiriyor mu?
4. GT bbox yerine YOLO bbox kullanınca ek kayıp ne kadar oluyor?
5. Etki overlap ve toplam mask-area koşullarında değişiyor mu?
6. Plane bulgusu farklı ve yoğun bir küçük nesne sınıfında tekrarlanıyor mu?

Dondurulmuş hipotezler:

- **H1 - Skor enflasyonu:** Pseudo referans, insan referansına göre daha
  yüksek skor üretebilir.
- **H2 - Üretici yakınlığı:** Artış en fazla pseudo referansı üreten SAM1'de
  görülür.
- **H3 - Sıralama duyarlılığı:** Model sıralaması referans kaynağına göre
  değişebilir.
- **H4 - Detector kaybı:** YOLO bbox koşulu GT bbox koşulundan daha düşük
  maske başarısı üretir.
- **H5 - Sahne duyarlılığı:** Overlap ve mask alanı, başarı farklarını etkiler.

## 5. Ne İddia Ediyoruz, Ne İddia Etmiyoruz?

İddia:

- Model üretimli test referansı bağımsız insan ground truth'u değildir.
- Böyle bir referans mutlak skorları ve model sıralamasını yanlılaştırabilir.
- Bu risk, aynı iSAID tahminlerini iki referansa karşı ölçen eşlenmiş tasarımda
  doğrudan gösterilebilir.

İddia edilmeyenler:

- "Bütün pseudo etiketler işe yaramaz" denmiyor.
- "SAMRS veri seti değersizdir" denmiyor.
- "SAM1 her zaman SAM2/SAM3'ten kötüdür veya iyidir" denmiyor.
- SAMRS pseudo sonuçları insan ground truth başarısı diye sunulmuyor.
- Farklı veri setlerindeki ham skor farkı tek başına nedensel kanıt sayılmıyor.
- Plane ve small-vehicle detector farkları doğrudan teacher bias olarak
  yorumlanmıyor; detector eğitim veri miktarları ve nesne dağılımları farklı.

## 6. Kanonik Çalışmalar

### 6.1 Tamamlanan plane çalışması

Klasör:

- `studies/teacher_reference_bias_v2_512/`

Üç full-metric rapor:

- `reports/full_metrics/isaid_plane_human/`
- `reports/full_metrics/isaid_plane_pseudo_sam1/`
- `reports/full_metrics/samrs_sota_plane/`

Her klasörde Markdown, renkli DOCX, renkli PDF, tablo CSV'leri, görseller ve
hash manifesti vardır. Raporlar yalnız sabit YOLO seed `42` kullanılarak
yeniden üretildi.

### 6.2 Tamamlanan small-vehicle tekrarı

Klasör:

- `studies/teacher_reference_bias_small_vehicle_v1_512/`

Üretilen üç full-metric rapor:

- `reports/full_metrics/isaid_small_vehicle_human/`
- `reports/full_metrics/isaid_small_vehicle_pseudo_sam1/`
- `reports/full_metrics/samrs_sota_small_vehicle/`

Small-vehicle protokolü, `study_id` dışında plane protokolüyle aynı olacak
şekilde denetlendi. Audit:

- `results/audits/plane_protocol_equivalence.json`

## 7. Birebir Eşlenmiş Deney Protokolü

### 7.1 Ortak model ve inference koşulları

- Giriş çözünürlüğü: `1024×1024`
- Segmenterler:
  - SAM1 `facebook/sam-vit-huge`
  - SAM2.1 `facebook/sam2.1-hiera-large`
  - SAM3 yerel sabit checkpoint
- SAM modellerinde fine-tune yok
- Prompt yalnız bbox
- Bbox kaynakları:
  - özgün anotasyondan GT bbox
  - veri setine özel eğitilmiş YOLO26x bbox
- Detector seed: yalnız `42`
- YOLO26x üst sınır: 100 epoch
- Batch: 12
- Patience: 30
- Confidence seçimi: yalnız validation kümesinde
- Test kümesi model veya confidence seçimi için kullanılmaz

Gerçek konfigürasyon:

- `studies/teacher_reference_bias_small_vehicle_v1_512/configs/protocol.yaml`

### 7.2 Referanslar

iSAID için aynı tahmin iki ayrı referansa karşı ölçülür:

1. Resmi insan instance maskesi
2. SAM1 GT-bbox tahmininden dondurulmuş kontrollü pseudo maske

Sabit kalanlar: görüntü, instance kimliği, bbox istemi, SAM tahmini ve detector
seed. Değişen tek şey değerlendirme referansıdır. Teacher-reference bias için
asıl kontrollü kanıt budur.

SAMRS SOTA için yayımlanan SAM1 kaynaklı pseudo maskeler kullanılır. Bu dış
pseudo-reference tekrarıdır; iSAID eşlenmiş karşılaştırması kadar kontrollü
bir insan/pseudo karşılaştırması değildir.

### 7.3 Test büyüklüğü ve alt gruplar

Her veri setinde test kümesi 512 görüntüdür:

| Alt grup | Görüntü sayısı |
|---|---:|
| No Overlap × Low Mask Area | 128 |
| No Overlap × High Mask Area | 128 |
| Overlap × Low Mask Area | 128 |
| Overlap × High Mask Area | 128 |
| Overall | 512 |

Gruplar birbirini dışlar. `Overall`, dört grubun birleşimidir.

Overlap tanımı:

- `No Overlap`: görüntüdeki bütün GT bbox çiftlerinin IoU'su tam `0`
- `Overlap`: en az bir GT bbox çiftinin IoU'su `≥ 0,001`

Low/High Mask Area, görüntüdeki hedef sınıfa ait toplam referans maskesi
alanının görüntü alanına oranıdır. Eşikler model tahminine bakılmadan veri
hazırlama sırasında dondurulur.

### 7.4 Bütün nesneler modele gider

Bir görüntüde çok sayıda uçak veya küçük araç varsa yalnız bir tanesi
seçilmez. Bütün instance bbox'ları modele verilir, her instance ayrı
değerlendirilir ve nitel rapor görsellerinde bütün GT kutular/maskeler birlikte
gösterilir.

### 7.5 Maske değerlendirmesi instance düzeyindedir

Her nesne için ayrı hesaplanır:

- `IoU = TP / (TP + FP + FN)`
- `Dice = 2TP / (2TP + FP + FN)`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `IoU ≥ 0.50`, `IoU ≥ 0.75`, `IoU ≥ 0.90` başarı oranları

Sonra nesneler eşit ağırlıkla ortalanır. Büyük nesneler daha çok piksele sahip
olduğu için küçük nesnelerin skorunu perdeleyemez.

YOLO'nun kaçırdığı GT nesne boş maske olarak değerlendirilir ve o instance'ın
maske skorları sıfır olur. Yanlış pozitif detector kutuları maske tablosuna
sahte GT instance eklemez; etkileri detector tablosunda ölçülür.

IoU eşik başarı oranları **mask AP veya mAP değildir**.

### 7.6 Detector metrikleri

YOLO tablosunda gerçek COCO bbox metrikleri kullanılır:

- BBox mAP50
- BBox mAP75
- BBox mAP90
- BBox mAP50-95
- Sabit confidence noktasında Precision/Recall @ IoU 0.50/0.75/0.90

`mAP proxy`, mask IoU eşik oranını mAP diye adlandırma veya tanımsız başka
metrik kullanılmaz.

## 8. Neden Plane ve Small Vehicle?

Plane seçimi, uçak geometrisinin yatay bbox'a tam oturmaması nedeniyle
segmenter sınır kalitesini daha görünür kılar. Fakat yalnız plane kullanmak,
bulgunun sınıfa özgü olduğu itirazını açık bırakır.

Small vehicle seçimi:

- Her iki veri setinde ortak sınıftır.
- Remote sensing small-object problemine doğrudan uyar.
- Kalabalık ve overlap içeren sahneler üretir.
- Plane ile aynı dört strata tasarımını kurmaya yeterli örnek sağlar.
- İnsan ve pseudo referans etkisini daha yoğun instance dağılımında sınar.

Bu iki hedef birlikte, hipotezin yalnız "uçak maskesi kolaydı" veya "tek bir
nesne türüne özgüydü" açıklamasına indirgenmesini zorlaştırır.

## 9. Tamamlanan Plane Sonuçları

### 9.1 GT-bbox Overall instance IoU

| Referans | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan | 0,653 | 0,629 | 0,655 |
| Aynı iSAID görüntülerinde SAM1 pseudo | 1,000 | 0,827 | 0,795 |
| SAMRS resmi SAM1 pseudo | 0,991 | 0,781 | 0,611 |

### 9.2 YOLO-bbox Overall instance IoU

| Referans | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan | 0,597 | 0,574 | 0,595 |
| Aynı iSAID görüntülerinde SAM1 pseudo | 0,873 | 0,750 | 0,721 |
| SAMRS resmi SAM1 pseudo | 0,813 | 0,679 | 0,537 |

### 9.3 Kontrollü iSAID skor artışı

GT-bbox koşulunda aynı tahminler insan yerine SAM1 pseudo referansa karşı
ölçülünce IoU değişimleri:

- SAM1: `+0,347`
- SAM2: `+0,198`
- SAM3: `+0,140`

İnsan referansındaki sıra `SAM3 > SAM1 > SAM2`, pseudo referanstaki sıra
`SAM1 > SAM2 > SAM3` oldu. Yani yalnız skor yükselmedi; model sıralaması da
değişti.

### 9.4 Plane detector sonuçları, seed 42

| Veri seti | mAP50 | mAP75 | mAP90 | mAP50-95 |
|---|---:|---:|---:|---:|
| iSAID plane | 0,920 | 0,847 | 0,545 | 0,762 |
| SAMRS SOTA plane | 0,913 | 0,797 | 0,209 | 0,665 |

## 10. Small-Vehicle Çalışmasının Güncel Durumu

### 10.1 Veri hazırlama tamamlandı

iSAID Small Vehicle:

| Split | Görüntü | Instance |
|---|---:|---:|
| Train | 5.930 | 359.927 |
| Validation | 1.353 | 71.275 |
| Test | 512 | 12.051 |

- Test 31 bağımsız kaynak sahneden gelir.
- Train/validation/test kaynak sahne kesişimi sıfırdır.
- Test tam `4×128` strata içerir.

SAMRS SOTA Small Vehicle:

| Split | Görüntü | Instance |
|---|---:|---:|
| Train | 7.824 | 304.414 |
| Validation | 1.567 | 49.792 |
| Test | 512 | 7.659 |

- Test 17 bağımsız kaynak sahneden gelir.
- Train/validation/test kaynak sahne kesişimi sıfırdır.
- Test tam `4×128` strata içerir.

### 10.2 Final segmentasyon sonuçları

GT-bbox Overall instance IoU değerleri:

| Referans | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan | 0,658 | 0,645 | 0,370 |
| Aynı iSAID görüntülerinde SAM1 pseudo | 1,000 | 0,749 | 0,419 |
| SAMRS resmi SAM1 pseudo | 0,998 | 0,846 | 0,685 |

YOLO-bbox Overall instance IoU değerleri:

| Referans | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan | 0,478 | 0,461 | 0,299 |
| Aynı iSAID görüntülerinde SAM1 pseudo | 0,655 | 0,550 | 0,341 |
| SAMRS resmi SAM1 pseudo | 0,782 | 0,707 | 0,560 |

Aynı iSAID GT-bbox tahminleri insan yerine SAM1 pseudo referansla ölçülünce
SAM1/SAM2/SAM3 için IoU değişimi `+0,342 / +0,103 / +0,049` oldu. Model
sırası her iki referansta da `SAM1 > SAM2 > SAM3` kaldı. Dolayısıyla bu
sınıftaki kanıt, sıralama değişiminden değil, referans üreticisi SAM1 lehine
çok daha büyük skor artışından gelir. SAM1'in pseudo referanstaki `1,000`
değeri yine bağımsız başarı değil kimlik kontrolüdür.

### 10.3 Final detector sonuçları

- iSAID small vehicle YOLO26x seed 42 eğitimi 100 epoch tamamladı; en iyi
  validation mAP50-95 `0,33355`, en iyi epoch 79'dur.
- SAMRS small vehicle YOLO26x seed 42 eğitimi 69. epoch'ta erken durdu; en iyi
  validation mAP50-95 `0,44472`, en iyi epoch 39'dur.

512 görüntülük test kümelerindeki gerçek COCO bbox sonuçları:

| Veri seti | mAP50 | mAP75 | mAP90 | mAP50-95 |
|---|---:|---:|---:|---:|
| iSAID small vehicle | 0,609 | 0,358 | 0,021 | 0,346 |
| SAMRS SOTA small vehicle | 0,819 | 0,534 | 0,072 | 0,502 |

### 10.4 Neden eğitim uzun sürüyor?

Hiperparametre taraması yapılmıyor. Toplam yalnız iki yeni detector eğitimi
vardır: bir iSAID, bir SAMRS.

Sürenin nedeni:

- YOLO26x büyük modeldir.
- Giriş çözünürlüğü `1024×1024`tür.
- Bir görüntüde yüzlerce küçük araç olabilir.
- Eğitim setlerinde yaklaşık 360 bin ve 304 bin bbox vardır.
- Dense batch'lerde Ultralytics `TaskAlignedAssigner` matrisi çok büyür.
- Bazı aşırı yoğun batch'ler güvenli biçimde CPU eşleştiriciye düşer; bu veri
  veya doğruluk kaybı yaratmaz, yalnız epoch süresini uzatır.
- Her epoch sonunda büyük validation split'i de işlenir.

Protokol eşitliğini bozmamak için çözünürlük, batch, veri veya patience sırf
hız kazanmak amacıyla değiştirilmemiştir.

## 11. Uygulanan Bilimsel ve Teknik Güvenceler

- Tek detector seed: `42`
- Hiperparametre sweep yok
- Testten confidence/model seçimi yok
- Kaynak sahne düzeyinde split; tile leakage yok
- iSAID insan maskeleri resmi polygonlardan kayıpsız COCO RLE'ye rasterize
  edildi
- Pseudo referans kimliği checkpoint SHA-256 ile sabitlendi
- Plane ve small-vehicle protokol eşitliği otomatik audit ile doğrulandı
- Dense SAM bbox çıkarımı bütün instance'ları koruyarak en fazla 16 kutuluk
  hesap batch'lerine bölündü
- Dense YOLO validation batch'i train batch'iyle aynı 12 değerinde tutuldu
- Bir görüntüdeki bütün GT instance'lar modele ve rapor görseline giriyor
- Maske metrikleri pixel toplamından değil instance başına hesaplanıyor
- Detector metrikleri gerçek COCO bbox AP; proxy mAP yok
- Full-metric rapor üreticisi eksik koşulda hata veriyor
- Rapor validator'ı tablo, sayım, kaynak, hash ve format kapılarını kontrol
  ediyor
- Taşınabilir paketler ham görüntü, cache, log ve checkpoint kopyalarını
  dışlıyor
- Plane kanonik raporları yalnız seed 42 sonucunu kullanıyor

## 12. Otomatik Tamamlama Zinciri

Detector eğitimleri terminalden bağımsız süreçlerde çalışır. Her veri seti için
bir post-training worker detector manifestini bekler. Eğitim tamamlanınca:

1. En iyi `best.pt` checkpoint'i doğrulanır.
2. Validation üzerinde confidence seçilir ve dondurulur.
3. 512 görüntülük testte gerçek COCO bbox metrikleri hesaplanır.
4. SAM1, SAM2 ve SAM3 YOLO-bbox segmentasyonları çalıştırılır.
5. İnsan/pseudo referanslarına karşı instance değerlendirmeleri yazılır.

İki veri seti tamamlanınca ayrı finalizer:

1. Kanonik analiz tablolarını birleştirir.
2. Figürleri üretir.
3. Üç full-metric MD/DOCX/PDF raporu yazar.
4. Rapor validator'ını çalıştırır.
5. Görüntüsüz taşınabilir paketleri üretir.
6. SHA-256 varlık kontrolünü çalıştırır.

Durum dosyaları:

- `results/detectors/isaid_small_vehicle/seed_42/manifest.json`
- `results/detectors/samrs_sota_small_vehicle/seed_42/manifest.json`
- `results/post_training/isaid_small_vehicle/manifest.json`
- `results/post_training/samrs_sota_small_vehicle/manifest.json`
- `results/finalization/manifest.json`

Worker giriş noktaları:

- `scripts/complete_yolo_condition_after_training.py`
- `scripts/finalize_after_post_training.py`

Bu dosyalar
`studies/teacher_reference_bias_small_vehicle_v1_512/` köküne göredir.

## 13. Tamamlanma Durumu

Deneysel çalışma tamamlandı:

1. İki seed-42 YOLO26x eğitimi ve validation confidence seçimi tamamlandı.
2. İki veri setinde detector testi ile bütün GT/YOLO-bbox SAM koşulları
   tamamlandı.
3. Kanonik analiz 190.566 instance satırı ve 90 aggregate satırıyla üretildi.
4. Üç full-metric rapor MD, renkli DOCX ve renkli PDF olarak üretildi.
5. PDF'ler 14/14/13 sayfa olarak görsel denetimden geçti; dört nitel örnekte
   bütün instance kutuları ve birleşik maskeler gösteriliyor.
6. Rapor validator'ı, protokol eşitliği, manifest, hash ve taşınabilir paket
   kontrolleri geçti.
7. `best.pt`, raporlar, auditler ve görüntüsüz aktarım paketleri Git LFS/Git
   kapsamına alındı.

Bundan sonraki iş deney çalıştırmak değil, tamamlanan kanıtları altı sayfalık
bildiri anlatısına dönüştürmek ve gerekirse ek ablation tasarlamaktır.

## 14. Yerel Codex'in Önce Okuması Gereken Dosyalar

Genel repo:

1. `README.md`
2. `docs/REPOSITORY_ARCHITECTURE.md`
3. `docs/WORKLOG.md`
4. Bu belge

Plane kanonik çalışma:

1. `studies/teacher_reference_bias_v2_512/README.md`
2. `studies/teacher_reference_bias_v2_512/docs/EXPERIMENT_PLAN.md`
3. `studies/teacher_reference_bias_v2_512/docs/METHOD.md`
4. `studies/teacher_reference_bias_v2_512/docs/QA_CHECKLIST.md`

Small-vehicle tamamlanmış çalışma:

1. `studies/teacher_reference_bias_small_vehicle_v1_512/README.md`
2. `studies/teacher_reference_bias_small_vehicle_v1_512/docs/EXPERIMENT_PLAN.md`
3. `studies/teacher_reference_bias_small_vehicle_v1_512/docs/METHOD.md`
4. `studies/teacher_reference_bias_small_vehicle_v1_512/docs/QA_CHECKLIST.md`
5. `studies/teacher_reference_bias_small_vehicle_v1_512/configs/protocol.yaml`

## 15. Yerel Bilgisayara Aktarım Notu

Repository normal Git ve Git LFS ile çekilmelidir:

```bash
git pull
git lfs pull
```

Ham/private görüntüler GitHub'a gönderilmez. Kayıtlı manifestteki seed-42
detector ağırlıkları, kanonik sonuç paketi ve görüntüsüz prepared metadata
paketi yerel varlık yöneticisiyle doğrulanabilir.

Plane için ayrıntılı RTX 4060 8 GB inference notu:

- `studies/teacher_reference_bias_v2_512/docs/LOCAL_INFERENCE.md`

Small-vehicle eşleniği:

- `studies/teacher_reference_bias_small_vehicle_v1_512/docs/LOCAL_INFERENCE.md`

Yerel 8 GB VRAM ortamında detector inference batch `1`, SAM inference ise
gerektiğinde `float16` ve küçük bbox compute batch ile çalıştırılmalıdır.
Training sonuçlarını yeniden üretmek zorunlu değildir; `best.pt` ağırlıkları
Git LFS ile taşınır.

## 16. Şu Anda Savunulabilen Ana Çıkarım

İki hedef sınıf birlikte şu sonucu destekliyor:

> Aynı iSAID görüntüsü, aynı bbox ve aynı SAM tahmini yalnızca insan maskesi
> yerine SAM1 pseudo maskesine karşı ölçüldüğünde bütün modellerin skoru
> yükselmiş ve en büyük artışı SAM1 almıştır. Plane koşulunda model sırası da
> değişmiş, small-vehicle koşulunda ise sıra korunmuştur. Bu, model üretimli
> test referansının tarafsız ground truth olmadığını gösterir.

Plane koşulunda model sıralaması da değişti. Small-vehicle koşulunda sıralama
değişmedi, fakat SAM1'in referans değişiminden aldığı `+0,342` IoU artışı
SAM2'nin `+0,103` ve SAM3'ün `+0,049` artışından çok daha büyüktü. Böylece
teacher-reference bias iki farklı hedef sınıfta mutlak skor enflasyonu olarak
tekrarlandı; sıralama değişiminin ise her sınıfta zorunlu olmadığı görüldü.

## 17. Muhtemel Bildiri İskeleti

Altı sayfalık kısa bildiri için önerilen yapı:

1. **Introduction:** Remote sensing segmentasyonunda pahalı insan maskeleri,
   SAM ile otomatik etiketleme eğilimi ve bağımsız değerlendirme riski
2. **Related Work:** SAMRS/SAM tabanlı pseudo etiketleme, model-generated
   benchmark ve evaluation leakage literatürü
3. **Method:** Aynı iSAID tahminini insan/pseudo referansa karşı ölçen eşlenmiş
   tasarım; plane ve small-vehicle tekrarı
4. **Experimental Setup:** 512 görüntü, 4×128 strata, SAM1/2/3, GT/YOLO bbox,
   tek seed 42 ve metrik sözleşmesi
5. **Results:** Skor enflasyonu, model sıralaması değişimi, GT/YOLO bbox farkı
   ve strata analizi
6. **Discussion:** Pseudo etiketlerin kullanım alanı ile bağımsız benchmark
   referansı olma iddiasının ayrılması
7. **Limitations:** İki hedef sınıf, tek pseudo öğretmeni, tek detector seed ve
   resmi tam benchmark yerine kontrollü dengeli test alt kümeleri
8. **Conclusion:** Model-generated test labels require disclosure and an
   independent human-reference audit

## 18. Yerel Codex İçin Kısa Başlangıç Talimatı

Yerel Codex'e şu çerçeve verilebilir:

> `docs/summary/TEACHER_REFERENCE_BIAS_HANDOFF.md` dosyasını, ardından plane
> ve small-vehicle `EXPERIMENT_PLAN.md` dosyalarını oku. Çalışmanın asıl
> nedensel kanıtının aynı iSAID tahminlerinin insan ve SAM1 pseudo referansa
> karşı eşlenmiş değerlendirmesi olduğunu koru. SAMRS skorlarını bağımsız
> insan başarısı olarak yorumlama. Mevcut metrikleri değiştirme, proxy mAP
> ekleme, test setinden model/confidence seçme ve farklı seed sonuçlarını
> ortalama. Önce mevcut manifest, rapor ve Git LFS varlıklarını doğrula; sonra
> bildiri anlatısını bu kanıtlara dayanarak geliştir.
