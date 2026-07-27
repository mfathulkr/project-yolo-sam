# iSAID ve SAMRS İçin Eşlenmiş Teacher-Reference Bias Deney Planı

## 1. Araştırma Problemi

İlk iSAID deneyinde SAM tabanlı segmentation sonuçları görece düşük, SAM ile pseudo mask üretilmiş olduğu düşünülen SAMRS/SOTA deneyinde ise özellikle SAM1 sonuçları çok yüksek görünmüştür. Ancak mevcut iki deney:

- Farklı sınıfları,
- Farklı örnek sayılarını,
- Farklı overlap eşiklerini,
- Farklı YOLO eğitim protokollerini,
- Farklı split stratejilerini,
- Başlangıçta kimliği ayrıca doğrulanması gereken bir SAMRS alt veri setini

birlikte değiştirdiği için bu fark doğrudan pseudo-label etkisine bağlanamaz.

Exhaustive resmi-anotasyon denetimi daha sonra SAMRS SOTA veri kimliğini,
numeric sınıf ID'sini ve RBox/RHBox geometrisini doğrulamıştır. Dolayısıyla
final risk veri kimliği değil; eşlenmemiş protokol, source-scene leakage,
mask-derived bbox ve pseudo referans bağımsızlığıdır.

Bu çalışmanın amacı, yalnızca veri setini veya referans anotasyon kaynağını değiştirerek ölçümün ne kadar değiştiğini kontrollü biçimde göstermektir.

Ana iddia dikkatli kurulacaktır:

> Model tarafından üretilen pseudo maskeler eğitim veya ön eğitim için yararlı olabilir; ancak aynı modelin veya yakın model ailesinin değerlendirme ground truth'u olarak kullanıldığında tarafsız bir benchmark oluşturmayabilir ve skorları yapay biçimde yükseltebilir.

Bu olgu klasik train-test leakage ile tamamen aynı değildir. Bu dokümanda daha doğru terim olarak **teacher-reference bias** veya **generator-reference bias** kullanılacaktır.

## 2. Bildiri İçin Önerilen Başlık

İngilizce çalışma başlığı:

**Grading SAM with SAM: Teacher-Induced Evaluation Bias in Remote-Sensing Segmentation**

Alternatif:

**When Pseudo Labels Become the Benchmark: Measuring SAM-Generated Reference Bias in Aerial Image Segmentation**

Türkçe çalışma karşılığı:

**SAM'i SAM ile Değerlendirmek: Uzaktan Algılama Segmentasyonunda Üretici-Referans Yanlılığı**

## 3. Araştırma Soruları

### RQ1

Aynı prediction, human mask yerine SAM1 tarafından üretilmiş pseudo mask ile değerlendirildiğinde skor ne kadar değişiyor?

### RQ2

Pseudo referansı üreten SAM1, bu referansa karşı SAM2 ve SAM3'ten orantısız biçimde daha fazla avantaj kazanıyor mu?

### RQ3

Model sıralaması human ve pseudo referanslar arasında değişiyor mu?

### RQ4

Teacher-reference bias küçük nesne, büyük nesne, overlap ve no-overlap gruplarında farklı mı?

### RQ5

GT bbox ile segmenter kalitesi izole edildiğinde görülen etki, YOLO bbox kullanılan pratik pipeline'da da korunuyor mu?

### RQ6

Pseudo maskelerin benchmark olarak taraflı olması, eğitim verisi olarak yararsız oldukları anlamına mı geliyor?

RQ6, zaman ve hesaplama bütçesi uygunsa ayrı downstream utility deneyiyle cevaplanacaktır.

## 4. Hipotezler

### H1

SAM tabanlı modeller pseudo reference'a karşı human reference'tan daha yüksek skor alacaktır.

### H2

SAM1'in skor artışı SAM2 ve SAM3'ün skor artışından daha büyük olacaktır.

### H3

Human referansa göre oluşan model sıralaması pseudo referansta değişebilir.

### H4

Bias, küçük ve sınırı belirsiz nesnelerde daha yüksek olacaktır.

### H5

YOLO bbox hatası eklendiğinde segmenter-reference ilişkisi detector hatası tarafından kısmen maskelenecektir.

### H6

Pseudo maskeler bağımsız human test setinde değerlendirilmek şartıyla training/pretraining için yine yararlı olabilir.

## 5. Neden Mevcut Sonuçlar Bildiri Kanıtı Değildir?

Mevcut sonuçlar hipotezi oluşturmak için değerlidir, fakat final sonuç olarak kullanılamaz:

1. iSAID hedefi birleşik small vehicle + large vehicle iken ikinci deneyde plane hedeflenmiştir.
2. İki deneyde sample sayısı ve strata eşikleri farklıdır.
3. YOLO training budget aynı değildir.
4. SAMRS split'inde aynı source scene train ve evaluation'a dağılmıştır.
5. İlk pickle incelemesinde kategori metinleri FAIR1M/FAST benzeri görünmüş,
   fakat resmi detection anotasyonlarıyla exhaustive karşılaştırma numeric
   class ID `4`'ün DOTA `plane` sınıfı olduğunu doğrulamıştır.
6. Eski deneydeki sorun veri kimliği değil; eşlenmemiş split, farklı protokol
   ve bazı bbox'ların pseudo maskeden türetilmesidir.
7. Bbox bazı durumlarda pseudo maskeden türetilmiştir.
8. Mevcut metrikler ağırlıklı olarak image-level merged mask ölçümüdür.

Bu sonuçlar "neden kontrollü deney gerekli?" motivasyonu olarak kullanılabilir, fakat sayısal ana tabloya taşınmamalıdır.

## 6. Veri Setleri

### 6.1 iSAID

iSAID human-annotated aerial instance segmentation referansı olarak kullanılacaktır.

Hedef sınıf:

- `plane`

Plane seçiminin nedenleri:

- Boxy değildir; yönü ve kanat geometrisi vardır.
- Hoca tarafından istenen zor, bbox'a tam oturmayan nesne koşuluna uygundur.
- iSAID ve DOTA/SAMRS tarafında ortak sınıftır.
- Overlap ve farklı mask area örnekleri bulunur.
- Urban/remote-sensing bağlamından uzaklaşmaz.

### 6.2 SAMRS SOTA

Yalnızca resmi ve doğrulanmış SOTA arşivi kullanılacaktır.

Deney başlamadan:

- Resmi indirme kaynağı kaydedilecek.
- Archive hash hesaplanacak.
- Resmi image ve mask sayıları kontrol edilecek.
- Category map annotation içinden doğrulanacak.
- SOTA'nın DOTA kaynak görüntüleriyle ilişkisi kontrol edilecek.
- Mask provenance, kullanılan SAM sürümü ve üretim protokolü makaleden kaydedilecek.

Mevcut yerel `data/samrs_raw/sota` klasörü bu doğrulamaları geçmeden kullanılmayacaktır.

### 6.3 Eşleştirilebilir görüntüler

Güçlü deney, aynı aerial image veya aynı source scene üzerindeki:

- Human iSAID plane maskesi
- SAMRS/SOTA SAM1 pseudo plane maskesi

çiftlerini bulmayı hedefler.

Eşleme sırası:

1. Resmi source image ID
2. Crop/tile koordinat dönüşümü
3. Sınıf eşleşmesi
4. Original bbox geometrisi
5. Bbox IoU ve merkez mesafesi

Eşleşmeyen instance'lar atılacak, sayıları ve nedenleri raporlanacaktır.

## 7. İki Katmanlı Deney Tasarımı

### 7.1 Deney A: Birebir matched dataset replication

Bu deney hocanın "aynı deney, yalnızca veri seti değişsin" talebini karşılar.

Sabit tutulacaklar:

- Hedef sınıf: plane
- Canonical görüntü boyutu
- Source-scene ayrıklığı ve hedef train/validation/test oranları
- YOLO model ailesi ve başlangıç checkpoint'i
- Epoch, batch, optimizer ve augmentation
- Random seed listesi
- SAM checkpoint'leri
- Bbox padding
- Confidence ve NMS threshold
- SAM mask threshold
- Overlap eşiği
- Low/high area tanımı
- Örnek sayısı
- Metrikler
- Tablo düzeni

Değişen tek ana faktör:

- Veri seti: iSAID veya doğrulanmış SAMRS SOTA

Bu deney veri seti düzeyinde gözlenen farkı kontrollü biçimde yeniden üretir.

Ham kaynak split mekanizması veri setlerinin yayın yapısı nedeniyle birebir
aynı değildir: iSAID'in insan etiketli resmi validation sahneleri test havuzu
olarak korunur ve resmi train sahneleri train/validation'a ayrılır; SAMRS SOTA
ise tüm kaynaklarda `source_scene_id` düzeyinde grouped `70/15/15` ayrılır.
İki tarafta da sibling tile sızıntısı sıfır, test örnekleme ve strata kuralı
aynıdır. Buna rağmen bu ham split farkı cross-dataset mutlak skor
karşılaştırmasının bir sınırlılığıdır. Teacher-reference bias için nedensel ana
kanıt, aynı SAMRS görüntü ve tahminlerini insan/pseudo referansla karşılaştıran
ortak görüntü denetimidir.

### 7.2 Deney B: Aynı prediction, iki referans

Bu deney teacher-reference bias iddiasının ana nedensel testidir.

Her matched instance için:

1. Aynı görüntü kullanılır.
2. Aynı original bbox kullanılır.
3. SAM1, SAM2 ve SAM3 birer prediction üretir.
4. Prediction değiştirilmeden human mask ile karşılaştırılır.
5. Aynı prediction pseudo mask ile karşılaştırılır.

Bu tasarımda input, bbox ve prediction sabit; yalnızca evaluator referansı değişir. Böylece ölçülen fark doğrudan reference source etkisidir.

## 8. Ana Pipeline Matrisi

Ana bildiri kapsamı yalnızca detection + segmentation mantığına doğrudan hizmet eden pipeline'ları içerir.

| Pipeline | Bbox kaynağı | Segmenter | Amaç |
|---|---|---|---|
| SAM1 + GT bbox | Original annotation | SAM1 | Oracle localization altında segmenter kalitesi |
| SAM2 + GT bbox | Original annotation | SAM2 | Oracle localization altında segmenter kalitesi |
| SAM3 + GT bbox | Original annotation | SAM3 | Oracle localization altında segmenter kalitesi |
| SAM1 + YOLO bbox | YOLO prediction | SAM1 | Pratik detection + segmentation |
| SAM2 + YOLO bbox | YOLO prediction | SAM2 | Pratik detection + segmentation |
| SAM3 + YOLO bbox | YOLO prediction | SAM3 | Pratik detection + segmentation |

Final protokolde SAM1/SAM2 Hugging Face revision'ları ve üç segmenter'ın
checkpoint SHA-256 değerleri pinlidir. Model adı tek başına provenance kabul
edilmez.

SAM3 çoklu bbox inference çıktıları input sırasına göre bağlanmayacaktır.
Filtrelenmiş/eksik output nedeniyle instance kaymasını önlemek için mask output
bbox'ları prompt bbox'lara global birebir maksimum-IoU atamasıyla eşlenecek;
prompt bbox ile hiç kesişmeyen output reddedilecektir. Final audit, non-empty
her prediction maskesinin kendi prompt bbox'ıyla piksel kesişimi olduğunu
doğrulayacaktır.

### 8.1 Ana karşılaştırmadan çıkarılacaklar

- RingMoSAM
- GroundingDINO + SAM
- SegEarth + SAM
- RemoteSAM text-only
- SAM3 text-only
- SAM3 hybrid text + bbox

Gerekçe:

- Bunlar ana detection + bbox-prompted segmentation sorusuna birebir uymuyor.
- Model ve prompt kaynağına ek confound getiriyor.
- Altı sayfalık bildirinin odağını dağıtıyor.

Text-only sonuçlar korunacaksa yalnızca appendix veya ek analiz olarak sunulacaktır.

## 9. Bbox Kuralları

### 9.1 Reference-independent GT bbox

- Aynı veri setindeki human ve pseudo referans değerlendirmesinde aynı GT bbox
  kullanılacak; referans değişirken prompt değişmeyecek.
- iSAID'de resmi insan instance annotation'ındaki eksen hizalı bbox
  kullanılacak. Bu bbox insan polygonunun resmi envelope alanıdır.
- SAMRS'de özgün DOTA oriented detection kutusunun resmi RHBox envelope'u
  kullanılacak.
- SAM1 pseudo-maskeden yeniden tight bbox türetilmeyecek.
- Bu provenance farkı nedeniyle cross-dataset GT-bbox mutlak skorları
  betimleyicidir; teacher-reference etkisinin nedensel kanıtı sayılmaz.

### 9.2 YOLO bbox

- YOLO yalnızca train split üzerinde eğitilecek.
- Hyperparameter seçimi validation split üzerinde yapılacak.
- Final test prediction threshold'u validation üzerinde dondurulacak.
- Gerçekleştirilen seçim kuralı: her veri seti ve detector seed için
  validation tahminlerinde `bbox IoU = 0.50` altında F1'i en yüksek yapan
  confidence eşiği seçilir. Aynı eşik test detector operating point ve
  YOLO-bbox SAM inference için değişmeden kullanılır.
- Her GT instance, ortak matching kuralıyla en fazla bir prediction ile eşleşecek.
- Eşleşmeyen detector prediction, detector AP/precision hesabında FP olarak
  ve image-level union maskesinde tahmin olarak korunacak; herhangi bir GT
  instance satırına yapay olarak atanmayacak.
- Eşleşmeyen GT için boş maske yazılacak ve instance-level sonuçta sıfır skor
  alarak detector FN etkisi segmentation özetine yansıtılacak.
- Bu nedenle instance-level mask tablosu COCO mask AP değildir. Detector
  object FP etkisi gerçek bbox AP tablosunda, piksel FP etkisi ise ikincil
  image-level union tablosunda ayrıca görülecektir.

### 9.3 Bbox padding

Padding tek sabit yüzde veya piksel kuralıyla uygulanacaktır. Ana deneyde model başına farklı padding ayarlanmayacaktır. Padding etkisi araştırılacaksa ayrı ablation olarak yapılacaktır.

## 10. Split Protokolü

### 10.1 Split birimi

Split birimi `source_scene_id` olacaktır. Aynı büyük aerial scene'in tile'ları farklı split'lere dağıtılamaz.

### 10.2 Önerilen oran

- Train: %70
- Validation: %15
- Test: %15

Nihai oran, plane instance ve strata yeterliliğine göre küçük ölçüde değişebilir; iki veri setinde aynı kural uygulanacaktır.

### 10.3 Test setinin dondurulması

- Test manifest deneyden önce oluşturulur.
- Model veya threshold seçimi için test kullanılmaz.
- Test manifest hash'i final run manifestlerine eklenir.

### 10.4 Tekrarlanan YOLO eğitimi

YOLO en az üç seed ile eğitilecektir:

```text
42
123
2026
```

SAM inference deterministik ayarlarla tek kez çalıştırılabilir; nondeterminism varsa aynı seed politikası uygulanır.

## 11. Örnek Sayısı ve Strata Dengesi

İlk hedef stratum başına 128 örnekti. Ancak iSAID plane audit'inde dar grubun bu sayıyı desteklemediği görülmüştür.

Mevcut feasibility bulgusu:

- `overlap_threshold = 0.001` civarında en dar grup yaklaşık 46 tile.
- Daha yüksek threshold'larda en dar grup daha da küçülmektedir.
- Source-scene-safe split sonrasında sayı azalabilir.

Bu nedenle:

1. İki veri setinde de ortak erişilebilir maksimum `n` hesaplanacak.
2. Tekrarlı veya replacement sampling yapılmayacak.
3. Audit sonucunda güvenli ortak alt sınır olarak stratum başına `32` dondurulmuştur; dört stratum toplamı veri seti başına `128` görüntüdür.
4. Bildiride hem instance sayısı hem source scene sayısı yazılacak.
5. Güven aralıkları source scene seviyesinde clustered bootstrap ile hesaplanacak.

Yüksek sample görüntüsü vermek için bilimsel bağımsızlık bozulmayacaktır.

## 12. Overlap ve Mask Area Tanımı

### 12.1 Overlap

Bir evaluation görüntüsü/tile için:

```text
crowding_overlap =
    max IoU(target_bbox_i, target_bbox_j), i != j
```

Final örneklemede `crowding_overlap = 0` ise No Overlap,
`crowding_overlap >= 0.001` ise Overlap olarak atanır. `0` ile `0,001`
arasındaki belirsiz görüntüler evaluation havuzuna alınmaz.

Buradaki IoU:

- Bbox'lar arasındaki geometrik çakışmayı ölçer.
- Segmentation başarısını ölçen mask IoU değildir.
- Stratum oluşturmak için kullanılır.

Overlap threshold bütün veri setlerinde aynı olacaktır. Aday eşik `0.001` olarak değerlendirilecek; final değer veri dağılımı görülüp sonuçlara bakılmadan dondurulacaktır.

### 12.2 Low ve High Mask Area

- Stratum birimi instance değil görüntü/tile'dır.
- Alan oranı, görüntüdeki bütün hedef instance mask alanları toplamının görüntü
  piksel sayısına bölünmesiyle hesaplanır.
- Aynı kural iki veri setinde uygulanır; eşik her veri setinin uygun test
  havuzundaki alan oranı medyanıdır.
- Dondurulmuş eşik iSAID için `0,016714`, SAMRS için `0,011932`'dir.
- iSAID alanı insan maskesinden, SAMRS alanı official SAM1 pseudo-maskeden
  gelir. Bu nedenle cross-dataset dört-strata sonucu betimleyici zorluk
  analizi olarak sunulur; reference etkisinin nedensel ana kanıtı sayılmaz.
- Stratum tablosundaki mask metrikleri yine instance-level hesaplanır ve her
  instance eşit ağırlık alır; aynı görüntüdeki instance'lar aynı görüntü
  stratum'unu paylaşır.

### 12.3 Beş tablo

Her ana sonuç ailesi:

1. Overall
2. No Overlap × Low Mask Area
3. No Overlap × High Mask Area
4. Overlap × Low Mask Area
5. Overlap × High Mask Area

olarak sunulur.

## 13. Metrikler

### 13.1 Birincil instance-level mask metrikleri

#### Mask IoU

Prediction ve reference maskin kesişiminin birleşime oranıdır.

```text
IoU = TP / (TP + FP + FN)
```

#### Dice

Kesişimi iki maskenin toplam alanına göre ölçer.

```text
Dice = 2TP / (2TP + FP + FN)
```

#### Pixel Precision

Modelin plane diye işaretlediği piksellerin ne kadarının gerçekten reference plane maskesinde olduğunu ölçer.

```text
Precision = TP / (TP + FP)
```

#### Pixel Recall

Reference plane piksellerinin ne kadarının prediction içinde yakalandığını ölçer.

```text
Recall = TP / (TP + FN)
```

#### Boundary IoU

Maskenin yalnızca genel alanını değil, sınırlarının ne kadar doğru çizildiğini ölçer. Uçak kanadı ve kuyruk gibi bbox'a tam oturmayan geometriler için önemlidir.

### 13.2 Success oranları

- `Success@IoU50`
- `Success@IoU75`
- `Success@IoU90`

Bu değerler, instance'ların yüzde kaçının ilgili IoU eşiğini geçtiğini gösterir. Bunlar AP veya mAP değildir.

### 13.3 İkincil image-level union metrikleri

Görüntüdeki bütün plane maskeleri birleştirilerek IoU, Dice, precision ve recall hesaplanır. Bu metrikler geçmiş raporla bağlantı kurar, fakat ana bilimsel sonuç instance-level olacaktır.

### 13.4 Detector metrikleri

YOLO için:

- Bbox Precision
- Bbox Recall
- AP50
- AP75
- AP50-95

raporlanır. Bunlar mask metriklerinden ayrı tabloda yer alır.

## 14. Teacher-Reference Bias Ölçüleri

### 14.1 Reference inflation

Her model ve metric için:

```text
Inflation =
    Score(pseudo_reference) - Score(human_reference)
```

Pozitif değer pseudo referansın daha yüksek skor verdiğini gösterir.

### 14.2 Teacher advantage

SAM1 teacher kabul edilirse:

```text
Teacher Advantage =
    Inflation(SAM1) - mean(Inflation(SAM2), Inflation(SAM3))
```

### 14.3 Ranking reversal

Model sıralaması human ve pseudo referanslarda ayrı çıkarılacaktır.

Raporlanacaklar:

- Sıra tablosu
- Spearman rank correlation
- Kendall tau
- Sıra değişikliği sayısı

### 14.4 Instance-level fark dağılımı

Her instance için:

```text
delta_i =
    metric_i(pseudo_reference) - metric_i(human_reference)
```

Histogram, violin/box plot ve strata kırılımı üretilecektir.

## 15. İstatistiksel Analiz

### 15.1 Confidence interval

- Source scene clustered bootstrap
- En az 10.000 bootstrap örneklemi
- %95 confidence interval

### 15.2 Paired karşılaştırma

Aynı prediction iki referansla değerlendirildiği için testler paired olacaktır.

Öneriler:

- Bootstrap confidence interval of paired differences
- Eşlenmiş model farkları için kaynak-sahne ortalamaları üzerinde Wilcoxon
  signed-rank testi
- Aynı karşılaştırma ailesindeki üç model çifti için Holm çoklu-test düzeltmesi
- Wilcoxon signed-rank, varsayımlar uygunsa
- Etki büyüklüğü ve median fark

P-value tek başına yorumlanmayacaktır.

### 15.3 Çoklu karşılaştırma

Çok sayıda model, metric ve stratum için:

- Ana hipotezler önceden belirlenir.
- Secondary analizler exploratory olarak işaretlenir.
- Gerekirse Holm düzeltmesi uygulanır.

### 15.4 YOLO seed varyansı

- Her seed için detector ve downstream mask sonucu ayrı tutulur.
- Ortalama ± standart sapma raporlanır.
- Ana conclusion tek bir şanslı seed'e dayanmaz.

## 16. Optional Downstream Utility Deneyi

Bu deney, "pseudo benchmark yanlı olabilir" sonucunun "pseudo training data yararsızdır" şeklinde yanlış yorumlanmasını engeller.

Tasarım:

1. Aynı hafif segmentation modeli iki kez eğitilir.
2. Bir eğitim human maskelerle yapılır.
3. Diğer eğitim SAM1 pseudo maskelerle yapılır.
4. İki model de yalnızca bağımsız human-annotated test setinde değerlendirilir.
5. Eğitim sample sayısı, augmentation ve budget eşit tutulur.

Bu deney ana altı sayfalık bildiride yer yoksa appendix veya gelecek çalışma olarak bırakılabilir.

## 17. Görsel İnceleme

Her stratum için sabit sayıda örnek seçilecektir:

- En yüksek human IoU
- En düşük human IoU
- En yüksek pseudo-human delta
- En düşük pseudo-human delta
- Model sıralamasının değiştiği örnekler

Her görselde:

- Original image
- Original bbox
- YOLO bbox, ilgiliyse
- Human mask
- Pseudo mask
- SAM1 prediction
- SAM2 prediction
- SAM3 prediction

aynı koordinat sisteminde gösterilecektir.

Görsel seçimi yalnızca başarılı örneklere dayanmayacaktır.

## 18. Uygulama Aşamaları

### Aşama 1: Veri doğrulama

- [x] Resmi iSAID kaynağını ve checksum'larını kaydet.
- [x] Gerçek SAMRS SOTA arşivini resmi kaynaktan edin.
- [x] SOTA image, mask ve annotation sayısını doğrula.
- [x] Sınıf eşlemesini annotation içinden doğrula.
- [x] Plane instance örneklerini görsel olarak kontrol et.
- [x] Pseudo mask üreticisi ve checkpoint provenance'ını kaydet.

Çıkış kapısı: Veri seti kimliği doğrulanmadan sonraki aşamaya geçilmez.

### Aşama 2: Common plane corpus

- [x] iSAID plane instance manifesti oluştur.
- [x] SAMRS SOTA plane instance manifesti oluştur.
- [x] Original bbox'ları çıkar.
- [x] Mask-derived bbox kayıtlarını reddet.
- [x] İnsan referans maskelerini kayıpsız COCO RLE ile kodla; decoded area ve
  boş maske denetimini zorunlu tut.
- [x] Source scene ID'lerini üret.
- [x] Plane alt sınıf birleştirmelerini belgeleyip test et.

Çıkış kapısı: Rastgele seçilen örneklerde image, bbox, mask ve class birlikte doğru görünür.

### Aşama 3: Source-scene split ve feasibility

- [x] Group-aware split üret.
- [x] Train/validation/test kesişimini doğrula.
- [x] Dört strata sayısını hesapla.
- [x] Ortak `n` değerini dondur.
- [x] Overlap ve area eşiklerini dondur.
- [x] Evaluation manifest hash'ini kaydet.

Çıkış kapısı: Her stratum yeterli örneğe sahip ve source scene leakage sıfırdır.

### Aşama 4: YOLO eğitimi

- [x] Aynı YOLO config'ini iki veri setinde çözümle.
- [x] Config diff kontrolü yap.
- [x] Üç seed ile eğit.
- [x] Validation üzerinden threshold seç.
- [x] Test detector metriklerini üret.
- [x] Prediction manifestlerini dondur.
- [x] Eğitim geçmişi ve sonlu olmayan değer denetimini otomatikleştir.
- [x] Altı koşulun 100 epok ve sonlu final metriklerle tamamlandığını doğrula.

Çıkış kapısı: Her run tamamlanmış manifest ve checkpoint hash'i içerir.

Detector eğitim manifesti yalnız train/validation görüntüleri, YOLO bbox
label'ları ve `data.yaml` dosyasını kapsayan scoped content tree hash'ine
bağlanır. Test split'i ve segmentation RLE'si bu eğitim girdisine dahil
edilmez. Başlangıç ve bitiş input hash'leri ayrı tutulur; drift varsa run final
kanıt olarak kabul edilmez. Şema eklenmeden önce başlatılmış pahalı detector
run'ları yalnız özgün manifest arşivi ve açık repair audit'i ile taşınabilir.
Diğer pipeline aşamaları onarılmaz, yeniden çalıştırılır.

### Aşama 5: SAM inference

- [x] SAM1 + GT bbox
- [x] SAM2 + GT bbox
- [x] SAM3 + GT bbox
- [x] SAM1 + YOLO bbox
- [x] SAM2 + YOLO bbox
- [x] SAM3 + YOLO bbox
- [x] Empty ve failed prediction audit
- [x] Non-empty her maskenin kendi prompt bbox'ıyla piksel kesişimini doğrula.

Çıkış kapısı: Bütün modeller aynı instance listesi ve ortak prediction schema ile çalışmıştır.

### Aşama 6: Dual-reference evaluation

- [x] Human reference metriklerini hesapla.
- [x] Pseudo reference metriklerini hesapla.
- [x] Instance-level paired delta üret.
- [x] Overall ve dört strata tablolarını üret.
- [x] Ranking ve inflation analizini üret.
- [x] Bootstrap confidence interval hesapla.

Çıkış kapısı: Aynı prediction hash'inin iki reference evaluation'ında değişmediği doğrulanır.

### Aşama 7: Rapor ve bildiri

- [x] Dataset audit tablosu
- [x] Detector tablosu
- [x] GT bbox segmentation tablosu
- [x] YOLO bbox segmentation tablosu
- [x] Human-pseudo inflation tablosu
- [x] Ranking karşılaştırma tablosu
- [x] Strata tabloları
- [x] Representative görseller
- [x] Limitations
- [x] Reproducibility appendix
- [x] Canonical instance, aggregate ve detector koşul matrisi kalite kapısı

Çıkış kapısı: Her tablo canonical metrics dosyasından otomatik yeniden üretilebilir.

## 19. Sonuç Tablosu Şablonları

### 19.1 GT bbox, human reference

| Model | N | IoU | Dice | Precision | Recall | Boundary IoU | Success@50 | Success@75 | Success@90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 19.2 YOLO bbox, human reference

| Model | N | IoU | Dice | Precision | Recall | Boundary IoU | Success@50 | Success@75 | Success@90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 19.3 Reference inflation

| Model | Bbox | Human IoU | Pseudo IoU | Delta | %95 CI | Human Rank | Pseudo Rank |
|---|---|---:|---:|---:|---:|---:|---:|

### 19.4 YOLO detector

| Dataset | Seed | Precision | Recall | AP50 | AP75 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|

Her tablonun üzerinde instance sayısı ve benzersiz source scene sayısı yazılacaktır.

## 20. Altı Sayfalık Bildiri Taslağı

### 1. Introduction

- Remote-sensing küçük nesne segmentation problemi
- Pseudo label üretiminin yaygınlaşması
- Generator ile benchmark reference arasındaki bağımsızlık sorunu
- Katkılar

### 2. Related Work

- SAM ve aerial segmentation
- SAMRS ve pseudo-mask data generation
- Pseudo-label noise ve confirmation bias
- Benchmark annotation error ve test-set validity

### 3. Methodology

- iSAID ve SAMRS/SOTA
- Matched plane instances
- Source-scene split
- SAM1/2/3 ve bbox koşulları
- Human vs pseudo dual-reference protocol
- Metrikler ve istatistik

### 4. Results

- Matched dataset replication
- Reference inflation
- Teacher advantage
- Ranking reversal
- Overlap ve area strata
- GT bbox vs YOLO bbox

### 5. Discussion and Limitations

- Pseudo labels training için yararlı olabilir
- Pseudo labels bağımsız benchmark yerine geçmemeli
- Matching coverage ve annotation uncertainty
- Model-family bias
- Aynı YOLO ayarları ve epok sayısına karşın farklı train görüntü sayılarının
  farklı optimizer adımı üretmesi; cross-dataset detector farkının yalnız
  veri zorluğuna bağlanamaması
- Yalnızca plane sınıfı ve aerial domain sınırı

### 6. Conclusion

- Bağımsız human test reference gerekliliği
- Dataset dokümantasyonunda generator provenance gerekliliği
- Pseudo-label benchmarklar için önerilen audit protokolü

## 21. Beklenen Bilimsel Katkılar

1. Remote-sensing segmentation'da SAM-generated reference bias için kontrollü ölçüm.
2. Aynı prediction'ı human ve pseudo referansla değerlendiren dual-reference protokol.
3. Teacher modelin kendi pseudo etiketine karşı avantajını ölçen inflation ve teacher advantage metrikleri.
4. Source-scene leakage'i önleyen tekrar üretilebilir split protokolü.
5. Overlap ve mask area koşullarında bias davranışının analizi.
6. Pseudo-label training utility ile pseudo-label benchmark validity kavramlarının açık ayrımı.

## 22. İddia Sınırları

Bildiri şu iddiaları yapmayacaktır:

- "AI ile etiketlenmiş bütün veriler işe yaramaz."
- "SAMRS değersiz bir veri setidir."
- "Yüksek SAMRS skoru yalnızca cheating sonucudur."
- "SAM1 her durumda SAM2 ve SAM3'ten iyidir."

Savunulacak daha dar ve kanıtlanabilir sonuç:

> Pseudo-mask generator ile değerlendirilen model aynı veya yakın model ailesindeyse, referans maskesi model sıralamasını ve mutlak skoru etkileyebilir. Bu nedenle pseudo maskeler, bağımsız human audit veya human test seti olmadan tarafsız ground truth kabul edilmemelidir.

## 23. Başarısızlık Durumları ve Alternatifler

### Gerçek SOTA arşivi bulunamazsa - çözüldü

Bu fallback'e gerek kalmamıştır. Resmi arşiv hash'i, `17.555` dosya,
`615.407` instance, numeric class ID ve RBox/RHBox geometrileri exhaustive
olarak doğrulanmıştır. Kontrollü iSAID SAM1 pseudo-reference deneyi yine de
bağımsız bir iç kontrol olarak korunmuştur.

### iSAID ve SOTA instance'ları güvenilir eşleşmezse - kısmen çözüldü

SAMRS testindeki `128` görüntünün `126` tanesi piksel düzeyinde iSAID
kaynaklarına eşlenmiş, `1.375` plane instance'ın `1.033` tanesi bağımsız insan
maskesiyle birleştirilmiştir. SAMRS'nin örtüşen tile yapısı nedeniyle bu
`1.033` tile-instance görünümü `770` benzersiz insan-anotasyonlu uçağa aittir.
Ana analiz kaynak-sahne düzeyinde kümelenir; ayrıca aynı uçağın crop
görünümleri önce kendi içinde ortalanarak benzersiz nesne ağırlıklı duyarlılık
analizi raporlanır. Eşlenemeyen iki kaynak ve dışarıda kalan instance'lar
limitation olarak raporlanır; kontrollü iSAID pseudo-reference deneyi de
korunur.

### Dört strata için yeterli örnek yoksa - çözüldü

Replacement sampling yapılmadan ortak değer stratum başına `32`, veri seti
başına toplam `128` görüntü olarak dondurulmuştur.

### Hesaplama bütçesi sınırlıysa

Öncelik sırası:

1. GT bbox + SAM1/2/3 dual-reference
2. YOLO bbox + SAM1/2/3
3. Üç YOLO seed
4. Downstream utility deneyi

## 24. Deney Tamamlanma Tanımı

Deney aşağıdaki koşullar sağlandığında bildiride kullanılabilir:

- [x] İki veri setinin kimliği ve annotation provenance'ı doğrulanmış.
- [x] Ortak hedef sınıf plane.
- [x] Source-scene leakage sıfır.
- [x] Ortak sample ve strata protokolü kullanılmış.
- [x] YOLO config farkı yalnızca veri yolu ve izin verilen dataset alanları.
- [x] Reference-independent GT bbox kullanılmış; SAM1 pseudo-mask-derived bbox
  yok ve iki veri setindeki bbox provenance farkı açıkça belgelenmiş.
- [x] SAM1, SAM2 ve SAM3 aynı instance ve bbox'larla çalışmış.
- [x] Human ve pseudo reference aynı prediction üzerinde değerlendirilmiş.
- [x] Instance-level metrikler ana sonuç.
- [x] Detector AP metrikleri ayrı.
- [x] En az üç YOLO seed raporlanmış.
- [x] Scene-clustered %95 confidence interval verilmiş.
- [x] Model sıralaması iki referansta karşılaştırılmış.
- [x] Overall ve dört strata sonucu mevcut.
- [x] Representative failure ve success görselleri mevcut.
- [x] Bütün final sonuçlar run manifest ve input hash'lerine bağlı.
- [x] Ana iddia verinin training utility'si ile benchmark validity'sini karıştırmıyor.
