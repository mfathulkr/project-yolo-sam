# Kritik Uygulama Kararları ve Tekrar Üretim Rehberi

Bu belge, teacher-reference-bias çalışmasında sonucu değiştirebilecek fakat
yalnız kısa bir config adına bakınca anlaşılmayan bütün kararları tek yerde
sabitler. Amaç, başka bir araştırmacının aynı bilimsel deneyi kurabilmesi ve
yanlışlıkla farklı bir deney çalıştırmamasıdır.

Kanonik çalışma üç dizine yayılır:

- `teacher_reference_bias_v2_512`: Plane için insan, SAM1 pseudo ve SAMRS
  deneyleri;
- `teacher_reference_bias_small_vehicle_v1_512`: Small Vehicle için aynı
  protokol;
- `teacher_reference_bias_multiteacher_v1_512`: aynı iSAID tahminlerini insan,
  SAM1, SAM2 ve SAM3 referanslarıyla çaprazlayan bildiri analizi.

## 1. En Kısa Deney Tanımı

Her iSAID hedef nesnesi için SAM1, SAM2 ve SAM3'e bir bbox verilir ve bir maske
üretilir. Bu tahmin maskeleri dondurulur. Sonra aynı tahminler sırasıyla insan,
SAM1, SAM2 ve SAM3 referans maskelerine karşı ölçülür. Böylece değişen şey
model tahmini değil, yalnız cetvel olarak kullanılan referans maskedir.

İki bbox koşulu vardır:

- `GT bbox`: insan anotasyonundaki gerçek nesne kutusu;
- `YOLO bbox`: yalnız eğitim verisiyle eğitilmiş detector'ın testte bulduğu
  kutu.

GT-bbox diagonal hücresi, bir modelin kendi çıktısını yine kendi çıktısıyla
karşılaştırır. Bu bir başarı sonucu değil, `identity control` yani veri
soy-zincirinin doğru kurulduğunu gösteren kimlik kontrolüdür. Bilimsel ana
kanıt, kutu ve maske birebir aynı olmadığı halde etkinin sürdüğü YOLO-bbox
koşuludur.

## 2. Terimler Sözlüğü

### BBox ve istem

`BBox`, nesnenin çevresindeki eksenlere paralel dikdörtgen kutudur. `Prompt`
veya Türkçe metindeki karşılığıyla istem, SAM'e "şu kutudaki nesnenin maskesini
çıkar" demektir. Çalışmada nokta, metin, oriented bbox veya mask prompt
kullanılmaz.

### Instance

`Instance`, tek bir nesne örneğidir. Aynı görüntüde 20 uçak varsa 20 ayrı
instance vardır. Her biri kendi kutusuyla segmentlenir ve kendi maskesiyle
ölçülür. Görüntüdeki nesneler tek bir birleşik maske gibi ana metrikte
değerlendirilmez.

### Teacher ve candidate

`Teacher`, pseudo referans maskeyi üreten modeldir. `Candidate`, bu referansa
karşı puanlanan modeldir. Aynı SAM modeli bir tabloda teacher, başka bir
hücrede candidate olabilir.

### Pseudo referans

İnsan eliyle çizilmiş maske yerine, bir modelin ürettiği ve sonradan
değiştirilmeden saklanan maskedir. Bu çalışmada bütün kontrollü SAM1/SAM2/SAM3
pseudo referansları insan GT bbox istemiyle üretilir. YOLO bbox ile pseudo
referans üretilmez.

### PVS

`PVS`, **Promptable Visual Segmentation** kısaltmasıdır. Türkçesi yaklaşık
olarak "görsel istemle belirli nesne segmentasyonu"dur. Kutu, nokta veya maske
gibi uzamsal bir istem verilir; model her istem için o belirli nesnenin
maskesini döndürür. SAM1 ve SAM2'nin kutu kullanımına denk davranış budur.

Bu çalışmada SAM3 PVS yolu:

1. `Sam3TrackerProcessor` görüntüyü ve bbox listesini hazırlar.
2. `Sam3TrackerModel` aynı checkpoint'i PVS davranışıyla çalıştırır.
3. `multimask_output=False` ile her bbox için tek aday maske istenir.
4. Logit değeri `0,0` veya üzerindeki pikseller maskeye alınır.
5. Dönen maske sayısının verilen bbox sayısına eşit olması zorunludur.

İlgili kod:

- `src/yolo_sam/models/sam3_tracker_local.py`
- `src/yolo_sam/segmentation/box_segmenters.py`
- `src/yolo_sam/segmentation/factory.py`

Resmi Transformers dokümantasyonu da PVS'yi her görsel istem için belirli bir
nesne örneği segmentasyonu olarak tanımlar:
https://huggingface.co/docs/transformers/model_doc/sam3_tracker

### PCS

`PCS`, **Promptable Concept Segmentation** kısaltmasıdır. Türkçesi "istemle
kavram segmentasyonu"dur. Metin veya görsel örnekle "bu kavrama benzeyen bütün
nesneleri bul" görevi çözülür. Örneğin bir uçak kutusunu görsel örnek olarak
vermek, yalnız o uçağı segmentle anlamına gelmeyebilir; model görüntüde aynı
kavrama uyan başka uçakları da arayabilir.

Bu nedenle `Sam3Model` + visual-exemplar PCS yolu, SAM1/SAM2 bbox-only
deneyinin eşdeğeri değildir. İlk tarihsel uygulama bu yolu kullandığı ve ayrıca
`0,5` çıktı olasılığı filtresi uyguladığı için küçük nesnelerde çok sayıda boş
maske üretmişti. O SAM3 çıktıları bilimsel kullanımdan çıkarıldı. Tarihsel PCS
kodları yanlışlıkla yeniden kullanılmasın diye açık hata verir:

- `src/yolo_sam/pipelines/gt_box_sam3.py`
- `src/yolo_sam/pipelines/yolo_sam3.py`

SAM3 makalesi PCS'yi kısa isim, görsel örnek veya ikisinin birleşimiyle eşleşen
bütün nesneleri bulma görevi olarak tanımlar:
https://arxiv.org/abs/2511.16719

### Identity control

Bir modelin dondurulmuş GT-bbox maskesinin, aynı maskeden oluşturulan pseudo
referansa karşı ölçülmesidir. Dolu maskede matematiksel olarak skor `1,0`
olmalıdır. Bu sonuç "model insan gerçeğini kusursuz buldu" anlamına gelmez.
Yalnız aynı verinin iki tarafta gerçekten aynı olduğunu doğrular.

### Known-positive empty

İnsan anotasyonundan nesnenin var olduğunu bildiğimiz halde teacher'ın GT bbox
verildiğinde boş maske üretmesidir. Bu gerçek negatif değil, eksik pseudo
etikettir. Tahmin de boş olsa bile başarı verilmez; bütün maske metrikleri
`0,0` olur.

### Source scene

iSAID'daki büyük özgün hava görüntüsüdür. Bir source scene daha sonra çok sayıda
1024 x 1024 tile'a bölünebilir. Aynı özgün sahnenin tile'ları birbirine çok
benzediği için split ve bootstrap işlemlerinde kaynak sahne kimliği korunur.

## 3. Dondurulmuş Bilimsel Sözleşme

| Karar | Kanonik değer |
|---|---|
| Hedefler | iSAID `plane` ve `Small_Vehicle` |
| Görüntü boyutu | 1024 x 1024 tile |
| Test görüntüsü | Hedef başına 512 |
| Tabaka | 4 tabaka x 128 görüntü |
| Modeller | SAM1 ViT-H, SAM2.1 Hiera-Large, SAM3 |
| BBox koşulları | insan GT bbox, seed-42 YOLO bbox |
| Referanslar | human, pseudo-SAM1, pseudo-SAM2, pseudo-SAM3 |
| Ana metrik | instance-macro ortalama IoU |
| Detector seed | 42 |
| Bootstrap | kaynak sahne kümeli 10.000 tekrar, yüzde 95 güven aralığı |
| Maske eşiği | bütün SAM modellerinde logit `0,0` |
| Multimask | kapalı; bbox başına tek maske |

## 4. Veri Hazırlama ve Örnekleme

### 4.1 Neden 1024 x 1024?

iSAID çalışma girdileri zaten 1024 x 1024 tile'lardır. Pipeline sahneyi yeniden
crop etmez veya küçük nesneyi elle büyütmez. PIL RGB görüntü doğrudan modelin
kendi processor sınıfına verilir. Processor kendi modelinin gerekli normalize
ve dahili resize işlemini uygular; bbox koordinatları tekrar özgün tile
koordinatlarına bağlanır.

Bu karar SOPSeg tarzı region-adaptive magnification çalıştırmadığımız anlamına
gelir. Sonuçlar frozen SAM + bbox baseline'ıdır; crop/magnification etkisiyle
karıştırılmaz.

### 4.2 Kaynak sahne güvenli split

Split birimi tile değil source scene'dir. Aynı büyük görüntüden kesilmiş iki
tile train ve test arasında paylaşılmaz. Test için önce dört tabakayı
doldurabilecek kaynak sahneler deterministik seçilir, sonra her tabakadan tam
128 görüntü seed 42 ile alınır. Kalan sahneler train ve validation'a ayrılır.

Denetim kaynakları:

- `data/prepared/*/source_scene_split.csv`
- `data/prepared/*/master_provenance.json`
- `data/prepared/*/content_manifest.json`

### 4.3 Overlap ne demektir?

Overlap, instance maskelerinin değil GT bbox çiftlerinin kesişimine göre
tanımlanır:

- `No Overlap`: görüntüdeki bütün GT bbox çiftlerinin IoU değeri tam `0`;
- `Overlap`: en az bir bbox çiftinin IoU değeri `0,001` veya daha büyük;
- aradaki sayısal belirsiz bölge test seçimine alınmaz.

Eşik çok küçüktür; amaç iki kutunun en ufak gerçek kesişimini bile kalabalık
sahne olarak ayırmaktır. Bu değişken "nesne maskeleri gerçekten birbirinin
üstüne bindi" diye yorumlanmamalıdır.

### 4.4 Low/High Mask Area ne demektir?

Her görüntüdeki hedef instance maskelerinin piksel alanları toplanır ve tile
alanına bölünür. Bu bir union alanı değildir; iki instance maskesi ortak piksel
içerirse o piksel iki instance'ın alanında ayrı sayılabilir.

Model çalıştırılmadan önce dondurulan iSAID eşikleri:

- Plane: `0,0167140960693359`;
- Small Vehicle: `0,0018463134765625`.

Eşikler insan/dataset-native anotasyona göre belirlenir. SAM1/SAM2/SAM3 pseudo
referansı değiştiğinde bir görüntünün tabakası değiştirilmez. Aksi halde hem
referans hem örnek grubu birlikte değişmiş olurdu.

### 4.5 Bütün nesneler kullanılır

Seçilen 512 görüntüdeki tek bir örnek nesne alınmaz. Plane için 5.447, Small
Vehicle için 12.051 instance'ın tamamı çalıştırılır. Nitel figürde ayrı
instance maskeleri yalnız görüntüyü okunabilir göstermek için union edilir;
tablo hesabı her instance için ayrıdır.

### 4.6 SAMRS SOTA-RBB özel durumu

SAMRS deneyleri dört referanslı ana iSAID matrisinin yerine geçmez; yayımlanmış
SAM1 kaynaklı pseudo maskelerdeki davranışı dış bağlam olarak gösterir. SAMRS
GT-bbox istemi pseudo maskeden yeniden hesaplanmaz. Yayımlanan detection
anotasyonundaki `rhbox` alanı `[xmin, ymin, xmax, ymax]` olarak okunur, COCO
`[x, y, width, height]` biçimine çevrilir ve gerekiyorsa 1024 x 1024 görüntü
sınırına kırpılır. `rhbox` adı burada rotated/oriented kutu kullanıldığı
anlamına gelmez; segmentere verilen istem eksenlere paralel horizontal bbox'tır.

Bir SAMRS instance'ında özgün detection kutusu yoksa maskeden kutu türetilmez;
hazırlama hata verir. Bu koruma, teacher maskesinin sınırlarından kusursuz kutu
çıkarıp segmentere haksız bilgi vermeyi engeller. SAMRS resmi maskeleri ayrı bir
SAM1 üretim hattından geldiği için kontrollü iSAID SAM1 identity hücresi kadar
byte-level özdeşlik beklenmez ve sonuç insan doğruluğu diye sunulmaz.

## 5. Detector Sözleşmesi

### 5.1 Eğitim

Her hedef için ayrı YOLO26x detector sıfırdan değil aynı `yolo26x.pt`
başlangıç ağırlığından fine-tune edilir:

- seed `42`;
- 1024 x 1024 giriş;
- en çok 100 epok;
- training batch 12;
- patience 30;
- optimizer `auto`;
- `deterministic=True`;
- minimum instance alanı 8 piksel.

Small Vehicle validation batch'i de 12'dir. Ultralytics'in validation batch'ini
iki katına çıkaran varsayılan davranışı yoğun sahnelerde bellek taşmasına yol
açtığı için kapatılmıştır. Bu yalnız aynı görüntülerin kaç parçada işlendiğini
değiştirir; ağırlık güncellemesini veya metriği değiştirmez.

### 5.2 İki confidence eşiği neden var?

`evaluation_confidence_threshold=0,001`, detector'ın validation/test için çok
geniş bir aday listesi çıkarmasını sağlar. Sonra validation kümesinde bbox IoU
0,50 kabul kuralıyla F1'i en yüksek yapan confidence seçilir. Test ve SAM
inference bu sabit eşiği kullanır:

- Plane: `0,28115004301071167`;
- Small Vehicle: `0,2740148901939392`.

`confidence_threshold=0,20` config'teki başlangıç/operasyon değeridir; kanonik
YOLO-bbox değerlendirmesinin nihai eşiği validation'da seçilen yukarıdaki
değerdir. Test sonucuna bakarak eşik ayarlanmaz.

Detector inference ayrıca `NMS IoU=0,70` ve görüntü başına en çok 500 detection
kullanır. NMS, aynı nesne üzerine yığılmış benzer kutuları eler.

### 5.3 YOLO kutusu GT instance'a nasıl bağlanır?

1. Detection'lar confidence yüksekten düşüğe sıralanır.
2. Her detection, henüz eşleşmemiş GT kutular arasından bbox IoU'su en yüksek
   olana aday olur.
3. IoU `0,50` veya üzerindeyse bire bir eşleşme kabul edilir.
4. Bir GT ve bir detection en fazla bir eşleşmede kullanılabilir.
5. Eşleşmeyen GT instance `missing_bbox` olur ve sıfır maske alır.
6. Eşleşmeyen detector kutusu false positive olarak ayrı dosyada saklanır.

Bu greedy eşleştirme Hungarian/global-optimum eşleştirme değildir. Yöntem
değiştirilirse YOLO-bbox segmentasyon satırları da değişebilir.

False-positive kutular ana instance-macro maske ortalamasına sahte bir GT
nesnesi olarak eklenmez. Bunların cezası detector precision/recall/AP'de ve
ikincil image-union analizinde görünür. Bu nedenle mask IoU tablosu tek başına
uçtan uca detector AP değildir.

## 6. SAM1, SAM2 ve SAM3 Sözleşmesi

### 6.1 Ortak davranış

- Modeller iSAID üzerinde fine-tune edilmez; frozen inference yapılır.
- Görüntü RGB olarak model processor'ına verilir.
- BBox biçimi `xyxy`dir.
- `multimask_output=False`: her kutu için tek maske.
- `mask_threshold=0,0`: sıfır ve üzeri logit foreground kabul edilir.
- Çıktı maskesi processor ile özgün 1024 x 1024 boyuta taşınır.
- Verilen bbox sayısı ile dönen maske sayısı eşit değilse koşu hata verir.

### 6.2 Checkpoint kimlikleri

| Model | Kimlik / revision | Checkpoint SHA-256 |
|---|---|---|
| SAM1 | `facebook/sam-vit-huge`, `87aecf0df4ce6b30cd7de76e87673c49644bdf67` | `edfb0462392541fca9af44ff039bfb32dbd0c939997f3abb77a26e23af7afd7c` |
| SAM2 | `facebook/sam2.1-hiera-large`, `665f8e2ad61cf5f53d65644ff27c8ee525124610` | `dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230` |
| SAM3 | local `model.safetensors` | `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a` |

Revision adı yeterli kabul edilmez; dosyanın gerçek SHA-256 değeri inference
öncesi doğrulanır. Processor/config dosyalarının ayrı hash ağacı da
`segmenter_provenance.input.json` içinde saklanır.

### 6.3 Batch püf noktası

SAM1 ve SAM2 kutuları en çok 16'lık parçalarda çalıştırılır. SAM3 A6000
koşusunda 128'lik parça kullanır. Bunlar farklı bilimsel koşullar değildir;
aynı görüntü ve sıralı bbox listesini daha küçük hesap parçalarına ayırarak
bellek kullanımını sınırlar. Instance sırası korunur ve bbox başına tam bir
çıktı zorunludur.

Daha küçük GPU'da batch düşürülebilir. Bilimsel eşdeğerliği kabul etmeden önce
aynı checkpoint, `float32`, aynı mask threshold ve çıktı RLE hash'lerinin
korunduğu doğrulanmalıdır. Kanonik koşular `float32`dir; düşük VRAM için var
olan `float16` yerel profil kanonik sayıları yeniden üretme garantisi vermez.

## 7. Pseudo Referans Üretimi

Her teacher için kaynak dosya, o modelin insan GT bbox ile oluşturduğu frozen
`predictions.jsonl` dosyasıdır. Pseudo referans üreticisi yeniden inference
çalıştırmaz; prediction RLE'sini birebir kopyalar ve soy bilgisi ekler.

Zorunlu kontroller:

- model kimliği doğru teacher ile eşleşir;
- prompt türü `gt_bbox`dır;
- her instance ID benzersizdir;
- instance kümeleri human referansla aynıdır;
- status yalnız `ok` veya `empty_mask` olabilir;
- status ile gerçek RLE piksel alanı uyumludur;
- pseudo RLE ile kaynak prediction RLE byte düzeyinde aynıdır;
- kaynak prediction, manifest ve config hash'leri saklanır.

Boş pseudo maskeler filtrelenmez. Filtrelemek teacher'ın zorlandığı örnekleri
testten çıkartarak skoru ayrıca şişirirdi.

## 8. Metrik Sözleşmesi

### 8.1 Piksel sayıları

Bir instance için:

- TP: hem tahmin hem referansta foreground olan piksel;
- FP: yalnız tahminde foreground olan piksel;
- FN: yalnız referansta foreground olan piksel.

Sonra IoU, Dice, precision ve recall hesaplanır. Bunlar önce instance başına
hesaplanıp aritmetik ortalanır. Bu `instance-macro` ortalamadır. Bütün
pikselleri önce havuza döken `pixel-micro` hesap değildir; büyük nesneler küçük
nesneleri piksel sayısıyla bastıramaz.

### 8.2 Eşik başarı oranları

`IoU >= 0,50/0,75/0,90`, bu eşiği geçen instance sayısının bütün instance
sayısına oranıdır. Bunlar confidence sıralı precision-recall eğrisi
hesaplamadığı için COCO mask AP veya mAP değildir.

### 8.3 Detector metriği ile maske metriği ayrımı

YOLO tablosundaki bbox mAP50, mAP75, mAP90 ve mAP50-95 gerçek COCO detection
AP'dir. Segmentasyon tablosundaki Avg IoU ise eşleştirilmiş instance maskesinin
geometrik örtüşmesidir. Aynı adla raporlanmaz ve birbirinin yerine geçmez.

### 8.4 Image-union neden ikincildir?

Her görüntüde bütün tahmin maskeleri ve bütün referans maskeleri ayrı ayrı
birleştirilerek ikincil bir image-union metriği de saklanır. YOLO false-positive
maskeleri bu birleşime dahil edilir. Ancak ana araştırma nesne örneği
düzeyindeki teacher affinity olduğu için bildiri sonucu instance-macro IoU'dur.

### 8.5 Boundary IoU

Eski ortak evaluator Boundary IoU hesaplayabilir. Fakat yeni SAM2/SAM3 RLE
uzantısı Boundary IoU üretmediği için dört referanslı bildiri matrisinde bu
metrik raporlanmaz. Eksik hücreleri uydurma veya eski çalışmadan taşıma
yapılmaz.

## 9. İstatistik ve Model Sıralaması

İnsan ve pseudo skor farkı aynı `instance_id` üzerinde eşlenir:

`delta = IoU(pseudo referans) - IoU(insan referansı)`.

Güven aralığında tek tek instance veya tile örneklenmez. Önce source scene
kümeleri seçilir; seçilen sahnenin bütün instance'ları birlikte gelir. 10.000
bootstrap örneği seed 42 ile üretilir ve yüzde 2,5 ile 97,5 quantile'ları yüzde
95 güven aralığını verir.

`Teacher advantage`, teacher modelin kendi pseudo referansındaki ortalama IoU
değeri ile diğer iki aday modelin ortalama IoU değerinin farkıdır. `Ranking`
ise aynı referansta üç modelin Avg IoU değerlerinin büyükten küçüğe
sıralanmasıdır. Eşitlikte model adı deterministik tie-break olarak kullanılır.

## 10. Raporlama Kararları

- Her raporda Overall ve dört tabaka ayrı tablo olarak verilir.
- Başarı rengi yalnız 0 ile 1 arasındaki metrik hücrelerine uygulanır.
- Tabloda olmayan metrik rapor yazarı tarafından elle üretilmez.
- Nitel örnekler tabaka içindeki median zorluğa yakın görüntülerden
  deterministik seçilir; en iyi örnek elle seçilmez.
- Nitel panelde görüntüdeki bütün GT bbox'lar ve bütün hedef instance maskeleri
  gösterilir.
- Rapor crop'u sabit piksel konumuyla değil panel sınırına göre yapılır.
- PDF/DOCX/CSV ve figürlerin SHA-256 değerleri report/paper manifestlerinde
  saklanır.

## 11. Provenance: Hangi Dosya Neyi Kanıtlar?

| Dosya | Kanıtladığı şey |
|---|---|
| `configs/protocol.yaml` | amaçlanan bilimsel parametreler |
| `effective_config.input.json` | koşuda gerçekten kullanılan çözülmüş config |
| `segmenter_provenance.input.json` | checkpoint, processor ve config hash'leri |
| prediction `manifest.json` | run ID, ortam, giriş/çıkış hash'leri, sayaçlar |
| pseudo `.manifest.json` | teacher, GT-bbox kaynağı ve RLE soy zinciri |
| evaluation `manifest.json` | prediction/reference hash'leri ve metrik çıktısı |
| analysis `manifest.json` | bütün metrik küpünün input/output hash'leri |
| report/paper `manifest.json` | kullanıcıya gösterilen varlığın kaynak hash'leri |

Manifestler başlangıçta ve bitişte input fingerprint alır. Koşu sırasında bir
girdi değişirse `input_drift` oluşur. Tamamlanmış bir çıktı `--force` olmadan
ezilmez; aynı dizine iki süreç yazmasın diye writer lock kullanılır.

Kanonik environment manifestlerde kaydedilmiştir: Python 3.12.3, Torch 2.11.0,
Transformers 5.6.2, Ultralytics 8.4.41, NumPy 2.4.4, pycocotools 2.0.11 ve RTX
A6000. `requirements.txt` minimum sürümleri verir; birebir byte-level tekrar
için manifestteki kesin sürümler kullanılmalıdır.

## 12. Tekrar Üretim Sırası

Önce Plane ve Small Vehicle ana çalışmalarında aynı sıra izlenir:

1. `preflight`: config, veri ve gerekli dosyaları denetle.
2. `prepare`: source-scene-safe split ve 4 x 128 test seçimini üret.
3. `validate-prepared`: COCO, YOLO label, instance ve split bütünlüğünü denetle.
4. `model-provenance`: SAM checkpoint/config hash'lerini doğrula.
5. `train-detector`: seed-42 YOLO detector'ı eğit.
6. `detect`: validation threshold seçimini ve test bbox metriklerini üret.
7. `infer`: üç model x iki bbox koşulunu çalıştır.
8. `build-pseudo-reference`: SAM1 GT-bbox çıktısını dondur.
9. `evaluate`: aynı tahminleri human ve SAM1 referansa karşı ölç.
10. `analyze`, `figures` ve full-metric rapor üretimi.

Ana CLI:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
  .venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/study.py --help
```

Small Vehicle için yalnız study yolu ve `PYTHONPATH` içindeki çalışma paketi
değişir. Ardından SAM2/SAM3 referans uzantısı şu sırayla çalıştırılır:

```bash
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/build_pseudo_references.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/evaluate_pseudo_references.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/compile_analysis.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/generate_figures.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/write_full_metric_reports.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/write_teacher_comparison_report.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/generate_paper_assets.py
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/validate_study.py
```

## 13. Tekrar Üretimde Kırmızı Bayraklar

Aşağıdakilerden biri görülürse sonuç kanonik çalışmayla aynı sayılmaz:

- SAM3 için `Sam3Model`/PCS veya visual exemplar kullanılması;
- SAM3'te `output_prob_thresh=0,5` ile aday eleme;
- `multimask_output=True` ve en iyi maskeyi farklı bir kuralla seçme;
- pseudo referansı YOLO bbox ile üretme;
- pseudo maskeden yeniden bbox türetme;
- boş pseudo referansları veri kümesinden çıkarma;
- bilinen pozitif boş-boş çiftine IoU 1,0 verme;
- yalnız görüntü başına tek hedef nesne çalıştırma;
- tabakayı pseudo referans alanına göre yeniden belirleme;
- test confidence eşiğini test skoruna bakarak seçme;
- YOLO false positive'larını gerçek instance gibi maske ortalamasına ekleme;
- instance-macro yerine bütün pikselleri birleştirerek ana skor üretme;
- source-scene yerine tile/instance düzeyinde bağımsız bootstrap yapma;
- detector bbox AP ile maske eşik başarı oranını aynı mAP adıyla verme;
- SAM2/SAM3 uzantısına hesaplanmamış Boundary IoU değeri taşıma;
- checkpoint revision veya SHA-256 uyuşmadan inference çalıştırma.

## 14. Bilinen Sınırlılıklar

- Yalnız iki iSAID hedef sınıfı vardır.
- Detector için tek seed 42 kullanılır; seed varyansı ölçülmez.
- Pseudo referanslar insan GT bbox yardımıyla üretilir.
- İnsan inter-rater maskeleri yoktur; "human" tek resmi iSAID anotasyonudur.
- PVS, crop/magnification ve boundary refinement olmadan frozen baseline'dır.
- Dört referanslı yeni küpte Boundary IoU yoktur.
- GT diagonal kimlik kontrolüdür; bağımsız doğruluk kanıtı değildir.

Bu sınırlılıklar sonuçları geçersiz kılmaz, fakat bildiride iddianın sınırını
belirler: çalışma pseudo etiketlerin eğitim değerini değil, model üretilmiş
referansla model karşılaştırmanın bağımsızlık riskini gösterir.
