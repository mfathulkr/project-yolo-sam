# iSAID Small Vehicle Human Reference Full Metric Document

## Scope

- Veri seti iSAID, hedef sınıf Small_Vehicle ve model giriş çözünürlüğü 1024×1024 pikseldir.
- Test kümesi 512 görüntüdür. Dört overlap × mask-area grubunun her birinde tam 128 görüntü vardır.
- Strata tanımı gereği 512 test görüntüsünün tamamında en az bir küçük araç vardır; detector tablosu negatif arka plan görüntülerini içeren resmi tam benchmark değil, bu dengeli pozitif test alt kümesindeki gerçek COCO bbox değerlendirmesidir.
- No Overlap, görüntüdeki hiçbir iki GT bbox'un kesişmemesi; Overlap ise en az bir bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.
- Low/High Mask Area ayrımı, görüntüdeki toplam resmi iSAID insan küçük araç maskesi alanının veri seti için testten önce dondurulan eşiğin altında veya üstünde olmasına göre yapılır.
- SAM1, SAM2 ve SAM3 aynı görüntülerde hem GT bbox hem YOLO bbox istemiyle çalıştırılmıştır.
- YOLO detector her veri setinde ayrıca eğitilmiştir; SAM1, SAM2 ve SAM3 bu veri setlerinde yeniden eğitilmeden veya ince ayar yapılmadan yalnız bbox istemiyle kullanılmıştır.
- Detector protokolü aynı olsa da iSAID eğitim bölümü 5.930, SAMRS eğitim bölümü 7.824 görüntüdür; bu nedenle veri setleri arasındaki detector skoru farkı yalnız referans kaynağına bağlanan kontrollü bir etki değildir.
- YOLO bbox sonuçları deney başlamadan önce sabitlenen seed 42 ile eğitilmiş tek YOLO26x detector sonucudur.
- Maske metrikleri küçük araç örneği düzeyinde hesaplanır; büyük nesneler küçük nesnelerin sonucunu piksel sayısıyla baskılamaz.

## Metric Logic

- TP, modelin doğru biçimde nesne olarak işaretlediği pikseldir. FP, nesne olmadığı hâlde nesne diye işaretlenen; FN ise nesne olduğu hâlde kaçırılan pikseldir.
- IoU = TP / (TP + FP + FN). Tahmin ve referans maskenin ortak alanını birleşim alanına böler; 1 kusursuz, 0 hiç örtüşme yok demektir.
- Dice = 2TP / (2TP + FP + FN). IoU ile aynı davranışı farklı ölçekle ifade eder.
- Precision = TP / (TP + FP). Modelin boyadığı piksellerin ne kadarının gerçekten nesne olduğunu gösterir; fazla alan boyamak precision değerini düşürür.
- Recall = TP / (TP + FN). Gerçek nesne piksellerinin ne kadarının yakalandığını gösterir; eksik maske recall değerini düşürür.
- Dört ortalama maske metriği nesne örneği düzeyinde (instance-level) önce her küçük araç için hesaplanır, sonra bütün örnekler eşit ağırlıkla ortalanır. Büyük nesneler küçük nesnelerin sonucunu perdelemez.
- IoU ≥ 0.50/0.75/0.90 sütunları, ilgili IoU eşiğini geçen küçük araç maskelerinin oranıdır. Bunlar mAP değildir ve raporda mAP gibi adlandırılmaz.
- YOLO'nun kaçırdığı bir gerçek küçük araç, YOLO-bbox maske tablosunda boş tahmin olarak değerlendirilir ve o örneğin maske skorları sıfır olur. Herhangi bir gerçek nesneyle eşleşmeyen yanlış pozitif YOLO kutuları ise instance maske ortalamasına sahte bir referans örneği olarak eklenmez; bunların etkisi detector Precision, Recall ve mAP değerlerinde ölçülür.
- Maske tabloları her GT küçük araç örneğini değerlendirir; YOLO'nun eşleştiremediği GT örnekleri de boş tahmin ve sıfır skorla hesaba katılır. Bu değerlendirme gerçek COCO segmentation AP ile aynı değildir. Confidence sırasındaki bütün maskeleri ve yanlış pozitifleri kullanan uçtan uca COCO mask AP bu raporda ayrıca çalıştırılmadığı için IoU eşik oranları AP veya mAP diye yeniden adlandırılmamıştır.
- Overall tablosu 512 görüntüyü, diğer tabloların her biri 128 görüntüyü kapsar.
- GT-bbox satırları tek sabit koşuldur. YOLO-bbox satırlarındaki değerler sabit seed 42 sonucudur.

## Dataset Context

- iSAID maskeleri insanlar tarafından çizildiği için bu rapor bağımsız insan ground truth sonucudur.
- iSAID veri seti kaynağı: https://captain-whu.github.io/iSAID/
- iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images makalesi: https://arxiv.org/abs/1905.12886
- Bu rapordaki GT bbox, resmi iSAID insan instance anotasyonunda verilen kutudur.
- Bu rapor yalnız SAM1/SAM2/SAM3 ve GT/YOLO bbox koşullarını içerir.
- Detector mAP değerleri bbox ölçümüdür. IoU, Dice, Precision ve Recall ise piksel maskesi ölçümüdür.
- Beş tam SAM1 pseudo referans tablosu ayrı belgede verilmiştir. Bu belgede yalnız aynı tahminlerin referans değişimine duyarlılığını gösteren kısa karşılaştırma özeti bulunur.

## YOLO Detector BBox Metrics

- Bu tablo yalnız YOLO detector kutularını değerlendirir; burada ölçülen bbox başarısıdır, maske başarısı değildir.
- BBox mAP50/mAP75/mAP90, tahmin kutusunun GT kutuyla sırasıyla en az 0,50/0,75/0,90 IoU yaptığı eşiklerde confidence sıralaması boyunca hesaplanan gerçek average precision değeridir.
- BBox mAP50-95, 0,50 ile 0,95 arasındaki on bbox IoU eşiğinin AP ortalamasıdır.
- BBox Precision ve Recall değerleri, doğrulama kümesinde seçilip testten önce sabitlenen güven eşiğinde hesaplanır.
- Tablodaki değerler sabit seed 42 sonucudur.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (seed 42) | 512 | 0.609 | 0.358 | 0.021 | 0.346 | 0.528 | 0.716 | 0.350 | 0.474 | 0.055 | 0.075 |

## İnsan Referansı

Değerlendirme resmi iSAID insan çizimli instance maskelerine karşı yapılmıştır.

### Overall

Referans: İnsan Referansı. Bu tablo 512 görüntüdeki 12.051 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.658 | 0.779 | 0.733 | 0.877 | 0.838 | 0.341 | 0.018 |
| SAM1 YOLO bbox | 512 | 0.478 | 0.564 | 0.529 | 0.635 | 0.607 | 0.259 | 0.014 |
| SAM2 GT bbox | 512 | 0.645 | 0.771 | 0.693 | 0.910 | 0.802 | 0.318 | 0.013 |
| SAM2 YOLO bbox | 512 | 0.461 | 0.552 | 0.499 | 0.647 | 0.575 | 0.223 | 0.008 |
| SAM3 GT bbox | 512 | 0.370 | 0.438 | 0.397 | 0.508 | 0.480 | 0.188 | 0.004 |
| SAM3 YOLO bbox | 512 | 0.299 | 0.354 | 0.323 | 0.405 | 0.392 | 0.153 | 0.003 |

### No Overlap × Low Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 522 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.680 | 0.800 | 0.744 | 0.902 | 0.941 | 0.341 | 0.015 |
| SAM1 YOLO bbox | 128 | 0.466 | 0.549 | 0.510 | 0.616 | 0.644 | 0.226 | 0.008 |
| SAM2 GT bbox | 128 | 0.680 | 0.804 | 0.725 | 0.924 | 0.946 | 0.328 | 0.006 |
| SAM2 YOLO bbox | 128 | 0.451 | 0.539 | 0.486 | 0.624 | 0.634 | 0.178 | 0.000 |
| SAM3 GT bbox | 128 | 0.586 | 0.704 | 0.639 | 0.812 | 0.793 | 0.205 | 0.000 |
| SAM3 YOLO bbox | 128 | 0.385 | 0.465 | 0.419 | 0.538 | 0.523 | 0.128 | 0.000 |

### No Overlap × High Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 1.512 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.753 | 0.853 | 0.807 | 0.928 | 0.975 | 0.574 | 0.060 |
| SAM1 YOLO bbox | 128 | 0.627 | 0.710 | 0.673 | 0.770 | 0.817 | 0.476 | 0.042 |
| SAM2 GT bbox | 128 | 0.774 | 0.869 | 0.819 | 0.940 | 0.982 | 0.674 | 0.048 |
| SAM2 YOLO bbox | 128 | 0.630 | 0.713 | 0.677 | 0.769 | 0.821 | 0.505 | 0.034 |
| SAM3 GT bbox | 128 | 0.730 | 0.826 | 0.780 | 0.893 | 0.942 | 0.608 | 0.026 |
| SAM3 YOLO bbox | 128 | 0.608 | 0.691 | 0.660 | 0.741 | 0.798 | 0.472 | 0.016 |

### Overlap × Low Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 1.582 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.565 | 0.702 | 0.648 | 0.852 | 0.677 | 0.155 | 0.004 |
| SAM1 YOLO bbox | 128 | 0.298 | 0.371 | 0.334 | 0.457 | 0.359 | 0.076 | 0.003 |
| SAM2 GT bbox | 128 | 0.564 | 0.705 | 0.616 | 0.884 | 0.677 | 0.131 | 0.004 |
| SAM2 YOLO bbox | 128 | 0.286 | 0.361 | 0.312 | 0.465 | 0.350 | 0.055 | 0.001 |
| SAM3 GT bbox | 128 | 0.346 | 0.435 | 0.380 | 0.548 | 0.429 | 0.058 | 0.000 |
| SAM3 YOLO bbox | 128 | 0.220 | 0.277 | 0.242 | 0.347 | 0.267 | 0.035 | 0.000 |

### Overlap × High Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 8.435 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.656 | 0.779 | 0.735 | 0.872 | 0.837 | 0.333 | 0.014 |
| SAM1 YOLO bbox | 128 | 0.486 | 0.576 | 0.541 | 0.646 | 0.613 | 0.256 | 0.012 |
| SAM2 GT bbox | 128 | 0.635 | 0.764 | 0.683 | 0.908 | 0.784 | 0.288 | 0.009 |
| SAM2 YOLO bbox | 128 | 0.464 | 0.559 | 0.504 | 0.661 | 0.569 | 0.207 | 0.005 |
| SAM3 GT bbox | 128 | 0.297 | 0.353 | 0.317 | 0.412 | 0.387 | 0.137 | 0.001 |
| SAM3 YOLO bbox | 128 | 0.254 | 0.301 | 0.272 | 0.347 | 0.335 | 0.120 | 0.001 |

## Reference Bias Comparison

Görüntü, küçük araç örneği, bbox ve model tahmini aynıdır; yalnız karşılaştırılan referans maske insan etiketinden SAM1 pseudo etiketine değişir. Bu nedenle fark, referans kaynağına duyarlılığı doğrudan gösterir.

| Model | BBox | Human IoU | SAM1 Pseudo IoU | IoU Artışı |
| --- | --- | --- | --- | --- |
| SAM1 | GT bbox | 0.658 | 1.000 | +0.342 |
| SAM1 | YOLO bbox | 0.478 | 0.655 | +0.177 |
| SAM2 | GT bbox | 0.645 | 0.749 | +0.103 |
| SAM2 | YOLO bbox | 0.461 | 0.550 | +0.089 |
| SAM3 | GT bbox | 0.370 | 0.419 | +0.049 |
| SAM3 | YOLO bbox | 0.299 | 0.341 | +0.041 |

## Qualitative Examples

Her sayfa bir gruptan tek görüntüyü gösterir. Görüntüdeki bütün GT küçük araç kutuları modele ayrı istemler olarak verilmiş ve instance maskeleri yalnız bu görsel için birleştirilmiştir. Tablolar instance-level kalır. Yeşil TP, turuncu FP ve pembe FN piksellerini gösterir.

### No Overlap / Low Mask Area

![No Overlap / Low Mask Area](qualitative/no_overlap__low_mask_area.png)

### No Overlap / High Mask Area

![No Overlap / High Mask Area](qualitative/no_overlap__high_mask_area.png)

### Overlap / Low Mask Area

![Overlap / Low Mask Area](qualitative/overlap__low_mask_area.png)

### Overlap / High Mask Area

![Overlap / High Mask Area](qualitative/overlap__high_mask_area.png)

## Discussion

- İnsan referansında GT-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,658/0,645/0,370 olarak ölçülmüştür.
- İnsan referansında YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,478/0,461/0,299 olarak ölçülmüştür; bunlar sabit seed 42 detector sonuçlarıdır.
- En yüksek insan-referanslı Overall IoU, GT bbox koşulunda SAM1 için 0.658; YOLO bbox koşulunda SAM1 için 0.478 olmuştur.
- GT bbox yerine YOLO bbox kullanıldığında Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla 0,180/0,184/0,071 olmuştur.
- GT-bbox üç-model ortalamasında en yüksek alt grup No Overlap × High Mask Area (0.753), en düşük alt grup Overlap × Low Mask Area (0.492) olmuştur.
- SAM1 gt bbox koşulunda Overall Precision 0.733, Recall 0.877 olmuştur. Recall daha yüksek olduğu için model nesne piksellerini büyük ölçüde yakalarken hedef dışına taşan pikseller precision değerini düşürmektedir.
- GT bbox koşulu segmenter sınır kalitesini daha doğrudan, YOLO bbox koşulu ise detection ve segmentation hatalarının birleşik etkisini gösterir.
- Overlap ve mask-area alt tabloları aynı toplam test kümesinin dengeli, birbirini dışlayan dört parçasıdır; her tabloda 128 görüntü vardır.
- Bu insan etiketli sonuçlar, model kaynaklı pseudo etiket değerlendirmesine karşı birincil bağımsız karşılaştırma noktasıdır.
- GT-bbox model sıralaması insan referansında SAM1 > SAM2 > SAM3, aynı tahminlerin SAM1 pseudo referansında SAM1 > SAM2 > SAM3 biçimindedir.
- Bu sıralamalar tablo nokta tahminleridir; birbirine yakın insan-referanslı skorlar tek başına istatistiksel üstünlük iddiası değildir.
