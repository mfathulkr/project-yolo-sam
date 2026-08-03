# iSAID Plane Human Reference Full Metric Document

## Scope

- Veri seti iSAID, hedef sınıf plane ve model giriş çözünürlüğü 1024×1024 pikseldir.
- Test kümesi 512 görüntüdür. Dört overlap × mask-area grubunun her birinde tam 128 görüntü vardır.
- Strata tanımı gereği 512 test görüntüsünün tamamında en az bir uçak vardır; detector tablosu negatif arka plan görüntülerini içeren resmi tam benchmark değil, bu dengeli pozitif test alt kümesindeki gerçek COCO bbox değerlendirmesidir.
- No Overlap, görüntüdeki hiçbir iki GT bbox'un kesişmemesi; Overlap ise en az bir bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.
- Low/High Mask Area ayrımı, görüntüdeki toplam resmi iSAID insan uçak maskesi alanının veri seti için testten önce dondurulan eşiğin altında veya üstünde olmasına göre yapılır.
- SAM1, SAM2 ve SAM3 aynı görüntülerde hem GT bbox hem YOLO bbox istemiyle çalıştırılmıştır.
- YOLO detector her veri setinde ayrıca eğitilmiştir; SAM1, SAM2 ve SAM3 bu veri setlerinde yeniden eğitilmeden veya ince ayar yapılmadan yalnız bbox istemiyle kullanılmıştır.
- Detector protokolü aynı olsa da iSAID eğitim bölümü 1.571, SAMRS eğitim bölümü 2.191 görüntüdür; bu nedenle veri setleri arasındaki detector skoru farkı yalnız referans kaynağına bağlanan kontrollü bir etki değildir.
- YOLO bbox sonuçları deney başlamadan önce sabitlenen seed 42 ile eğitilmiş tek YOLO26x detector sonucudur.
- Maske metrikleri uçak örneği düzeyinde hesaplanır; büyük nesneler küçük nesnelerin sonucunu piksel sayısıyla baskılamaz.

## Metric Logic

- TP, modelin doğru biçimde nesne olarak işaretlediği pikseldir. FP, nesne olmadığı hâlde nesne diye işaretlenen; FN ise nesne olduğu hâlde kaçırılan pikseldir.
- IoU = TP / (TP + FP + FN). Tahmin ve referans maskenin ortak alanını birleşim alanına böler; 1 kusursuz, 0 hiç örtüşme yok demektir.
- Dice = 2TP / (2TP + FP + FN). IoU ile aynı davranışı farklı ölçekle ifade eder.
- Precision = TP / (TP + FP). Modelin boyadığı piksellerin ne kadarının gerçekten nesne olduğunu gösterir; fazla alan boyamak precision değerini düşürür.
- Recall = TP / (TP + FN). Gerçek nesne piksellerinin ne kadarının yakalandığını gösterir; eksik maske recall değerini düşürür.
- Dört ortalama maske metriği nesne örneği düzeyinde (instance-level) önce her uçak için hesaplanır, sonra bütün örnekler eşit ağırlıkla ortalanır. Büyük nesneler küçük nesnelerin sonucunu perdelemez.
- IoU ≥ 0.50/0.75/0.90 sütunları, ilgili IoU eşiğini geçen uçak maskelerinin oranıdır. Bunlar mAP değildir ve raporda mAP gibi adlandırılmaz.
- YOLO'nun kaçırdığı bir gerçek uçak, YOLO-bbox maske tablosunda boş tahmin olarak değerlendirilir ve o örneğin maske skorları sıfır olur. Herhangi bir gerçek nesneyle eşleşmeyen yanlış pozitif YOLO kutuları ise instance maske ortalamasına sahte bir referans örneği olarak eklenmez; bunların etkisi detector Precision, Recall ve mAP değerlerinde ölçülür.
- Maske tabloları her GT uçak örneğini değerlendirir; YOLO'nun eşleştiremediği GT örnekleri de boş tahmin ve sıfır skorla hesaba katılır. Bu değerlendirme gerçek COCO segmentation AP ile aynı değildir. Confidence sırasındaki bütün maskeleri ve yanlış pozitifleri kullanan uçtan uca COCO mask AP bu raporda ayrıca çalıştırılmadığı için IoU eşik oranları AP veya mAP diye yeniden adlandırılmamıştır.
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
| YOLO26x (seed 42) | 512 | 0.920 | 0.847 | 0.545 | 0.762 | 0.925 | 0.896 | 0.868 | 0.840 | 0.632 | 0.612 |

## İnsan Referansı

Değerlendirme resmi iSAID insan çizimli instance maskelerine karşı yapılmıştır.

### Overall

Referans: İnsan Referansı. Bu tablo 512 görüntüdeki 5.447 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.653 | 0.780 | 0.686 | 0.935 | 0.905 | 0.195 | 0.005 |
| SAM1 YOLO bbox | 512 | 0.597 | 0.711 | 0.627 | 0.842 | 0.837 | 0.178 | 0.003 |
| SAM2 GT bbox | 512 | 0.629 | 0.761 | 0.651 | 0.952 | 0.871 | 0.168 | 0.004 |
| SAM2 YOLO bbox | 512 | 0.574 | 0.692 | 0.595 | 0.852 | 0.807 | 0.146 | 0.004 |
| SAM3 GT bbox | 512 | 0.655 | 0.776 | 0.683 | 0.924 | 0.890 | 0.265 | 0.008 |
| SAM3 YOLO bbox | 512 | 0.595 | 0.703 | 0.621 | 0.828 | 0.820 | 0.238 | 0.006 |

### No Overlap × Low Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 439 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.593 | 0.727 | 0.626 | 0.926 | 0.802 | 0.144 | 0.014 |
| SAM1 YOLO bbox | 128 | 0.543 | 0.662 | 0.571 | 0.824 | 0.749 | 0.128 | 0.009 |
| SAM2 GT bbox | 128 | 0.565 | 0.704 | 0.584 | 0.951 | 0.747 | 0.123 | 0.000 |
| SAM2 YOLO bbox | 128 | 0.519 | 0.643 | 0.537 | 0.834 | 0.706 | 0.105 | 0.000 |
| SAM3 GT bbox | 128 | 0.612 | 0.735 | 0.640 | 0.910 | 0.809 | 0.253 | 0.000 |
| SAM3 YOLO bbox | 128 | 0.567 | 0.677 | 0.597 | 0.810 | 0.763 | 0.235 | 0.000 |

### No Overlap × High Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 622 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.649 | 0.772 | 0.681 | 0.935 | 0.894 | 0.249 | 0.002 |
| SAM1 YOLO bbox | 128 | 0.611 | 0.726 | 0.642 | 0.862 | 0.850 | 0.228 | 0.003 |
| SAM2 GT bbox | 128 | 0.670 | 0.787 | 0.695 | 0.954 | 0.912 | 0.323 | 0.024 |
| SAM2 YOLO bbox | 128 | 0.625 | 0.734 | 0.652 | 0.871 | 0.855 | 0.288 | 0.023 |
| SAM3 GT bbox | 128 | 0.722 | 0.823 | 0.761 | 0.933 | 0.916 | 0.523 | 0.042 |
| SAM3 YOLO bbox | 128 | 0.655 | 0.747 | 0.695 | 0.833 | 0.839 | 0.477 | 0.039 |

### Overlap × Low Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 1.708 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.633 | 0.767 | 0.671 | 0.926 | 0.881 | 0.139 | 0.001 |
| SAM1 YOLO bbox | 128 | 0.577 | 0.695 | 0.611 | 0.827 | 0.814 | 0.128 | 0.001 |
| SAM2 GT bbox | 128 | 0.586 | 0.731 | 0.612 | 0.938 | 0.826 | 0.046 | 0.000 |
| SAM2 YOLO bbox | 128 | 0.532 | 0.661 | 0.559 | 0.831 | 0.770 | 0.043 | 0.000 |
| SAM3 GT bbox | 128 | 0.604 | 0.738 | 0.635 | 0.902 | 0.841 | 0.113 | 0.001 |
| SAM3 YOLO bbox | 128 | 0.550 | 0.667 | 0.581 | 0.800 | 0.780 | 0.108 | 0.001 |

### Overlap × High Mask Area

Referans: İnsan Referansı. Bu tablo 128 görüntüdeki 2.678 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.676 | 0.799 | 0.707 | 0.941 | 0.941 | 0.226 | 0.006 |
| SAM1 YOLO bbox | 128 | 0.616 | 0.726 | 0.642 | 0.849 | 0.863 | 0.207 | 0.003 |
| SAM2 GT bbox | 128 | 0.658 | 0.782 | 0.676 | 0.960 | 0.912 | 0.217 | 0.003 |
| SAM2 YOLO bbox | 128 | 0.597 | 0.710 | 0.614 | 0.863 | 0.835 | 0.186 | 0.002 |
| SAM3 GT bbox | 128 | 0.680 | 0.796 | 0.702 | 0.938 | 0.928 | 0.304 | 0.006 |
| SAM3 YOLO bbox | 128 | 0.614 | 0.720 | 0.634 | 0.847 | 0.849 | 0.267 | 0.003 |

## Reference Bias Comparison

Görüntü, uçak örneği, bbox ve model tahmini aynıdır; yalnız karşılaştırılan referans maske insan etiketinden SAM1 pseudo etiketine değişir. Bu nedenle fark, referans kaynağına duyarlılığı doğrudan gösterir.

| Model | BBox | Human IoU | SAM1 Pseudo IoU | IoU Artışı |
| --- | --- | --- | --- | --- |
| SAM1 | GT bbox | 0.653 | 1.000 | +0.347 |
| SAM1 | YOLO bbox | 0.597 | 0.873 | +0.276 |
| SAM2 | GT bbox | 0.629 | 0.827 | +0.198 |
| SAM2 | YOLO bbox | 0.574 | 0.750 | +0.177 |
| SAM3 | GT bbox | 0.655 | 0.795 | +0.140 |
| SAM3 | YOLO bbox | 0.595 | 0.721 | +0.126 |

## Qualitative Examples

Her sayfa bir gruptan tek görüntüyü gösterir. Görüntüdeki bütün GT uçak kutuları modele ayrı istemler olarak verilmiş ve instance maskeleri yalnız bu görsel için birleştirilmiştir. Tablolar instance-level kalır. Yeşil TP, turuncu FP ve pembe FN piksellerini gösterir.

### No Overlap / Low Mask Area

![No Overlap / Low Mask Area](qualitative/no_overlap__low_mask_area.png)

### No Overlap / High Mask Area

![No Overlap / High Mask Area](qualitative/no_overlap__high_mask_area.png)

### Overlap / Low Mask Area

![Overlap / Low Mask Area](qualitative/overlap__low_mask_area.png)

### Overlap / High Mask Area

![Overlap / High Mask Area](qualitative/overlap__high_mask_area.png)

## Discussion

- İnsan referansında GT-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,653/0,629/0,655 olarak ölçülmüştür.
- İnsan referansında YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,597/0,574/0,595 olarak ölçülmüştür; bunlar sabit seed 42 detector sonuçlarıdır.
- En yüksek insan-referanslı Overall IoU, GT bbox koşulunda SAM3 için 0.655; YOLO bbox koşulunda SAM1 için 0.597 olmuştur.
- GT bbox yerine YOLO bbox kullanıldığında Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla 0,056/0,056/0,061 olmuştur.
- GT-bbox üç-model ortalamasında en yüksek alt grup No Overlap × High Mask Area (0.680), en düşük alt grup No Overlap × Low Mask Area (0.590) olmuştur.
- SAM3 gt bbox koşulunda Overall Precision 0.683, Recall 0.924 olmuştur. Recall daha yüksek olduğu için model nesne piksellerini büyük ölçüde yakalarken hedef dışına taşan pikseller precision değerini düşürmektedir.
- GT bbox koşulu segmenter sınır kalitesini daha doğrudan, YOLO bbox koşulu ise detection ve segmentation hatalarının birleşik etkisini gösterir.
- Overlap ve mask-area alt tabloları aynı toplam test kümesinin dengeli, birbirini dışlayan dört parçasıdır; her tabloda 128 görüntü vardır.
- Bu insan etiketli sonuçlar, model kaynaklı pseudo etiket değerlendirmesine karşı birincil bağımsız karşılaştırma noktasıdır.
- GT-bbox model sıralaması insan referansında SAM3 > SAM1 > SAM2, aynı tahminlerin SAM1 pseudo referansında SAM1 > SAM2 > SAM3 biçimindedir.
- Bu sıralamalar tablo nokta tahminleridir; birbirine yakın insan-referanslı skorlar tek başına istatistiksel üstünlük iddiası değildir.
