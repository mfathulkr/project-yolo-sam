# iSAID Small Vehicle SAM1 Pseudo Reference Full Metric Document

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

- Bu belge bağımsız benchmark sonucu değildir; referans kaynağı yanlılığını ölçen kontrollü deneydir.
- iSAID veri seti kaynağı: https://captain-whu.github.io/iSAID/
- iSAID: A Large-scale Dataset for Instance Segmentation in Aerial Images makalesi: https://arxiv.org/abs/1905.12886
- Görüntüler, bbox istemleri ve model tahminleri insan referansı deneyindekiyle aynıdır. Yalnız değerlendirme referansı SAM1 pseudo maskesidir.
- Bu rapordaki GT bbox, resmi iSAID insan instance anotasyonunda verilen kutudur; pseudo maskeden türetilmemiştir.
- Referansı üreten SAM1'in kendi pseudo maskelerine yüksek benzerlik göstermesi teacher-reference bias beklentisidir.
- Beş tam bağımsız iSAID insan referans tablosu ayrı full metric belgede verilmiştir. Bu belgede yalnız aynı tahminlerin referans değişimine duyarlılığını gösteren kısa karşılaştırma özeti bulunur.

## YOLO Detector BBox Metrics

- Bu tablo yalnız YOLO detector kutularını değerlendirir; burada ölçülen bbox başarısıdır, maske başarısı değildir.
- BBox mAP50/mAP75/mAP90, tahmin kutusunun GT kutuyla sırasıyla en az 0,50/0,75/0,90 IoU yaptığı eşiklerde confidence sıralaması boyunca hesaplanan gerçek average precision değeridir.
- BBox mAP50-95, 0,50 ile 0,95 arasındaki on bbox IoU eşiğinin AP ortalamasıdır.
- BBox Precision ve Recall değerleri, doğrulama kümesinde seçilip testten önce sabitlenen güven eşiğinde hesaplanır.
- Tablodaki değerler sabit seed 42 sonucudur.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (seed 42) | 512 | 0.609 | 0.358 | 0.021 | 0.346 | 0.528 | 0.716 | 0.350 | 0.474 | 0.055 | 0.075 |

## Kontrollü SAM1 Pseudo Referansı

Aynı iSAID görüntülerindeki GT bbox'lar SAM1'e verilmiş ve dondurulan çıktılar pseudo referans olarak kullanılmıştır.

### Overall

Referans: Kontrollü SAM1 Pseudo Referansı. Bu tablo 512 görüntüdeki 12.051 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM1 YOLO bbox | 512 | 0.655 | 0.682 | 0.678 | 0.692 | 0.713 | 0.661 | 0.508 |
| SAM2 GT bbox | 512 | 0.749 | 0.842 | 0.824 | 0.898 | 0.906 | 0.602 | 0.186 |
| SAM2 YOLO bbox | 512 | 0.550 | 0.615 | 0.598 | 0.655 | 0.662 | 0.465 | 0.150 |
| SAM3 GT bbox | 512 | 0.419 | 0.474 | 0.458 | 0.506 | 0.526 | 0.333 | 0.040 |
| SAM3 YOLO bbox | 512 | 0.341 | 0.383 | 0.373 | 0.404 | 0.426 | 0.286 | 0.035 |

### No Overlap × Low Mask Area

Referans: Kontrollü SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 522 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM1 YOLO bbox | 128 | 0.640 | 0.660 | 0.656 | 0.666 | 0.682 | 0.667 | 0.557 |
| SAM2 GT bbox | 128 | 0.799 | 0.881 | 0.873 | 0.912 | 0.969 | 0.755 | 0.186 |
| SAM2 YOLO bbox | 128 | 0.546 | 0.602 | 0.586 | 0.632 | 0.667 | 0.527 | 0.123 |
| SAM3 GT bbox | 128 | 0.649 | 0.751 | 0.743 | 0.783 | 0.866 | 0.406 | 0.011 |
| SAM3 YOLO bbox | 128 | 0.436 | 0.502 | 0.493 | 0.526 | 0.577 | 0.289 | 0.015 |

### No Overlap × High Mask Area

Referans: Kontrollü SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.512 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM1 YOLO bbox | 128 | 0.793 | 0.810 | 0.813 | 0.809 | 0.829 | 0.811 | 0.732 |
| SAM2 GT bbox | 128 | 0.864 | 0.924 | 0.932 | 0.926 | 0.988 | 0.892 | 0.479 |
| SAM2 YOLO bbox | 128 | 0.721 | 0.769 | 0.777 | 0.768 | 0.826 | 0.751 | 0.397 |
| SAM3 GT bbox | 128 | 0.791 | 0.865 | 0.869 | 0.871 | 0.956 | 0.805 | 0.130 |
| SAM3 YOLO bbox | 128 | 0.665 | 0.728 | 0.741 | 0.724 | 0.807 | 0.675 | 0.108 |

### Overlap × Low Mask Area

Referans: Kontrollü SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.582 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM1 YOLO bbox | 128 | 0.463 | 0.490 | 0.480 | 0.506 | 0.521 | 0.451 | 0.310 |
| SAM2 GT bbox | 128 | 0.692 | 0.800 | 0.792 | 0.861 | 0.860 | 0.472 | 0.064 |
| SAM2 YOLO bbox | 128 | 0.376 | 0.431 | 0.414 | 0.473 | 0.466 | 0.265 | 0.044 |
| SAM3 GT bbox | 128 | 0.419 | 0.496 | 0.471 | 0.548 | 0.547 | 0.205 | 0.004 |
| SAM3 YOLO bbox | 128 | 0.269 | 0.317 | 0.302 | 0.348 | 0.351 | 0.140 | 0.004 |

### Overlap × High Mask Area

Referans: Kontrollü SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 8.435 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM1 YOLO bbox | 128 | 0.668 | 0.697 | 0.692 | 0.707 | 0.731 | 0.673 | 0.502 |
| SAM2 GT bbox | 128 | 0.735 | 0.833 | 0.808 | 0.899 | 0.896 | 0.566 | 0.157 |
| SAM2 YOLO bbox | 128 | 0.553 | 0.622 | 0.601 | 0.670 | 0.669 | 0.448 | 0.127 |
| SAM3 GT bbox | 128 | 0.339 | 0.383 | 0.364 | 0.416 | 0.424 | 0.269 | 0.032 |
| SAM3 YOLO bbox | 128 | 0.290 | 0.326 | 0.312 | 0.350 | 0.363 | 0.243 | 0.029 |

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

- SAM1 pseudo referansında GT-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 1,000/0,749/0,419 olarak ölçülmüştür.
- Kontrollü pseudo referans doğrudan SAM1 GT-bbox tahmininden dondurulduğu için SAM1 GT-bbox satırı bir kimlik kontrolüdür; bu satır bağımsız segmentasyon başarısı olarak yorumlanmaz.
- SAM1 pseudo referansında YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,655/0,550/0,341 olarak ölçülmüştür.
- Aynı GT-bbox tahminleri insan yerine SAM1 pseudo referansla ölçüldüğünde IoU değişimi SAM1/SAM2/SAM3 için sırasıyla 0,342/0,103/0,049 olmuştur.
- GT-bbox model sıralamasının lideri insan referansında SAM1 (0.658), SAM1 pseudo referansında SAM1 (1.000) olmuştur.
- Tam GT-bbox sıralaması her iki referansta da SAM1 > SAM2 > SAM3 biçiminde korunmuştur; buna rağmen skor düzeylerindeki değişim pseudo referansın ölçülen başarı büyüklüğünü etkilediğini gösterir.
- Bu sıralamalar tablo nokta tahminleridir; birbirine yakın insan-referanslı skorlar tek başına istatistiksel üstünlük iddiası değildir.
- GT bbox yerine YOLO bbox kullanıldığında pseudo-referanslı Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla 0,345/0,198/0,079 olmuştur.
- GT-bbox üç-model ortalamasında en yüksek alt grup No Overlap × High Mask Area (0.885), en düşük alt grup Overlap × High Mask Area (0.691) olmuştur.
- SAM1 gt bbox koşulunda Overall Precision 1.000, Recall 1.000 olmuştur. Precision ve Recall dengelidir.
- SAM1'in kendisinin ürettiği referansa yakınlığı, gerçek insan çizimli sınır doğruluğuyla aynı şey değildir.
- Pseudo etiketler eğitim veya ön-etiketleme için kullanılabilir; ancak bağımsız test ground truth'u yerine kullanıldığında sonuç model ailesine yanlı görünebilir.
