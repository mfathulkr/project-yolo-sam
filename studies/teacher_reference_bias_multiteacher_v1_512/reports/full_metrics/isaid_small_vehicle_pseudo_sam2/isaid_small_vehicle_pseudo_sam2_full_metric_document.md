# Isaid Small Vehicle SAM2 Pseudo Reference Full Metric Document

## Scope

- Veri seti iSAID, hedef sınıf küçük araç ve giriş çözünürlüğü 1024×1024 pikseldir.
- Test kümesi 512 görüntüdür. Dört Overlap × Mask Area grubunun her birinde tam 128 görüntü vardır.
- No Overlap, görüntüdeki hiçbir iki insan GT bbox'un kesişmemesi; Overlap ise en az bir bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.
- Low/High Mask Area ayrımı insan çizimli hedef maskelerinin görüntüdeki toplam alanına göre, testten önce dondurulmuş veri setine özgü eşikle yapılmıştır. Referans değişse bile stratum üyeliği değiştirilmemiştir.
- SAM1, SAM2 ve SAM3 aynı 512 görüntüde hem GT bbox hem seed 42 YOLO bbox istemiyle çalıştırılmıştır.
- Yeni inference yapılmamıştır. İnsan, SAM1, SAM2 ve SAM3 değerlendirmelerinde aynı dondurulmuş model tahminleri kullanılmış; yalnız karşılaştırılan referans maske değiştirilmiştir.
- SAM2/SAM3 pseudo referansları ilgili öğretmenin insan GT bbox istemiyle ürettiği instance maskeleridir. İnsan kutusunun kullanılması nedeniyle bu referanslar insan lokalizasyonundan tamamen bağımsız değildir.
- Maske metrikleri instance-level hesaplanır; her hedef örnek eşit ağırlıktadır. Büyük nesneler küçük nesneleri piksel sayısıyla perdelemez.
- YOLO detector sonuçları referans maskeden bağımsızdır ve bütün pseudo raporlarda aynı gerçek bbox mAP değerleri tekrar kullanılır.

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

- Bu belge bağımsız ground truth performansı değil, değerlendirme referansının model skorunu nasıl değiştirdiğini gösteren kontrollü bir referans duyarlılığı deneyidir.
- Referans öğretmeni SAM2. referans istemi insan GT bbox ve örnek sayısı 12.051.
- Referans kümesinde 0 boş maske vardır (0.0%).
- Bir öğretmenin GT-bbox tahminini aynı tahmin maskesine karşı ölçen diagonal hücre özdeşlik gereği 1,000 olur; bu sonuç model başarısı olarak yorumlanamaz.
- Detector mAP değerleri bbox ölçümüdür. Avg IoU, Dice, Precision, Recall ve IoU eşik oranları piksel maskesi ölçümüdür; IoU eşik oranları mAP değildir.

## YOLO Detector BBox Metrics

- Bu tablo yalnız YOLO detector kutularını değerlendirir; burada ölçülen bbox başarısıdır, maske başarısı değildir.
- BBox mAP50/mAP75/mAP90, tahmin kutusunun GT kutuyla sırasıyla en az 0,50/0,75/0,90 IoU yaptığı eşiklerde confidence sıralaması boyunca hesaplanan gerçek average precision değeridir.
- BBox mAP50-95, 0,50 ile 0,95 arasındaki on bbox IoU eşiğinin AP ortalamasıdır.
- BBox Precision ve Recall değerleri, doğrulama kümesinde seçilip testten önce sabitlenen güven eşiğinde hesaplanır.
- Tablodaki değerler sabit seed 42 sonucudur.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (seed 42) | 512 | 0.609 | 0.358 | 0.021 | 0.346 | 0.528 | 0.716 | 0.350 | 0.474 | 0.055 | 0.075 |

## SAM2 Pseudo Referansı

Değerlendirme SAM2 modelinin insan GT bbox istemiyle ürettiği dondurulmuş instance maskelerine karşı yapılmıştır.

### Overall

Referans: SAM2 Pseudo Referansı. Bu tablo 512 görüntüdeki 12.051 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.749 | 0.842 | 0.898 | 0.824 | 0.906 | 0.602 | 0.186 |
| SAM1 YOLO bbox | 512 | 0.551 | 0.615 | 0.644 | 0.607 | 0.668 | 0.461 | 0.148 |
| SAM2 GT bbox | 512 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM2 YOLO bbox | 512 | 0.624 | 0.663 | 0.663 | 0.674 | 0.702 | 0.629 | 0.369 |
| SAM3 GT bbox | 512 | 0.420 | 0.474 | 0.466 | 0.494 | 0.529 | 0.338 | 0.038 |
| SAM3 YOLO bbox | 512 | 0.339 | 0.382 | 0.378 | 0.395 | 0.428 | 0.283 | 0.031 |

### No Overlap × Low Mask Area

Referans: SAM2 Pseudo Referansı. Bu tablo 128 görüntüdeki 522 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.799 | 0.881 | 0.912 | 0.873 | 0.969 | 0.755 | 0.186 |
| SAM1 YOLO bbox | 128 | 0.546 | 0.602 | 0.620 | 0.597 | 0.669 | 0.511 | 0.126 |
| SAM2 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM2 YOLO bbox | 128 | 0.614 | 0.645 | 0.641 | 0.653 | 0.678 | 0.653 | 0.450 |
| SAM3 GT bbox | 128 | 0.665 | 0.762 | 0.770 | 0.773 | 0.875 | 0.475 | 0.013 |
| SAM3 YOLO bbox | 128 | 0.444 | 0.508 | 0.509 | 0.517 | 0.592 | 0.316 | 0.010 |

### No Overlap × High Mask Area

Referans: SAM2 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.512 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.864 | 0.924 | 0.926 | 0.932 | 0.988 | 0.892 | 0.479 |
| SAM1 YOLO bbox | 128 | 0.718 | 0.768 | 0.770 | 0.773 | 0.827 | 0.742 | 0.384 |
| SAM2 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM2 YOLO bbox | 128 | 0.763 | 0.793 | 0.797 | 0.794 | 0.829 | 0.787 | 0.610 |
| SAM3 GT bbox | 128 | 0.796 | 0.869 | 0.871 | 0.876 | 0.954 | 0.821 | 0.133 |
| SAM3 YOLO bbox | 128 | 0.667 | 0.729 | 0.739 | 0.728 | 0.808 | 0.681 | 0.099 |

### Overlap × Low Mask Area

Referans: SAM2 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.582 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.692 | 0.800 | 0.861 | 0.792 | 0.860 | 0.472 | 0.064 |
| SAM1 YOLO bbox | 128 | 0.378 | 0.433 | 0.450 | 0.434 | 0.480 | 0.272 | 0.035 |
| SAM2 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM2 YOLO bbox | 128 | 0.444 | 0.477 | 0.469 | 0.494 | 0.513 | 0.436 | 0.204 |
| SAM3 GT bbox | 128 | 0.422 | 0.498 | 0.488 | 0.527 | 0.566 | 0.215 | 0.003 |
| SAM3 YOLO bbox | 128 | 0.269 | 0.316 | 0.311 | 0.333 | 0.361 | 0.139 | 0.003 |

### Overlap × High Mask Area

Referans: SAM2 Pseudo Referansı. Bu tablo 128 görüntüdeki 8.435 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.735 | 0.833 | 0.899 | 0.808 | 0.896 | 0.566 | 0.157 |
| SAM1 YOLO bbox | 128 | 0.553 | 0.623 | 0.659 | 0.611 | 0.675 | 0.443 | 0.129 |
| SAM2 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM2 YOLO bbox | 128 | 0.633 | 0.676 | 0.676 | 0.687 | 0.716 | 0.636 | 0.352 |
| SAM3 GT bbox | 128 | 0.337 | 0.382 | 0.371 | 0.403 | 0.425 | 0.266 | 0.029 |
| SAM3 YOLO bbox | 128 | 0.287 | 0.324 | 0.317 | 0.339 | 0.363 | 0.236 | 0.025 |

## Reference Bias Comparison

Görüntü, instance, bbox istemi ve model tahmini aynıdır; yalnız değerlendirme referansı insan maskesinden SAM2 pseudo maskesine değişmiştir.

| Model | BBox | Human IoU | Reference IoU | IoU Farkı |
| --- | --- | --- | --- | --- |
| SAM1 | GT bbox | 0.658 | 0.749 | +0.091 |
| SAM1 | YOLO bbox | 0.478 | 0.551 | +0.073 |
| SAM2 | GT bbox | 0.645 | 1.000 | +0.355 |
| SAM2 | YOLO bbox | 0.461 | 0.624 | +0.163 |
| SAM3 | GT bbox | 0.370 | 0.420 | +0.050 |
| SAM3 | YOLO bbox | 0.299 | 0.339 | +0.040 |

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

- SAM2 GT-bbox diagonal kontrolü 1.000 değerindedir. Bu tam skor beklenen özdeşlik kontrolüdür.
- Aynı SAM2 modeli insan referansında GT-bbox Avg IoU 0.645 verir; referansın kendi çıktısına çevrilmesi görünürde +0.355 artış üretir.
- YOLO bbox kullanıldığında SAM2 modeli kendi pseudo referansında 0.624 Avg IoU verir; bbox değişmesine rağmen aynı model ailesine ait referans avantajı sürmektedir.
- İnsan referansı GT-bbox sıralaması SAM1 > SAM2 > SAM3; bu pseudo referans sıralaması SAM2 > SAM1 > SAM3 biçimindedir.
- 0 boş öğretmen maskesi. özellikle boş tahmin–boş referans eşleşmelerinde öz-skoru yükseltebilir. Bu nedenle boş referans oranı skorlarla birlikte raporlanmalıdır.
- Sonuç, pseudo etiketlerin kullanılamaz olduğunu değil; pseudo etiket üreticisiyle aday modelin aynı veya yakın aileden olduğu değerlendirmelerde bağımsız insan referansı olmadan model seçimi yapılamayacağını gösterir.
