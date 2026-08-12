# Isaid Small Vehicle - İnsan referansı Full Metric Document

## Scope

- Veri kaynağı iSAID, hedef sınıf küçük araç ve model giriş çözünürlüğü 1024×1024 pikseldir.
- Test kümesi 512 görüntüdür. Dört Overlap × Mask Area grubunun her birinde tam 128 görüntü vardır.
- No Overlap, görüntüdeki hedef GT bbox çiftlerinin kesişmemesi; Overlap ise en az bir hedef bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.
- Low/High Mask Area ayrımı yayımlanmış temel instance maskelerinin görüntü içindeki toplam alan oranına göre, testten önce dondurulmuş veri setine özgü eşikle yapılmıştır. Referans türü değişse bile stratum üyeliği değişmez.
- SAM1, SAM2 ve SAM3 tahminleri aynı 512 görüntüde hem GT bbox hem de seed 42 ile eğitilmiş YOLO bbox istemiyle bir kez üretilmiş ve bütün referanslara karşı değişmeden yeniden değerlendirilmiştir.
- SAM3 bbox koşulu Sam3Tracker PVS arayüzüyle, multimask_output=False ve mask_threshold=0.0 ayarlarıyla çalıştırılmıştır. PCS kavram örneği arayüzü kullanılmamıştır.
- Maske metrikleri instance-level hesaplanır; her hedef örnek eşit ağırlıktadır. Büyük nesneler küçük nesneleri piksel sayısıyla perdelemez.
- YOLO detector tablosu referans maskeden bağımsızdır; bütün referans raporlarında aynı dondurulmuş bbox sonuçları kullanılır.
- Nitel görsellerde seçilen görüntüdeki bütün hedef instance'lar modele ayrı GT bbox istemleri olarak verilir ve maskeler yalnız gösterim amacıyla birleştirilir.

## Metric Logic

- TP, modelin doğru biçimde nesne olarak işaretlediği pikseldir. FP, nesne olmadığı hâlde nesne diye işaretlenen; FN ise nesne olduğu hâlde kaçırılan pikseldir.
- IoU = TP / (TP + FP + FN). Tahmin ve referans maskenin ortak alanını birleşim alanına böler; 1 kusursuz, 0 hiç örtüşme yok demektir.
- Dice = 2TP / (2TP + FP + FN). IoU ile aynı davranışı farklı ölçekle ifade eder.
- Precision = TP / (TP + FP). Modelin boyadığı piksellerin ne kadarının gerçekten nesne olduğunu gösterir; fazla alan boyamak precision değerini düşürür.
- Recall = TP / (TP + FN). Gerçek nesne piksellerinin ne kadarının yakalandığını gösterir; eksik maske recall değerini düşürür.
- Dört ortalama maske metriği nesne örneği düzeyinde (instance-level) önce her küçük araç için hesaplanır, sonra bütün örnekler eşit ağırlıkla ortalanır. Büyük nesneler küçük nesnelerin sonucunu perdelemez.
- Her satırda temel veri seti anotasyonuyla varlığı bilinen bir küçük araç vardır. Bu nedenle boş pseudo referans eksik etikettir; tahmin de boş olsa bile maske metrikleri 0 kabul edilir ve referans kapsama kaybı ayrıca raporlanır.
- IoU ≥ 0.50/0.75/0.90 sütunları, ilgili IoU eşiğini geçen küçük araç maskelerinin oranıdır. Bunlar mAP değildir ve raporda mAP gibi adlandırılmaz.
- YOLO'nun kaçırdığı bir gerçek küçük araç, YOLO-bbox maske tablosunda boş tahmin olarak değerlendirilir ve o örneğin maske skorları sıfır olur. Herhangi bir gerçek nesneyle eşleşmeyen yanlış pozitif YOLO kutuları ise instance maske ortalamasına sahte bir referans örneği olarak eklenmez; bunların etkisi detector Precision, Recall ve mAP değerlerinde ölçülür.
- Maske tabloları her GT küçük araç örneğini değerlendirir; YOLO'nun eşleştiremediği GT örnekleri de boş tahmin ve sıfır skorla hesaba katılır. Bu değerlendirme gerçek COCO segmentation AP ile aynı değildir. Confidence sırasındaki bütün maskeleri ve yanlış pozitifleri kullanan uçtan uca COCO mask AP bu raporda ayrıca çalıştırılmadığı için IoU eşik oranları AP veya mAP diye yeniden adlandırılmamıştır.
- Overall tablosu 512 görüntüyü, diğer tabloların her biri 128 görüntüyü kapsar.
- GT-bbox satırları tek sabit koşuldur. YOLO-bbox satırlarındaki değerler sabit seed 42 sonucudur.

## Dataset Context

- Bu belgede değerlendirme referansı İnsan referansı ve değerlendirilen instance sayısı 12.051.
- Referans kümesinde 0 boş maske vardır (0.00%). Bilinen pozitif nesnede boş pseudo maske başarı sayılmaz ve 0 puanlanır.
- Detector mAP değerleri bbox ölçümüdür. Avg IoU, Dice, Precision, Recall ve IoU eşik oranları piksel maskesi ölçümüdür; IoU eşik oranları mAP değildir.
- Bu iSAID insan anotasyonu bağımsız kontrol referansıdır ve model kalitesine ilişkin ana bilimsel yorum bu referansa dayanır.

## YOLO Detector BBox Metrics

- Bu tablo yalnız YOLO detector kutularını değerlendirir; burada ölçülen bbox başarısıdır, maske başarısı değildir.
- BBox mAP50/mAP75/mAP90, tahmin kutusunun GT kutuyla sırasıyla en az 0,50/0,75/0,90 IoU yaptığı eşiklerde confidence sıralaması boyunca hesaplanan gerçek average precision değeridir.
- BBox mAP50-95, 0,50 ile 0,95 arasındaki on bbox IoU eşiğinin AP ortalamasıdır.
- BBox Precision ve Recall değerleri, doğrulama kümesinde seçilip testten önce sabitlenen güven eşiğinde hesaplanır.
- Tablodaki değerler sabit seed 42 sonucudur.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (seed 42) | 512 | 0.609 | 0.358 | 0.021 | 0.346 | 0.528 | 0.716 | 0.350 | 0.474 | 0.055 | 0.075 |

## İnsan referansı

Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve İnsan referansı ile değerlendirilmiştir.

### Overall

Referans: İnsan referansı. Bu tablo 512 görüntüdeki 12.051 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.658 | 0.779 | 0.733 | 0.877 | 0.838 | 0.341 | 0.018 |
| SAM1 YOLO bbox | 512 | 0.478 | 0.564 | 0.529 | 0.635 | 0.607 | 0.259 | 0.014 |
| SAM2 GT bbox | 512 | 0.645 | 0.771 | 0.693 | 0.910 | 0.802 | 0.318 | 0.013 |
| SAM2 YOLO bbox | 512 | 0.461 | 0.552 | 0.499 | 0.647 | 0.575 | 0.223 | 0.008 |
| SAM3 GT bbox | 512 | 0.698 | 0.815 | 0.756 | 0.907 | 0.920 | 0.416 | 0.019 |
| SAM3 YOLO bbox | 512 | 0.491 | 0.577 | 0.540 | 0.640 | 0.636 | 0.280 | 0.011 |

### No Overlap × Low Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 522 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.680 | 0.800 | 0.744 | 0.902 | 0.941 | 0.341 | 0.015 |
| SAM1 YOLO bbox | 128 | 0.466 | 0.549 | 0.510 | 0.616 | 0.644 | 0.226 | 0.008 |
| SAM2 GT bbox | 128 | 0.680 | 0.804 | 0.725 | 0.924 | 0.946 | 0.328 | 0.006 |
| SAM2 YOLO bbox | 128 | 0.451 | 0.539 | 0.486 | 0.624 | 0.634 | 0.178 | 0.000 |
| SAM3 GT bbox | 128 | 0.715 | 0.829 | 0.770 | 0.916 | 0.977 | 0.398 | 0.010 |
| SAM3 YOLO bbox | 128 | 0.474 | 0.556 | 0.518 | 0.616 | 0.649 | 0.234 | 0.002 |

### No Overlap × High Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 1.512 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.753 | 0.853 | 0.807 | 0.928 | 0.975 | 0.574 | 0.060 |
| SAM1 YOLO bbox | 128 | 0.627 | 0.710 | 0.673 | 0.770 | 0.817 | 0.476 | 0.042 |
| SAM2 GT bbox | 128 | 0.774 | 0.869 | 0.819 | 0.940 | 0.982 | 0.674 | 0.048 |
| SAM2 YOLO bbox | 128 | 0.630 | 0.713 | 0.677 | 0.769 | 0.821 | 0.505 | 0.034 |
| SAM3 GT bbox | 128 | 0.791 | 0.880 | 0.849 | 0.927 | 0.987 | 0.738 | 0.062 |
| SAM3 YOLO bbox | 128 | 0.643 | 0.722 | 0.700 | 0.758 | 0.825 | 0.552 | 0.039 |

### Overlap × Low Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 1.582 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.565 | 0.702 | 0.648 | 0.852 | 0.677 | 0.155 | 0.004 |
| SAM1 YOLO bbox | 128 | 0.298 | 0.371 | 0.334 | 0.457 | 0.359 | 0.076 | 0.003 |
| SAM2 GT bbox | 128 | 0.564 | 0.705 | 0.616 | 0.884 | 0.677 | 0.131 | 0.004 |
| SAM2 YOLO bbox | 128 | 0.286 | 0.361 | 0.312 | 0.465 | 0.350 | 0.055 | 0.001 |
| SAM3 GT bbox | 128 | 0.611 | 0.748 | 0.665 | 0.895 | 0.791 | 0.173 | 0.002 |
| SAM3 YOLO bbox | 128 | 0.307 | 0.380 | 0.337 | 0.464 | 0.384 | 0.074 | 0.001 |

### Overlap × High Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 8.435 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.656 | 0.779 | 0.735 | 0.872 | 0.837 | 0.333 | 0.014 |
| SAM1 YOLO bbox | 128 | 0.486 | 0.576 | 0.541 | 0.646 | 0.613 | 0.256 | 0.012 |
| SAM2 GT bbox | 128 | 0.635 | 0.764 | 0.683 | 0.908 | 0.784 | 0.288 | 0.009 |
| SAM2 YOLO bbox | 128 | 0.464 | 0.559 | 0.504 | 0.661 | 0.569 | 0.207 | 0.005 |
| SAM3 GT bbox | 128 | 0.697 | 0.815 | 0.755 | 0.905 | 0.929 | 0.405 | 0.015 |
| SAM3 YOLO bbox | 128 | 0.500 | 0.589 | 0.550 | 0.653 | 0.648 | 0.273 | 0.008 |

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

- Bu referansta Overall GT-bbox sıralaması SAM3 > SAM1 > SAM2; YOLO-bbox sıralaması SAM3 > SAM1 > SAM2 biçimindedir.
- GT bbox ile YOLO bbox arasındaki fark, segmenterden önceki detection hatasının uçtan uca sisteme etkisini gösterir.
- Bu insan referansı model ailelerinden bağımsız olduğu için modeller arası kalite karşılaştırmasının güvenilir kontrolüdür.
- Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir.
