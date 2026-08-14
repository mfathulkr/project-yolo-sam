# iSAID Small Vehicle - SAM3 pseudo referansı Full Metric Document

## Scope

- Veri kaynağı iSAID, hedef sınıf küçük araç ve model giriş çözünürlüğü 1024×1024 pikseldir.
- Test kümesi 512 görüntüdür. Dört Overlap × Mask Area grubunun her birinde tam 128 görüntü vardır.
- Bu 512 görüntünün tamamında en az bir hedef instance vardır. Detector tablosu bu nedenle resmi, hedef-negatif görüntüler de içeren benchmark AP'si değil pozitif test alt kümesindeki deney içi kontroldür.
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

- Bu belgede değerlendirme referansı SAM3 pseudo referansı ve değerlendirilen instance sayısı 12.051.
- Referans kümesinde 0 boş maske vardır (0.00%). Bilinen pozitif nesnede boş pseudo maske başarı sayılmaz ve 0 puanlanır.
- Detector mAP değerleri bbox ölçümüdür. Avg IoU, Dice, Precision, Recall ve IoU eşik oranları piksel maskesi ölçümüdür; IoU eşik oranları mAP değildir.
- Bu pseudo referans SAM3 modeline insan/yayımlanmış GT bbox verilerek instance başına üretilmiştir; lokalizasyon kutusu referans veri setinden gelir.
- GT-bbox diagonal hücre aynı dondurulmuş tahmin ile kendi pseudo referansını karşılaştıran özdeşlik/kapsama kontrolüdür. Bu hücre bağımsız segmentasyon başarısı değildir.

## YOLO Detector BBox Metrics

- Bu tablo yalnız YOLO detector kutularını değerlendirir; burada ölçülen bbox başarısıdır, maske başarısı değildir.
- Detector testi, her birinde en az bir hedef nesne bulunan seçilmiş 512 görüntüden oluşur. Hedef-negatif görüntü içermediği için bu değerler resmi veri seti benchmark AP'si değil deney içi detector kontrolüdür.
- Her detector yalnız tek hedef sınıf için eğitilip değerlendirildiğinden, sınıflar üzerindeki ortalama olan mAP bu deneyde o tek sınıfın AP değerine eşittir.
- BBox mAP50/mAP75/mAP90, tahmin kutusunun GT kutuyla sırasıyla en az 0,50/0,75/0,90 IoU yaptığı eşiklerde confidence sıralaması boyunca hesaplanan gerçek average precision değeridir.
- BBox mAP50-95, 0,50 ile 0,95 arasındaki on bbox IoU eşiğinin AP ortalamasıdır.
- BBox Precision ve Recall değerleri, doğrulama kümesinde seçilip testten önce sabitlenen güven eşiğinde hesaplanır.
- Tablodaki değerler sabit seed 42 sonucudur.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (seed 42) | 512 | 0.609 | 0.358 | 0.021 | 0.346 | 0.528 | 0.716 | 0.350 | 0.474 | 0.055 | 0.075 |

## SAM3 pseudo referansı

Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve SAM3 pseudo referansı ile değerlendirilmiştir.

### Overall

Referans: SAM3 pseudo referansı. Bu tablo 512 görüntüdeki 12.051 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.766 | 0.856 | 0.867 | 0.874 | 0.931 | 0.652 | 0.169 |
| SAM1 YOLO bbox | 512 | 0.560 | 0.623 | 0.620 | 0.642 | 0.681 | 0.492 | 0.140 |
| SAM2 GT bbox | 512 | 0.771 | 0.863 | 0.833 | 0.916 | 0.952 | 0.643 | 0.165 |
| SAM2 YOLO bbox | 512 | 0.551 | 0.617 | 0.592 | 0.661 | 0.677 | 0.460 | 0.113 |
| SAM3 GT bbox | 512 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 512 | 0.633 | 0.670 | 0.667 | 0.679 | 0.714 | 0.647 | 0.392 |

### No Overlap × Low Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 522 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.800 | 0.881 | 0.880 | 0.903 | 0.962 | 0.778 | 0.161 |
| SAM1 YOLO bbox | 128 | 0.549 | 0.604 | 0.600 | 0.621 | 0.667 | 0.538 | 0.113 |
| SAM2 GT bbox | 128 | 0.819 | 0.897 | 0.871 | 0.935 | 0.992 | 0.818 | 0.155 |
| SAM2 YOLO bbox | 128 | 0.547 | 0.604 | 0.581 | 0.639 | 0.676 | 0.529 | 0.065 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.615 | 0.646 | 0.641 | 0.654 | 0.682 | 0.659 | 0.423 |

### No Overlap × High Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 1.512 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.862 | 0.923 | 0.902 | 0.955 | 0.989 | 0.898 | 0.428 |
| SAM1 YOLO bbox | 128 | 0.718 | 0.768 | 0.750 | 0.793 | 0.827 | 0.746 | 0.349 |
| SAM2 GT bbox | 128 | 0.876 | 0.932 | 0.912 | 0.960 | 0.997 | 0.951 | 0.460 |
| SAM2 YOLO bbox | 128 | 0.720 | 0.769 | 0.755 | 0.791 | 0.829 | 0.765 | 0.332 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.774 | 0.799 | 0.802 | 0.800 | 0.829 | 0.807 | 0.662 |

### Overlap × Low Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 1.582 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.705 | 0.810 | 0.838 | 0.832 | 0.875 | 0.499 | 0.066 |
| SAM1 YOLO bbox | 128 | 0.381 | 0.436 | 0.435 | 0.455 | 0.482 | 0.288 | 0.035 |
| SAM2 GT bbox | 128 | 0.728 | 0.834 | 0.816 | 0.879 | 0.920 | 0.534 | 0.071 |
| SAM2 YOLO bbox | 128 | 0.377 | 0.433 | 0.413 | 0.471 | 0.475 | 0.267 | 0.028 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.447 | 0.480 | 0.471 | 0.494 | 0.520 | 0.448 | 0.210 |

### Overlap × High Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 8.435 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.758 | 0.851 | 0.865 | 0.865 | 0.930 | 0.629 | 0.142 |
| SAM1 YOLO bbox | 128 | 0.567 | 0.633 | 0.633 | 0.651 | 0.693 | 0.483 | 0.123 |
| SAM2 GT bbox | 128 | 0.757 | 0.853 | 0.819 | 0.914 | 0.948 | 0.598 | 0.131 |
| SAM2 YOLO bbox | 128 | 0.553 | 0.625 | 0.597 | 0.674 | 0.688 | 0.438 | 0.093 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.644 | 0.684 | 0.681 | 0.693 | 0.732 | 0.655 | 0.375 |

## Reference Bias Comparison

Görüntü, instance, bbox istemi ve model tahmini aynıdır; yalnız değerlendirme referansı İnsan referansı yerine SAM3 pseudo referansı olarak değiştirilmiştir.

| Model | BBox | Temel Referans IoU | Reference IoU | IoU Farkı |
| --- | --- | --- | --- | --- |
| SAM1 | GT bbox | 0.658 | 0.766 | +0.108 |
| SAM1 | YOLO bbox | 0.478 | 0.560 | +0.082 |
| SAM2 | GT bbox | 0.645 | 0.771 | +0.125 |
| SAM2 | YOLO bbox | 0.461 | 0.551 | +0.090 |
| SAM3 | GT bbox | 0.698 | 1.000 | +0.302 |
| SAM3 | YOLO bbox | 0.491 | 0.633 | +0.142 |

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

- Bu referansta Overall GT-bbox sıralaması SAM3 > SAM2 > SAM1; YOLO-bbox sıralaması SAM3 > SAM1 > SAM2 biçimindedir.
- GT bbox ile YOLO bbox arasındaki fark, eşleşen veya kaçırılan GT instance'lar üzerindeki lokalizasyon/recall etkisini gösterir. Eşleşmeyen detector yanlış pozitifleri maske ortalamasına eklenmediği için bu fark tam uçtan uca instance-segmentation performansı değildir.
- SAM3 modeli YOLO bbox koşulunda temel referansa karşı 0.491, kendi öğretmen ailesinin referansına karşı 0.633 Avg IoU verir; görünür fark +0.142'tür.
- Bu temel-referans farkı tek başına teacher affinity kanıtı değildir; pseudo referansların genel olarak daha kolay olması da fark yaratabilir. Ana exploratory kontrast, deney içi cross-analysis belgesinde aynı dondurulmuş checkpoint'in kendi pseudo referansı ile diğer iki öğretmenin pseudo referanslarını ve göreli model avantajını eşleşmiş olarak karşılaştırır; preregistered confirmatory test değildir.
- Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir.
