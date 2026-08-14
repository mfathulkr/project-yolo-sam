# SAMRS SOTA Small Vehicle - SAM3 pseudo referansı Full Metric Document

## Scope

- Veri kaynağı SAMRS SOTA, hedef sınıf küçük araç ve model giriş çözünürlüğü 1024×1024 pikseldir.
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

- Bu belgede değerlendirme referansı SAM3 pseudo referansı ve değerlendirilen instance sayısı 7.659.
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
| YOLO26x (seed 42) | 512 | 0.819 | 0.534 | 0.072 | 0.502 | 0.720 | 0.824 | 0.529 | 0.605 | 0.131 | 0.150 |

## SAM3 pseudo referansı

Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve SAM3 pseudo referansı ile değerlendirilmiştir.

### Overall

Referans: SAM3 pseudo referansı. Bu tablo 512 görüntüdeki 7.659 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.851 | 0.915 | 0.923 | 0.919 | 0.983 | 0.860 | 0.402 |
| SAM1 YOLO bbox | 512 | 0.705 | 0.756 | 0.763 | 0.759 | 0.810 | 0.725 | 0.345 |
| SAM2 GT bbox | 512 | 0.856 | 0.918 | 0.898 | 0.949 | 0.989 | 0.875 | 0.403 |
| SAM2 YOLO bbox | 512 | 0.705 | 0.757 | 0.749 | 0.773 | 0.816 | 0.718 | 0.330 |
| SAM3 GT bbox | 512 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 512 | 0.760 | 0.789 | 0.798 | 0.784 | 0.823 | 0.790 | 0.597 |

### No Overlap × Low Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 596 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.832 | 0.904 | 0.913 | 0.910 | 0.987 | 0.839 | 0.252 |
| SAM1 YOLO bbox | 128 | 0.557 | 0.598 | 0.591 | 0.609 | 0.648 | 0.607 | 0.213 |
| SAM2 GT bbox | 128 | 0.838 | 0.907 | 0.875 | 0.956 | 0.977 | 0.856 | 0.304 |
| SAM2 YOLO bbox | 128 | 0.560 | 0.599 | 0.592 | 0.612 | 0.646 | 0.602 | 0.206 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.598 | 0.621 | 0.631 | 0.614 | 0.648 | 0.634 | 0.451 |

### No Overlap × High Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 1.478 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.892 | 0.939 | 0.938 | 0.949 | 0.986 | 0.926 | 0.643 |
| SAM1 YOLO bbox | 128 | 0.752 | 0.790 | 0.780 | 0.804 | 0.833 | 0.790 | 0.555 |
| SAM2 GT bbox | 128 | 0.899 | 0.945 | 0.937 | 0.959 | 0.991 | 0.960 | 0.652 |
| SAM2 YOLO bbox | 128 | 0.754 | 0.791 | 0.782 | 0.804 | 0.833 | 0.802 | 0.555 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.799 | 0.816 | 0.814 | 0.820 | 0.834 | 0.825 | 0.751 |

### Overlap × Low Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 1.884 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.804 | 0.887 | 0.900 | 0.890 | 0.982 | 0.760 | 0.166 |
| SAM1 YOLO bbox | 128 | 0.582 | 0.641 | 0.652 | 0.640 | 0.705 | 0.571 | 0.123 |
| SAM2 GT bbox | 128 | 0.822 | 0.899 | 0.874 | 0.937 | 0.990 | 0.813 | 0.194 |
| SAM2 YOLO bbox | 128 | 0.587 | 0.645 | 0.640 | 0.658 | 0.713 | 0.568 | 0.119 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.640 | 0.676 | 0.689 | 0.669 | 0.719 | 0.670 | 0.391 |

### Overlap × High Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 3.701 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.862 | 0.921 | 0.930 | 0.923 | 0.981 | 0.888 | 0.451 |
| SAM1 YOLO bbox | 128 | 0.772 | 0.827 | 0.840 | 0.825 | 0.881 | 0.797 | 0.394 |
| SAM2 GT bbox | 128 | 0.858 | 0.920 | 0.898 | 0.951 | 0.990 | 0.876 | 0.427 |
| SAM2 YOLO bbox | 128 | 0.769 | 0.826 | 0.817 | 0.844 | 0.888 | 0.779 | 0.367 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.832 | 0.863 | 0.875 | 0.856 | 0.899 | 0.862 | 0.664 |

## Reference Bias Comparison

Görüntü, instance, bbox istemi ve model tahmini aynıdır; yalnız değerlendirme referansı Yayınlanmış SAMRS referansı yerine SAM3 pseudo referansı olarak değiştirilmiştir.

| Model | BBox | Temel Referans IoU | Reference IoU | IoU Farkı |
| --- | --- | --- | --- | --- |
| SAM1 | GT bbox | 0.998 | 0.851 | -0.147 |
| SAM1 | YOLO bbox | 0.782 | 0.705 | -0.077 |
| SAM2 | GT bbox | 0.846 | 0.856 | +0.009 |
| SAM2 | YOLO bbox | 0.707 | 0.705 | -0.002 |
| SAM3 | GT bbox | 0.851 | 1.000 | +0.149 |
| SAM3 | YOLO bbox | 0.707 | 0.760 | +0.054 |

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

- Bu referansta Overall GT-bbox sıralaması SAM3 > SAM2 > SAM1; YOLO-bbox sıralaması SAM3 > SAM2 > SAM1 biçimindedir.
- GT bbox ile YOLO bbox arasındaki fark, eşleşen veya kaçırılan GT instance'lar üzerindeki lokalizasyon/recall etkisini gösterir. Eşleşmeyen detector yanlış pozitifleri maske ortalamasına eklenmediği için bu fark tam uçtan uca instance-segmentation performansı değildir.
- SAM3 modeli YOLO bbox koşulunda temel referansa karşı 0.707, kendi öğretmen ailesinin referansına karşı 0.760 Avg IoU verir; görünür fark +0.054'tür.
- Bu temel-referans farkı tek başına teacher affinity kanıtı değildir; pseudo referansların genel olarak daha kolay olması da fark yaratabilir. Ana exploratory kontrast, deney içi cross-analysis belgesinde aynı dondurulmuş checkpoint'in kendi pseudo referansı ile diğer iki öğretmenin pseudo referanslarını ve göreli model avantajını eşleşmiş olarak karşılaştırır; preregistered confirmatory test değildir.
- Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir.
