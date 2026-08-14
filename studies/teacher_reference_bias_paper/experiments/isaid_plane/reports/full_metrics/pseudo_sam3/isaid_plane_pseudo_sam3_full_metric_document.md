# iSAID Plane - SAM3 pseudo referansı Full Metric Document

## Scope

- Veri kaynağı iSAID, hedef sınıf uçak ve model giriş çözünürlüğü 1024×1024 pikseldir.
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
- Dört ortalama maske metriği nesne örneği düzeyinde (instance-level) önce her uçak için hesaplanır, sonra bütün örnekler eşit ağırlıkla ortalanır. Büyük nesneler küçük nesnelerin sonucunu perdelemez.
- Her satırda temel veri seti anotasyonuyla varlığı bilinen bir uçak vardır. Bu nedenle boş pseudo referans eksik etikettir; tahmin de boş olsa bile maske metrikleri 0 kabul edilir ve referans kapsama kaybı ayrıca raporlanır.
- IoU ≥ 0.50/0.75/0.90 sütunları, ilgili IoU eşiğini geçen uçak maskelerinin oranıdır. Bunlar mAP değildir ve raporda mAP gibi adlandırılmaz.
- YOLO'nun kaçırdığı bir gerçek uçak, YOLO-bbox maske tablosunda boş tahmin olarak değerlendirilir ve o örneğin maske skorları sıfır olur. Herhangi bir gerçek nesneyle eşleşmeyen yanlış pozitif YOLO kutuları ise instance maske ortalamasına sahte bir referans örneği olarak eklenmez; bunların etkisi detector Precision, Recall ve mAP değerlerinde ölçülür.
- Maske tabloları her GT uçak örneğini değerlendirir; YOLO'nun eşleştiremediği GT örnekleri de boş tahmin ve sıfır skorla hesaba katılır. Bu değerlendirme gerçek COCO segmentation AP ile aynı değildir. Confidence sırasındaki bütün maskeleri ve yanlış pozitifleri kullanan uçtan uca COCO mask AP bu raporda ayrıca çalıştırılmadığı için IoU eşik oranları AP veya mAP diye yeniden adlandırılmamıştır.
- Overall tablosu 512 görüntüyü, diğer tabloların her biri 128 görüntüyü kapsar.
- GT-bbox satırları tek sabit koşuldur. YOLO-bbox satırlarındaki değerler sabit seed 42 sonucudur.

## Dataset Context

- Bu belgede değerlendirme referansı SAM3 pseudo referansı ve değerlendirilen instance sayısı 5.447.
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
| YOLO26x (seed 42) | 512 | 0.920 | 0.847 | 0.545 | 0.762 | 0.925 | 0.896 | 0.868 | 0.840 | 0.632 | 0.612 |

## SAM3 pseudo referansı

Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve SAM3 pseudo referansı ile değerlendirilmiştir.

### Overall

Referans: SAM3 pseudo referansı. Bu tablo 512 görüntüdeki 5.447 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.820 | 0.895 | 0.858 | 0.949 | 0.975 | 0.810 | 0.263 |
| SAM1 YOLO bbox | 512 | 0.742 | 0.808 | 0.774 | 0.855 | 0.882 | 0.740 | 0.235 |
| SAM2 GT bbox | 512 | 0.784 | 0.872 | 0.809 | 0.962 | 0.966 | 0.709 | 0.110 |
| SAM2 YOLO bbox | 512 | 0.706 | 0.785 | 0.729 | 0.862 | 0.872 | 0.639 | 0.094 |
| SAM3 GT bbox | 512 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 512 | 0.863 | 0.877 | 0.880 | 0.876 | 0.891 | 0.875 | 0.803 |

### No Overlap × Low Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 439 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.774 | 0.862 | 0.819 | 0.934 | 0.943 | 0.718 | 0.130 |
| SAM1 YOLO bbox | 128 | 0.696 | 0.771 | 0.728 | 0.834 | 0.847 | 0.651 | 0.118 |
| SAM2 GT bbox | 128 | 0.746 | 0.845 | 0.770 | 0.960 | 0.943 | 0.551 | 0.059 |
| SAM2 YOLO bbox | 128 | 0.666 | 0.751 | 0.685 | 0.845 | 0.843 | 0.510 | 0.043 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.845 | 0.859 | 0.861 | 0.859 | 0.875 | 0.852 | 0.788 |

### No Overlap × High Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 622 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.778 | 0.864 | 0.813 | 0.945 | 0.955 | 0.662 | 0.211 |
| SAM1 YOLO bbox | 128 | 0.721 | 0.800 | 0.754 | 0.869 | 0.886 | 0.609 | 0.199 |
| SAM2 GT bbox | 128 | 0.790 | 0.874 | 0.819 | 0.955 | 0.960 | 0.728 | 0.172 |
| SAM2 YOLO bbox | 128 | 0.726 | 0.803 | 0.755 | 0.874 | 0.883 | 0.658 | 0.164 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.873 | 0.888 | 0.895 | 0.887 | 0.894 | 0.870 | 0.817 |

### Overlap × Low Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 1.708 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.800 | 0.884 | 0.843 | 0.944 | 0.978 | 0.787 | 0.084 |
| SAM1 YOLO bbox | 128 | 0.722 | 0.796 | 0.758 | 0.847 | 0.882 | 0.731 | 0.081 |
| SAM2 GT bbox | 128 | 0.751 | 0.854 | 0.776 | 0.962 | 0.976 | 0.577 | 0.019 |
| SAM2 YOLO bbox | 128 | 0.674 | 0.765 | 0.698 | 0.856 | 0.876 | 0.524 | 0.015 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.853 | 0.871 | 0.876 | 0.868 | 0.891 | 0.876 | 0.768 |

### Overlap × High Mask Area

Referans: SAM3 pseudo referansı. Bu tablo 128 görüntüdeki 2.678 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.850 | 0.914 | 0.885 | 0.955 | 0.983 | 0.874 | 0.410 |
| SAM1 YOLO bbox | 128 | 0.767 | 0.823 | 0.797 | 0.860 | 0.886 | 0.791 | 0.361 |
| SAM2 GT bbox | 128 | 0.809 | 0.887 | 0.833 | 0.965 | 0.965 | 0.814 | 0.162 |
| SAM2 YOLO bbox | 128 | 0.728 | 0.799 | 0.751 | 0.866 | 0.872 | 0.729 | 0.137 |
| SAM3 GT bbox | 128 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SAM3 YOLO bbox | 128 | 0.869 | 0.882 | 0.883 | 0.882 | 0.894 | 0.880 | 0.825 |

## Reference Bias Comparison

Görüntü, instance, bbox istemi ve model tahmini aynıdır; yalnız değerlendirme referansı İnsan referansı yerine SAM3 pseudo referansı olarak değiştirilmiştir.

| Model | BBox | Temel Referans IoU | Reference IoU | IoU Farkı |
| --- | --- | --- | --- | --- |
| SAM1 | GT bbox | 0.653 | 0.820 | +0.167 |
| SAM1 | YOLO bbox | 0.597 | 0.742 | +0.145 |
| SAM2 | GT bbox | 0.629 | 0.784 | +0.154 |
| SAM2 | YOLO bbox | 0.574 | 0.706 | +0.132 |
| SAM3 | GT bbox | 0.700 | 1.000 | +0.300 |
| SAM3 | YOLO bbox | 0.638 | 0.863 | +0.224 |

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

- Bu referansta Overall GT-bbox sıralaması SAM3 > SAM1 > SAM2; YOLO-bbox sıralaması SAM3 > SAM1 > SAM2 biçimindedir.
- GT bbox ile YOLO bbox arasındaki fark, eşleşen veya kaçırılan GT instance'lar üzerindeki lokalizasyon/recall etkisini gösterir. Eşleşmeyen detector yanlış pozitifleri maske ortalamasına eklenmediği için bu fark tam uçtan uca instance-segmentation performansı değildir.
- SAM3 modeli YOLO bbox koşulunda temel referansa karşı 0.638, kendi öğretmen ailesinin referansına karşı 0.863 Avg IoU verir; görünür fark +0.224'tür.
- Bu temel-referans farkı tek başına teacher affinity kanıtı değildir; pseudo referansların genel olarak daha kolay olması da fark yaratabilir. Ana exploratory kontrast, deney içi cross-analysis belgesinde aynı dondurulmuş checkpoint'in kendi pseudo referansı ile diğer iki öğretmenin pseudo referanslarını ve göreli model avantajını eşleşmiş olarak karşılaştırır; preregistered confirmatory test değildir.
- Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir.
