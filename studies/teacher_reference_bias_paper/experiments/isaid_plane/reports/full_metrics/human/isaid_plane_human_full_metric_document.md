# Isaid Plane - İnsan referansı Full Metric Document

## Scope

- Veri kaynağı iSAID, hedef sınıf uçak ve model giriş çözünürlüğü 1024×1024 pikseldir.
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
- Dört ortalama maske metriği nesne örneği düzeyinde (instance-level) önce her uçak için hesaplanır, sonra bütün örnekler eşit ağırlıkla ortalanır. Büyük nesneler küçük nesnelerin sonucunu perdelemez.
- Her satırda temel veri seti anotasyonuyla varlığı bilinen bir uçak vardır. Bu nedenle boş pseudo referans eksik etikettir; tahmin de boş olsa bile maske metrikleri 0 kabul edilir ve referans kapsama kaybı ayrıca raporlanır.
- IoU ≥ 0.50/0.75/0.90 sütunları, ilgili IoU eşiğini geçen uçak maskelerinin oranıdır. Bunlar mAP değildir ve raporda mAP gibi adlandırılmaz.
- YOLO'nun kaçırdığı bir gerçek uçak, YOLO-bbox maske tablosunda boş tahmin olarak değerlendirilir ve o örneğin maske skorları sıfır olur. Herhangi bir gerçek nesneyle eşleşmeyen yanlış pozitif YOLO kutuları ise instance maske ortalamasına sahte bir referans örneği olarak eklenmez; bunların etkisi detector Precision, Recall ve mAP değerlerinde ölçülür.
- Maske tabloları her GT uçak örneğini değerlendirir; YOLO'nun eşleştiremediği GT örnekleri de boş tahmin ve sıfır skorla hesaba katılır. Bu değerlendirme gerçek COCO segmentation AP ile aynı değildir. Confidence sırasındaki bütün maskeleri ve yanlış pozitifleri kullanan uçtan uca COCO mask AP bu raporda ayrıca çalıştırılmadığı için IoU eşik oranları AP veya mAP diye yeniden adlandırılmamıştır.
- Overall tablosu 512 görüntüyü, diğer tabloların her biri 128 görüntüyü kapsar.
- GT-bbox satırları tek sabit koşuldur. YOLO-bbox satırlarındaki değerler sabit seed 42 sonucudur.

## Dataset Context

- Bu belgede değerlendirme referansı İnsan referansı ve değerlendirilen instance sayısı 5.447.
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
| YOLO26x (seed 42) | 512 | 0.920 | 0.847 | 0.545 | 0.762 | 0.925 | 0.896 | 0.868 | 0.840 | 0.632 | 0.612 |

## İnsan referansı

Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve İnsan referansı ile değerlendirilmiştir.

### Overall

Referans: İnsan referansı. Bu tablo 512 görüntüdeki 5.447 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.653 | 0.780 | 0.686 | 0.935 | 0.905 | 0.195 | 0.005 |
| SAM1 YOLO bbox | 512 | 0.597 | 0.711 | 0.627 | 0.842 | 0.837 | 0.178 | 0.003 |
| SAM2 GT bbox | 512 | 0.629 | 0.761 | 0.651 | 0.952 | 0.871 | 0.168 | 0.004 |
| SAM2 YOLO bbox | 512 | 0.574 | 0.692 | 0.595 | 0.852 | 0.807 | 0.146 | 0.004 |
| SAM3 GT bbox | 512 | 0.700 | 0.814 | 0.746 | 0.920 | 0.920 | 0.396 | 0.010 |
| SAM3 YOLO bbox | 512 | 0.638 | 0.740 | 0.682 | 0.825 | 0.849 | 0.366 | 0.008 |

### No Overlap × Low Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 439 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.593 | 0.727 | 0.626 | 0.926 | 0.802 | 0.144 | 0.014 |
| SAM1 YOLO bbox | 128 | 0.543 | 0.662 | 0.571 | 0.824 | 0.749 | 0.128 | 0.009 |
| SAM2 GT bbox | 128 | 0.565 | 0.704 | 0.584 | 0.951 | 0.747 | 0.123 | 0.000 |
| SAM2 YOLO bbox | 128 | 0.519 | 0.643 | 0.537 | 0.834 | 0.706 | 0.105 | 0.000 |
| SAM3 GT bbox | 128 | 0.656 | 0.776 | 0.699 | 0.917 | 0.859 | 0.308 | 0.002 |
| SAM3 YOLO bbox | 128 | 0.602 | 0.707 | 0.643 | 0.807 | 0.802 | 0.296 | 0.002 |

### No Overlap × High Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 622 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.649 | 0.772 | 0.681 | 0.935 | 0.894 | 0.249 | 0.002 |
| SAM1 YOLO bbox | 128 | 0.611 | 0.726 | 0.642 | 0.862 | 0.850 | 0.228 | 0.003 |
| SAM2 GT bbox | 128 | 0.670 | 0.787 | 0.695 | 0.954 | 0.912 | 0.323 | 0.024 |
| SAM2 YOLO bbox | 128 | 0.625 | 0.734 | 0.652 | 0.871 | 0.855 | 0.288 | 0.023 |
| SAM3 GT bbox | 128 | 0.735 | 0.832 | 0.781 | 0.923 | 0.905 | 0.611 | 0.037 |
| SAM3 YOLO bbox | 128 | 0.688 | 0.777 | 0.735 | 0.846 | 0.862 | 0.569 | 0.040 |

### Overlap × Low Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 1.708 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.633 | 0.767 | 0.671 | 0.926 | 0.881 | 0.139 | 0.001 |
| SAM1 YOLO bbox | 128 | 0.577 | 0.695 | 0.611 | 0.827 | 0.814 | 0.128 | 0.001 |
| SAM2 GT bbox | 128 | 0.586 | 0.731 | 0.612 | 0.938 | 0.826 | 0.046 | 0.000 |
| SAM2 YOLO bbox | 128 | 0.532 | 0.661 | 0.559 | 0.831 | 0.770 | 0.043 | 0.000 |
| SAM3 GT bbox | 128 | 0.672 | 0.797 | 0.729 | 0.901 | 0.902 | 0.292 | 0.002 |
| SAM3 YOLO bbox | 128 | 0.611 | 0.720 | 0.666 | 0.800 | 0.827 | 0.275 | 0.002 |

### Overlap × High Mask Area

Referans: İnsan referansı. Bu tablo 128 görüntüdeki 2.678 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.676 | 0.799 | 0.707 | 0.941 | 0.941 | 0.226 | 0.006 |
| SAM1 YOLO bbox | 128 | 0.616 | 0.726 | 0.642 | 0.849 | 0.863 | 0.207 | 0.003 |
| SAM2 GT bbox | 128 | 0.658 | 0.782 | 0.676 | 0.960 | 0.912 | 0.217 | 0.003 |
| SAM2 YOLO bbox | 128 | 0.597 | 0.710 | 0.614 | 0.863 | 0.835 | 0.186 | 0.002 |
| SAM3 GT bbox | 128 | 0.716 | 0.827 | 0.757 | 0.933 | 0.945 | 0.428 | 0.010 |
| SAM3 YOLO bbox | 128 | 0.650 | 0.750 | 0.687 | 0.839 | 0.867 | 0.388 | 0.005 |

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
- GT bbox ile YOLO bbox arasındaki fark, segmenterden önceki detection hatasının uçtan uca sisteme etkisini gösterir.
- Bu insan referansı model ailelerinden bağımsız olduğu için modeller arası kalite karşılaştırmasının güvenilir kontrolüdür.
- Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir.
