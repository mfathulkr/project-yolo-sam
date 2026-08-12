# Samrs Small Vehicle - Yayınlanmış SAMRS referansı Full Metric Document

## Scope

- Veri kaynağı SAMRS SOTA, hedef sınıf küçük araç ve model giriş çözünürlüğü 1024×1024 pikseldir.
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

- Bu belgede değerlendirme referansı Yayınlanmış SAMRS referansı ve değerlendirilen instance sayısı 7.659.
- Referans kümesinde 0 boş maske vardır (0.00%). Bilinen pozitif nesnede boş pseudo maske başarı sayılmaz ve 0 puanlanır.
- Detector mAP değerleri bbox ölçümüdür. Avg IoU, Dice, Precision, Recall ve IoU eşik oranları piksel maskesi ölçümüdür; IoU eşik oranları mAP değildir.
- Bu etiketler insan ground truth değildir. SAMRS veri seti tarafından yayımlanmış, SAM tabanlı otomatik üretim hattından gelen referanslardır.
- Yayımlanmış SAMRS etiketi ile bu çalışmada güncel SAM1 checkpoint'i kullanılarak yeniden üretilen referans birbirine çok yakın olabilir; fakat aynı dosya veya bağımsız insan anotasyonu değildir.

## YOLO Detector BBox Metrics

- Bu tablo yalnız YOLO detector kutularını değerlendirir; burada ölçülen bbox başarısıdır, maske başarısı değildir.
- BBox mAP50/mAP75/mAP90, tahmin kutusunun GT kutuyla sırasıyla en az 0,50/0,75/0,90 IoU yaptığı eşiklerde confidence sıralaması boyunca hesaplanan gerçek average precision değeridir.
- BBox mAP50-95, 0,50 ile 0,95 arasındaki on bbox IoU eşiğinin AP ortalamasıdır.
- BBox Precision ve Recall değerleri, doğrulama kümesinde seçilip testten önce sabitlenen güven eşiğinde hesaplanır.
- Tablodaki değerler sabit seed 42 sonucudur.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (seed 42) | 512 | 0.819 | 0.534 | 0.072 | 0.502 | 0.720 | 0.824 | 0.529 | 0.605 | 0.131 | 0.150 |

## Yayınlanmış SAMRS referansı

Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve Yayınlanmış SAMRS referansı ile değerlendirilmiştir.

### Overall

Referans: Yayınlanmış SAMRS referansı. Bu tablo 512 görüntüdeki 7.659 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.998 | 0.999 | 1.000 | 0.999 | 1.000 | 0.999 | 0.998 |
| SAM1 YOLO bbox | 512 | 0.782 | 0.801 | 0.807 | 0.798 | 0.819 | 0.800 | 0.714 |
| SAM2 GT bbox | 512 | 0.846 | 0.911 | 0.892 | 0.945 | 0.978 | 0.831 | 0.427 |
| SAM2 YOLO bbox | 512 | 0.707 | 0.757 | 0.752 | 0.772 | 0.809 | 0.714 | 0.369 |
| SAM3 GT bbox | 512 | 0.851 | 0.915 | 0.919 | 0.922 | 0.983 | 0.860 | 0.403 |
| SAM3 YOLO bbox | 512 | 0.707 | 0.757 | 0.770 | 0.754 | 0.811 | 0.727 | 0.348 |

### No Overlap × Low Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 596 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.999 | 0.999 | 1.000 | 0.999 | 1.000 | 1.000 | 0.997 |
| SAM1 YOLO bbox | 128 | 0.616 | 0.631 | 0.637 | 0.627 | 0.648 | 0.641 | 0.554 |
| SAM2 GT bbox | 128 | 0.817 | 0.891 | 0.865 | 0.944 | 0.955 | 0.800 | 0.305 |
| SAM2 YOLO bbox | 128 | 0.556 | 0.597 | 0.603 | 0.595 | 0.646 | 0.586 | 0.218 |
| SAM3 GT bbox | 128 | 0.832 | 0.904 | 0.911 | 0.912 | 0.987 | 0.837 | 0.250 |
| SAM3 YOLO bbox | 128 | 0.551 | 0.594 | 0.619 | 0.576 | 0.648 | 0.581 | 0.186 |

### No Overlap × High Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 1.478 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.999 | 0.999 | 1.000 | 0.999 | 0.999 | 0.999 | 0.999 |
| SAM1 YOLO bbox | 128 | 0.809 | 0.820 | 0.819 | 0.824 | 0.833 | 0.822 | 0.775 |
| SAM2 GT bbox | 128 | 0.904 | 0.946 | 0.945 | 0.955 | 0.988 | 0.938 | 0.702 |
| SAM2 YOLO bbox | 128 | 0.762 | 0.795 | 0.796 | 0.799 | 0.833 | 0.798 | 0.608 |
| SAM3 GT bbox | 128 | 0.892 | 0.939 | 0.949 | 0.938 | 0.986 | 0.926 | 0.643 |
| SAM3 YOLO bbox | 128 | 0.752 | 0.790 | 0.798 | 0.786 | 0.832 | 0.791 | 0.566 |

### Overlap × Low Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 1.884 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.998 | 0.999 | 0.999 | 0.999 | 1.000 | 0.999 | 0.998 |
| SAM1 YOLO bbox | 128 | 0.672 | 0.694 | 0.702 | 0.689 | 0.717 | 0.697 | 0.575 |
| SAM2 GT bbox | 128 | 0.797 | 0.881 | 0.856 | 0.926 | 0.974 | 0.725 | 0.178 |
| SAM2 YOLO bbox | 128 | 0.581 | 0.639 | 0.636 | 0.655 | 0.703 | 0.554 | 0.120 |
| SAM3 GT bbox | 128 | 0.804 | 0.887 | 0.890 | 0.899 | 0.982 | 0.761 | 0.167 |
| SAM3 YOLO bbox | 128 | 0.581 | 0.640 | 0.655 | 0.636 | 0.708 | 0.562 | 0.115 |

### Overlap × High Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 3.701 küçük araç örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.998 | 0.999 | 1.000 | 0.998 | 1.000 | 0.999 | 0.997 |
| SAM1 YOLO bbox | 128 | 0.854 | 0.875 | 0.883 | 0.870 | 0.893 | 0.869 | 0.786 |
| SAM2 GT bbox | 128 | 0.853 | 0.915 | 0.894 | 0.951 | 0.979 | 0.848 | 0.463 |
| SAM2 YOLO bbox | 128 | 0.774 | 0.828 | 0.817 | 0.849 | 0.880 | 0.784 | 0.424 |
| SAM3 GT bbox | 128 | 0.862 | 0.921 | 0.923 | 0.929 | 0.982 | 0.887 | 0.451 |
| SAM3 YOLO bbox | 128 | 0.777 | 0.830 | 0.841 | 0.829 | 0.882 | 0.809 | 0.405 |

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

- Bu referansta Overall GT-bbox sıralaması SAM1 > SAM3 > SAM2; YOLO-bbox sıralaması SAM1 > SAM2 > SAM3 biçimindedir.
- GT bbox ile YOLO bbox arasındaki fark, segmenterden önceki detection hatasının uçtan uca sisteme etkisini gösterir.
- Bu yayımlanmış SAMRS referansı bağımsız insan kontrolü değildir; sonuçlar model başarısından çok SAM-türevi referansla uyumu da içerir.
- Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir.
