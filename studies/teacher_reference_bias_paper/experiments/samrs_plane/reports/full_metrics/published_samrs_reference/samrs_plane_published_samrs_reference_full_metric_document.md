# Samrs Plane - Yayınlanmış SAMRS referansı Full Metric Document

## Scope

- Veri kaynağı SAMRS SOTA, hedef sınıf uçak ve model giriş çözünürlüğü 1024×1024 pikseldir.
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

- Bu belgede değerlendirme referansı Yayınlanmış SAMRS referansı ve değerlendirilen instance sayısı 3.713.
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
| YOLO26x (seed 42) | 512 | 0.913 | 0.797 | 0.209 | 0.665 | 0.917 | 0.843 | 0.851 | 0.782 | 0.375 | 0.344 |

## Yayınlanmış SAMRS referansı

Bütün SAM1/2/3 tahminleri değişmeden tutulmuş ve Yayınlanmış SAMRS referansı ile değerlendirilmiştir.

### Overall

Referans: Yayınlanmış SAMRS referansı. Bu tablo 512 görüntüdeki 3.713 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.991 | 0.993 | 0.994 | 0.993 | 0.993 | 0.991 | 0.987 |
| SAM1 YOLO bbox | 512 | 0.813 | 0.824 | 0.827 | 0.824 | 0.835 | 0.821 | 0.782 |
| SAM2 GT bbox | 512 | 0.781 | 0.866 | 0.813 | 0.952 | 0.935 | 0.691 | 0.239 |
| SAM2 YOLO bbox | 512 | 0.679 | 0.744 | 0.705 | 0.805 | 0.804 | 0.639 | 0.242 |
| SAM3 GT bbox | 512 | 0.808 | 0.885 | 0.925 | 0.863 | 0.969 | 0.778 | 0.226 |
| SAM3 YOLO bbox | 512 | 0.691 | 0.754 | 0.793 | 0.727 | 0.823 | 0.680 | 0.213 |

### No Overlap × Low Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 286 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.979 | 0.981 | 0.981 | 0.984 | 0.979 | 0.979 | 0.972 |
| SAM1 YOLO bbox | 128 | 0.765 | 0.780 | 0.773 | 0.792 | 0.794 | 0.766 | 0.727 |
| SAM2 GT bbox | 128 | 0.769 | 0.854 | 0.795 | 0.952 | 0.916 | 0.724 | 0.203 |
| SAM2 YOLO bbox | 128 | 0.631 | 0.698 | 0.649 | 0.777 | 0.752 | 0.573 | 0.203 |
| SAM3 GT bbox | 128 | 0.812 | 0.886 | 0.908 | 0.878 | 0.972 | 0.822 | 0.182 |
| SAM3 YOLO bbox | 128 | 0.659 | 0.719 | 0.734 | 0.717 | 0.780 | 0.675 | 0.168 |

### No Overlap × High Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 412 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.979 | 0.984 | 0.987 | 0.983 | 0.981 | 0.976 | 0.966 |
| SAM1 YOLO bbox | 128 | 0.901 | 0.914 | 0.914 | 0.918 | 0.922 | 0.913 | 0.862 |
| SAM2 GT bbox | 128 | 0.827 | 0.892 | 0.885 | 0.913 | 0.947 | 0.850 | 0.330 |
| SAM2 YOLO bbox | 128 | 0.780 | 0.842 | 0.834 | 0.865 | 0.888 | 0.784 | 0.337 |
| SAM3 GT bbox | 128 | 0.788 | 0.867 | 0.940 | 0.817 | 0.949 | 0.728 | 0.250 |
| SAM3 YOLO bbox | 128 | 0.743 | 0.818 | 0.889 | 0.769 | 0.896 | 0.682 | 0.218 |

### Overlap × Low Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 1.209 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.997 | 0.998 | 0.999 | 0.997 | 0.999 | 0.998 | 0.996 |
| SAM1 YOLO bbox | 128 | 0.676 | 0.690 | 0.697 | 0.687 | 0.706 | 0.682 | 0.633 |
| SAM2 GT bbox | 128 | 0.702 | 0.815 | 0.734 | 0.957 | 0.897 | 0.444 | 0.035 |
| SAM2 YOLO bbox | 128 | 0.511 | 0.588 | 0.532 | 0.680 | 0.648 | 0.369 | 0.031 |
| SAM3 GT bbox | 128 | 0.778 | 0.870 | 0.895 | 0.865 | 0.970 | 0.704 | 0.070 |
| SAM3 YOLO bbox | 128 | 0.562 | 0.623 | 0.650 | 0.609 | 0.691 | 0.533 | 0.062 |

### Overlap × High Mask Area

Referans: Yayınlanmış SAMRS referansı. Bu tablo 128 görüntüdeki 1.806 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.991 | 0.993 | 0.993 | 0.993 | 0.993 | 0.991 | 0.988 |
| SAM1 YOLO bbox | 128 | 0.891 | 0.900 | 0.903 | 0.899 | 0.909 | 0.902 | 0.872 |
| SAM2 GT bbox | 128 | 0.825 | 0.895 | 0.853 | 0.958 | 0.961 | 0.815 | 0.360 |
| SAM2 YOLO bbox | 128 | 0.775 | 0.834 | 0.801 | 0.880 | 0.896 | 0.798 | 0.367 |
| SAM3 GT bbox | 128 | 0.831 | 0.900 | 0.944 | 0.869 | 0.973 | 0.833 | 0.331 |
| SAM3 YOLO bbox | 128 | 0.771 | 0.832 | 0.877 | 0.798 | 0.903 | 0.779 | 0.321 |

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

- Bu referansta Overall GT-bbox sıralaması SAM1 > SAM3 > SAM2; YOLO-bbox sıralaması SAM1 > SAM3 > SAM2 biçimindedir.
- GT bbox ile YOLO bbox arasındaki fark, segmenterden önceki detection hatasının uçtan uca sisteme etkisini gösterir.
- Bu yayımlanmış SAMRS referansı bağımsız insan kontrolü değildir; sonuçlar model başarısından çok SAM-türevi referansla uyumu da içerir.
- Ana sonuç yalnız Overall tablosuna dayandırılmamalıdır; aynı yönün dört Overlap × Mask Area tabakasında korunup korunmadığı deney içi çapraz analiz belgesinde ayrıca gösterilir.
