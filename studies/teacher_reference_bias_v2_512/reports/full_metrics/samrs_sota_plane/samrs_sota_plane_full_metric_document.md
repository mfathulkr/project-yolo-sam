# SAMRS SOTA Plane Full Metric Document

## Scope

- Veri seti SAMRS SOTA-RBB, hedef sınıf plane ve model giriş çözünürlüğü 1024×1024 pikseldir.
- Test kümesi 512 görüntüdür. Dört overlap × mask-area grubunun her birinde tam 128 görüntü vardır.
- Strata tanımı gereği 512 test görüntüsünün tamamında en az bir uçak vardır; detector tablosu negatif arka plan görüntülerini içeren resmi tam benchmark değil, bu dengeli pozitif test alt kümesindeki gerçek COCO bbox değerlendirmesidir.
- No Overlap, görüntüdeki hiçbir iki GT bbox'un kesişmemesi; Overlap ise en az bir bbox çiftinin IoU değerinin 0,001 veya üstünde olmasıdır.
- Low/High Mask Area ayrımı, görüntüdeki toplam SAMRS pseudo uçak maskesi alanının veri seti için testten önce dondurulan eşiğin altında veya üstünde olmasına göre yapılır.
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

- SAMRS SOTA-RBB görüntüleri DOTA v2.0 remote-sensing sahnelerinden gelir.
- SAMRS veri seti ve üretim kodu kaynağı: https://github.com/ViTAE-Transformer/SAMRS
- SAMRS: Scaling-up Remote Sensing Segmentation Dataset with Segment Anything Model makalesi: https://arxiv.org/abs/2305.02034
- Yayımlanan segmentasyon maskeleri, mevcut detection istemleri SAM1 ViT-H modeline verilerek otomatik üretilmiştir.
- Bu rapordaki GT bbox, yayımlanan özgün SAMRS detection anotasyonudur; pseudo maskeden yeniden türetilmemiştir.
- Bu nedenle raporlanan maske başarısı insan çizimli bağımsız ground truth değil, SAM1 kaynaklı pseudo referansa uyumdur.
- Detector mAP değerleri bbox ölçümüdür. IoU, Dice, Precision ve Recall ise piksel maskesi ölçümüdür.

## YOLO Detector BBox Metrics

- Bu tablo yalnız YOLO detector kutularını değerlendirir; burada ölçülen bbox başarısıdır, maske başarısı değildir.
- BBox mAP50/mAP75/mAP90, tahmin kutusunun GT kutuyla sırasıyla en az 0,50/0,75/0,90 IoU yaptığı eşiklerde confidence sıralaması boyunca hesaplanan gerçek average precision değeridir.
- BBox mAP50-95, 0,50 ile 0,95 arasındaki on bbox IoU eşiğinin AP ortalamasıdır.
- BBox Precision ve Recall değerleri, doğrulama kümesinde seçilip testten önce sabitlenen güven eşiğinde hesaplanır.
- Tablodaki değerler sabit seed 42 sonucudur.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (seed 42) | 512 | 0.913 | 0.797 | 0.209 | 0.665 | 0.917 | 0.843 | 0.851 | 0.782 | 0.375 | 0.344 |

## Resmi SAMRS SAM1 Pseudo Referansı

SAMRS SOTA maskeleri SAM1 ViT-H ve özgün detection istemlerinden üretilmiş pseudo maskelerdir.

### Overall

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 512 görüntüdeki 3.713 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.991 | 0.993 | 0.994 | 0.993 | 0.993 | 0.991 | 0.987 |
| SAM1 YOLO bbox | 512 | 0.813 | 0.824 | 0.827 | 0.824 | 0.835 | 0.821 | 0.782 |
| SAM2 GT bbox | 512 | 0.781 | 0.866 | 0.813 | 0.952 | 0.935 | 0.691 | 0.239 |
| SAM2 YOLO bbox | 512 | 0.679 | 0.744 | 0.705 | 0.805 | 0.804 | 0.639 | 0.242 |
| SAM3 GT bbox | 512 | 0.611 | 0.725 | 0.637 | 0.930 | 0.688 | 0.370 | 0.071 |
| SAM3 YOLO bbox | 512 | 0.537 | 0.629 | 0.556 | 0.800 | 0.597 | 0.364 | 0.105 |

### No Overlap × Low Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 286 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.979 | 0.981 | 0.981 | 0.984 | 0.979 | 0.979 | 0.972 |
| SAM1 YOLO bbox | 128 | 0.765 | 0.780 | 0.773 | 0.792 | 0.794 | 0.766 | 0.727 |
| SAM2 GT bbox | 128 | 0.769 | 0.854 | 0.795 | 0.952 | 0.916 | 0.724 | 0.203 |
| SAM2 YOLO bbox | 128 | 0.631 | 0.698 | 0.649 | 0.777 | 0.752 | 0.573 | 0.203 |
| SAM3 GT bbox | 128 | 0.632 | 0.745 | 0.666 | 0.924 | 0.745 | 0.392 | 0.028 |
| SAM3 YOLO bbox | 128 | 0.467 | 0.568 | 0.494 | 0.754 | 0.507 | 0.231 | 0.031 |

### No Overlap × High Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 412 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.979 | 0.984 | 0.987 | 0.983 | 0.981 | 0.976 | 0.966 |
| SAM1 YOLO bbox | 128 | 0.901 | 0.914 | 0.914 | 0.918 | 0.922 | 0.913 | 0.862 |
| SAM2 GT bbox | 128 | 0.827 | 0.892 | 0.885 | 0.913 | 0.947 | 0.850 | 0.330 |
| SAM2 YOLO bbox | 128 | 0.780 | 0.842 | 0.834 | 0.865 | 0.888 | 0.784 | 0.337 |
| SAM3 GT bbox | 128 | 0.625 | 0.736 | 0.675 | 0.904 | 0.680 | 0.398 | 0.119 |
| SAM3 YOLO bbox | 128 | 0.590 | 0.691 | 0.635 | 0.851 | 0.638 | 0.415 | 0.153 |

### Overlap × Low Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.209 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.997 | 0.998 | 0.999 | 0.997 | 0.999 | 0.998 | 0.996 |
| SAM1 YOLO bbox | 128 | 0.676 | 0.690 | 0.697 | 0.687 | 0.706 | 0.682 | 0.633 |
| SAM2 GT bbox | 128 | 0.702 | 0.815 | 0.734 | 0.957 | 0.897 | 0.444 | 0.036 |
| SAM2 YOLO bbox | 128 | 0.511 | 0.588 | 0.532 | 0.680 | 0.648 | 0.369 | 0.031 |
| SAM3 GT bbox | 128 | 0.552 | 0.683 | 0.587 | 0.903 | 0.651 | 0.175 | 0.000 |
| SAM3 YOLO bbox | 128 | 0.407 | 0.501 | 0.428 | 0.670 | 0.483 | 0.154 | 0.000 |

### Overlap × High Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.806 uçak örneğini kapsar. YOLO bbox değerleri sabit seed 42 sonucudur.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.991 | 0.993 | 0.993 | 0.993 | 0.993 | 0.991 | 0.988 |
| SAM1 YOLO bbox | 128 | 0.891 | 0.900 | 0.903 | 0.899 | 0.909 | 0.902 | 0.872 |
| SAM2 GT bbox | 128 | 0.825 | 0.895 | 0.853 | 0.958 | 0.961 | 0.815 | 0.360 |
| SAM2 YOLO bbox | 128 | 0.775 | 0.834 | 0.801 | 0.880 | 0.896 | 0.798 | 0.367 |
| SAM3 GT bbox | 128 | 0.645 | 0.748 | 0.657 | 0.955 | 0.705 | 0.491 | 0.114 |
| SAM3 YOLO bbox | 128 | 0.623 | 0.711 | 0.633 | 0.883 | 0.677 | 0.513 | 0.176 |

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

- Resmi SAMRS pseudo referansında GT-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,991/0,781/0,611 olarak ölçülmüştür.
- YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,813/0,679/0,537 olarak ölçülmüştür; bunlar sabit seed 42 detector sonuçlarıdır.
- GT bbox yerine YOLO bbox kullanıldığında Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla 0,178/0,102/0,075 olmuştur.
- SAM1'in GT-bbox IoU değeri, SAM2 ve SAM3 ortalamasından 0.295 daha yüksektir; bu fark resmi referansın SAM1 ile üretilmiş olmasıyla birlikte yorumlanmalıdır.
- SAMRS maskeleri ayrı bir resmi üretim hattında SAM1 ViT-H ile oluşturulduğu için buradaki SAM1 satırı kontrollü iSAID kimlik kontrolü kadar birebir değildir; yine de aynı model ailesine çok güçlü yakınlık gösterir.
- GT-bbox üç-model ortalamasında en yüksek alt grup Overlap × High Mask Area (0.820), en düşük alt grup Overlap × Low Mask Area (0.750) olmuştur.
- SAM1 gt bbox koşulunda Overall Precision 0.994, Recall 0.993 olmuştur. Precision daha yüksek olduğu için model boyadığı bölgelerde daha temizdir; buna karşılık bazı gerçek nesne piksellerini eksik bırakmaktadır.
- SAM1'in yüksek skoru, aynı model ailesinin ürettiği pseudo maskelere biçimsel yakınlık içerebilir; tek başına insan etiketli gerçek performans kanıtı değildir.
- GT bbox ile YOLO bbox arasındaki fark, otomatik detector hatasının segmentasyon zincirine eklediği kaybı gösterir.
- SAMRS pseudo etiketleri eğitim ve ölçeklenebilir ön-etiketleme için yararlı olabilir; nihai model karşılaştırması bağımsız insan etiketli test kümesiyle yapılmalıdır.
