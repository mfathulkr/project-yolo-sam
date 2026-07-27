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
- YOLO bbox sonuçları üç bağımsız YOLO eğitiminin ortalaması ± standart sapmasıdır.
- Maske metrikleri uçak örneği düzeyinde hesaplanır; büyük nesneler küçük nesnelerin sonucunu piksel sayısıyla baskılamaz.

## Metric Logic

- TP, modelin doğru biçimde nesne olarak işaretlediği pikseldir. FP, nesne olmadığı hâlde nesne diye işaretlenen; FN ise nesne olduğu hâlde kaçırılan pikseldir.
- IoU = TP / (TP + FP + FN). Tahmin ve referans maskenin ortak alanını birleşim alanına böler; 1 kusursuz, 0 hiç örtüşme yok demektir.
- Dice = 2TP / (2TP + FP + FN). IoU ile aynı davranışı farklı ölçekle ifade eder.
- Precision = TP / (TP + FP). Modelin boyadığı piksellerin ne kadarının gerçekten nesne olduğunu gösterir; fazla alan boyamak precision değerini düşürür.
- Recall = TP / (TP + FN). Gerçek nesne piksellerinin ne kadarının yakalandığını gösterir; eksik maske recall değerini düşürür.
- Dört ortalama maske metriği nesne örneği düzeyinde (instance-level) önce her uçak için hesaplanır, sonra bütün uçaklar eşit ağırlıkla ortalanır. Büyük uçaklar küçük uçakların sonucunu perdelemez.
- IoU ≥ 0.50/0.75/0.90 sütunları, ilgili IoU eşiğini geçen uçak maskelerinin oranıdır. Bunlar mAP değildir ve raporda mAP gibi adlandırılmaz.
- YOLO'nun kaçırdığı bir gerçek uçak, YOLO-bbox maske tablosunda boş tahmin olarak değerlendirilir ve o örneğin maske skorları sıfır olur. Herhangi bir gerçek uçakla eşleşmeyen yanlış pozitif YOLO kutuları ise instance maske ortalamasına sahte bir referans örneği olarak eklenmez; bunların etkisi detector Precision, Recall ve mAP değerlerinde ölçülür.
- Maske tabloları her GT uçak örneğini değerlendirir; YOLO'nun eşleştiremediği GT örnekleri de boş tahmin ve sıfır skorla hesaba katılır. Bu değerlendirme gerçek COCO segmentation AP ile aynı değildir. Confidence sırasındaki bütün maskeleri ve yanlış pozitifleri kullanan uçtan uca COCO mask AP bu raporda ayrıca çalıştırılmadığı için IoU eşik oranları AP veya mAP diye yeniden adlandırılmamıştır.
- Overall tablosu 512 görüntüyü, diğer tabloların her biri 128 görüntüyü kapsar.
- GT-bbox satırları tek sabit koşuldur. YOLO-bbox satırlarındaki değerler üç ayrı YOLO eğitiminin ortalaması ± standart sapmasıdır.

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
- Tablodaki değerler üç ayrı YOLO eğitiminin ortalaması ± standart sapmasıdır.

| Detector | Images | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26x (3 seed) | 512 | 0.918 ± 0.006 | 0.808 ± 0.009 | 0.212 ± 0.004 | 0.671 ± 0.007 | 0.928 ± 0.010 | 0.844 ± 0.013 | 0.865 ± 0.013 | 0.787 ± 0.012 | 0.384 ± 0.009 | 0.350 ± 0.006 |

## Resmi SAMRS SAM1 Pseudo Referansı

SAMRS SOTA maskeleri SAM1 ViT-H ve özgün detection istemlerinden üretilmiş pseudo maskelerdir.

### Overall

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 512 görüntüdeki 3.713 uçak örneğini kapsar. YOLO bbox değerleri üç seed ortalaması ± standart sapmadır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 512 | 0.991 | 0.993 | 0.994 | 0.993 | 0.993 | 0.991 | 0.987 |
| SAM1 YOLO bbox | 512 | 0.814 ± 0.012 | 0.826 ± 0.013 | 0.829 ± 0.012 | 0.825 ± 0.013 | 0.836 ± 0.014 | 0.822 ± 0.013 | 0.782 ± 0.009 |
| SAM2 GT bbox | 512 | 0.781 | 0.866 | 0.813 | 0.952 | 0.935 | 0.691 | 0.239 |
| SAM2 YOLO bbox | 512 | 0.679 ± 0.009 | 0.745 ± 0.010 | 0.706 ± 0.009 | 0.806 ± 0.013 | 0.803 ± 0.012 | 0.641 ± 0.003 | 0.241 ± 0.002 |
| SAM3 GT bbox | 512 | 0.611 | 0.725 | 0.637 | 0.930 | 0.688 | 0.370 | 0.071 |
| SAM3 YOLO bbox | 512 | 0.537 ± 0.006 | 0.630 ± 0.008 | 0.555 ± 0.006 | 0.802 ± 0.012 | 0.596 ± 0.006 | 0.360 ± 0.005 | 0.102 ± 0.006 |

### No Overlap × Low Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 286 uçak örneğini kapsar. YOLO bbox değerleri üç seed ortalaması ± standart sapmadır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.979 | 0.981 | 0.981 | 0.984 | 0.979 | 0.979 | 0.972 |
| SAM1 YOLO bbox | 128 | 0.765 ± 0.015 | 0.779 ± 0.017 | 0.773 ± 0.015 | 0.788 ± 0.019 | 0.791 ± 0.018 | 0.768 ± 0.011 | 0.728 ± 0.019 |
| SAM2 GT bbox | 128 | 0.769 | 0.854 | 0.795 | 0.952 | 0.916 | 0.724 | 0.203 |
| SAM2 YOLO bbox | 128 | 0.630 ± 0.010 | 0.696 ± 0.012 | 0.649 ± 0.009 | 0.771 ± 0.018 | 0.752 ± 0.010 | 0.584 ± 0.010 | 0.198 ± 0.005 |
| SAM3 GT bbox | 128 | 0.632 | 0.745 | 0.666 | 0.924 | 0.745 | 0.392 | 0.028 |
| SAM3 YOLO bbox | 128 | 0.465 ± 0.009 | 0.565 ± 0.010 | 0.492 ± 0.011 | 0.748 ± 0.013 | 0.505 ± 0.018 | 0.239 ± 0.007 | 0.030 ± 0.005 |

### No Overlap × High Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 412 uçak örneğini kapsar. YOLO bbox değerleri üç seed ortalaması ± standart sapmadır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.979 | 0.984 | 0.987 | 0.983 | 0.981 | 0.976 | 0.966 |
| SAM1 YOLO bbox | 128 | 0.901 ± 0.002 | 0.915 ± 0.002 | 0.914 ± 0.002 | 0.920 ± 0.003 | 0.922 ± 0.006 | 0.913 ± 0.002 | 0.861 ± 0.006 |
| SAM2 GT bbox | 128 | 0.827 | 0.892 | 0.885 | 0.913 | 0.947 | 0.850 | 0.330 |
| SAM2 YOLO bbox | 128 | 0.782 ± 0.003 | 0.843 ± 0.002 | 0.835 ± 0.002 | 0.865 ± 0.003 | 0.889 ± 0.004 | 0.790 ± 0.006 | 0.340 ± 0.004 |
| SAM3 GT bbox | 128 | 0.625 | 0.736 | 0.675 | 0.904 | 0.680 | 0.398 | 0.119 |
| SAM3 YOLO bbox | 128 | 0.591 ± 0.001 | 0.691 ± 0.001 | 0.630 ± 0.005 | 0.855 ± 0.006 | 0.633 ± 0.010 | 0.420 ± 0.005 | 0.158 ± 0.005 |

### Overlap × Low Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.209 uçak örneğini kapsar. YOLO bbox değerleri üç seed ortalaması ± standart sapmadır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.997 | 0.998 | 0.999 | 0.997 | 0.999 | 0.998 | 0.996 |
| SAM1 YOLO bbox | 128 | 0.676 ± 0.024 | 0.689 ± 0.024 | 0.696 ± 0.023 | 0.686 ± 0.025 | 0.704 ± 0.026 | 0.680 ± 0.026 | 0.629 ± 0.017 |
| SAM2 GT bbox | 128 | 0.702 | 0.815 | 0.734 | 0.957 | 0.897 | 0.444 | 0.036 |
| SAM2 YOLO bbox | 128 | 0.510 ± 0.016 | 0.586 ± 0.019 | 0.531 ± 0.016 | 0.678 ± 0.025 | 0.645 ± 0.022 | 0.371 ± 0.003 | 0.031 ± 0.002 |
| SAM3 GT bbox | 128 | 0.552 | 0.683 | 0.587 | 0.903 | 0.651 | 0.175 | 0.000 |
| SAM3 YOLO bbox | 128 | 0.407 ± 0.013 | 0.501 ± 0.016 | 0.428 ± 0.014 | 0.668 ± 0.023 | 0.485 ± 0.013 | 0.143 ± 0.012 | 0.001 ± 0.000 |

### Overlap × High Mask Area

Referans: Resmi SAMRS SAM1 Pseudo Referansı. Bu tablo 128 görüntüdeki 1.806 uçak örneğini kapsar. YOLO bbox değerleri üç seed ortalaması ± standart sapmadır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | IoU ≥ 0.50 | IoU ≥ 0.75 | IoU ≥ 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM1 GT bbox | 128 | 0.991 | 0.993 | 0.993 | 0.993 | 0.993 | 0.991 | 0.988 |
| SAM1 YOLO bbox | 128 | 0.895 ± 0.007 | 0.904 ± 0.007 | 0.907 ± 0.007 | 0.902 ± 0.007 | 0.912 ± 0.008 | 0.905 ± 0.008 | 0.874 ± 0.005 |
| SAM2 GT bbox | 128 | 0.825 | 0.895 | 0.853 | 0.958 | 0.961 | 0.815 | 0.360 |
| SAM2 YOLO bbox | 128 | 0.777 ± 0.006 | 0.836 ± 0.006 | 0.803 ± 0.006 | 0.883 ± 0.007 | 0.897 ± 0.007 | 0.798 ± 0.002 | 0.366 ± 0.003 |
| SAM3 GT bbox | 128 | 0.645 | 0.748 | 0.657 | 0.955 | 0.705 | 0.491 | 0.114 |
| SAM3 YOLO bbox | 128 | 0.624 ± 0.002 | 0.713 ± 0.004 | 0.634 ± 0.002 | 0.887 ± 0.007 | 0.676 ± 0.002 | 0.511 ± 0.003 | 0.169 ± 0.015 |

## Qualitative Examples

Her örnek GT-bbox koşulundandır. Yeşil TP, turuncu FP ve pembe FN piksellerini gösterir.

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
- YOLO-bbox Overall IoU değerleri SAM1/SAM2/SAM3 sırasıyla 0,814/0,679/0,537 olarak ölçülmüştür; bunlar üç detector seed ortalamasıdır.
- GT bbox yerine YOLO bbox kullanıldığında Overall IoU kaybı SAM1/SAM2/SAM3 için sırasıyla 0,176/0,102/0,074 olmuştur.
- SAM1'in GT-bbox IoU değeri, SAM2 ve SAM3 ortalamasından 0.295 daha yüksektir; bu fark resmi referansın SAM1 ile üretilmiş olmasıyla birlikte yorumlanmalıdır.
- SAMRS maskeleri ayrı bir resmi üretim hattında SAM1 ViT-H ile oluşturulduğu için buradaki SAM1 satırı kontrollü iSAID kimlik kontrolü kadar birebir değildir; yine de aynı model ailesine çok güçlü yakınlık gösterir.
- GT-bbox üç-model ortalamasında en yüksek alt grup Overlap × High Mask Area (0.820), en düşük alt grup Overlap × Low Mask Area (0.750) olmuştur.
- SAM1 gt bbox koşulunda Overall Precision 0.994, Recall 0.993 olmuştur. Precision daha yüksek olduğu için model boyadığı bölgelerde daha temizdir; buna karşılık bazı gerçek nesne piksellerini eksik bırakmaktadır.
- SAM1'in yüksek skoru, aynı model ailesinin ürettiği pseudo maskelere biçimsel yakınlık içerebilir; tek başına insan etiketli gerçek performans kanıtı değildir.
- GT bbox ile YOLO bbox arasındaki fark, otomatik detector hatasının segmentasyon zincirine eklediği kaybı gösterir.
- SAMRS pseudo etiketleri eğitim ve ölçeklenebilir ön-etiketleme için yararlı olabilir; nihai model karşılaştırması bağımsız insan etiketli test kümesiyle yapılmalıdır.
