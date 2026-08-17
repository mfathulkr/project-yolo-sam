# Teacher-Reference Bias Main Cross Analysis · GT bbox

## Ana Sonuç

- Dört deney aynı 512 görüntü / dört eşit 128 görüntülük sahne grubu / seed 42 / 1024×1024 / SAM1-2-3 / GT ve YOLO bbox protokolünü kullanır.
- iSAID Plane ve Small Vehicle deneylerinde insan anotasyonu bağımsız kontrol referansıdır. Ana karşılaştırma, aynı dondurulmuş model tahmininin kendi ürettiği etikette aldığı IoU ile diğer iki SAM etiketinde aldığı ortalama IoU arasındaki farktır.
- SAMRS Plane ve Small Vehicle deneylerinde yayımlanmış etiket insan GT değildir; SAM tabanlı üretim hattından geldiği için bu iki deney destekleyici SAM1-benzeri referans yakınlığı analizi olarak yorumlanır.
- Bu rapor yalnız GT bbox sonuçlarını gösterir. Aynı modelin kendi GT-bbox pseudo referansına karşı 1,000 IoU alması beklenen matematiksel özdeşlik kontrolüdür; bağımsız segmentasyon başarısı değildir.
- Dört deney tek bir ortalamada birleştirilmez. iSAID ve SAMRS ayrı anotasyon ürünleridir ancak ikisi de DOTA kökenli görüntüler içerdiği ve test kümeleri kısmen örtüştüğü için bağımsız dört replikasyon olarak yorumlanmaz.
- Sonuçlar, bu dondurulmuş checkpoint'ler ve seçilmiş test kapsamı içinde, pseudo etiket üreticisiyle bağımlı test referansının skor ve model seçimini etkileyebildiğini destekler; pseudo etiketlemenin eğitimde yararsız olduğunu göstermez.

## Dört Deneyde Temel Referans Sonuçları

| Deney | Temel Referans | Model | BBox | Avg IoU | Nesne Sayısı |
| --- | --- | --- | --- | --- | --- |
| iSAID Plane | İnsan | SAM1 | GT bbox | 0.653 | 5447 |
| iSAID Plane | İnsan | SAM2 | GT bbox | 0.629 | 5447 |
| iSAID Plane | İnsan | SAM3 | GT bbox | 0.700 | 5447 |
| iSAID Small Vehicle | İnsan | SAM1 | GT bbox | 0.658 | 12051 |
| iSAID Small Vehicle | İnsan | SAM2 | GT bbox | 0.645 | 12051 |
| iSAID Small Vehicle | İnsan | SAM3 | GT bbox | 0.698 | 12051 |
| SAMRS SOTA Plane | Yayımlanmış SAMRS | SAM1 | GT bbox | 0.991 | 3713 |
| SAMRS SOTA Plane | Yayımlanmış SAMRS | SAM2 | GT bbox | 0.781 | 3713 |
| SAMRS SOTA Plane | Yayımlanmış SAMRS | SAM3 | GT bbox | 0.808 | 3713 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS | SAM1 | GT bbox | 0.998 | 7659 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS | SAM2 | GT bbox | 0.846 | 7659 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS | SAM3 | GT bbox | 0.851 | 7659 |

## iSAID: Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor?

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU.

| Deney | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| iSAID Plane | SAM1 | 1.000 | 0.824 | +0.176 |
| iSAID Plane | SAM2 | 1.000 | 0.805 | +0.195 |
| iSAID Plane | SAM3 | 1.000 | 0.802 | +0.198 |
| iSAID Small Vehicle | SAM1 | 0.998 | 0.757 | +0.241 |
| iSAID Small Vehicle | SAM2 | 1.000 | 0.760 | +0.240 |
| iSAID Small Vehicle | SAM3 | 1.000 | 0.768 | +0.232 |

## iSAID: İnsan Etiketinden Kendi Etiketine Geçince Ne Değişiyor?

Aynı tahmin sabit tutulur; yalnız puanın hesaplandığı referans maske değişir.

| Deney | Model | İnsan Etiketiyle IoU | Kendi Etiketiyle IoU | Puan Değişimi |
| --- | --- | --- | --- | --- |
| iSAID Plane | SAM1 | 0.653 | 1.000 | +0.347 |
| iSAID Plane | SAM2 | 0.629 | 1.000 | +0.371 |
| iSAID Plane | SAM3 | 0.700 | 1.000 | +0.300 |
| iSAID Small Vehicle | SAM1 | 0.658 | 0.998 | +0.341 |
| iSAID Small Vehicle | SAM2 | 0.645 | 1.000 | +0.355 |
| iSAID Small Vehicle | SAM3 | 0.698 | 1.000 | +0.302 |

## SAMRS: Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor?

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU.

| Deney | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| SAMRS SOTA Plane | SAM1 | 1.000 | 0.799 | +0.201 |
| SAMRS SOTA Plane | SAM2 | 1.000 | 0.768 | +0.232 |
| SAMRS SOTA Plane | SAM3 | 1.000 | 0.782 | +0.218 |
| SAMRS SOTA Small Vehicle | SAM1 | 1.000 | 0.849 | +0.151 |
| SAMRS SOTA Small Vehicle | SAM2 | 1.000 | 0.851 | +0.149 |
| SAMRS SOTA Small Vehicle | SAM3 | 1.000 | 0.853 | +0.147 |

## SAMRS Yayımlanmış Referans Yakınlığı

| Deney | Referans Çifti | Referans Anlaşması | Nesne Sayısı |
| --- | --- | --- | --- |
| SAMRS SOTA Plane | Yayımlanmış SAMRS ↔ yeniden SAM1 | 0.991 | 3713 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS ↔ yeniden SAM1 | 0.998 | 7659 |

## Sınırlılıklar

- Çalışma iki DOTA kökenli anotasyon ürünü ve iki hedef sınıfla sınırlıdır; dört deney bağımsız veri replikasyonları değildir.
- 512 görüntülük testlerin tamamı hedef-pozitiftir. GT bbox koşulu, nesne konumunun kusursuz bilindiği oracle-localization analizidir; uçtan uca detection + segmentation başarısı değildir.
- SAMRS için bağımsız insan instance maskesi bulunmadığından mutlak segmentasyon kalitesi iddiası kurulamaz.
- GT bbox koşulunda hem pseudo referanslar hem aday maskeler aynı anotasyon kutularından üretilir. Bu, detector hatasını kaldırır; ancak aynı modelin kendi pseudo referansındaki diagonal hücrelerini özdeşlik kontrolüne dönüştürür.
- Ana kendi-etiketi karşılaştırmaları ilk sonuçlar görüldükten sonra geliştirilmiş destekleyici analizlerdir; önceden kaydedilmiş doğrulayıcı test değildir ve çoklu karşılaştırma düzeltmesi uygulanmamıştır.
- Aynı-üretici etkisi aynı dondurulmuş SAM1/2/3 checkpoint'leri için ölçülmüştür. Farklı seed/checkpoint veya model ailesi düzeyinde genelleme bu çalışmada test edilmemiştir.
- Bu GT bbox raporunda detector kaçırmaları, yanlış pozitifleri ve kutu konum hataları yoktur; değerler yalnız verilen doğru kutu içindeki maske davranışını gösterir.
- GT bbox değerleri YOLO bbox değerlerinden doğal olarak daha yüksektir; bu fark model kalitesindeki artış olarak yorumlanmamalıdır.
