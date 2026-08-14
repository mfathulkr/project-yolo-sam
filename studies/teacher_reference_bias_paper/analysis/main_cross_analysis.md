# Teacher-Reference Bias Main Cross Analysis

## Ana Sonuç

- Dört deney aynı 512 görüntü / dört eşit 128 görüntülük sahne grubu / seed 42 / 1024×1024 / SAM1-2-3 / GT ve YOLO bbox protokolünü kullanır.
- iSAID Plane ve Small Vehicle deneylerinde insan anotasyonu bağımsız kontrol referansıdır. Ana karşılaştırma, aynı dondurulmuş model tahmininin kendi ürettiği etikette aldığı IoU ile diğer iki SAM etiketinde aldığı ortalama IoU arasındaki farktır.
- SAMRS Plane ve Small Vehicle deneylerinde yayımlanmış etiket insan GT değildir; SAM tabanlı üretim hattından geldiği için bu iki deney destekleyici SAM1-benzeri referans yakınlığı analizi olarak yorumlanır.
- GT bbox ile bir modelin kendi ürettiği maske yine aynı maskeye karşı ölçüldüğünde IoU'nun 1,000 olması beklenen matematiksel sonuçtur. Ana değerlendirme bu doğrudan eşitliği kıran YOLO bbox koşuludur.
- Dört deney tek bir ortalamada birleştirilmez. iSAID ve SAMRS ayrı anotasyon ürünleridir ancak ikisi de DOTA kökenli görüntüler içerdiği ve test kümeleri kısmen örtüştüğü için bağımsız dört replikasyon olarak yorumlanmaz.
- Sonuçlar, bu dondurulmuş checkpoint'ler ve seçilmiş test kapsamı içinde, pseudo etiket üreticisiyle bağımlı test referansının skor ve model seçimini etkileyebildiğini destekler; pseudo etiketlemenin eğitimde yararsız olduğunu göstermez.

## Dört Deneyde Temel Referans Sonuçları

| Deney | Temel Referans | Model | BBox | Avg IoU | Nesne Sayısı |
| --- | --- | --- | --- | --- | --- |
| iSAID Plane | İnsan | SAM1 | YOLO bbox | 0.597 | 5447 |
| iSAID Plane | İnsan | SAM2 | YOLO bbox | 0.574 | 5447 |
| iSAID Plane | İnsan | SAM3 | YOLO bbox | 0.638 | 5447 |
| iSAID Small Vehicle | İnsan | SAM1 | YOLO bbox | 0.478 | 12051 |
| iSAID Small Vehicle | İnsan | SAM2 | YOLO bbox | 0.461 | 12051 |
| iSAID Small Vehicle | İnsan | SAM3 | YOLO bbox | 0.491 | 12051 |
| SAMRS SOTA Plane | Yayımlanmış SAMRS | SAM1 | YOLO bbox | 0.813 | 3713 |
| SAMRS SOTA Plane | Yayımlanmış SAMRS | SAM2 | YOLO bbox | 0.679 | 3713 |
| SAMRS SOTA Plane | Yayımlanmış SAMRS | SAM3 | YOLO bbox | 0.691 | 3713 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS | SAM1 | YOLO bbox | 0.782 | 7659 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS | SAM2 | YOLO bbox | 0.707 | 7659 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS | SAM3 | YOLO bbox | 0.707 | 7659 |

## iSAID: Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor?

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU.

| Deney | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| iSAID Plane | SAM1 | 0.873 | 0.745 | +0.128 |
| iSAID Plane | SAM2 | 0.853 | 0.728 | +0.124 |
| iSAID Plane | SAM3 | 0.863 | 0.722 | +0.141 |
| iSAID Small Vehicle | SAM1 | 0.654 | 0.556 | +0.098 |
| iSAID Small Vehicle | SAM2 | 0.624 | 0.550 | +0.074 |
| iSAID Small Vehicle | SAM3 | 0.633 | 0.559 | +0.075 |

## iSAID: İnsan Etiketinden Kendi Etiketine Geçince Ne Değişiyor?

Aynı tahmin sabit tutulur; yalnız puanın hesaplandığı referans maske değişir.

| Deney | Model | İnsan Etiketiyle IoU | Kendi Etiketiyle IoU | Puan Değişimi |
| --- | --- | --- | --- | --- |
| iSAID Plane | SAM1 | 0.597 | 0.873 | +0.276 |
| iSAID Plane | SAM2 | 0.574 | 0.853 | +0.279 |
| iSAID Plane | SAM3 | 0.638 | 0.863 | +0.224 |
| iSAID Small Vehicle | SAM1 | 0.478 | 0.654 | +0.176 |
| iSAID Small Vehicle | SAM2 | 0.461 | 0.624 | +0.163 |
| iSAID Small Vehicle | SAM3 | 0.491 | 0.633 | +0.142 |

## SAMRS: Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor?

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU.

| Deney | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| SAMRS SOTA Plane | SAM1 | 0.818 | 0.685 | +0.133 |
| SAMRS SOTA Plane | SAM2 | 0.785 | 0.663 | +0.122 |
| SAMRS SOTA Plane | SAM3 | 0.799 | 0.667 | +0.132 |
| SAMRS SOTA Small Vehicle | SAM1 | 0.782 | 0.701 | +0.081 |
| SAMRS SOTA Small Vehicle | SAM2 | 0.749 | 0.706 | +0.043 |
| SAMRS SOTA Small Vehicle | SAM3 | 0.760 | 0.700 | +0.060 |

## SAMRS Yayımlanmış Referans Yakınlığı

| Deney | Referans Çifti | Referans Anlaşması | Nesne Sayısı |
| --- | --- | --- | --- |
| SAMRS SOTA Plane | Yayımlanmış SAMRS ↔ yeniden SAM1 | 0.991 | 3713 |
| SAMRS SOTA Small Vehicle | Yayımlanmış SAMRS ↔ yeniden SAM1 | 0.998 | 7659 |

## Sınırlılıklar

- Çalışma iki DOTA kökenli anotasyon ürünü ve iki hedef sınıfla sınırlıdır; dört deney bağımsız veri replikasyonları değildir.
- 512 görüntülük testlerin tamamı hedef-pozitif olarak seçilmiştir. Detector AP değerleri resmi, negatif görüntüler de içeren benchmark AP'si değil bu seçilmiş pozitif test kapsamının kontrol metriğidir.
- SAMRS için bağımsız insan instance maskesi bulunmadığından mutlak segmentasyon kalitesi iddiası kurulamaz.
- Pseudo referanslar insan/yayımlanmış anotasyon kutularından gelen GT bbox ile, YOLO aday maskeleri ise tahmin kutularıyla üretilmiştir. Bu nedenle ölçülen fark yalnız sınır stilini izole etmez; checkpoint kimliği, GT/YOLO kutu farkı, prompt hassasiyeti ve maske biçiminin ortak etkileşimidir. Deney tam otomatik pseudo-etiketleme hattı değildir.
- Ana kendi-etiketi karşılaştırmaları ilk sonuçlar görüldükten sonra geliştirilmiş destekleyici analizlerdir; önceden kaydedilmiş doğrulayıcı test değildir ve çoklu karşılaştırma düzeltmesi uygulanmamıştır.
- Aynı-üretici etkisi aynı dondurulmuş SAM1/2/3 checkpoint'leri için ölçülmüştür. Farklı seed/checkpoint veya model ailesi düzeyinde genelleme bu çalışmada test edilmemiştir.
- YOLO yanlış pozitifleri detector mAP/precision/recall tablosunda ölçülür; instance maske ortalamasına sahte bir GT örneği olarak eklenmez. Bu nedenle maske tabloları tam uçtan uca instance-segmentation AP'si değildir.
- Her detector tek hedef sınıflıdır; bu nedenle detector mAP değeri o tek sınıfın AP değerine eşittir.
