# SAMRS SOTA Small Vehicle Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM1 (0.782) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT bbox ile bir modelin kendi ürettiği maske yine aynı maskeye karşı ölçüldüğünde IoU'nun 1,000 olması beklenir; bu bağımsız başarı sonucu değildir.
- YOLO bbox koşulunda her modelin kendi ürettiği etikette kazandığı ek IoU, aynı tahminlerin diğer iki SAM etiketindeki ortalaması çıkarılarak hesaplandı; Overall aralık +0.043 ile +0.081 arasındadır.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- Yayımlanmış SAMRS referansı insan GT değildir. Bu nedenle sonuç, mutlak kalite karşılaştırmasından çok SAM-türevi referans yakınlığı analizidir.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü, dört sahne grubunun her birinde 128 görüntü ve toplam 7.659 küçük araç nesnesi.
- Avg IoU her nesne için ayrı hesaplanır ve bütün nesneler eşit ağırlıkla ortalanır.
- Model–referans matrislerinde satır değerlendirilen modeli, sütun kullanılan referans maskeyi, hücre ise Avg IoU değerini gösterir.
- Kendi Etiketiyle IoU, örneğin SAM2 tahmininin SAM2 tarafından üretilen referans maskeye göre puanıdır.
- Diğer SAM Etiketleriyle Ortalama IoU, aynı tahminin diğer iki SAM modelinin ürettiği maskelere göre aldığı iki puanın ortalamasıdır.
- Ek IoU, bu iki değerin farkıdır. Pozitif değer, modelin kendi etiketine göre ölçüldüğünde daha yüksek puan aldığını gösterir.
- Güven aralıkları ve ayrıntılı istatistiksel kontroller analiz CSV'lerinde saklanır; okunabilirliği korumak için bu özet tablolara basılmaz.
- Bu karşılaştırmalar ilk sonuçlar görüldükten sonra geliştirilmiştir; önceden kaydedilmiş doğrulayıcı test değildir ve çoklu karşılaştırma düzeltmesi uygulanmamıştır.
- Aynı-model karşılaştırması aynı dondurulmuş checkpoint ile sınırlıdır; farklı eğitim seed'i/checkpoint'i veya model ailesi düzeyinde genelleme test edilmemiştir.
- Bilinen pozitif nesnedeki boş pseudo referans eksik etikettir ve 0 puanlanır.

## Overall

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.998 | 1.000 | 0.846 | 0.851 |
| SAM2 | 0.846 | 0.846 | 1.000 | 0.856 |
| SAM3 | 0.851 | 0.851 | 0.856 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.782 | 0.782 | 0.697 | 0.705 |
| SAM2 | 0.707 | 0.707 | 0.749 | 0.705 |
| SAM3 | 0.707 | 0.707 | 0.694 | 0.760 |

## No Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.999 | 1.000 | 0.817 | 0.832 |
| SAM2 | 0.817 | 0.817 | 1.000 | 0.838 |
| SAM3 | 0.832 | 0.832 | 0.838 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.616 | 0.616 | 0.555 | 0.557 |
| SAM2 | 0.556 | 0.556 | 0.593 | 0.560 |
| SAM3 | 0.551 | 0.551 | 0.549 | 0.598 |

## No Overlap × High Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.999 | 1.000 | 0.904 | 0.892 |
| SAM2 | 0.904 | 0.904 | 1.000 | 0.899 |
| SAM3 | 0.892 | 0.892 | 0.899 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.809 | 0.809 | 0.759 | 0.752 |
| SAM2 | 0.762 | 0.762 | 0.792 | 0.754 |
| SAM3 | 0.752 | 0.752 | 0.751 | 0.799 |

## Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.998 | 1.000 | 0.797 | 0.804 |
| SAM2 | 0.797 | 0.797 | 1.000 | 0.822 |
| SAM3 | 0.804 | 0.804 | 0.822 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.672 | 0.672 | 0.571 | 0.582 |
| SAM2 | 0.581 | 0.581 | 0.629 | 0.587 |
| SAM3 | 0.581 | 0.581 | 0.573 | 0.640 |

## Overlap × High Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.998 | 1.000 | 0.853 | 0.862 |
| SAM2 | 0.853 | 0.853 | 1.000 | 0.858 |
| SAM3 | 0.862 | 0.862 | 0.858 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.854 | 0.854 | 0.759 | 0.772 |
| SAM2 | 0.774 | 0.774 | 0.817 | 0.769 |
| SAM3 | 0.777 | 0.777 | 0.757 | 0.832 |

## Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor? · Overall · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- |
| SAM1 | 0.782 | 0.701 | +0.081 |
| SAM2 | 0.749 | 0.706 | +0.043 |
| SAM3 | 0.760 | 0.700 | +0.060 |

## Sahne Gruplarına Göre Kendi Etiketindeki Ek Puan · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Sahne Grubu | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| No Overlap × Low Mask Area | SAM1 | 0.616 | 0.556 | +0.060 |
| No Overlap × Low Mask Area | SAM2 | 0.593 | 0.558 | +0.036 |
| No Overlap × Low Mask Area | SAM3 | 0.598 | 0.550 | +0.048 |
| No Overlap × High Mask Area | SAM1 | 0.809 | 0.756 | +0.053 |
| No Overlap × High Mask Area | SAM2 | 0.792 | 0.758 | +0.034 |
| No Overlap × High Mask Area | SAM3 | 0.799 | 0.752 | +0.048 |
| Overlap × Low Mask Area | SAM1 | 0.672 | 0.577 | +0.095 |
| Overlap × Low Mask Area | SAM2 | 0.629 | 0.584 | +0.045 |
| Overlap × Low Mask Area | SAM3 | 0.640 | 0.577 | +0.063 |
| Overlap × High Mask Area | SAM1 | 0.854 | 0.765 | +0.089 |
| Overlap × High Mask Area | SAM2 | 0.817 | 0.771 | +0.046 |
| Overlap × High Mask Area | SAM3 | 0.832 | 0.767 | +0.065 |

## Temel Referanstan Kendi Etiketine Geçince Puan Değişimi · YOLO bbox

Aynı tahmin sabit tutulur; yalnız puanın hesaplandığı referans maske değişir.

| Model | Yayımlanmış Etiketle IoU | Kendi Etiketiyle IoU | Puan Değişimi |
| --- | --- | --- | --- |
| SAM1 | 0.782 | 0.782 | +0.000 |
| SAM2 | 0.707 | 0.749 | +0.042 |
| SAM3 | 0.707 | 0.760 | +0.054 |

## Referans Maskeler Birbirine Ne Kadar Benziyor?

Bu tablo model başarısını değil, iki referans maske kümesinin birbirine benzerliğini gösterir.

| Referans A | Referans B | Maskeler Arası Ortalama IoU | Nesne Sayısı |
| --- | --- | --- | --- |
| Yayımlanmış SAMRS | Yeniden SAM1 | 0.998 | 7659 |
| Yayımlanmış SAMRS | SAM2 pseudo | 0.846 | 7659 |
| Yayımlanmış SAMRS | SAM3 pseudo | 0.851 | 7659 |
| Yeniden SAM1 | SAM2 pseudo | 0.846 | 7659 |
| Yeniden SAM1 | SAM3 pseudo | 0.851 | 7659 |
| SAM2 pseudo | SAM3 pseudo | 0.856 | 7659 |

## Referansa Göre Model Sırası

| BBox | Referans | Sıralama |
| --- | --- | --- |
| GT bbox | Yayımlanmış SAMRS | SAM1 > SAM3 > SAM2 |
| GT bbox | Yeniden SAM1 | SAM1 > SAM3 > SAM2 |
| GT bbox | SAM2 pseudo | SAM2 > SAM3 > SAM1 |
| GT bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 |
| YOLO bbox | Yayımlanmış SAMRS | SAM1 > SAM2 > SAM3 |
| YOLO bbox | Yeniden SAM1 | SAM1 > SAM2 > SAM3 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 |

## Boş Üretilen Referans Maskeler

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| SAM2 pseudo | 0 | 0.000 |
| SAM3 pseudo | 0 | 0.000 |
| Yayımlanmış SAMRS | 0 | 0.000 |
| Yeniden SAM1 | 0 | 0.000 |
