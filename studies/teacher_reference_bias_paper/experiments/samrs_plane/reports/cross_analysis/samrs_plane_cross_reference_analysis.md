# SAMRS SOTA Plane Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM1 (0.813) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT bbox ile bir modelin kendi ürettiği maske yine aynı maskeye karşı ölçüldüğünde IoU'nun 1,000 olması beklenir; bu bağımsız başarı sonucu değildir.
- YOLO bbox koşulunda her modelin kendi ürettiği etikette kazandığı ek IoU, aynı tahminlerin diğer iki SAM etiketindeki ortalaması çıkarılarak hesaplandı; Overall aralık +0.122 ile +0.133 arasındadır.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- Yayımlanmış SAMRS referansı insan GT değildir. Bu nedenle sonuç, mutlak kalite karşılaştırmasından çok SAM-türevi referans yakınlığı analizidir.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü, dört sahne grubunun her birinde 128 görüntü ve toplam 3.713 uçak nesnesi.
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
| SAM1 | 0.991 | 1.000 | 0.785 | 0.813 |
| SAM2 | 0.781 | 0.785 | 1.000 | 0.751 |
| SAM3 | 0.808 | 0.813 | 0.751 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.813 | 0.818 | 0.676 | 0.695 |
| SAM2 | 0.679 | 0.682 | 0.785 | 0.643 |
| SAM3 | 0.691 | 0.695 | 0.639 | 0.799 |

## No Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.979 | 1.000 | 0.784 | 0.825 |
| SAM2 | 0.769 | 0.784 | 1.000 | 0.761 |
| SAM3 | 0.812 | 0.825 | 0.761 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.765 | 0.784 | 0.647 | 0.671 |
| SAM2 | 0.631 | 0.646 | 0.745 | 0.622 |
| SAM3 | 0.659 | 0.672 | 0.624 | 0.759 |

## No Overlap × High Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.979 | 1.000 | 0.837 | 0.798 |
| SAM2 | 0.827 | 0.837 | 1.000 | 0.801 |
| SAM3 | 0.788 | 0.798 | 0.801 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.901 | 0.909 | 0.786 | 0.744 |
| SAM2 | 0.780 | 0.789 | 0.893 | 0.746 |
| SAM3 | 0.743 | 0.752 | 0.751 | 0.901 |

## Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.997 | 1.000 | 0.701 | 0.780 |
| SAM2 | 0.702 | 0.701 | 1.000 | 0.685 |
| SAM3 | 0.778 | 0.780 | 0.685 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.676 | 0.678 | 0.505 | 0.564 |
| SAM2 | 0.511 | 0.511 | 0.630 | 0.487 |
| SAM3 | 0.562 | 0.563 | 0.482 | 0.655 |

## Overlap × High Mask Area

### GT bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.991 | 1.000 | 0.829 | 0.836 |
| SAM2 | 0.825 | 0.829 | 1.000 | 0.782 |
| SAM3 | 0.831 | 0.836 | 0.782 | 1.000 |

### YOLO bbox Avg IoU

| Model | Yayımlanmış SAMRS | Yeniden SAM1 | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.891 | 0.897 | 0.769 | 0.775 |
| SAM2 | 0.775 | 0.778 | 0.870 | 0.728 |
| SAM3 | 0.771 | 0.775 | 0.721 | 0.879 |

## Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor? · Overall · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- |
| SAM1 | 0.818 | 0.685 | +0.133 |
| SAM2 | 0.785 | 0.663 | +0.122 |
| SAM3 | 0.799 | 0.667 | +0.132 |

## Sahne Gruplarına Göre Kendi Etiketindeki Ek Puan · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Sahne Grubu | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| No Overlap × Low Mask Area | SAM1 | 0.784 | 0.659 | +0.125 |
| No Overlap × Low Mask Area | SAM2 | 0.745 | 0.634 | +0.111 |
| No Overlap × Low Mask Area | SAM3 | 0.759 | 0.648 | +0.111 |
| No Overlap × High Mask Area | SAM1 | 0.909 | 0.765 | +0.144 |
| No Overlap × High Mask Area | SAM2 | 0.893 | 0.767 | +0.126 |
| No Overlap × High Mask Area | SAM3 | 0.901 | 0.752 | +0.150 |
| Overlap × Low Mask Area | SAM1 | 0.678 | 0.534 | +0.143 |
| Overlap × Low Mask Area | SAM2 | 0.630 | 0.499 | +0.131 |
| Overlap × Low Mask Area | SAM3 | 0.655 | 0.523 | +0.133 |
| Overlap × High Mask Area | SAM1 | 0.897 | 0.772 | +0.124 |
| Overlap × High Mask Area | SAM2 | 0.870 | 0.753 | +0.117 |
| Overlap × High Mask Area | SAM3 | 0.879 | 0.748 | +0.131 |

## Temel Referanstan Kendi Etiketine Geçince Puan Değişimi · YOLO bbox

Aynı tahmin sabit tutulur; yalnız puanın hesaplandığı referans maske değişir.

| Model | Yayımlanmış Etiketle IoU | Kendi Etiketiyle IoU | Puan Değişimi |
| --- | --- | --- | --- |
| SAM1 | 0.813 | 0.818 | +0.005 |
| SAM2 | 0.679 | 0.785 | +0.106 |
| SAM3 | 0.691 | 0.799 | +0.108 |

## Referans Maskeler Birbirine Ne Kadar Benziyor?

Bu tablo model başarısını değil, iki referans maske kümesinin birbirine benzerliğini gösterir.

| Referans A | Referans B | Maskeler Arası Ortalama IoU | Nesne Sayısı |
| --- | --- | --- | --- |
| Yayımlanmış SAMRS | Yeniden SAM1 | 0.991 | 3713 |
| Yayımlanmış SAMRS | SAM2 pseudo | 0.781 | 3713 |
| Yayımlanmış SAMRS | SAM3 pseudo | 0.808 | 3713 |
| Yeniden SAM1 | SAM2 pseudo | 0.785 | 3713 |
| Yeniden SAM1 | SAM3 pseudo | 0.813 | 3713 |
| SAM2 pseudo | SAM3 pseudo | 0.751 | 3713 |

## Referansa Göre Model Sırası

| BBox | Referans | Sıralama |
| --- | --- | --- |
| GT bbox | Yayımlanmış SAMRS | SAM1 > SAM3 > SAM2 |
| GT bbox | Yeniden SAM1 | SAM1 > SAM3 > SAM2 |
| GT bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 |
| GT bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 |
| YOLO bbox | Yayımlanmış SAMRS | SAM1 > SAM3 > SAM2 |
| YOLO bbox | Yeniden SAM1 | SAM1 > SAM3 > SAM2 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 |

## Boş Üretilen Referans Maskeler

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| SAM2 pseudo | 0 | 0.000 |
| SAM3 pseudo | 0 | 0.000 |
| Yayımlanmış SAMRS | 0 | 0.000 |
| Yeniden SAM1 | 0 | 0.000 |
