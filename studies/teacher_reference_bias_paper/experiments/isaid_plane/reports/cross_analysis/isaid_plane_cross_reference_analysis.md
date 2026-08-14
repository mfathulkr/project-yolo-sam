# iSAID Plane Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM3 (0.638) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT bbox ile bir modelin kendi ürettiği maske yine aynı maskeye karşı ölçüldüğünde IoU'nun 1,000 olması beklenir; bu bağımsız başarı sonucu değildir.
- YOLO bbox koşulunda her modelin kendi ürettiği etikette kazandığı ek IoU, aynı tahminlerin diğer iki SAM etiketindeki ortalaması çıkarılarak hesaplandı; Overall aralık +0.124 ile +0.141 arasındadır.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- İnsan referansı bağımsız kontroldür; pseudo referans kaynaklı sıralama veya skor değişimi buna göre yorumlanır.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü, dört sahne grubunun her birinde 128 görüntü ve toplam 5.447 uçak nesnesi.
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

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.653 | 1.000 | 0.827 | 0.820 |
| SAM2 | 0.629 | 0.827 | 1.000 | 0.784 |
| SAM3 | 0.700 | 0.820 | 0.784 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.597 | 0.873 | 0.749 | 0.742 |
| SAM2 | 0.574 | 0.750 | 0.853 | 0.706 |
| SAM3 | 0.638 | 0.741 | 0.702 | 0.863 |

## No Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.593 | 1.000 | 0.795 | 0.774 |
| SAM2 | 0.565 | 0.795 | 1.000 | 0.746 |
| SAM3 | 0.656 | 0.774 | 0.746 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.543 | 0.852 | 0.719 | 0.696 |
| SAM2 | 0.519 | 0.720 | 0.828 | 0.666 |
| SAM3 | 0.602 | 0.698 | 0.666 | 0.845 |

## No Overlap × High Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.649 | 1.000 | 0.820 | 0.778 |
| SAM2 | 0.670 | 0.820 | 1.000 | 0.790 |
| SAM3 | 0.735 | 0.778 | 0.790 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.611 | 0.881 | 0.765 | 0.721 |
| SAM2 | 0.625 | 0.758 | 0.872 | 0.726 |
| SAM3 | 0.688 | 0.716 | 0.723 | 0.873 |

## Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.633 | 1.000 | 0.808 | 0.800 |
| SAM2 | 0.586 | 0.808 | 1.000 | 0.751 |
| SAM3 | 0.672 | 0.800 | 0.751 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.577 | 0.868 | 0.728 | 0.722 |
| SAM2 | 0.532 | 0.733 | 0.840 | 0.674 |
| SAM3 | 0.611 | 0.719 | 0.666 | 0.853 |

## Overlap × High Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.676 | 1.000 | 0.846 | 0.850 |
| SAM2 | 0.658 | 0.846 | 1.000 | 0.809 |
| SAM3 | 0.716 | 0.850 | 0.809 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.616 | 0.878 | 0.763 | 0.767 |
| SAM2 | 0.597 | 0.765 | 0.860 | 0.728 |
| SAM3 | 0.650 | 0.768 | 0.727 | 0.869 |

## Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor? · Overall · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- |
| SAM1 | 0.873 | 0.745 | +0.128 |
| SAM2 | 0.853 | 0.728 | +0.124 |
| SAM3 | 0.863 | 0.722 | +0.141 |

## Sahne Gruplarına Göre Kendi Etiketindeki Ek Puan · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Sahne Grubu | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| No Overlap × Low Mask Area | SAM1 | 0.852 | 0.707 | +0.144 |
| No Overlap × Low Mask Area | SAM2 | 0.828 | 0.693 | +0.136 |
| No Overlap × Low Mask Area | SAM3 | 0.845 | 0.682 | +0.163 |
| No Overlap × High Mask Area | SAM1 | 0.881 | 0.743 | +0.138 |
| No Overlap × High Mask Area | SAM2 | 0.872 | 0.742 | +0.130 |
| No Overlap × High Mask Area | SAM3 | 0.873 | 0.719 | +0.154 |
| Overlap × Low Mask Area | SAM1 | 0.868 | 0.725 | +0.143 |
| Overlap × Low Mask Area | SAM2 | 0.840 | 0.703 | +0.137 |
| Overlap × Low Mask Area | SAM3 | 0.853 | 0.692 | +0.161 |
| Overlap × High Mask Area | SAM1 | 0.878 | 0.765 | +0.113 |
| Overlap × High Mask Area | SAM2 | 0.860 | 0.747 | +0.113 |
| Overlap × High Mask Area | SAM3 | 0.869 | 0.748 | +0.121 |

## Temel Referanstan Kendi Etiketine Geçince Puan Değişimi · YOLO bbox

Aynı tahmin sabit tutulur; yalnız puanın hesaplandığı referans maske değişir.

| Model | İnsan Etiketiyle IoU | Kendi Etiketiyle IoU | Puan Değişimi |
| --- | --- | --- | --- |
| SAM1 | 0.597 | 0.873 | +0.276 |
| SAM2 | 0.574 | 0.853 | +0.279 |
| SAM3 | 0.638 | 0.863 | +0.224 |

## Referans Maskeler Birbirine Ne Kadar Benziyor?

Bu tablo model başarısını değil, iki referans maske kümesinin birbirine benzerliğini gösterir.

| Referans A | Referans B | Maskeler Arası Ortalama IoU | Nesne Sayısı |
| --- | --- | --- | --- |
| İnsan | SAM1 pseudo | 0.653 | 5447 |
| İnsan | SAM2 pseudo | 0.629 | 5447 |
| İnsan | SAM3 pseudo | 0.700 | 5447 |
| SAM1 pseudo | SAM2 pseudo | 0.827 | 5447 |
| SAM1 pseudo | SAM3 pseudo | 0.820 | 5447 |
| SAM2 pseudo | SAM3 pseudo | 0.784 | 5447 |

## Referansa Göre Model Sırası

| BBox | Referans | Sıralama |
| --- | --- | --- |
| GT bbox | İnsan | SAM3 > SAM1 > SAM2 |
| GT bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 |
| GT bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 |
| GT bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 |
| YOLO bbox | İnsan | SAM3 > SAM1 > SAM2 |
| YOLO bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 |

## Boş Üretilen Referans Maskeler

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| İnsan | 0 | 0.000 |
| SAM1 pseudo | 0 | 0.000 |
| SAM2 pseudo | 0 | 0.000 |
| SAM3 pseudo | 0 | 0.000 |
