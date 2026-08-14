# iSAID Small Vehicle Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM3 (0.491) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT bbox ile bir modelin kendi ürettiği maske yine aynı maskeye karşı ölçüldüğünde IoU'nun 1,000 olması beklenir; bu bağımsız başarı sonucu değildir.
- YOLO bbox koşulunda her modelin kendi ürettiği etikette kazandığı ek IoU, aynı tahminlerin diğer iki SAM etiketindeki ortalaması çıkarılarak hesaplandı; Overall aralık +0.074 ile +0.098 arasındadır.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- İnsan referansı bağımsız kontroldür; pseudo referans kaynaklı sıralama veya skor değişimi buna göre yorumlanır.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü, dört sahne grubunun her birinde 128 görüntü ve toplam 12.051 küçük araç nesnesi.
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
| SAM1 | 0.658 | 0.998 | 0.749 | 0.766 |
| SAM2 | 0.645 | 0.749 | 1.000 | 0.771 |
| SAM3 | 0.698 | 0.766 | 0.771 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.478 | 0.654 | 0.551 | 0.560 |
| SAM2 | 0.461 | 0.549 | 0.624 | 0.551 |
| SAM3 | 0.491 | 0.562 | 0.555 | 0.633 |

## No Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.680 | 1.000 | 0.799 | 0.800 |
| SAM2 | 0.680 | 0.799 | 1.000 | 0.819 |
| SAM3 | 0.715 | 0.800 | 0.819 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.466 | 0.640 | 0.546 | 0.549 |
| SAM2 | 0.451 | 0.546 | 0.614 | 0.547 |
| SAM3 | 0.474 | 0.554 | 0.554 | 0.615 |

## No Overlap × High Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.753 | 1.000 | 0.864 | 0.862 |
| SAM2 | 0.774 | 0.864 | 1.000 | 0.876 |
| SAM3 | 0.791 | 0.862 | 0.876 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.627 | 0.793 | 0.718 | 0.718 |
| SAM2 | 0.630 | 0.721 | 0.763 | 0.720 |
| SAM3 | 0.643 | 0.718 | 0.717 | 0.774 |

## Overlap × Low Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.565 | 0.997 | 0.692 | 0.705 |
| SAM2 | 0.564 | 0.692 | 1.000 | 0.728 |
| SAM3 | 0.611 | 0.705 | 0.728 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.298 | 0.461 | 0.378 | 0.381 |
| SAM2 | 0.286 | 0.373 | 0.444 | 0.377 |
| SAM3 | 0.307 | 0.383 | 0.383 | 0.447 |

## Overlap × High Mask Area

### GT bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.656 | 0.998 | 0.735 | 0.758 |
| SAM2 | 0.635 | 0.735 | 1.000 | 0.757 |
| SAM3 | 0.697 | 0.758 | 0.757 | 1.000 |

### YOLO bbox Avg IoU

| Model | İnsan | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.486 | 0.666 | 0.553 | 0.567 |
| SAM2 | 0.464 | 0.551 | 0.633 | 0.553 |
| SAM3 | 0.500 | 0.568 | 0.558 | 0.644 |

## Model Kendi Etiketiyle Ne Kadar Ek Puan Alıyor? · Overall · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- |
| SAM1 | 0.654 | 0.556 | +0.098 |
| SAM2 | 0.624 | 0.550 | +0.074 |
| SAM3 | 0.633 | 0.559 | +0.075 |

## Sahne Gruplarına Göre Kendi Etiketindeki Ek Puan · YOLO bbox

Ek IoU = Kendi Etiketiyle IoU − Diğer SAM Etiketleriyle Ortalama IoU. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.

| Sahne Grubu | Model | Kendi Etiketiyle IoU | Diğer SAM Etiketleriyle Ortalama IoU | Ek IoU |
| --- | --- | --- | --- | --- |
| No Overlap × Low Mask Area | SAM1 | 0.640 | 0.548 | +0.092 |
| No Overlap × Low Mask Area | SAM2 | 0.614 | 0.546 | +0.067 |
| No Overlap × Low Mask Area | SAM3 | 0.615 | 0.554 | +0.061 |
| No Overlap × High Mask Area | SAM1 | 0.793 | 0.718 | +0.075 |
| No Overlap × High Mask Area | SAM2 | 0.763 | 0.720 | +0.043 |
| No Overlap × High Mask Area | SAM3 | 0.774 | 0.718 | +0.056 |
| Overlap × Low Mask Area | SAM1 | 0.461 | 0.380 | +0.081 |
| Overlap × Low Mask Area | SAM2 | 0.444 | 0.375 | +0.069 |
| Overlap × Low Mask Area | SAM3 | 0.447 | 0.383 | +0.064 |
| Overlap × High Mask Area | SAM1 | 0.666 | 0.560 | +0.106 |
| Overlap × High Mask Area | SAM2 | 0.633 | 0.552 | +0.081 |
| Overlap × High Mask Area | SAM3 | 0.644 | 0.563 | +0.081 |

## Temel Referanstan Kendi Etiketine Geçince Puan Değişimi · YOLO bbox

Aynı tahmin sabit tutulur; yalnız puanın hesaplandığı referans maske değişir.

| Model | İnsan Etiketiyle IoU | Kendi Etiketiyle IoU | Puan Değişimi |
| --- | --- | --- | --- |
| SAM1 | 0.478 | 0.654 | +0.176 |
| SAM2 | 0.461 | 0.624 | +0.163 |
| SAM3 | 0.491 | 0.633 | +0.142 |

## Referans Maskeler Birbirine Ne Kadar Benziyor?

Bu tablo model başarısını değil, iki referans maske kümesinin birbirine benzerliğini gösterir.

| Referans A | Referans B | Maskeler Arası Ortalama IoU | Nesne Sayısı |
| --- | --- | --- | --- |
| İnsan | SAM1 pseudo | 0.658 | 12051 |
| İnsan | SAM2 pseudo | 0.645 | 12051 |
| İnsan | SAM3 pseudo | 0.698 | 12051 |
| SAM1 pseudo | SAM2 pseudo | 0.749 | 12051 |
| SAM1 pseudo | SAM3 pseudo | 0.766 | 12051 |
| SAM2 pseudo | SAM3 pseudo | 0.771 | 12051 |

## Referansa Göre Model Sırası

| BBox | Referans | Sıralama |
| --- | --- | --- |
| GT bbox | İnsan | SAM3 > SAM1 > SAM2 |
| GT bbox | SAM1 pseudo | SAM1 > SAM3 > SAM2 |
| GT bbox | SAM2 pseudo | SAM2 > SAM3 > SAM1 |
| GT bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 |
| YOLO bbox | İnsan | SAM3 > SAM1 > SAM2 |
| YOLO bbox | SAM1 pseudo | SAM1 > SAM3 > SAM2 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM3 > SAM1 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 |

## Boş Üretilen Referans Maskeler

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| İnsan | 0 | 0.000 |
| SAM1 pseudo | 19 | 0.002 |
| SAM2 pseudo | 0 | 0.000 |
| SAM3 pseudo | 0 | 0.000 |
