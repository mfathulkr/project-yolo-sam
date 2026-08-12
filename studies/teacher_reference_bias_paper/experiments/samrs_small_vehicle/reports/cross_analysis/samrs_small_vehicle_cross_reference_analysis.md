# Samrs Small Vehicle Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM1 (0.782) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT-bbox öz-referans diagonali bağımsız performans değil coverage-aware identity control'dür; ana teacher-affinity kanıtı YOLO-bbox hücreleridir.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- Yayımlanmış SAMRS referansı insan GT değildir. Bu nedenle sonuç, mutlak kalite karşılaştırmasından çok SAM-türevi referans yakınlığı analizidir.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü. her dört tabakada 128 görüntü ve toplam 7.659 küçük araç instance.
- Avg IoU instance başına hesaplanır ve bütün instance'lar eşit ağırlıkla ortalanır.
- Teacher advantage, ilgili pseudo referansta öğretmen modelin IoU değeri ile diğer iki modelin ortalaması arasındaki farktır.
- Eşleşmiş fark güven aralıkları kaynak sahne kümeli bootstrap ile hesaplanır; aynı büyük sahneden gelen crop'lar bağımsız sayılmaz.
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

## Eşleşmiş Referans Etkileri

| Model | BBox | Karşılaştırılan Referans | Temel IoU | Referans IoU | IoU Farkı (%95 GA) |
| --- | --- | --- | --- | --- | --- |
| SAM1 | GT bbox | SAM2 pseudo | 0.998 | 0.846 | -0.152 [-0.187, -0.115] |
| SAM1 | GT bbox | SAM3 pseudo | 0.998 | 0.851 | -0.147 [-0.179, -0.118] |
| SAM1 | GT bbox | Yeniden SAM1 | 0.998 | 1.000 | +0.002 [+0.001, +0.002] |
| SAM1 | YOLO bbox | SAM2 pseudo | 0.782 | 0.697 | -0.085 [-0.106, -0.060] |
| SAM1 | YOLO bbox | SAM3 pseudo | 0.782 | 0.705 | -0.077 [-0.092, -0.061] |
| SAM1 | YOLO bbox | Yeniden SAM1 | 0.782 | 0.782 | +0.000 [+0.000, +0.000] |
| SAM2 | GT bbox | SAM2 pseudo | 0.846 | 1.000 | +0.154 [+0.117, +0.189] |
| SAM2 | GT bbox | SAM3 pseudo | 0.846 | 0.856 | +0.009 [-0.002, +0.021] |
| SAM2 | GT bbox | Yeniden SAM1 | 0.846 | 0.846 | -0.000 [-0.000, +0.000] |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.707 | 0.749 | +0.042 [+0.030, +0.054] |
| SAM2 | YOLO bbox | SAM3 pseudo | 0.707 | 0.705 | -0.002 [-0.010, +0.005] |
| SAM2 | YOLO bbox | Yeniden SAM1 | 0.707 | 0.707 | -0.000 [-0.000, +0.000] |
| SAM3 | GT bbox | SAM2 pseudo | 0.851 | 0.856 | +0.005 [-0.007, +0.018] |
| SAM3 | GT bbox | SAM3 pseudo | 0.851 | 1.000 | +0.149 [+0.120, +0.181] |
| SAM3 | GT bbox | Yeniden SAM1 | 0.851 | 0.851 | +0.000 [-0.000, +0.000] |
| SAM3 | YOLO bbox | SAM2 pseudo | 0.707 | 0.694 | -0.012 [-0.023, -0.001] |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.707 | 0.760 | +0.054 [+0.044, +0.063] |
| SAM3 | YOLO bbox | Yeniden SAM1 | 0.707 | 0.707 | +0.000 [-0.000, +0.000] |

## Öğretmen Avantajı

| Model | BBox | Referans | Öğretmen IoU | Diğerleri Ort. | Öğretmen Avantajı | Identity Control |
| --- | --- | --- | --- | --- | --- | --- |
| SAM2 | GT bbox | SAM2 pseudo | 1.000 | 0.851 | +0.149 | Evet |
| SAM3 | GT bbox | SAM3 pseudo | 1.000 | 0.853 | +0.147 | Evet |
| SAM1 | GT bbox | Yayımlanmış SAMRS | 0.998 | 0.849 | +0.150 | Hayır |
| SAM1 | GT bbox | Yeniden SAM1 | 1.000 | 0.849 | +0.151 | Evet |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.749 | 0.696 | +0.053 | Hayır |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.760 | 0.705 | +0.055 | Hayır |
| SAM1 | YOLO bbox | Yayımlanmış SAMRS | 0.782 | 0.707 | +0.075 | Hayır |
| SAM1 | YOLO bbox | Yeniden SAM1 | 0.782 | 0.707 | +0.075 | Hayır |

## Referanslar Arası Anlaşma

| Referans A | Referans B | Referans Anlaşması | Instance |
| --- | --- | --- | --- |
| Yayımlanmış SAMRS | Yeniden SAM1 | 0.998 | 7659 |
| Yayımlanmış SAMRS | SAM2 pseudo | 0.846 | 7659 |
| Yayımlanmış SAMRS | SAM3 pseudo | 0.851 | 7659 |
| Yeniden SAM1 | SAM2 pseudo | 0.846 | 7659 |
| Yeniden SAM1 | SAM3 pseudo | 0.851 | 7659 |
| SAM2 pseudo | SAM3 pseudo | 0.856 | 7659 |

## Model Sıralamaları

| BBox | Referans | Sıralama | Temele Göre Değişen Sıra |
| --- | --- | --- | --- |
| GT bbox | Yayımlanmış SAMRS | SAM1 > SAM3 > SAM2 | 0 |
| GT bbox | Yeniden SAM1 | SAM1 > SAM3 > SAM2 | 0 |
| GT bbox | SAM2 pseudo | SAM2 > SAM3 > SAM1 | 2 |
| GT bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 | 3 |
| YOLO bbox | Yayımlanmış SAMRS | SAM1 > SAM2 > SAM3 | 0 |
| YOLO bbox | Yeniden SAM1 | SAM1 > SAM2 > SAM3 | 0 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 2 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 | 2 |

## Boş Referans Denetimi

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| SAM2 pseudo | 0 | 0.0000 |
| SAM3 pseudo | 0 | 0.0000 |
| Yayımlanmış SAMRS | 0 | 0.0000 |
| Yeniden SAM1 | 0 | 0.0000 |
