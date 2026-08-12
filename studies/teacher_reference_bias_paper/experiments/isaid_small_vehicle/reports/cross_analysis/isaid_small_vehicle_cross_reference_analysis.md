# Isaid Small Vehicle Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM3 (0.491) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT-bbox öz-referans diagonali bağımsız performans değil coverage-aware identity control'dür; ana teacher-affinity kanıtı YOLO-bbox hücreleridir.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- İnsan referansı bağımsız kontroldür; pseudo referans kaynaklı sıralama veya skor değişimi buna göre yorumlanır.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü. her dört tabakada 128 görüntü ve toplam 12.051 küçük araç instance.
- Avg IoU instance başına hesaplanır ve bütün instance'lar eşit ağırlıkla ortalanır.
- Teacher advantage, ilgili pseudo referansta öğretmen modelin IoU değeri ile diğer iki modelin ortalaması arasındaki farktır.
- Eşleşmiş fark güven aralıkları kaynak sahne kümeli bootstrap ile hesaplanır; aynı büyük sahneden gelen crop'lar bağımsız sayılmaz.
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

## Eşleşmiş Referans Etkileri

| Model | BBox | Karşılaştırılan Referans | Temel IoU | Referans IoU | IoU Farkı (%95 GA) |
| --- | --- | --- | --- | --- | --- |
| SAM1 | GT bbox | SAM1 pseudo | 0.658 | 0.998 | +0.341 [+0.292, +0.379] |
| SAM1 | GT bbox | SAM2 pseudo | 0.658 | 0.749 | +0.091 [+0.075, +0.109] |
| SAM1 | GT bbox | SAM3 pseudo | 0.658 | 0.766 | +0.108 [+0.094, +0.123] |
| SAM1 | YOLO bbox | SAM1 pseudo | 0.478 | 0.654 | +0.176 [+0.159, +0.194] |
| SAM1 | YOLO bbox | SAM2 pseudo | 0.478 | 0.551 | +0.073 [+0.058, +0.087] |
| SAM1 | YOLO bbox | SAM3 pseudo | 0.478 | 0.560 | +0.082 [+0.072, +0.093] |
| SAM2 | GT bbox | SAM1 pseudo | 0.645 | 0.749 | +0.103 [+0.084, +0.121] |
| SAM2 | GT bbox | SAM2 pseudo | 0.645 | 1.000 | +0.355 [+0.295, +0.401] |
| SAM2 | GT bbox | SAM3 pseudo | 0.645 | 0.771 | +0.125 [+0.105, +0.142] |
| SAM2 | YOLO bbox | SAM1 pseudo | 0.461 | 0.549 | +0.088 [+0.076, +0.099] |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.461 | 0.624 | +0.163 [+0.143, +0.180] |
| SAM2 | YOLO bbox | SAM3 pseudo | 0.461 | 0.551 | +0.090 [+0.079, +0.101] |
| SAM3 | GT bbox | SAM1 pseudo | 0.698 | 0.766 | +0.067 [+0.052, +0.085] |
| SAM3 | GT bbox | SAM2 pseudo | 0.698 | 0.771 | +0.072 [+0.057, +0.090] |
| SAM3 | GT bbox | SAM3 pseudo | 0.698 | 1.000 | +0.302 [+0.256, +0.338] |
| SAM3 | YOLO bbox | SAM1 pseudo | 0.491 | 0.562 | +0.071 [+0.060, +0.083] |
| SAM3 | YOLO bbox | SAM2 pseudo | 0.491 | 0.555 | +0.064 [+0.049, +0.077] |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.491 | 0.633 | +0.142 [+0.128, +0.158] |

## Öğretmen Avantajı

| Model | BBox | Referans | Öğretmen IoU | Diğerleri Ort. | Öğretmen Avantajı | Identity Control |
| --- | --- | --- | --- | --- | --- | --- |
| SAM1 | GT bbox | SAM1 pseudo | 0.998 | 0.757 | +0.241 | Evet |
| SAM2 | GT bbox | SAM2 pseudo | 1.000 | 0.760 | +0.240 | Evet |
| SAM3 | GT bbox | SAM3 pseudo | 1.000 | 0.768 | +0.232 | Evet |
| SAM1 | YOLO bbox | SAM1 pseudo | 0.654 | 0.556 | +0.098 | Hayır |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.624 | 0.553 | +0.071 | Hayır |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.633 | 0.556 | +0.078 | Hayır |

## Referanslar Arası Anlaşma

| Referans A | Referans B | Referans Anlaşması | Instance |
| --- | --- | --- | --- |
| İnsan | SAM1 pseudo | 0.658 | 12051 |
| İnsan | SAM2 pseudo | 0.645 | 12051 |
| İnsan | SAM3 pseudo | 0.698 | 12051 |
| SAM1 pseudo | SAM2 pseudo | 0.749 | 12051 |
| SAM1 pseudo | SAM3 pseudo | 0.766 | 12051 |
| SAM2 pseudo | SAM3 pseudo | 0.771 | 12051 |

## Model Sıralamaları

| BBox | Referans | Sıralama | Temele Göre Değişen Sıra |
| --- | --- | --- | --- |
| GT bbox | İnsan | SAM3 > SAM1 > SAM2 | 0 |
| GT bbox | SAM1 pseudo | SAM1 > SAM3 > SAM2 | 2 |
| GT bbox | SAM2 pseudo | SAM2 > SAM3 > SAM1 | 3 |
| GT bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 | 2 |
| YOLO bbox | İnsan | SAM3 > SAM1 > SAM2 | 0 |
| YOLO bbox | SAM1 pseudo | SAM1 > SAM3 > SAM2 | 2 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM3 > SAM1 | 3 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 | 0 |

## Boş Referans Denetimi

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| İnsan | 0 | 0.0000 |
| SAM1 pseudo | 19 | 0.0016 |
| SAM2 pseudo | 0 | 0.0000 |
| SAM3 pseudo | 0 | 0.0000 |
