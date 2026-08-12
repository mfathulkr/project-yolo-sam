# Isaid Plane Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM3 (0.638) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT-bbox öz-referans diagonali bağımsız performans değil coverage-aware identity control'dür; ana teacher-affinity kanıtı YOLO-bbox hücreleridir.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- İnsan referansı bağımsız kontroldür; pseudo referans kaynaklı sıralama veya skor değişimi buna göre yorumlanır.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü. her dört tabakada 128 görüntü ve toplam 5.447 uçak instance.
- Avg IoU instance başına hesaplanır ve bütün instance'lar eşit ağırlıkla ortalanır.
- Teacher advantage, ilgili pseudo referansta öğretmen modelin IoU değeri ile diğer iki modelin ortalaması arasındaki farktır.
- Eşleşmiş fark güven aralıkları kaynak sahne kümeli bootstrap ile hesaplanır; aynı büyük sahneden gelen crop'lar bağımsız sayılmaz.
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

## Eşleşmiş Referans Etkileri

| Model | BBox | Karşılaştırılan Referans | Temel IoU | Referans IoU | IoU Farkı (%95 GA) |
| --- | --- | --- | --- | --- | --- |
| SAM1 | GT bbox | SAM1 pseudo | 0.653 | 1.000 | +0.347 [+0.331, +0.369] |
| SAM1 | GT bbox | SAM2 pseudo | 0.653 | 0.827 | +0.175 [+0.159, +0.189] |
| SAM1 | GT bbox | SAM3 pseudo | 0.653 | 0.820 | +0.167 [+0.151, +0.182] |
| SAM1 | YOLO bbox | SAM1 pseudo | 0.597 | 0.873 | +0.276 [+0.257, +0.297] |
| SAM1 | YOLO bbox | SAM2 pseudo | 0.597 | 0.749 | +0.152 [+0.136, +0.166] |
| SAM1 | YOLO bbox | SAM3 pseudo | 0.597 | 0.742 | +0.145 [+0.130, +0.158] |
| SAM2 | GT bbox | SAM1 pseudo | 0.629 | 0.827 | +0.198 [+0.180, +0.215] |
| SAM2 | GT bbox | SAM2 pseudo | 0.629 | 1.000 | +0.371 [+0.352, +0.396] |
| SAM2 | GT bbox | SAM3 pseudo | 0.629 | 0.784 | +0.154 [+0.142, +0.168] |
| SAM2 | YOLO bbox | SAM1 pseudo | 0.574 | 0.750 | +0.177 [+0.158, +0.195] |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.574 | 0.853 | +0.279 [+0.258, +0.302] |
| SAM2 | YOLO bbox | SAM3 pseudo | 0.574 | 0.706 | +0.132 [+0.120, +0.144] |
| SAM3 | GT bbox | SAM1 pseudo | 0.700 | 0.820 | +0.120 [+0.097, +0.140] |
| SAM3 | GT bbox | SAM2 pseudo | 0.700 | 0.784 | +0.084 [+0.066, +0.101] |
| SAM3 | GT bbox | SAM3 pseudo | 0.700 | 1.000 | +0.300 [+0.285, +0.320] |
| SAM3 | YOLO bbox | SAM1 pseudo | 0.638 | 0.741 | +0.103 [+0.080, +0.121] |
| SAM3 | YOLO bbox | SAM2 pseudo | 0.638 | 0.702 | +0.064 [+0.046, +0.081] |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.638 | 0.863 | +0.224 [+0.208, +0.240] |

## Öğretmen Avantajı

| Model | BBox | Referans | Öğretmen IoU | Diğerleri Ort. | Öğretmen Avantajı | Identity Control |
| --- | --- | --- | --- | --- | --- | --- |
| SAM1 | GT bbox | SAM1 pseudo | 1.000 | 0.824 | +0.176 | Evet |
| SAM2 | GT bbox | SAM2 pseudo | 1.000 | 0.805 | +0.195 | Evet |
| SAM3 | GT bbox | SAM3 pseudo | 1.000 | 0.802 | +0.198 | Evet |
| SAM1 | YOLO bbox | SAM1 pseudo | 0.873 | 0.746 | +0.128 | Hayır |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.853 | 0.726 | +0.127 | Hayır |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.863 | 0.724 | +0.139 | Hayır |

## Referanslar Arası Anlaşma

| Referans A | Referans B | Referans Anlaşması | Instance |
| --- | --- | --- | --- |
| İnsan | SAM1 pseudo | 0.653 | 5447 |
| İnsan | SAM2 pseudo | 0.629 | 5447 |
| İnsan | SAM3 pseudo | 0.700 | 5447 |
| SAM1 pseudo | SAM2 pseudo | 0.827 | 5447 |
| SAM1 pseudo | SAM3 pseudo | 0.820 | 5447 |
| SAM2 pseudo | SAM3 pseudo | 0.784 | 5447 |

## Model Sıralamaları

| BBox | Referans | Sıralama | Temele Göre Değişen Sıra |
| --- | --- | --- | --- |
| GT bbox | İnsan | SAM3 > SAM1 > SAM2 | 0 |
| GT bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 | 3 |
| GT bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 2 |
| GT bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 | 0 |
| YOLO bbox | İnsan | SAM3 > SAM1 > SAM2 | 0 |
| YOLO bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 | 3 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 2 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 | 0 |

## Boş Referans Denetimi

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| İnsan | 0 | 0.0000 |
| SAM1 pseudo | 0 | 0.0000 |
| SAM2 pseudo | 0 | 0.0000 |
| SAM3 pseudo | 0 | 0.0000 |
