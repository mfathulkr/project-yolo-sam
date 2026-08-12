# Samrs Plane Cross-Reference Analysis

## Teknik Özet

- Temel referansta YOLO-bbox Overall Avg IoU bakımından en yüksek model SAM1 (0.813) olmuştur.
- Aynı dondurulmuş tahminler bütün referanslara karşı değerlendirildiği için sütunlar arasındaki fark yalnız referans maskesi değişiminin etkisini gösterir.
- GT-bbox öz-referans diagonali bağımsız performans değil coverage-aware identity control'dür; ana teacher-affinity kanıtı YOLO-bbox hücreleridir.
- Tabaka sonuçları, etkinin kalabalık/örtüşen sahne ve hedef alanı koşullarında tutarlı olup olmadığını kontrol eder.
- Yayımlanmış SAMRS referansı insan GT değildir. Bu nedenle sonuç, mutlak kalite karşılaştırmasından çok SAM-türevi referans yakınlığı analizidir.

## Kapsam ve Tanımlar

- Kapsam: 512 görüntü. her dört tabakada 128 görüntü ve toplam 3.713 uçak instance.
- Avg IoU instance başına hesaplanır ve bütün instance'lar eşit ağırlıkla ortalanır.
- Teacher advantage, ilgili pseudo referansta öğretmen modelin IoU değeri ile diğer iki modelin ortalaması arasındaki farktır.
- Eşleşmiş fark güven aralıkları kaynak sahne kümeli bootstrap ile hesaplanır; aynı büyük sahneden gelen crop'lar bağımsız sayılmaz.
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

## Eşleşmiş Referans Etkileri

| Model | BBox | Karşılaştırılan Referans | Temel IoU | Referans IoU | IoU Farkı (%95 GA) |
| --- | --- | --- | --- | --- | --- |
| SAM1 | GT bbox | SAM2 pseudo | 0.991 | 0.785 | -0.206 [-0.262, -0.159] |
| SAM1 | GT bbox | SAM3 pseudo | 0.991 | 0.813 | -0.178 [-0.215, -0.146] |
| SAM1 | GT bbox | Yeniden SAM1 | 0.991 | 1.000 | +0.009 [+0.004, +0.017] |
| SAM1 | YOLO bbox | SAM2 pseudo | 0.813 | 0.676 | -0.137 [-0.167, -0.114] |
| SAM1 | YOLO bbox | SAM3 pseudo | 0.813 | 0.695 | -0.118 [-0.137, -0.099] |
| SAM1 | YOLO bbox | Yeniden SAM1 | 0.813 | 0.818 | +0.005 [+0.002, +0.011] |
| SAM2 | GT bbox | SAM2 pseudo | 0.781 | 1.000 | +0.219 [+0.173, +0.274] |
| SAM2 | GT bbox | SAM3 pseudo | 0.781 | 0.751 | -0.030 [-0.046, -0.013] |
| SAM2 | GT bbox | Yeniden SAM1 | 0.781 | 0.785 | +0.004 [+0.001, +0.010] |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.679 | 0.785 | +0.106 [+0.094, +0.123] |
| SAM2 | YOLO bbox | SAM3 pseudo | 0.679 | 0.643 | -0.035 [-0.048, -0.022] |
| SAM2 | YOLO bbox | Yeniden SAM1 | 0.679 | 0.682 | +0.004 [+0.001, +0.008] |
| SAM3 | GT bbox | SAM2 pseudo | 0.808 | 0.751 | -0.057 [-0.082, -0.037] |
| SAM3 | GT bbox | SAM3 pseudo | 0.808 | 1.000 | +0.192 [+0.161, +0.230] |
| SAM3 | GT bbox | Yeniden SAM1 | 0.808 | 0.813 | +0.005 [+0.002, +0.010] |
| SAM3 | YOLO bbox | SAM2 pseudo | 0.691 | 0.639 | -0.052 [-0.073, -0.035] |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.691 | 0.799 | +0.108 [+0.091, +0.128] |
| SAM3 | YOLO bbox | Yeniden SAM1 | 0.691 | 0.695 | +0.004 [+0.002, +0.008] |

## Öğretmen Avantajı

| Model | BBox | Referans | Öğretmen IoU | Diğerleri Ort. | Öğretmen Avantajı | Identity Control |
| --- | --- | --- | --- | --- | --- | --- |
| SAM2 | GT bbox | SAM2 pseudo | 1.000 | 0.768 | +0.232 | Evet |
| SAM3 | GT bbox | SAM3 pseudo | 1.000 | 0.782 | +0.218 | Evet |
| SAM1 | GT bbox | Yayımlanmış SAMRS | 0.991 | 0.794 | +0.196 | Hayır |
| SAM1 | GT bbox | Yeniden SAM1 | 1.000 | 0.799 | +0.201 | Evet |
| SAM2 | YOLO bbox | SAM2 pseudo | 0.785 | 0.657 | +0.128 | Hayır |
| SAM3 | YOLO bbox | SAM3 pseudo | 0.799 | 0.669 | +0.130 | Hayır |
| SAM1 | YOLO bbox | Yayımlanmış SAMRS | 0.813 | 0.685 | +0.128 | Hayır |
| SAM1 | YOLO bbox | Yeniden SAM1 | 0.818 | 0.689 | +0.129 | Hayır |

## Referanslar Arası Anlaşma

| Referans A | Referans B | Referans Anlaşması | Instance |
| --- | --- | --- | --- |
| Yayımlanmış SAMRS | Yeniden SAM1 | 0.991 | 3713 |
| Yayımlanmış SAMRS | SAM2 pseudo | 0.781 | 3713 |
| Yayımlanmış SAMRS | SAM3 pseudo | 0.808 | 3713 |
| Yeniden SAM1 | SAM2 pseudo | 0.785 | 3713 |
| Yeniden SAM1 | SAM3 pseudo | 0.813 | 3713 |
| SAM2 pseudo | SAM3 pseudo | 0.751 | 3713 |

## Model Sıralamaları

| BBox | Referans | Sıralama | Temele Göre Değişen Sıra |
| --- | --- | --- | --- |
| GT bbox | Yayımlanmış SAMRS | SAM1 > SAM3 > SAM2 | 0 |
| GT bbox | Yeniden SAM1 | SAM1 > SAM3 > SAM2 | 0 |
| GT bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 3 |
| GT bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 | 2 |
| YOLO bbox | Yayımlanmış SAMRS | SAM1 > SAM3 > SAM2 | 0 |
| YOLO bbox | Yeniden SAM1 | SAM1 > SAM3 > SAM2 | 0 |
| YOLO bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 3 |
| YOLO bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 | 2 |

## Boş Referans Denetimi

| Referans | Boş Maske | Boş Oranı |
| --- | --- | --- |
| SAM2 pseudo | 0 | 0.0000 |
| SAM3 pseudo | 0 | 0.0000 |
| Yayımlanmış SAMRS | 0 | 0.0000 |
| Yeniden SAM1 | 0 | 0.0000 |
