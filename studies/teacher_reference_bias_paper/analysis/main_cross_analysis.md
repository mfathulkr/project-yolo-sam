# Teacher-Reference Bias Main Cross Analysis

## Ana Sonuç

- Dört deney aynı 512 görüntü / dört eşit 128 görüntülük tabaka / seed 42 / 1024×1024 / SAM1-2-3 / GT ve YOLO bbox protokolünü kullanır.
- iSAID Plane ve Small Vehicle deneylerinde insan anotasyonu bağımsız kontrol referansıdır; teacher-reference bias iddiasının ana kanıtı bu iki deneydeki eşleşmiş pseudo−human farklarıdır.
- SAMRS Plane ve Small Vehicle deneylerinde yayımlanmış etiket insan GT değildir; SAM tabanlı üretim hattından geldiği için bu iki deney destekleyici model-aile yakınlığı analizi olarak yorumlanır.
- GT-bbox öz-referans diagonalinde 1,000 görmek beklenen matematiksel identity control'dür. Ana değerlendirme, teacher ve evaluated prediction bbox istemleri farklı olan YOLO-bbox koşuludur.
- Farklı veri setlerinin IoU değerleri tek bir ortalamada birleştirilmez; sınıf ve veri ailesi bazında ayrı raporlanır.
- Sonuç pseudo etiketlemenin yararsız olduğunu değil, pseudo etiket üreticisiyle aday modelin bağımlı olduğu test referansının model seçimini yanıltabileceğini gösterir.

## Dört Deneyde Temel Referans Sonuçları

| Deney | Temel Referans | Model | Avg IoU | Instance |
| --- | --- | --- | --- | --- |
| isaid_plane | İnsan | SAM1 | 0.597 | 5447 |
| isaid_plane | İnsan | SAM2 | 0.574 | 5447 |
| isaid_plane | İnsan | SAM3 | 0.638 | 5447 |
| isaid_small_vehicle | İnsan | SAM1 | 0.478 | 12051 |
| isaid_small_vehicle | İnsan | SAM2 | 0.461 | 12051 |
| isaid_small_vehicle | İnsan | SAM3 | 0.491 | 12051 |
| samrs_plane | Yayımlanmış SAMRS | SAM1 | 0.813 | 3713 |
| samrs_plane | Yayımlanmış SAMRS | SAM2 | 0.679 | 3713 |
| samrs_plane | Yayımlanmış SAMRS | SAM3 | 0.691 | 3713 |
| samrs_small_vehicle | Yayımlanmış SAMRS | SAM1 | 0.782 | 7659 |
| samrs_small_vehicle | Yayımlanmış SAMRS | SAM2 | 0.707 | 7659 |
| samrs_small_vehicle | Yayımlanmış SAMRS | SAM3 | 0.707 | 7659 |

## YOLO-BBox Eşleşmiş Referans Etkileri

| Deney | Model | Referans | Temel IoU | Referans IoU | Fark (%95 GA) |
| --- | --- | --- | --- | --- | --- |
| isaid_plane | SAM1 | SAM1 pseudo | 0.597 | 0.873 | +0.276 [+0.257, +0.297] |
| isaid_plane | SAM1 | SAM2 pseudo | 0.597 | 0.749 | +0.152 [+0.136, +0.166] |
| isaid_plane | SAM1 | SAM3 pseudo | 0.597 | 0.742 | +0.145 [+0.130, +0.158] |
| isaid_plane | SAM2 | SAM1 pseudo | 0.574 | 0.750 | +0.177 [+0.158, +0.195] |
| isaid_plane | SAM2 | SAM2 pseudo | 0.574 | 0.853 | +0.279 [+0.258, +0.302] |
| isaid_plane | SAM2 | SAM3 pseudo | 0.574 | 0.706 | +0.132 [+0.120, +0.144] |
| isaid_plane | SAM3 | SAM1 pseudo | 0.638 | 0.741 | +0.103 [+0.080, +0.121] |
| isaid_plane | SAM3 | SAM2 pseudo | 0.638 | 0.702 | +0.064 [+0.046, +0.081] |
| isaid_plane | SAM3 | SAM3 pseudo | 0.638 | 0.863 | +0.224 [+0.208, +0.240] |
| isaid_small_vehicle | SAM1 | SAM1 pseudo | 0.478 | 0.654 | +0.176 [+0.159, +0.194] |
| isaid_small_vehicle | SAM1 | SAM2 pseudo | 0.478 | 0.551 | +0.073 [+0.058, +0.087] |
| isaid_small_vehicle | SAM1 | SAM3 pseudo | 0.478 | 0.560 | +0.082 [+0.072, +0.093] |
| isaid_small_vehicle | SAM2 | SAM1 pseudo | 0.461 | 0.549 | +0.088 [+0.076, +0.099] |
| isaid_small_vehicle | SAM2 | SAM2 pseudo | 0.461 | 0.624 | +0.163 [+0.143, +0.180] |
| isaid_small_vehicle | SAM2 | SAM3 pseudo | 0.461 | 0.551 | +0.090 [+0.079, +0.101] |
| isaid_small_vehicle | SAM3 | SAM1 pseudo | 0.491 | 0.562 | +0.071 [+0.060, +0.083] |
| isaid_small_vehicle | SAM3 | SAM2 pseudo | 0.491 | 0.555 | +0.064 [+0.049, +0.077] |
| isaid_small_vehicle | SAM3 | SAM3 pseudo | 0.491 | 0.633 | +0.142 [+0.128, +0.158] |
| samrs_plane | SAM1 | SAM2 pseudo | 0.813 | 0.676 | -0.137 [-0.167, -0.114] |
| samrs_plane | SAM1 | SAM3 pseudo | 0.813 | 0.695 | -0.118 [-0.137, -0.099] |
| samrs_plane | SAM1 | Yeniden SAM1 | 0.813 | 0.818 | +0.005 [+0.002, +0.011] |
| samrs_plane | SAM2 | SAM2 pseudo | 0.679 | 0.785 | +0.106 [+0.094, +0.123] |
| samrs_plane | SAM2 | SAM3 pseudo | 0.679 | 0.643 | -0.035 [-0.048, -0.022] |
| samrs_plane | SAM2 | Yeniden SAM1 | 0.679 | 0.682 | +0.004 [+0.001, +0.008] |
| samrs_plane | SAM3 | SAM2 pseudo | 0.691 | 0.639 | -0.052 [-0.073, -0.035] |
| samrs_plane | SAM3 | SAM3 pseudo | 0.691 | 0.799 | +0.108 [+0.091, +0.128] |
| samrs_plane | SAM3 | Yeniden SAM1 | 0.691 | 0.695 | +0.004 [+0.002, +0.008] |
| samrs_small_vehicle | SAM1 | SAM2 pseudo | 0.782 | 0.697 | -0.085 [-0.106, -0.060] |
| samrs_small_vehicle | SAM1 | SAM3 pseudo | 0.782 | 0.705 | -0.077 [-0.092, -0.061] |
| samrs_small_vehicle | SAM1 | Yeniden SAM1 | 0.782 | 0.782 | +0.000 [+0.000, +0.000] |
| samrs_small_vehicle | SAM2 | SAM2 pseudo | 0.707 | 0.749 | +0.042 [+0.030, +0.054] |
| samrs_small_vehicle | SAM2 | SAM3 pseudo | 0.707 | 0.705 | -0.002 [-0.010, +0.005] |
| samrs_small_vehicle | SAM2 | Yeniden SAM1 | 0.707 | 0.707 | -0.000 [-0.000, +0.000] |
| samrs_small_vehicle | SAM3 | SAM2 pseudo | 0.707 | 0.694 | -0.012 [-0.023, -0.001] |
| samrs_small_vehicle | SAM3 | SAM3 pseudo | 0.707 | 0.760 | +0.054 [+0.044, +0.063] |
| samrs_small_vehicle | SAM3 | Yeniden SAM1 | 0.707 | 0.707 | +0.000 [-0.000, +0.000] |

## SAMRS Yayımlanmış Referans Yakınlığı

| Deney | Referans Çifti | Referans Anlaşması | Instance |
| --- | --- | --- | --- |
| samrs_plane | Yayımlanmış SAMRS ↔ yeniden SAM1 | 0.991 | 3713 |
| samrs_small_vehicle | Yayımlanmış SAMRS ↔ yeniden SAM1 | 0.998 | 7659 |

## Sınırlılıklar

- Çalışma iki remote-sensing veri ailesi ve iki hedef sınıfla sınırlıdır.
- SAMRS için bağımsız insan instance maskesi bulunmadığından mutlak segmentasyon kalitesi iddiası kurulamaz.
- Pseudo referanslar GT bbox ile üretildiği için localization hatası teacher tarafında kontrol edilmiştir; bu, yalnız maske sınırı kaynaklı affinity etkisini izole eder.
- YOLO yanlış pozitifleri detector mAP/precision/recall tablosunda ölçülür; instance maske ortalamasına sahte bir GT örneği olarak eklenmez.
