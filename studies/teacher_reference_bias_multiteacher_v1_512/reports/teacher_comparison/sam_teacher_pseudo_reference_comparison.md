# SAM1/SAM2/SAM3 Pseudo Referans Karşılaştırması

## Teknik Özet

- Her pseudo referans GT-bbox koşulunda kendi üreticisine 1,000 IoU verir. Bu matematiksel özdeşlik başarı değil, deneyin pozitif kontrolüdür.
- YOLO bbox koşulunda da her pseudo referans kendi öğretmen modelini en yüksek skora taşır. Plane için öğretmen avantajı 0,113–0,138; Small Vehicle için 0,179–0,299 IoU aralığındadır.
- İnsan referansındaki model sıralaması pseudo referansa göre değişebilir. Bu nedenle model seçimi, referansı üreten model ailesine bağımlı hâle gelir.
- SAM3 Small Vehicle öğretmen çıktılarının 5.345/12.051'i (%44,4) boştur. SAM3'ün kendi pseudo referansında yüksek görünmesi bağımsız doğruluk kanıtı değildir.

## Kapsam ve Tanımlar

- Kapsam: iSAID Plane'de 5.447, iSAID Small Vehicle'da 12.051 instance; her veri setinde 512 görüntü ve dört eşit 128 görüntülük stratum.
- Sabitler: görüntüler, instance'lar, insan GT bbox istemleri, seed 42 YOLO bbox istemleri ve SAM1/2/3 tahminleri aynıdır. Değişen tek değerlendirme girdisi referans maskedir.
- Avg IoU her instance için ayrı hesaplanır ve instance'lar eşit ağırlıkla ortalanır. Büyük nesneler sonucu piksel alanıyla baskılamaz.
- Pseudo referans, SAM1/SAM2/SAM3'ün insan GT bbox istemiyle ürettiği maskedir. Bu nedenle insan lokalizasyon bilgisi korunur; deney maske sınırı ve model-öğretmen uyumuna odaklanır.
- Scene-clustered bootstrap güven aralıkları aynı kaynak sahneden gelen crop'ların bağımsız kabul edilmesini önler.

## iSAID Plane

### GT bbox Overall Avg IoU

| Model | Human | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.653 | 1.000 | 0.827 | 0.795 |
| SAM2 | 0.629 | 0.827 | 1.000 | 0.793 |
| SAM3 | 0.655 | 0.795 | 0.793 | 1.000 |

### YOLO bbox Overall Avg IoU

| Model | Human | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.597 | 0.873 | 0.749 | 0.747 |
| SAM2 | 0.574 | 0.750 | 0.853 | 0.744 |
| SAM3 | 0.595 | 0.721 | 0.718 | 0.858 |

## iSAID Small Vehicle

### GT bbox Overall Avg IoU

| Model | Human | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.658 | 1.000 | 0.749 | 0.419 |
| SAM2 | 0.645 | 0.749 | 1.000 | 0.420 |
| SAM3 | 0.370 | 0.419 | 0.420 | 1.000 |

### YOLO bbox Overall Avg IoU

| Model | Human | SAM1 pseudo | SAM2 pseudo | SAM3 pseudo |
| --- | --- | --- | --- | --- |
| SAM1 | 0.478 | 0.655 | 0.551 | 0.539 |
| SAM2 | 0.461 | 0.550 | 0.624 | 0.539 |
| SAM3 | 0.299 | 0.341 | 0.339 | 0.838 |

## Referanslar Arası Anlaşma

| Dataset | Reference A | Reference B | Mean instance IoU | Instances |
| --- | --- | --- | --- | --- |
| iSAID Plane | Human | SAM1 pseudo | 0.653 | 5447 |
| iSAID Plane | Human | SAM2 pseudo | 0.629 | 5447 |
| iSAID Plane | Human | SAM3 pseudo | 0.655 | 5447 |
| iSAID Plane | SAM1 pseudo | SAM2 pseudo | 0.827 | 5447 |
| iSAID Plane | SAM1 pseudo | SAM3 pseudo | 0.795 | 5447 |
| iSAID Plane | SAM2 pseudo | SAM3 pseudo | 0.793 | 5447 |
| iSAID Small Vehicle | Human | SAM1 pseudo | 0.658 | 12051 |
| iSAID Small Vehicle | Human | SAM2 pseudo | 0.645 | 12051 |
| iSAID Small Vehicle | Human | SAM3 pseudo | 0.370 | 12051 |
| iSAID Small Vehicle | SAM1 pseudo | SAM2 pseudo | 0.749 | 12051 |
| iSAID Small Vehicle | SAM1 pseudo | SAM3 pseudo | 0.419 | 12051 |
| iSAID Small Vehicle | SAM2 pseudo | SAM3 pseudo | 0.420 | 12051 |

## Boş Maske Denetimi

| Dataset | Teacher | Instances | Empty masks | Empty rate |
| --- | --- | --- | --- | --- |
| iSAID Plane | SAM1 | 5447 | 0 | 0.000 |
| iSAID Plane | SAM2 | 5447 | 0 | 0.000 |
| iSAID Plane | SAM3 | 5447 | 133 | 0.024 |
| iSAID Small Vehicle | SAM1 | 12051 | 19 | 0.002 |
| iSAID Small Vehicle | SAM2 | 12051 | 0 | 0.000 |
| iSAID Small Vehicle | SAM3 | 12051 | 5345 | 0.444 |

## Model Sıralamaları

| Dataset | BBox | Reference | Ranking | Changes vs human |
| --- | --- | --- | --- | --- |
| iSAID Plane | gt_bbox | Human | SAM3 > SAM1 > SAM2 | 0 |
| iSAID Plane | gt_bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 | 3 |
| iSAID Plane | gt_bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 2 |
| iSAID Plane | gt_bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 | 0 |
| iSAID Plane | yolo_bbox | Human | SAM1 > SAM3 > SAM2 | 0 |
| iSAID Plane | yolo_bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 | 2 |
| iSAID Plane | yolo_bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 3 |
| iSAID Plane | yolo_bbox | SAM3 pseudo | SAM3 > SAM1 > SAM2 | 2 |
| iSAID Small Vehicle | gt_bbox | Human | SAM1 > SAM2 > SAM3 | 0 |
| iSAID Small Vehicle | gt_bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 | 0 |
| iSAID Small Vehicle | gt_bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 2 |
| iSAID Small Vehicle | gt_bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 | 2 |
| iSAID Small Vehicle | yolo_bbox | Human | SAM1 > SAM2 > SAM3 | 0 |
| iSAID Small Vehicle | yolo_bbox | SAM1 pseudo | SAM1 > SAM2 > SAM3 | 0 |
| iSAID Small Vehicle | yolo_bbox | SAM2 pseudo | SAM2 > SAM1 > SAM3 | 2 |
| iSAID Small Vehicle | yolo_bbox | SAM3 pseudo | SAM3 > SAM2 > SAM1 | 2 |

## Sınırlılıklar

- Bu çalışma pseudo etiketlerin eğitimde yararsız olduğunu kanıtlamaz. Test referansının bağımsız olmadığı durumda değerlendirme ve model sıralamasının bozulabildiğini gösterir.
- GT-bbox diagonal hücreleri tautological identity kontrolüdür; makalede ana performans sonucu olarak kullanılmamalıdır.
- YOLO-bbox sonuçları daha güçlü kanıttır, çünkü öğrenci istemi öğretmen referans isteminden farklıdır; ancak referans ve aday model aynı mimari aileden olduğu için hata korelasyonu hâlâ beklenir.
- SAM3 Small Vehicle boş maskeleri ayrı bir failure mode'dur. Boş referanslarda hem boş öğrenci tahmini 1,0 alabilir; boş oranı olmadan ortalama skor yanıltıcıdır.
- İki sınıf ve tek remote-sensing veri seti ailesiyle sınırlıyız. Sonuç genellenebilirlik iddiası değil, kontrollü bir ölçüm geçerliliği uyarısıdır.

## Önerilen Raporlama Protokolü

- Ana model karşılaştırmasını insan referansı üzerinde raporla; pseudo referans sonuçlarını duyarlılık analizi olarak ayrı göster.
- Pseudo referans üreticisini, sürümünü, istemini, boş maske oranını ve post-processing adımlarını açıkça raporla.
- Referans üreticisiyle aynı model ailesini değerlendirirken diagonal/self-reference sonucunu performans tablosundan ayır veya açıkça identity control olarak işaretle.
- Mümkünse küçük, kör ve bağımsız bir insan audit alt kümesi kullan; model sıralamasının bu alt kümede korunup korunmadığını kontrol et.
- Bildiri ana figürü olarak model–referans matrisini, ana istatistik olarak pseudo−human paired IoU değişimini ve güven aralığını kullan.
