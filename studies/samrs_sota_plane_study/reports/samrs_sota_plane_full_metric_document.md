# SAMRS SOTA Plane Segmentation Metric Report

## Scope

- Veri seti: SAMRS içindeki SOTA subset; hedef sınıf `plane`.
- SOTA, DOTA v2.0 kaynaklı yüksek çözünürlüklü remote sensing patch'lerinden türetilmiştir.
- Önemli sınır: SAMRS maskeleri insan tarafından çizilmiş kesin GT değil, SAM ile üretilmiş pseudo-mask etiketleridir. Bu rapor bu yüzden `SAMRS-SOTA pseudo-mask benchmark` olarak okunmalıdır.
- Değerlendirme kümesi hedef olarak her stratum için 128 pozitif görüntü seçer: Overall, No Overlap / Low Mask Area, No Overlap / High Mask Area, Overlap / Low Mask Area, Overlap / High Mask Area.
- Bu deney iSAID vehicle deneyinin aynı mantığını daha az boxy bir nesneye taşır. Plane nesnesinde kanat, kuyruk ve gövde nedeniyle bbox nesne maskesine tam oturmaz; bbox promptunun maskeye etkisi bu yüzden daha anlamlı test edilir.
- SAM1 bu deneyde ek baseline olarak eklendi. Böylece aynı bbox kalitesiyle SAM1, SAM2 ve SAM3 davranışı ayrıca görülebilir.

## Metric Logic

- Segmentasyon metrikleri görüntü seviyesinde birleştirilmiş ikili plane maskesi üzerinde hesaplanır.
- TP, modelin plane dediği ve pseudo-GT maskesinde de plane olan piksel sayısıdır.
- FP, modelin plane dediği ama pseudo-GT'de arka plan olan piksel sayısıdır. FP artarsa maske nesne dışına taşıyor demektir.
- FN, pseudo-GT'de plane olan ama modelin kaçırdığı piksel sayısıdır. FN artarsa model hedef maskeyi eksik yakalıyor demektir.
- IoU, `TP / (TP + FP + FN)` oranıdır. Pixel-level maske örtüşmesini ölçer.
- Dice, `2TP / (2TP + FP + FN)` oranıdır. IoU'ya benzer bir örtüşme skorudur, genellikle IoU'dan daha yüksek görünür.
- Precision, `TP / (TP + FP)` oranıdır. Düşük precision, maske tahmininin hedef dışına fazla taştığını gösterir.
- Recall, `TP / (TP + FN)` oranıdır. Düşük recall, hedef plane piksellerinin eksik yakalandığını gösterir.
- Ortalama metrikler önce her görüntü için ayrıca hesaplanır, sonra görüntü skorlarının ortalaması alınır. Böylece tek bir görüntü rapor ortalamasında ekstra ağırlık almaz.
- Yine de tek görüntü içindeki hesap piksel tabanlıdır. Büyük maskeler o görüntünün TP/FP/FN sayımlarını domine edebilir; bu yüzden low/high mask area tabloları ayrıca tutulur.
- Pred/GT Area, tahmin edilen plane piksel sayısının pseudo-GT plane piksel sayısına oranıdır. 1'in üstü over-segmentation, 1'in altı under-segmentation işaretidir.
- Segmentasyon mAP50 proxy, mAP75 proxy ve mAP90 proxy değerleri COCO AP değildir. Bunlar görüntü seviyesinde IoU eşik geçme oranlarıdır.
- mAP50-95 proxy, 0.50, 0.55, ..., 0.95 eşiklerindeki geçme oranlarının ortalamasıdır.
- YOLO detector metrikleri bbox metriğidir. Buradaki IoU, tahmin bbox'u ile pseudo-GT bbox'u arasındaki kutu örtüşmesidir; maske IoU değildir.

## Dataset Context

- [SAMRS resmi repo](https://github.com/ViTAE-Transformer/SAMRS), veri setinin SAM ve mevcut remote sensing detection veri setlerinden üretildiğini belirtir.
- [SAMRS NeurIPS 2023 makalesi](https://papers.nips.cc/paper_files/paper/2023/file/1be3843e534ee06d3a70c7f62b983b31-Paper-Datasets_and_Benchmarks.pdf), SAMRS'in 105.090 görüntü ve 1.668.241 instance içerdiğini raporlar.
- SOTA subset, DOTA v2.0 kaynaklı olduğu için kalabalık sahneler, küçük nesneler ve farklı ölçekler açısından iSAID vehicle deneyine yakın bir stres testi verir.
- Plane sınıfı vehicle'a göre daha az boxy olduğu için bbox-only prompt ile maskenin kanat/kuyruk gibi çıkıntıları ne kadar yakaladığı daha net görülür.

## YOLO Detector BBox Metrics

Not: Bu tablo YOLO'yu yalnızca detector olarak değerlendirir. Buradaki metrikler bbox metriğidir, maske metrikleri değildir.

| Split | Images | Detections | AP conf | Fixed conf | BBox mAP50 | BBox mAP75 | BBox mAP90 | BBox mAP50-95 | BBox Precision@0.50 | BBox Recall@0.50 | BBox Precision@0.75 | BBox Recall@0.75 | BBox Precision@0.90 | BBox Recall@0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| değerlendirme | 512 | 38911 | 0.0010 | 0.2000 | 0.9066 | 0.7975 | 0.5320 | 0.7327 | 0.8249 | 0.8936 | 0.7507 | 0.8132 | 0.5699 | 0.6174 |

## Segmentation Tables

DOCX/PDF çıktılarında 0.0-1.0 aralığındaki başarı metrikleri kırmızıdan sarıya, sarıdan yeşile giden renk ölçeğiyle boyanır.

### Overall

Not: Bu tablodaki segmentasyon mAP proxy değerleri, her model hattı için 512 görüntü üzerinden hesaplanan görüntü seviyesinde IoU eşik geçme oranlarıdır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | Pred/GT Area | mAP50 proxy | mAP75 proxy | mAP90 proxy | mAP50-95 proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM3 text only | 512 | 0.6459 | 0.7519 | 0.8057 | 0.7354 | 1.2525 | 0.8008 | 0.4219 | 0.0449 | 0.4352 |
| SAM3 YOLO bbox | 512 | 0.6565 | 0.7528 | 0.7612 | 0.7744 | 1.3234 | 0.8027 | 0.5078 | 0.0508 | 0.4701 |
| SAM3 GT bbox | 512 | 0.7326 | 0.8275 | 0.8152 | 0.8674 | 1.2324 | 0.8848 | 0.6406 | 0.0859 | 0.5598 |
| SAM3 hybrid YOLO bbox | 512 | 0.6614 | 0.7582 | 0.7659 | 0.7800 | 1.3759 | 0.8027 | 0.5215 | 0.0547 | 0.4777 |
| RemoteSAM text only | 512 | 0.7963 | 0.8758 | 0.8406 | 0.9256 | 1.3421 | 0.9512 | 0.7617 | 0.1562 | 0.6607 |
| RingMo-SAM GT bbox | 512 | 0.0046 | 0.0082 | 0.2286 | 0.0046 | 0.0070 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RingMo-SAM YOLO bbox | 512 | 0.0035 | 0.0064 | 0.2071 | 0.0035 | 0.0060 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SAM1 GT bbox | 512 | 0.9105 | 0.9444 | 0.9575 | 0.9393 | 1.0649 | 0.9727 | 0.9258 | 0.7793 | 0.8809 |
| SAM1 YOLO bbox | 512 | 0.7342 | 0.8012 | 0.8340 | 0.8011 | 0.9718 | 0.8086 | 0.6719 | 0.3945 | 0.6195 |
| SAM2 GT bbox | 512 | 0.8203 | 0.8893 | 0.8794 | 0.9093 | 1.0708 | 0.9570 | 0.8457 | 0.2480 | 0.7135 |
| SAM2 YOLO bbox | 512 | 0.6684 | 0.7604 | 0.7700 | 0.7837 | 1.0358 | 0.7910 | 0.5371 | 0.0938 | 0.4916 |

### No Overlap / Low Mask Area

Not: Bu tablodaki segmentasyon mAP proxy değerleri, her model hattı için 128 görüntü üzerinden hesaplanan görüntü seviyesinde IoU eşik geçme oranlarıdır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | Pred/GT Area | mAP50 proxy | mAP75 proxy | mAP90 proxy | mAP50-95 proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM3 text only | 128 | 0.6385 | 0.7369 | 0.7774 | 0.7296 | 2.1468 | 0.7969 | 0.4531 | 0.0859 | 0.4445 |
| SAM3 YOLO bbox | 128 | 0.6450 | 0.7391 | 0.7441 | 0.7676 | 2.1822 | 0.7969 | 0.4844 | 0.0703 | 0.4609 |
| SAM3 GT bbox | 128 | 0.7218 | 0.8110 | 0.8028 | 0.8560 | 1.4783 | 0.8672 | 0.6094 | 0.1172 | 0.5633 |
| SAM3 hybrid YOLO bbox | 128 | 0.6440 | 0.7378 | 0.7410 | 0.7687 | 2.4291 | 0.7891 | 0.4922 | 0.0781 | 0.4656 |
| RemoteSAM text only | 128 | 0.7824 | 0.8572 | 0.8311 | 0.9018 | 2.0113 | 0.9141 | 0.7891 | 0.1562 | 0.6602 |
| RingMo-SAM GT bbox | 128 | 0.0011 | 0.0022 | 0.1378 | 0.0012 | 0.0042 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RingMo-SAM YOLO bbox | 128 | 0.0011 | 0.0021 | 0.1203 | 0.0011 | 0.0065 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SAM1 GT bbox | 128 | 0.8964 | 0.9266 | 0.9409 | 0.9243 | 1.2530 | 0.9453 | 0.9219 | 0.8047 | 0.8766 |
| SAM1 YOLO bbox | 128 | 0.7413 | 0.7998 | 0.8056 | 0.8235 | 1.1132 | 0.8047 | 0.6719 | 0.4609 | 0.6430 |
| SAM2 GT bbox | 128 | 0.8055 | 0.8709 | 0.8607 | 0.8972 | 1.1519 | 0.9375 | 0.8125 | 0.2578 | 0.7023 |
| SAM2 YOLO bbox | 128 | 0.6735 | 0.7577 | 0.7411 | 0.8056 | 1.2071 | 0.7969 | 0.5312 | 0.1797 | 0.5164 |

### No Overlap / High Mask Area

Not: Bu tablodaki segmentasyon mAP proxy değerleri, her model hattı için 128 görüntü üzerinden hesaplanan görüntü seviyesinde IoU eşik geçme oranlarıdır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | Pred/GT Area | mAP50 proxy | mAP75 proxy | mAP90 proxy | mAP50-95 proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM3 text only | 128 | 0.6358 | 0.7418 | 0.7969 | 0.7213 | 0.9710 | 0.7969 | 0.3906 | 0.0703 | 0.4227 |
| SAM3 YOLO bbox | 128 | 0.6175 | 0.7056 | 0.7198 | 0.7203 | 0.9509 | 0.7578 | 0.4766 | 0.0781 | 0.4477 |
| SAM3 GT bbox | 128 | 0.7199 | 0.8169 | 0.8158 | 0.8517 | 1.2046 | 0.8672 | 0.6094 | 0.1328 | 0.5391 |
| SAM3 hybrid YOLO bbox | 128 | 0.6407 | 0.7291 | 0.7423 | 0.7432 | 0.9549 | 0.7891 | 0.5156 | 0.0859 | 0.4766 |
| RemoteSAM text only | 128 | 0.8080 | 0.8815 | 0.8378 | 0.9398 | 1.1328 | 0.9375 | 0.7812 | 0.2734 | 0.6867 |
| RingMo-SAM GT bbox | 128 | 0.0026 | 0.0047 | 0.1507 | 0.0026 | 0.0056 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RingMo-SAM YOLO bbox | 128 | 0.0025 | 0.0045 | 0.1416 | 0.0025 | 0.0049 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SAM1 GT bbox | 128 | 0.9152 | 0.9463 | 0.9547 | 0.9473 | 1.0344 | 0.9688 | 0.9062 | 0.8281 | 0.8867 |
| SAM1 YOLO bbox | 128 | 0.7001 | 0.7586 | 0.7812 | 0.7580 | 0.8945 | 0.7578 | 0.6328 | 0.4453 | 0.6023 |
| SAM2 GT bbox | 128 | 0.8070 | 0.8740 | 0.8793 | 0.8814 | 1.0260 | 0.9219 | 0.7734 | 0.3750 | 0.7023 |
| SAM2 YOLO bbox | 128 | 0.6385 | 0.7208 | 0.7411 | 0.7267 | 0.9008 | 0.7422 | 0.5234 | 0.1094 | 0.4836 |

### Overlap / Low Mask Area

Not: Bu tablodaki segmentasyon mAP proxy değerleri, her model hattı için 128 görüntü üzerinden hesaplanan görüntü seviyesinde IoU eşik geçme oranlarıdır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | Pred/GT Area | mAP50 proxy | mAP75 proxy | mAP90 proxy | mAP50-95 proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM3 text only | 128 | 0.6340 | 0.7475 | 0.8414 | 0.7104 | 0.8639 | 0.7969 | 0.3516 | 0.0000 | 0.4016 |
| SAM3 YOLO bbox | 128 | 0.6705 | 0.7787 | 0.8041 | 0.7825 | 1.0316 | 0.8438 | 0.4609 | 0.0078 | 0.4531 |
| SAM3 GT bbox | 128 | 0.7499 | 0.8492 | 0.8363 | 0.8798 | 1.0988 | 0.9453 | 0.6484 | 0.0391 | 0.5648 |
| SAM3 hybrid YOLO bbox | 128 | 0.6691 | 0.7778 | 0.8045 | 0.7831 | 1.0155 | 0.8281 | 0.4609 | 0.0078 | 0.4516 |
| RemoteSAM text only | 128 | 0.7701 | 0.8644 | 0.8381 | 0.9038 | 1.0879 | 0.9688 | 0.6484 | 0.0234 | 0.5961 |
| RingMo-SAM GT bbox | 128 | 0.0136 | 0.0237 | 0.3644 | 0.0136 | 0.0139 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RingMo-SAM YOLO bbox | 128 | 0.0095 | 0.0172 | 0.3280 | 0.0096 | 0.0104 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SAM1 GT bbox | 128 | 0.9103 | 0.9511 | 0.9785 | 0.9271 | 0.9496 | 0.9922 | 0.9609 | 0.6641 | 0.8695 |
| SAM1 YOLO bbox | 128 | 0.7168 | 0.8011 | 0.8819 | 0.7724 | 0.9094 | 0.8125 | 0.6094 | 0.2656 | 0.5617 |
| SAM2 GT bbox | 128 | 0.8312 | 0.9062 | 0.8775 | 0.9407 | 1.0790 | 0.9922 | 0.9297 | 0.1016 | 0.7156 |
| SAM2 YOLO bbox | 128 | 0.6571 | 0.7641 | 0.7912 | 0.7797 | 1.0200 | 0.7891 | 0.4453 | 0.0469 | 0.4367 |

### Overlap / High Mask Area

Not: Bu tablodaki segmentasyon mAP proxy değerleri, her model hattı için 128 görüntü üzerinden hesaplanan görüntü seviyesinde IoU eşik geçme oranlarıdır.

| Pipeline | Images | Avg IoU | Avg Dice | Avg Precision | Avg Recall | Pred/GT Area | mAP50 proxy | mAP75 proxy | mAP90 proxy | mAP50-95 proxy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SAM3 text only | 128 | 0.6752 | 0.7814 | 0.8072 | 0.7805 | 1.0286 | 0.8125 | 0.4922 | 0.0234 | 0.4719 |
| SAM3 YOLO bbox | 128 | 0.6929 | 0.7880 | 0.7768 | 0.8273 | 1.1288 | 0.8125 | 0.6094 | 0.0469 | 0.5188 |
| SAM3 GT bbox | 128 | 0.7391 | 0.8330 | 0.8060 | 0.8820 | 1.1479 | 0.8594 | 0.6953 | 0.0547 | 0.5719 |
| SAM3 hybrid YOLO bbox | 128 | 0.6920 | 0.7879 | 0.7758 | 0.8249 | 1.1040 | 0.8047 | 0.6172 | 0.0469 | 0.5172 |
| RemoteSAM text only | 128 | 0.8246 | 0.9002 | 0.8555 | 0.9571 | 1.1364 | 0.9844 | 0.8281 | 0.1719 | 0.7000 |
| RingMo-SAM GT bbox | 128 | 0.0011 | 0.0022 | 0.2614 | 0.0011 | 0.0043 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RingMo-SAM YOLO bbox | 128 | 0.0010 | 0.0019 | 0.2385 | 0.0010 | 0.0021 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SAM1 GT bbox | 128 | 0.9202 | 0.9536 | 0.9557 | 0.9587 | 1.0225 | 0.9844 | 0.9141 | 0.8203 | 0.8906 |
| SAM1 YOLO bbox | 128 | 0.7788 | 0.8451 | 0.8674 | 0.8504 | 0.9701 | 0.8594 | 0.7734 | 0.4062 | 0.6711 |
| SAM2 GT bbox | 128 | 0.8377 | 0.9060 | 0.9001 | 0.9178 | 1.0264 | 0.9766 | 0.8672 | 0.2578 | 0.7336 |
| SAM2 YOLO bbox | 128 | 0.7046 | 0.7989 | 0.8067 | 0.8227 | 1.0152 | 0.8359 | 0.6484 | 0.0391 | 0.5297 |

## Discussion

- `Overall` tablosunda en yüksek Avg IoU `SAM1 GT bbox` hattında `0.9105` olarak ölçüldü.
- SAM2 YOLO bbox ile SAM1 YOLO bbox farkı `-0.0658` Avg IoU.
- SAM3 YOLO bbox ile SAM2 YOLO bbox farkı `-0.0120` Avg IoU.
- SAM3 hybrid YOLO bbox, SAM3 YOLO bbox hattına göre Avg IoU değerini `+0.0050` değiştirdi.
- Hybrid prompt bbox-only sonucunun garantili iyileştirmesi değildir. Text + bbox birlikte verildiğinde modelin maske üretim davranışı değişebilir.
- Plane sınıfında bbox maskeye tam oturmadığı için kanat ve kuyruk gibi çıkıntılar bbox promptunun sınırlarını zorlar. Bu, iSAID vehicle deneyinden farklı ve daha zor bir geometri testidir.
- Bu rapordaki GT maskeler SAMRS pseudo-mask olduğu için, sonuçlar insan çizimli nihai ground truth liderlik tablosu gibi sunulmamalı; model davranışı karşılaştırması olarak okunmalıdır.
- RingMo-SAM çıktısı bu SOTA plane kurulumunda çok düşük kaldı. RingMo semantic class-map için `class_ids: [5]` kullanımı doğrulandı; düşük skor bu yüzden boş maske id hatası olarak değil, modelin pseudo-mask hedefleriyle zayıf örtüşmesi olarak yorumlanmalıdır.
- `No Overlap / Low Mask Area` grubunda en iyi Avg IoU `SAM1 GT bbox` hattında `0.8964` olarak ölçüldü.
- `No Overlap / High Mask Area` grubunda en iyi Avg IoU `SAM1 GT bbox` hattında `0.9152` olarak ölçüldü.
- `Overlap / Low Mask Area` grubunda en iyi Avg IoU `SAM1 GT bbox` hattında `0.9103` olarak ölçüldü.
- `Overlap / High Mask Area` grubunda en iyi Avg IoU `SAM1 GT bbox` hattında `0.9202` olarak ölçüldü.
- Low Mask Area tabloları küçük veya dar hedefleri gösterir. Bu kısımda düşen skorlar bbox ile maskenin ince detayları yakalayamadığına işaret edebilir.
- Overlap tabloları kalabalık sahnelerde birden fazla plane instance'ının birbirine yakın olduğu durumları ayırır. Burada FP/FN dengesi özellikle önemlidir.
- Segmentasyon mAP proxy kolonları görüntü seviyesinde eşik geçme oranıdır; COCO instance AP değildir. YOLO detector tablosundaki BBox mAP ise gerçek COCO bbox AP hesabıdır.

## Generated Artifacts

- Görüntü bazlı metrik CSV'si: `tables/full_metric_document/per_image_metrics_selected_pipelines.csv`
- Özet tablo CSV'si: `tables/full_metric_document/summary_all_tables_selected_pipelines.csv`