# Deep Scientific Audit

Durum: **completed_with_declared_limitations**

Bu denetim testlerin yalnızca çalışmasını değil; split sızıntısını, test kapsamını, deneyler arası bağımlılığı ve anotasyon kaynakları arasındaki doğrudan anlaşmayı kontrol eder.

## Split ve Örnekleme

| Deney | Train/Val/Test görüntü | Train/Val/Test sahne | Test kapsamı | Tabakalar |
| --- | --- | --- | --- | --- |
| isaid_plane | 1571/321/512 | 377/82/44 | yalnız hedef-pozitif | 4×128 |
| isaid_small_vehicle | 5930/1353/512 | 964/206/31 | yalnız hedef-pozitif | 4×128 |
| samrs_plane | 2191/481/512 | 527/113/24 | yalnız hedef-pozitif | 4×128 |
| samrs_small_vehicle | 7824/1567/512 | 1444/308/17 | yalnız hedef-pozitif | 4×128 |

Bütün deneylerde train/validation/test kaynak sahne kesişimi sıfırdır. Ancak detector test kümeleri hedef-negatif görüntü içermez; AP sonuçları resmi benchmark sonucu değil, seçilmiş pozitif test kümesindeki detector kontrolüdür.

## Deneyler Arası Bağımlılık

| experiment_a | experiment_b | exact_rgb_test_image_overlap | source_scene_id_overlap |
| --- | --- | --- | --- |
| isaid_plane | isaid_small_vehicle | 109 | 6 |
| isaid_plane | samrs_plane | 19 | 15 |
| isaid_plane | samrs_small_vehicle | 2 | 2 |
| isaid_small_vehicle | samrs_plane | 10 | 6 |
| isaid_small_vehicle | samrs_small_vehicle | 6 | 5 |
| samrs_plane | samrs_small_vehicle | 71 | 3 |

Dört deney tek bir ortalamada birleştirilmemeli ve bağımsız dört replikasyon gibi sunulmamalıdır. iSAID ile SAMRS farklı anotasyon ürünleri olsa da DOTA kökenini ve bazı görüntüleri paylaşır.

## Exploratory Human–SAMRS Anlaşması

| target | exact_rgb_image_count | human_instance_count | samrs_instance_count | bbox_matched_instance_count_at_iou_50 | human_unmatched_instance_count | samrs_unmatched_instance_count | matched_mask_mean_iou | matched_mask_success_at_iou_50 | matched_mask_success_at_iou_75 | matched_mask_success_at_iou_90 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plane | 19 | 118 | 88 | 48 | 70 | 40 | 0.629 | 0.812 | 0.167 | 0.000 | Exploratory exact-image subset only; post-hoc and not a representative cross-dataset benchmark. Unmatched instances expose annotation coverage differences that matched-mask IoU alone omits. |
| small_vehicle | 6 | 108 | 101 | 91 | 17 | 10 | 0.708 | 0.912 | 0.462 | 0.022 | Exploratory exact-image subset only; post-hoc and not a representative cross-dataset benchmark. Unmatched instances expose annotation coverage differences that matched-mask IoU alone omits. |

Bu analiz yalnız iki testte piksel olarak birebir aynı çıkan post-hoc görüntü alt kümesidir. Temsili benchmark değildir. Hem eşleşmiş maskelerin IoU'su hem de eşleşmeyen instance sayıları birlikte okunmalıdır.

## Detector Metriklerini Bağımsız Yeniden Hesaplama

| experiment_id | images | ground_truth_instances | detections_for_ap | maximum_absolute_metric_difference | status |
| --- | --- | --- | --- | --- | --- |
| isaid_plane | 512 | 5447 | 12840 | 0.000 | exact_match |
| isaid_small_vehicle | 512 | 12051 | 92270 | 0.000 | exact_match |
| samrs_plane | 512 | 3713 | 12790 | 0.000 | exact_match |
| samrs_small_vehicle | 512 | 7659 | 34599 | 0.000 | exact_match |

Dondurulmuş COCO detection dosyalarından AP50/AP75/AP90/AP50-95 ile sabit validation eşiğindeki precision/recall değerleri proje evaluator yardımcısı kullanılmadan yeniden hesaplanmıştır. Bütün değerler kaydedilmiş JSON ile tam eşleşmiştir.

## Tarihsel Metadata Errata

6 işlevsiz COCO açıklama/supercategory alanı tarihsel girişlerde hatalı adlandırılmıştır. Bu alanlar model girdisi veya metrik değildir. Eski run-manifest hash'lerini bozmamak için deney girdileri değiştirilmemiş, yeniden üretim kodu düzeltilmiş ve hata burada açıkça kaydedilmiştir.
