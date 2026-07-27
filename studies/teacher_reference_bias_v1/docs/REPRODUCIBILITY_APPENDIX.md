# Reproducibility Appendix

## Durum

Bu ek, `teacher_reference_bias_v1` çalışmasının final QA kapısından sonra
otomatik üretilmiştir.

- Üretim zamanı: `2026-07-26T19:41:14.743555+00:00`
- Frozen protokol: `studies/teacher_reference_bias_v1/configs/protocol.yaml`
- Protokol kimliği: `teacher_reference_bias_v1`
- Görüntü boyutu: `1024×1024`
- Detector seed'leri: `42, 123, 2026`
- Bootstrap: `10000`
- Tam hash envanteri: `studies/teacher_reference_bias_v1/results/reproducibility_manifest.json`

## Veri Kaynakları

| Veri | Sürüm | Referans | Sınıf | Arşiv SHA-256 | Prepared tree SHA-256 | Detector train tree SHA-256 |
| --- | --- | --- | --- | --- | --- | --- |
| isaid_plane | official_release | human | plane | configte yok | d95e77f33e86b846ebba8bef21bc871998ed039409b430793493f4852b362d96 | 2f5c95ab39c88176c9e8f25dc3dd06ad7ecbcbafb9e776adfb32966c72059114 |
| samrs_sota_plane | SAMRS-SOTA-DOTA2.0-2023 | pseudo_sam1 | plane | bbae2fb7f81b09dae3146dde2df406db1641716a8d7b3204dbafcabf8f00c706 | 0126325b065abb4eb77c8ea31c7b1e2eed8355ee4ae9d2c030737d5d3b55f5cc | 8f3c5ca65adfa88af02d2518e43e52dc22c1b432b3e7b824e0b44f284a85581d |

SAMRS arşiv kimliği numeric class ID, RBox ve RHBox geometri düzeyinde resmi
detection anotasyonlarıyla exhaustive olarak doğrulanmıştır. Pickle içindeki
metin kategori alanı authoritative kaynak değildir.

## Veri ve Manifest Migration Kayıtları

- iSAID insan maskeleri resmi polygonlardan OpenCV rastera ve kayıpsız
  compressed COCO RLE'ye geçirilmiştir. Train/validation/test için migration
  sonrası boş maske ve decoded-area uyuşmazlığı sıfırdır.
- Migration audit durumu: `pass`.
- Detector eğitim girdisi yalnız train/validation görüntü ve YOLO label
  ağacıdır; test split'i ile segmentation maskeleri bu kapsama girmez.
- Start/finish fingerprint şemasından önce başlayan
  `5` detector manifesti, byte düzeyindeki özgün
  manifest arşivi korunarak açık provenance repair auditinden geçirilmiştir.
- Başlangıç fingerprint'i zaten bulunan ve değişmeden bırakılan detector
  manifesti sayısı: `1`.
- Finalizer run manifestlerini değiştirmez; input drift veya kopuk repair
  zinciri final hatasıdır.

## Model Kimlikleri

| Model | Model kimliği | Revision | Checkpoint SHA-256 | Processor/config tree SHA-256 |
| --- | --- | --- | --- | --- |
| sam1 | facebook/sam-vit-huge | 87aecf0df4ce6b30cd7de76e87673c49644bdf67 | edfb0462392541fca9af44ff039bfb32dbd0c939997f3abb77a26e23af7afd7c | 720b6ec288e64a77c79f91f5b5868cceeae25b4d585d074c75ffff229456e112 |
| sam2 | facebook/sam2.1-hiera-large | 665f8e2ad61cf5f53d65644ff27c8ee525124610 | dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230 | c8d0dec638bbc5ae610e09d313b3c9444de780f241fc8c169f42ad235a55be56 |
| sam3 | sam3-local | local_checkpoint | 6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a | c95a22863afc3d4714cd653047fdc47dfe73048341f62c41e43058e8124d32b1 |

## Split Özeti

| Veri | Split | Görüntü | Instance | Kaynak sahne |
| --- | --- | --- | --- | --- |
| isaid_plane | train | 1685 | 10019 | 362 |
| isaid_plane | validation | 336 | 1961 | 78 |
| isaid_plane | test | 128 | 1045 | 45 |
| samrs_sota_plane | train | 2581 | 12734 | 505 |
| samrs_sota_plane | validation | 407 | 3556 | 111 |
| samrs_sota_plane | test | 128 | 1375 | 42 |

Split birimi tile değil kaynak sahnedir. Train, validation ve test kaynak
sahne kesişimi iki veri setinde de sıfırdır. Testte dört
`overlap × mask area` katmanının her birinde 32 görüntü vardır.

## Deney Matrisi

- GT-bbox: 2 veri seti × 3 SAM modeli = 6 koşul.
- YOLO-bbox: 2 veri seti × 3 detector seed × 3 SAM modeli = 18 koşul.
- iSAID değerlendirmesi: aynı tahmin üzerinde insan + SAM1 pseudo referans.
- SAMRS değerlendirmesi: resmi SAM1 pseudo referansı.
    - Ortak görüntü denetimi: 126 görüntü, 1.033 tile-instance görünümü,
      770 benzersiz insan-anotasyonlu uçak ve 35 kaynak sahne.

## Detector Özeti

| dataset_id | seed_count | fixed_confidence_threshold_mean | fixed_confidence_threshold_std | bbox_AP50_mean | bbox_AP50_std | bbox_AP75_mean | bbox_AP75_std | bbox_AP90_mean | bbox_AP90_std | bbox_AP50_95_mean | bbox_AP50_95_std | precision_at_bbox_iou50_mean | precision_at_bbox_iou50_std | recall_at_bbox_iou50_mean | recall_at_bbox_iou50_std | precision_at_bbox_iou75_mean | precision_at_bbox_iou75_std | recall_at_bbox_iou75_mean | recall_at_bbox_iou75_std | precision_at_bbox_iou90_mean | precision_at_bbox_iou90_std | recall_at_bbox_iou90_mean | recall_at_bbox_iou90_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isaid_plane | 3 | 0.3053 | 0.0331 | 0.9361 | 0.0039 | 0.8625 | 0.0054 | 0.6225 | 0.0078 | 0.7954 | 0.0011 | 0.9390 | 0.0055 | 0.9030 | 0.0064 | 0.8889 | 0.0042 | 0.8549 | 0.0062 | 0.6899 | 0.0062 | 0.6635 | 0.0020 |
| samrs_sota_plane | 3 | 0.6997 | 0.1075 | 0.9526 | 0.0054 | 0.8710 | 0.0090 | 0.2791 | 0.0277 | 0.7245 | 0.0057 | 0.9401 | 0.0174 | 0.8962 | 0.0156 | 0.8922 | 0.0260 | 0.8504 | 0.0124 | 0.4425 | 0.0202 | 0.4218 | 0.0177 |

## Detector Eğitim Sağlığı

Ara validation loss kayıtlarındaki geçici non-finite hücreler ayrıca
gösterilir; paylaşılabilir sonuç için son precision, recall ve AP değerlerinin
tamamı sonlu ve `[0,1]` aralığında olmalıdır.

| dataset_id | seed | results_file | epochs_completed | final_epoch | nonfinite_cells | nonfinite_validation_loss_cells | final_core_metrics_finite | final_precision | final_recall | final_ap50 | final_ap50_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isaid_plane | 123 | /home/ssyzai/projects/yolo-sam/project-yolo-sam/studies/teacher_reference_bias_v1/results/detectors/isaid_plane/seed_123/train/results.csv | 100 | 100 | 3 | 3 | True | 0.9466 | 0.9214 | 0.9536 | 0.8091 |
| isaid_plane | 2026 | /home/ssyzai/projects/yolo-sam/project-yolo-sam/studies/teacher_reference_bias_v1/results/detectors/isaid_plane/seed_2026/train/results.csv | 100 | 100 | 4 | 4 | True | 0.9662 | 0.9184 | 0.9558 | 0.8091 |
| isaid_plane | 42 | /home/ssyzai/projects/yolo-sam/project-yolo-sam/studies/teacher_reference_bias_v1/results/detectors/isaid_plane/seed_42/train/results.csv | 100 | 100 | 0 | 0 | True | 0.9678 | 0.9164 | 0.9482 | 0.8088 |
| samrs_sota_plane | 123 | /home/ssyzai/projects/yolo-sam/project-yolo-sam/studies/teacher_reference_bias_v1/results/detectors/samrs_sota_plane/seed_123/train/results.csv | 100 | 100 | 0 | 0 | True | 0.9689 | 0.9292 | 0.9693 | 0.7629 |
| samrs_sota_plane | 2026 | /home/ssyzai/projects/yolo-sam/project-yolo-sam/studies/teacher_reference_bias_v1/results/detectors/samrs_sota_plane/seed_2026/train/results.csv | 100 | 100 | 0 | 0 | True | 0.9615 | 0.9457 | 0.9670 | 0.7621 |
| samrs_sota_plane | 42 | /home/ssyzai/projects/yolo-sam/project-yolo-sam/studies/teacher_reference_bias_v1/results/detectors/samrs_sota_plane/seed_42/train/results.csv | 100 | 100 | 0 | 0 | True | 0.9629 | 0.9331 | 0.9674 | 0.7604 |

## YOLO-bbox Segmentation Özeti

| dataset_id | model | reference_type | seed_count | mean_iou_seed_mean | mean_iou_seed_std | mean_dice_seed_mean | mean_dice_seed_std | mean_precision_seed_mean | mean_precision_seed_std | mean_recall_seed_mean | mean_recall_seed_std | mean_boundary_iou_seed_mean | mean_boundary_iou_seed_std | success_at_iou_50_seed_mean | success_at_iou_50_seed_std | success_at_iou_75_seed_mean | success_at_iou_75_seed_std | success_at_iou_90_seed_mean | success_at_iou_90_seed_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| isaid_plane | sam1 | human | 3 | 0.6071 | 0.0033 | 0.7206 | 0.0042 | 0.6330 | 0.0031 | 0.8533 | 0.0068 | 0.6033 | 0.0032 | 0.8488 | 0.0058 | 0.1780 | 0.0010 | 0.0010 | 0.0000 |
| isaid_plane | sam1 | pseudo_sam1 | 3 | 0.8812 | 0.0070 | 0.8900 | 0.0071 | 0.8892 | 0.0065 | 0.8936 | 0.0075 | 0.8805 | 0.0069 | 0.8982 | 0.0091 | 0.8842 | 0.0078 | 0.8555 | 0.0058 |
| isaid_plane | sam2 | human | 3 | 0.5960 | 0.0021 | 0.7108 | 0.0031 | 0.6155 | 0.0023 | 0.8655 | 0.0057 | 0.5925 | 0.0020 | 0.8335 | 0.0029 | 0.1761 | 0.0035 | 0.0064 | 0.0022 |
| isaid_plane | sam2 | pseudo_sam1 | 3 | 0.7604 | 0.0044 | 0.8205 | 0.0047 | 0.8069 | 0.0045 | 0.8483 | 0.0049 | 0.7574 | 0.0047 | 0.8791 | 0.0047 | 0.7853 | 0.0077 | 0.3018 | 0.0080 |
| isaid_plane | sam3 | human | 3 | 0.6262 | 0.0030 | 0.7278 | 0.0042 | 0.6525 | 0.0027 | 0.8379 | 0.0071 | 0.6238 | 0.0028 | 0.8443 | 0.0055 | 0.3352 | 0.0045 | 0.0099 | 0.0020 |
| isaid_plane | sam3 | pseudo_sam1 | 3 | 0.7258 | 0.0058 | 0.7924 | 0.0058 | 0.8117 | 0.0051 | 0.7836 | 0.0068 | 0.7225 | 0.0060 | 0.8724 | 0.0054 | 0.7352 | 0.0100 | 0.1419 | 0.0074 |
| samrs_sota_plane | sam1 | pseudo_sam1 | 3 | 0.8694 | 0.0143 | 0.8806 | 0.0148 | 0.8781 | 0.0143 | 0.8855 | 0.0156 | 0.8694 | 0.0143 | 0.8919 | 0.0154 | 0.8766 | 0.0146 | 0.8378 | 0.0098 |
| samrs_sota_plane | sam2 | pseudo_sam1 | 3 | 0.7073 | 0.0103 | 0.7847 | 0.0119 | 0.7256 | 0.0106 | 0.8705 | 0.0153 | 0.7072 | 0.0103 | 0.8599 | 0.0111 | 0.6592 | 0.0069 | 0.1013 | 0.0022 |
| samrs_sota_plane | sam3 | pseudo_sam1 | 3 | 0.5915 | 0.0141 | 0.6892 | 0.0158 | 0.6171 | 0.0146 | 0.8446 | 0.0170 | 0.5896 | 0.0142 | 0.6892 | 0.0226 | 0.4325 | 0.0123 | 0.0359 | 0.0017 |

## Prediction Durum Denetimi

| Bbox kaynağı | Toplam | Başarılı | Boş maske | Eksik bbox | Inference hatası |
| --- | --- | --- | --- | --- | --- |
| gt_bbox | 7260 | 7182 | 78 | 0 | 0 |
| yolo_bbox | 21780 | 19485 | 99 | 2196 | 0 |

## Yeniden Üretim Sırası

```bash
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py preflight \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/isaid_plane.yaml \
  --dataset studies/teacher_reference_bias_v1/configs/datasets/samrs_sota_plane.yaml

.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py analyze
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py figures
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py paper
.venv/bin/python studies/teacher_reference_bias_v1/scripts/study.py finalize
```

Model, veri seti ve seed bazındaki ayrıntılı stage komutları `README.md`
içindedir. Her final run manifesti resolved config hash'i ile giriş/çıkış dosya
hash'lerini içerir.

## Değerlendirme Notu

YOLO-bbox instance tablosunda eşleşmeyen GT boş maskeyle sıfır skor alır.
Eşleşmeyen detector tahmini detector AP hesabında false positive ve
image-level union maskesinde tahmin olarak korunur. Instance mask tablosu COCO
mask AP değildir.
