# SAMRS SOTA Plane Çalışması

> **Durum: `invalid_for_paper_evidence` / legacy.** Bu klasör ilk, eşlenmemiş
> SAMRS deneyini korur. Source-scene-safe split, iki veri setinde aynı plane
> sınıfı, seed 42 ve çapraz referans kontrollerini kullanan otoritatif çalışma
> `teacher_reference_bias_paper`'dır. Güncel giriş noktaları
> için kök `README.md`, `docs/LEGACY_STATUS.md` ve
> `studies/teacher_reference_bias_paper/` kullanılmalıdır.

Bu klasör SAMRS/SOTA `plane` deneyi için ayrıldı. Eski iSAID vehicle çalışmasına karışmaması için veri, sonuç ve rapor path'leri `samrs_sota_plane` adıyla tutulur.

## Amaç

- iSAID vehicle deneyindeki YOLO + SAM mantığını, bbox'a daha az oturan `plane` nesnesi üzerinde tekrar etmek.
- Kanat, kuyruk ve gövde nedeniyle boxy olmayan nesnelerde bbox promptunun maske kalitesine etkisini ölçmek.
- Eski rapordaki kırılımı korumak: Overall, No Overlap / Low Mask Area, No Overlap / High Mask Area, Overlap / Low Mask Area, Overlap / High Mask Area.
- Önceki 32 örnek yerine her stratum için 128 örnek hedeflemek.
- SAM1'i de deney hattına ekleyerek SAM1 / SAM2 / SAM3 farkını görmek.

## Veri Seti Kararı

- Kaynak: SAMRS içindeki SOTA subset.
- SOTA, DOTA v2.0 kaynaklı remote sensing görüntülerinden üretilmiştir.
- Hedef sınıf: `plane`.
- SOTA/YOLO kategori eşlemesinde `plane` sıfır tabanlı class id `4` olarak geçer.
- RingMo-SAM'in ürettiği semantic class-map içinde `plane` sınıfı `5` olarak göründüğü için RingMo-SAM config'inde `class_ids: [5]` kullanılır. Bu değer SOTA/YOLO class id'siyle aynı kavram değildir.
- Önemli not: SAMRS maskeleri insan tarafından çizilmiş kesin GT değildir; SAM kullanılarak üretilmiş pseudo-mask etiketleridir. Bu yüzden raporda sonuçlar `SAMRS-SOTA pseudo-mask benchmark` olarak sunulmalıdır.

## Beklenen Ham Veri Yapısı

Tercih edilen ham yapı:

```text
data/samrs_raw/sota/
  trainval/
    images/
    rhbox_segs_init/ins/
    train.txt
    valid.txt
```

MTP/SAMRS COCO JSON formatı da desteklenir:

```text
data/samrs_raw/sota/
  trainval/
    images/
    sota_rbb_train_ins_segmentation.json
    sota_rbb_valid_ins_segmentation.json
```

Resmi SAMRS README, SOTA-RBB verisini OneDrive ve Baidu üzerinden veriyor. OneDrive tarayıcıda açılıyor ancak komut satırı indirme istekleri 403 döndürebiliyor; veri manuel indirildiğinde yukarıdaki yapıya yerleştirmek yeterli.

## Çalıştırma Sırası

Veriyi hazırla ve split audit'i al:

```bash
.venv/bin/python scripts/prepare_samrs_sota_plane_dataset.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/rebuild_samrs_sota_plane_eval_split.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/audit_samrs_sota_plane_dataset.py --config configs/samrs_sota_plane_yolo26x.yaml
```

YOLO detector'u yeniden eğit:

```bash
.venv/bin/python scripts/train_yolo.py --config configs/samrs_sota_plane_yolo26x.yaml
```

Segmentasyon hatlarını çalıştır:

```bash
.venv/bin/python scripts/run_sam3_text.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/run_yolo_sam3.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/run_gt_box_sam3.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/run_sam3_hybrid.py --config configs/samrs_sota_plane_yolo26x.yaml --box-source yolo

.venv/bin/python scripts/run_gt_box_sam1.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/run_yolo_sam1.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/run_gt_box_sam2.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/run_yolo_sam2.py --config configs/samrs_sota_plane_yolo26x.yaml

.venv/bin/python scripts/run_remotesam_text.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/run_ringmo_sam.py --config configs/samrs_sota_plane_yolo26x.yaml
```

YOLO bbox AP metriğini ve segmentasyon tablolarını üret:

```bash
.venv/bin/python scripts/evaluate_yolo_detector_coco.py \
  --config configs/samrs_sota_plane_yolo26x.yaml \
  --output-dir results/samrs_sota_plane_detector_metrics

.venv/bin/python scripts/evaluate_stratified_triplet.py --config configs/samrs_sota_plane_yolo26x.yaml

.venv/bin/python scripts/write_samrs_sota_plane_metric_document.py --config configs/samrs_sota_plane_yolo26x.yaml
.venv/bin/python scripts/export_samrs_sota_plane_metric_document_docx_pdf.py
.venv/bin/python scripts/validate_samrs_sota_plane_experiment_outputs.py --config configs/samrs_sota_plane_yolo26x.yaml
```

## Üretilen Ana Çıktılar

- Hazırlanmış veri: `data/samrs_sota_plane`
- YOLO eğitim çıktısı: `runs/yolo26x_samrs_sota_plane_s1024`
- Segmentasyon metrikleri: `results/samrs_sota_plane_metrics`
- Görsel örnekler: `results/samrs_sota_plane_visualizations`
- Rapor MD/DOCX/PDF: `studies/samrs_sota_plane_study/reports/`
- QA manifest: `studies/samrs_sota_plane_study/reports/QA_MANIFEST.md`
- Artifact manifesti: `studies/samrs_sota_plane_study/reports/ARTIFACT_MANIFEST.csv`

## Kontrol Listesi

- `AUDIT.md` içinde eval split gerçekten 5 tablo mantığını karşılıyor mu kontrol edilmeli.
- Her stratum için 128 görüntü yetmezse, raporda gerçek görüntü sayısı açıkça yazılmalı.
- YOLO detector tablosunda sadece bbox metrikleri olmalı; maske IoU ile karıştırılmamalı.
- Segmentasyon mAP proxy kolonları COCO AP gibi anlatılmamalı; bunlar görüntü seviyesinde IoU eşik geçme oranıdır.
- SAMRS pseudo-mask notu raporda mutlaka görünmeli.
- RingMo-SAM config'inde `class_ids: [5]` kullanılmalı; bu RingMo semantic çıktı sınıfıdır. SOTA/YOLO tarafındaki `plane` id'si `4` olarak kalır.
