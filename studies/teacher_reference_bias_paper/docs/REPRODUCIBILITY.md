# Reproducibility ve Operasyon Sözleşmesi

## Kanonik Ortam

- Repo kökü: `project-yolo-sam`
- Çalışma kökü: `studies/teacher_reference_bias_paper`
- Python: repo `.venv`
- Giriş boyutu: `1024 × 1024`
- Tek kanonik seed: `42`
- Detector: `YOLO26x`, 100 epok üst sınır, batch 12, patience 30
- NMS IoU: `0.70`, max detection: `500`
- Segmenterler frozen; iSAID/SAMRS üzerinde fine-tune edilmez.

Kanonik tam ayarlar `configs/protocol.yaml`, deney kaynakları `experiments/*/config.yaml`, ham veriden master havuz üretim ayarları ise `experiments/*/master_config.yaml` dosyalarındadır. `protocol.local_8gb.yaml` yalnız düşük VRAM çalışma profilidir; batching farkı nedeniyle runtime config hash'i kanonik koşuyla aynı değildir, fakat model checkpoint'i, giriş boyutu ve float32 çıkarım korunur.

## Model Kimlikleri

| Model | Model/revision | Checkpoint SHA-256 |
| --- | --- | --- |
| SAM1 | `facebook/sam-vit-huge@87aecf0...` | `edfb0462392541fca9af44ff039bfb32dbd0c939997f3abb77a26e23af7afd7c` |
| SAM2 | `facebook/sam2.1-hiera-large@665f8e2...` | `dc407dce21301fd94abb395c5099b4f2c455fdc8a8f261ac3d0ea6d4cd197230` |
| SAM3 | local `models/sam3_hf/model.safetensors` | `6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a` |

Revision adı tek başına yeterli değildir; checkpoint hash'i doğrulanmalıdır.

## SAM3 Püf Noktası: PVS ve PCS

`PVS`, Promptable Visual Segmentation'dır: bbox gibi uzamsal istemle belirli bir nesneyi segmentler. `PCS`, Promptable Concept Segmentation'dır: metin veya görsel örnekle bir kavramın bütün örneklerini arar.

Bu çalışma bbox instance segmentasyonu yaptığı için SAM3'te:

- `Sam3TrackerProcessor`;
- `Sam3TrackerModel`;
- PVS bbox yolu;
- `multimask_output=False`;
- `mask_threshold=0.0`;
- bbox başına tam bir çıktı

kullanılır. PCS visual-exemplar yolunu bbox segmentasyonu sanmak önceki hatalı koşunun temel nedeniydi; PCS çıktıları kanonik çalışmada kullanılmaz.

## Veri Hazırlama

1. Kaynak sahne kimliği korunur.
2. Train/validation/test bölünmesi kaynak sahne düzeyinde yapılır.
3. Tabaka seçimi temel veri seti anotasyonundan yapılır.
4. Her tabakadan 128 görüntü seçilir; test toplamı 512 olur.
5. COCO instance kimlikleri ve kaynak sahne kimlikleri manifestte sabitlenir.
6. `data.yaml` göreli yoldur; VM veya yerel makine mutlak yoluna bağlı değildir.

Ham iSAID/SAMRS verisi `datasets/` altında hazırlandıktan sonra örneğin iSAID Plane için iki aşamalı üretim:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py prepare-master --experiment isaid_plane
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py prepare-matched --experiment isaid_plane
```

İlk komut kaynak-sahne ayrık tam hedef havuzunu `data/master`, ikinci komut bildiride kullanılan dengeli `4×128=512` test kümesini `data/prepared` altında üretir. Aynı iki komut dört deney kimliği için geçerlidir. Var olan çıktının bilinçli olarak yeniden üretilmesi gerekiyorsa `--force` eklenir.

Hazır veriyi doğrulama:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py validate-data
```

Komut diğer üç `config.yaml` için de çalıştırılır.

8 GB VRAM profili komut adından önce seçilir:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py --profile local_8gb infer --experiment isaid_plane --model sam3 --bbox-source gt_bbox --device 0
```

Bu profil batch boyutunu küçültür; rapordaki kanonik artifact'lar `--profile canonical` varsayılanıyla üretilmiştir.

## Detector Eğitimi ve Değerlendirmesi

Her hedef/veri ailesi için ayrı YOLO detector vardır. Model seed 42 ile eğitilir. Test confidence eşiği test setinden seçilmez; validation setinde bbox IoU `0.50` kabul kuralıyla F1'i en yüksek yapan confidence dondurulur.

| Deney | Sabit confidence |
| --- | ---: |
| iSAID Plane | `0.28115004301071167` |
| iSAID Small Vehicle | `0.2740148901939392` |
| SAMRS Plane | `0.7607834339141846` |
| SAMRS Small Vehicle | `0.36182695627212524` |

Eğitim:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py train-detector --experiment isaid_plane --device 0
```

Tahmin ve test metriği:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py detect --experiment isaid_plane --device 0
```

## Segmenter Inference

GT bbox ve YOLO bbox tahminleri ayrı çalıştırılır. SAM1/SAM2 kutuları daha küçük batch'lere, SAM3 kutuları GPU belleğine göre parçalara bölünebilir; bbox sırası ve çıktı RLE'si korunmalıdır.

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py infer --experiment isaid_plane --model sam1 --bbox-source gt_bbox --device 0
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py infer --experiment isaid_plane --model sam1 --bbox-source yolo_bbox --device 0
```

`sam1`, `sam2`, `sam3` ve dört experiment config için yinelenir.

## Pseudo Referans Üretimi

Pseudo üretici yeniden model çalıştırmaz. İlgili modelin dondurulmuş GT-bbox `predictions.jsonl` RLE'sini referans dosyasına kopyalar ve kaynak hash/provenance ekler. Kontroller:

- her instance tam bir kez bulunur;
- kaynak model ve prompt `gt_bbox` ile eşleşir;
- pseudo RLE kaynak prediction RLE ile byte düzeyinde aynıdır;
- boş maskeler filtrelenmez;
- `status` ile gerçek RLE alanı uyuşur.

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/build_references.py
```

## Değerlendirme ve Raporlar

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/evaluate_reference_cubes.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/compile_experiment_analyses.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_experiment_figures.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_full_metric_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_cross_analysis_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_paper_assets.py
```

Figür manifestinde `qualitative_scope=all_target_instances_in_selected_images` olmalıdır. Nitel figürde seçilen görüntüdeki tek bir nesne değil bütün hedef instance'lar yer alır.

## Manifest ve Taşınabilirlik

Her ana artifact için SHA-256 manifesti vardır. `docs/MIGRATION_MANIFEST.json`, eski üç klasörden yeni yapıya taşınan büyük artifact'ların eski/yeni yollarını ve hash doğrulamasını saklar. `docs/RUN_MANIFEST_MIGRATION_AUDIT.json`, taşınan 36 çalışma manifestinin özgün manifest hash'ini ve yalnız yol/fingerprint onarımını kaydeder. Aktif manifest ve `train/args.yaml` yolları repository-relative'dir; eski mutlak yollar yalnız tarihsel taşıma denetiminde bulunur.

Kanonik detector ağırlığı her deneyde `results/detector/seed_42/train/weights/best.pt` yolundadır. Büyük ham/prepared veri Git'e eklenmez; içerik manifestleri ve yeniden hazırlama config'leri repoda tutulur.

## Son Doğrulama

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
.venv/bin/pytest -q studies/teacher_reference_bias_paper/tests
```

Doğrulayıcı dört veri kümesini, master→matched config zincirini, 36 strict çalışma manifestini, 16 full-metric rapordaki 80 mask tablosunu, 16 detector tablosunu, dört çapraz analiz raporunu, ana raporu, metric-cube cardinality'sini, coverage-aware identity kuralını, taşınabilir yolları ve PDF/DOCX varlığını denetler.
