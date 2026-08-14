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

Kanonik run manifestlerinde kaydedilen tam inference ortamı Python `3.12.3`, PyTorch `2.11.0`, Transformers `5.6.2`, Ultralytics `8.4.41`, NumPy `2.4.4`, pandas `3.0.2`, SciPy `1.17.1`, pycocotools `2.0.11`, Pillow `12.2.0` ve CUDA `13.0`'dır. `requirements.txt` yeniden kurulum için alt sınırlar verir; tam tarihsel sürümler run manifestlerinde otoritatiftir.

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

Kanonik split, resmi leaderboard split'i değildir. iSAID resmi train+validation anotasyonlarından; SAMRS yayıncı havuzundan kaynak sahne düzeyinde yeniden bölünür. Dört test kümesinin tamamı hedef-pozitiftir. Tabaka eşiği görüntü başına toplam hedef maske alanı oranıdır ve deney config'inde inference öncesi sabitlenmiştir.

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
| iSAID Plane | `0.281` |
| iSAID Small Vehicle | `0.274` |
| SAMRS Plane | `0.761` |
| SAMRS Small Vehicle | `0.362` |

Tablo okunabilirlik için üç ondalığa yuvarlanmıştır; çalıştırmada kullanılan tam değerler deney config dosyalarındadır.

Eğitim:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py train-detector --experiment isaid_plane --device 0
```

Tahmin ve test metriği:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py detect --experiment isaid_plane --device 0
```

Detector test AP'si 512 hedef-pozitif görüntü üzerindedir; resmi iSAID/SAMRS detector benchmark sonucu olarak kullanılmamalıdır. Mask ortalamasında eşleşmeyen detector yanlış pozitifleri yoktur; bunlar detector metriklerinde cezalandırılır. Dolayısıyla maske sonuçları tam uçtan uca COCO mask AP değildir.

`isaid_small_vehicle` ve `samrs_small_vehicle` detector eğitimleri tarihsel olarak resume edilmiştir. Resume başlangıç checkpoint'inin byte'ları korunmamış; başlangıç SHA-256/boyut kaydı, final `best.pt`, eğitim tablosu ve bütün değerlendirme giriş hash'leri korunmuştur. Final dondurulmuş ağırlıkla değerlendirme yeniden üretilebilir, fakat iki eğitimin ilk adımdan birebir trajectory yeniden üretimi garanti edilemez.

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

Pseudo üretim GT/published anotasyon bbox'ını lokalizasyon istemi olarak
kullanır. Bu nedenle düzenek tam otomatik bir pseudo-etiketleme hattı değildir
ve ölçülen fark yalnız maske sınırı stilini izole etmez. Sonuç; sabit üretici
checkpoint, referans maskesi, GT/YOLO bbox farkı ve prompt hassasiyetinin ortak
etkileşimi olarak yorumlanmalıdır.

## Değerlendirme ve Raporlar

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/evaluate_reference_cubes.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/compile_experiment_analyses.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_experiment_figures.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_full_metric_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_cross_analysis_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_paper_assets.py
```

Ana kendi etiketi ile diğer iki SAM etiketi arasındaki ek IoU ve ikincil
istatistiksel kontroller ilk sonuçlar görüldükten sonra geliştirilmiştir. Bunlar
önceden kaydedilmiş doğrulayıcı testler değil, çoklu karşılaştırma düzeltmesi
uygulanmamış destekleyici analizlerdir. Yorum yalnız kaydedilmiş SAM1/SAM2/SAM3
checkpoint'leri için geçerlidir; model ailesi, farklı seed veya farklı
checkpoint genellemesi yapılmaz.

Figür manifestinde `qualitative_scope=all_target_instances_in_selected_images`
olmalıdır. Nitel figürde seçilen görüntüdeki tek bir nesne değil bütün hedef
instance'lar yer alır. Dört gösterim görüntüsü model veya referans IoU'suna göre
seçilmez: her tabakada hazırlama metadata'sındaki `mask_area_ratio` medyanına en
yakın görüntü, farklı kaynak sahne koşuluyla deterministik seçilir. Aynı dört
görüntü deneydeki Human/yayımlanmış, SAM1, SAM2 ve SAM3 referans belgelerinde
değişmeden kullanılır. Manifest her panel için görüntü ID'sini, kaynak sahneyi,
hedef instance sayısını ve modele verilen bbox istemi sayısını kaydeder.

## Manifest ve Taşınabilirlik

Her ana artifact için SHA-256 manifesti vardır. `docs/MIGRATION_MANIFEST.json`, eski üç klasörden yeni yapıya taşınan büyük artifact'ların eski/yeni yollarını ve hash doğrulamasını saklar. `docs/RUN_MANIFEST_MIGRATION_AUDIT.json`, taşınan 36 çalışma manifestinin özgün manifest hash'ini ve yalnız yol/fingerprint onarımını kaydeder. Aktif manifest ve `train/args.yaml` yolları repository-relative'dir; eski mutlak yollar yalnız tarihsel taşıma denetiminde bulunur.

Taşınmış prediction klasörlerindeki `effective_config.input.json` ve `segmenter_provenance.input.json`, özgün çalıştırmanın tarihsel snapshot'ıdır. Bu snapshot'lardaki protocol hash'i taşıma sonrasında düzenlenen güncel `configs/protocol.yaml` hash'iyle aynı olmak zorunda değildir; exact run ayarını snapshot, güncel tekrar koşulunu kanonik protocol tanımlar. Eski run kimlikleri de tarihsel çalıştırma kimliğidir ve sonuçlar yeniden inference edilmeden değiştirilmez.

Kanonik detector ağırlığı her deneyde `results/detector/seed_42/train/weights/best.pt` yolundadır. Git+LFS ile analiz CSV'leri, tahminler, referanslar ve dondurulmuş detector ağırlıkları çekilerek metrik/rapor zinciri yeniden üretilebilir. Ham/prepared rasterlar ve SAM foundation checkpoint'leri Git'e girmez; inference için veri yeniden hazırlanmalı ve hash'i doğrulanan model checkpoint'leri ayrıca bulunmalıdır.

Tarihsel SAMRS Small Vehicle COCO `info.description` ve `supercategory` alanlarında işlevsiz adlandırma hatası vardır. Kutu, maske, kategori id, görüntü ve metrikleri etkilemez. Eski input hash'lerini bozmamak için artifact değiştirilmemiş; üretim kodu düzeltilmiş ve ayrıntı `DEEP_SCIENTIFIC_AUDIT.md` içinde kaydedilmiştir.

## Son Doğrulama

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/deep_scientific_audit.py
.venv/bin/pytest -q studies/teacher_reference_bias_paper/tests
```

Doğrulayıcı dört veri kümesini, master→matched config zincirini, 36 strict
çalışma manifestini, 16 full-metric rapordaki 80 mask tablosunu, 16 detector
tablosunu, dört çapraz analiz raporunu, ana raporu, metric-cube cardinality'sini,
coverage-aware identity kuralını, taşınabilir yolları ve PDF/DOCX varlığını
denetler.

Ek semantik denetim her model/bbox koşulundaki prediction satırlarını baştan
okur. Validation'da seçilen confidence eşiğini, YOLO confidence filtresini ve
greedy bire bir bbox eşlemesini bağımsız olarak yeniden kurar; her satırın
instance kimliğini, giriş bbox'ını, detector confidence/IoU'sunu, prompt
kaynağını, RLE boyutunu, boş maske durumunu, model/revision/checkpoint
provenance'ını ve manifest cardinality'sini doğrular. Pseudo referanslarda RLE,
kaynak GT-bbox prediction RLE'siyle doğrudan kimlik kontrolünden geçer. Son
denetimde 173.220 model×bbox prediction satırı ile bütün native/pseudo referans
zincirleri bu kontrollerden geçirilmiştir.

Taşıma sonrasında pseudo manifestlerinde saptanan eski GT-inference manifest
hash'leri `build_references.py` ile yenilenmiştir. Bu işlemde 12 pseudo JSONL
dosyasının SHA-256 değerleri değişmemiş; yalnız provenance manifestleri güncel
giriş hash'lerini gösterecek şekilde düzelmiştir.
