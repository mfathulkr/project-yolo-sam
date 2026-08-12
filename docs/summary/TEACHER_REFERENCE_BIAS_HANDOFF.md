# Teacher Reference Bias - Güncel Devir Özeti

**Tarih:** 12 Ağustos 2026
**Kanonik seed:** 42
**Durum:** Dört deney, 16 tam metrik rapor, 4 deney çapraz analizi, ana analiz,
bildiri varlıkları, hamdan yeniden hazırlama zinciri ve strict otomatik QA tek
çalışma altında tamamlandı.

## Tek Kanonik Konum

```text
studies/teacher_reference_bias_paper/
```

Eski Plane, Small Vehicle ve multi-teacher kökleri aktif çalışma değildir.
İçerikleri kayıpsız olarak kanonik çalışmanın
`archives/pre_unification/legacy_roots/` dizininde korunur.

## Bilimsel Soru

Bir SAM modelinin ürettiği pseudo maskeler bağımsız test ground truth'u gibi
kullanılırsa aynı model yapay olarak daha başarılı görünür ve model sıralaması
değişir mi?

Bu klasik train/test görüntü sızıntısı değildir. Sorun, değerlendirme
referansının değerlendirilen modelden bağımsız olmaması, yani ölçüm cetvelinin
aday modele bağlı olmasıdır. Pseudo maskelerin eğitim veya ön etiketleme için
yararsız olduğu iddia edilmez.

Parikh, Das ve Feragen'in *Biased Ruler* çalışması aynı genel problemi tıbbi
segmentasyonda göstermiştir. Bizim güvenli katkı iddiamız, uzaktan algılama
instance segmentasyonunda üç SAM öğretmeni ile üç SAM adayını iki hedef sınıf,
iki bbox kaynağı, insan kontrolü ve dört sahne tabakası altında çaprazlayan
model-referans matrisidir.

## Dört Deney

| Deney | Görüntü | Instance | Kaynak sahne | Referanslar |
|---|---:|---:|---:|---|
| iSAID Plane | 512 | 5.447 | 44 | Human, SAM1, SAM2, SAM3 pseudo |
| iSAID Small Vehicle | 512 | 12.051 | 31 | Human, SAM1, SAM2, SAM3 pseudo |
| SAMRS Plane | 512 | 3.713 | 24 | Published, reproduced SAM1, SAM2, SAM3 pseudo |
| SAMRS Small Vehicle | 512 | 7.659 | 17 | Published, reproduced SAM1, SAM2, SAM3 pseudo |

Her deney `No Overlap/Overlap × Low/High Mask Area` biçiminde dört ayrık
tabakaya sahiptir; her tabakada 128 görüntü vardır. Seçilen görüntüdeki tek bir
nesne değil bütün hedef instance'lar değerlendirilir.

## Sabit Protokol

- Adaylar: frozen SAM1 ViT-H, SAM2.1 Hiera-Large ve yerel frozen SAM3.
- İstemler: dataset-native GT bbox ve seed 42 YOLO bbox.
- Pseudo referanslar: aynı modelin GT-bbox prediction RLE'sinin dondurulmuş
  kopyasıdır; yeniden inference yapılmaz.
- Ana metrik: instance-macro IoU; her nesne eşit ağırlıktadır.
- İstatistik: aynı instance üzerindeki eşlenmiş fark ve kaynak-sahne kümeli
  10.000 bootstrap ile %95 güven aralığı.
- Bilinen pozitif nesnede boş pseudo referans, aday da boş olsa bile sıfır
  puanlanır.
- GT-bbox diagonalı kimlik/coverage kontrolüdür; bağımsız başarı sonucu
  değildir.
- YOLO bbox AP/precision/recall gerçek detection metrikleridir; maske eşik
  oranları mAP olarak adlandırılmaz.

SAM3 için SAM1/SAM2 bbox deneyinin karşılığı olan PVS arayüzü kullanılır:
`Sam3TrackerProcessor + Sam3TrackerModel`, `multimask_output=False` ve
`mask_threshold=0.0`. PCS kavram arama arayüzü bbox-instance deneyi değildir
ve aktif sonuçlarda kullanılmaz.

## Ana Sonuç

iSAID YOLO-bbox koşulunda kendi pseudo referansına geçişin eşlenmiş IoU
artışları:

| Hedef | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| Plane | +0,276 | +0,279 | +0,224 |
| Small Vehicle | +0,176 | +0,163 | +0,142 |

Altı %95 güven aralığının tamamı sıfırın üzerindedir. Bu, özdeş GT-bbox
diagonalından daha güçlü kontroldür; tahmin YOLO bbox ile değişmiş olsa da
öğretmen kendi referansıyla avantaj kazanır.

SAMRS yayımlanmış referansı ile yeniden üretilen SAM1 referansının ortalama
instance IoU'su Plane'de `0,990633`, Small Vehicle'da `0,998338`dir. SAMRS
yayımlanmış maskeleri SAM1 kökenlidir; bu deney insan doğruluğu kanıtı değil,
referans kökeni ve teacher affinity desteğidir.

## Çıktılar

- Ana çapraz analiz:
  `studies/teacher_reference_bias_paper/analysis/main_cross_analysis_colored.pdf`
- Deney raporları:
  `experiments/<id>/reports/full_metrics/<reference>/`
- Deney içi çapraz analiz:
  `experiments/<id>/reports/cross_analysis/`
- Bildiri planı:
  `paper_writing/PAPER_STRUCTURE.md`
- Overleaf:
  `paper_writing/overleaf/main.tex`
- Figür ve tablolar:
  `paper_writing/assets/`
- Literatür:
  `literature_review/LITERATURE_REVIEW.md`
- Arama denetimi:
  `literature_review/SEARCH_AUDIT.md`
- Teknik tekrar üretim:
  `docs/REPRODUCIBILITY.md`
- Bilimsel kararlar:
  `docs/SCIENTIFIC_PROTOCOL.md`
- Otomatik kalite raporu:
  `docs/QA_REPORT.md`

## Yeniden Üretim

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py --help
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py prepare-master --experiment isaid_plane
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py prepare-matched --experiment isaid_plane
.venv/bin/python studies/teacher_reference_bias_paper/scripts/build_references.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/evaluate_reference_cubes.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/compile_experiment_analyses.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_experiment_figures.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_full_metric_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_cross_analysis_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_paper_assets.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
.venv/bin/pytest -q studies/teacher_reference_bias_paper/tests
```

Model checkpointleri, thresholdlar, bbox eşleme, boş maske politikası, PVS
ayrıntısı ve 8 GB VRAM için gerçek CLI kullanımı `docs/REPRODUCIBILITY.md`
içindedir. Dört `master_config.yaml` ham veri havuzunu; dört `config.yaml`
bildirideki 4×128 matched test kümesini tanımlar. Toplam 36 run manifesti
repository-relative yollar ve güncel dosya hashleriyle doğrulanır; taşıma
zinciri `docs/RUN_MANIFEST_MIGRATION_AUDIT.json` içinde kayıtlıdır.
