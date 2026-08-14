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

Eski Plane, Small Vehicle, multi-teacher ve v1 kopyaları kanonik çalışma
doğrulandıktan sonra kaldırıldı. Aktif bilimsel dosyaların tek sahibi aşağıdaki
paper study dizinidir; tarihsel kopyalara yalnız Git geçmişinden ulaşılır.

## Bilimsel Soru

Bir SAM checkpoint'inin ürettiği pseudo maskeler bağımsız test ground truth'u
gibi kullanılırsa aynı dondurulmuş checkpoint daha başarılı görünür ve model
sıralaması değişir mi?

Bu klasik train/test görüntü sızıntısı değildir. Sorun, değerlendirme
referansının değerlendirilen modelden bağımsız olmaması, yani ölçüm cetvelinin
aday modele bağlı olmasıdır. Pseudo maskelerin eğitim veya ön etiketleme için
yararsız olduğu iddia edilmez.

Parikh, Das ve Feragen'in *Biased Ruler* çalışması aynı genel problemi tıbbi
segmentasyonda göstermiştir. Bizim güvenli katkı iddiamız, uzaktan algılama
instance segmentasyonunda üç sabit SAM üretici checkpoint'i ile üç sabit SAM
aday checkpoint'ini iki hedef sınıf, iki bbox kaynağı, insan kontrolü ve dört
sahne tabakası altında çaprazlayan model-referans matrisidir. Farklı checkpoint,
eğitim seed'i veya model ailesi düzeyinde genelleme bu deneyde sınanmamıştır.

## Dört Deney

| Deney | Görüntü | Nesne | Kaynak sahne | Referanslar |
|---|---:|---:|---:|---|
| iSAID Plane | 512 | 5.447 | 44 | Human, SAM1, SAM2, SAM3 pseudo |
| iSAID Small Vehicle | 512 | 12.051 | 31 | Human, SAM1, SAM2, SAM3 pseudo |
| SAMRS Plane | 512 | 3.713 | 24 | Published, reproduced SAM1, SAM2, SAM3 pseudo |
| SAMRS Small Vehicle | 512 | 7.659 | 17 | Published, reproduced SAM1, SAM2, SAM3 pseudo |

Her deney `No Overlap/Overlap × Low/High Mask Area` biçiminde dört ayrık
tabakaya sahiptir; her tabakada 128 görüntü vardır. Seçilen görüntüdeki tek bir
nesne değil bütün hedef nesneler değerlendirilir.

## Sabit Protokol

- Adaylar: frozen SAM1 ViT-H, SAM2.1 Hiera-Large ve yerel frozen SAM3.
- İstemler: dataset-native GT bbox ve seed 42 YOLO bbox.
- Pseudo referanslar: aynı modelin GT-bbox prediction RLE'sinin dondurulmuş
  kopyasıdır; yeniden inference yapılmaz.
- Ana metrik: nesne-ortalama IoU; her nesne eşit ağırlıktadır.
- İstatistik: aynı nesne üzerindeki eşlenmiş fark ve kaynak-sahne kümeli
  10.000 bootstrap ile %95 güven aralığı.
- Ana karşılaştırma, modelin kendi etiketindeki IoU'sundan diğer iki SAM
  etiketindeki ortalama IoU'yu çıkarır. İlk sonuçlar görüldükten sonra
  geliştirilmiştir; önceden kaydedilmiş doğrulayıcı test değildir ve çoklu
  karşılaştırma düzeltmesi uygulanmamıştır.
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

iSAID YOLO-bbox koşulunda aynı modelin kendi ürettiği etiketteki IoU'su eksi
diğer iki SAM etiketindeki ortalama IoU, yani **ek IoU**:

| Hedef | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| Plane | +0,128 | +0,124 | +0,141 |
| Small Vehicle | +0,098 | +0,074 | +0,075 |

Altı kaynak-sahne kümeli %95 güven aralığının tamamı sıfırın üzerindedir.
Ayrıntılı güven aralıkları ve ikincil istatistiksel kontrol analiz CSV'lerinde
saklanır. Karşılaştırma sonuçlar görüldükten sonra geliştirildiği için
doğrulayıcı nedensel kanıt gibi sunulmaz.

Ham kendi-pseudo eksi insan-referansı farkları Plane için
`+0,276/+0,279/+0,224`, Small Vehicle için `+0,176/+0,163/+0,142`dir. Bunlar
betimleyicidir; tek başına producer affinity kanıtı değildir.

SAMRS yayımlanmış referansı ile yeniden üretilen SAM1 referansının ortalama
nesne-ortalama IoU'su Plane'de `0,991`, Small Vehicle'da `0,998`dir. SAMRS
yayımlanmış maskeleri SAM1-türevli üretim hattından gelir; bu deney insan
doğruluğu veya orijinal üretim kodunun birebir yeniden çalıştırılması kanıtı
değil, SAM1-benzeri referans yakınlığı desteğidir.

Pseudo öğretmenler insan/yayımlanmış anotasyon kutularından gelen GT bbox ile,
YOLO aday maskeleri tahmin kutularıyla çalışır. Bu nedenle ölçülen fark yalnız
maske sınırını izole etmez; checkpoint kimliği, GT/YOLO kutu farkı, prompt
hassasiyeti ve maske biçiminin ortak etkileşimidir. Düzenek tam otomatik
pseudo-etiketleme hattı değildir.

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
- Derin split/dependency/detector denetimi:
  `docs/DEEP_SCIENTIFIC_AUDIT.md`

## Yeniden Üretim

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py --help
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py prepare-master --experiment isaid_plane
.venv/bin/python studies/teacher_reference_bias_paper/scripts/study.py prepare-matched --experiment isaid_plane
.venv/bin/python studies/teacher_reference_bias_paper/scripts/build_references.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/evaluate_reference_cubes.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/compile_experiment_analyses.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/deep_scientific_audit.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/audit_segmenter_provenance.py
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
