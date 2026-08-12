# Repository Architecture

## Temel Kural

Bir dosya yalnız bir araştırma çalışmasında kullanılıyorsa o çalışmanın
altında; birden fazla çalışmada aynı sözleşmeyle kullanılıyorsa kökteki ortak
katmanda bulunur. Config, hazırlanmış veri, prediction, analiz ve raporların
hangi bilimsel sonuca ait olduğu dizinden anlaşılmalıdır.

## Kök Dizinler

| Dizin | Sorumluluk |
|---|---|
| `src/yolo_sam/` | Veri setinden ve araştırma sorusundan bağımsız YOLO/SAM kodu |
| `tools/` | Birden fazla çalışmanın kullandığı bakım ve model araçları |
| `datasets/` | Paylaşılan, değiştirilemez ham veri kaynakları |
| `models/` | Ortak başlangıç checkpointleri ve yerel model cache'leri |
| `external_models/` | Harici model kaynak kodları |
| `docs/` | Depo mimarisi, ortak durum ve worklog |
| `studies/` | Araştırma sorusu bazında bütün çalışma sahipliği |
| `tests/` | Ortak kütüphane davranışı testleri |

Kökte genel `artifacts/`, `results/`, `runs/`, `data/` veya
`presentation_*` klasörü tutulmaz.

## Kanonik Bildiri Çalışması

```text
studies/teacher_reference_bias_paper/
├── README.md
├── configs/                  # Dört deney için ortak protokol
├── experiments/
│   ├── isaid_plane/
│   ├── isaid_small_vehicle/
│   ├── samrs_plane/
│   └── samrs_small_vehicle/
├── analysis/                 # Dört deney arası ana analiz
├── paper_writing/
│   ├── assets/               # Beş figür ve altı tablo
│   └── overleaf/             # Elektr şablonu ve BibTeX
├── literature_review/
├── docs/                     # Ortak yöntem, QA ve handoff
├── scripts/                  # Orchestration ve üretim komutları
├── src/                      # Bu araştırma sorusuna özgü kod
└── tests/
```

Her deney şu sözleşmeyi kullanır:

```text
experiments/<experiment_id>/
├── README.md
├── config.yaml              # Eşlenmiş 512 görüntülük deney configi
├── master_config.yaml       # Tam aday havuzu hazırlama configi
├── data/                     # master/prepared veri ve manifestler
├── docs/                     # Deneye özgü yöntem ve tekrar üretim
├── results/
│   ├── analysis/             # canonical metric cube ve özet CSV'ler
│   ├── detector/
│   ├── figures/
│   ├── predictions/
│   └── references/
└── reports/
    ├── full_metrics/<reference>/
    └── cross_analysis/
```

Birleştirme öncesindeki teacher-bias kökleri ve kopya arşivler kaldırılmıştır.
Aktif script ve configler yalnız kanonik dört deney yolunu kullanır; tarihsel
dosyalara ihtiyaç duyulursa Git geçmişi kullanılır.

## Deney Sahipliği

- `data/master`: Deneye özgü tam aday havuzu.
- `data/prepared`: Dondurulmuş train/validation/test türevi.
- `results/detector`: Yalnız o hedef için eğitilmiş YOLO ve bbox çıktıları.
- `results/predictions`: Model, bbox kaynağı ve seed koşulu tahminleri.
- `results/references`: Değerlendirme referans maskeleri.
- `results/analysis`: Instance düzeyi metrik küpü ve türetilmiş istatistikler.
- `reports`: İnsan tarafından okunacak MD, DOCX, PDF ve rapor tabloları.

Ham iSAID ve SAMRS kaynakları birden fazla çalışmada kullanıldığı için
`datasets/` altındadır. Eğitilmiş detector `best.pt` dosyaları deneye özgüdür
ve Git LFS ile ilgili deney altında tutulur.

## Rapor Sözleşmesi

Tam metrik raporlar tarihsel SAMRS raporunun okunaklı sayfa biçimini korur:

- `Overall` ve dört overlap × mask-area tablosu ayrı yatay sayfalardadır.
- Maske tablolarında Avg IoU, Dice, Precision, Recall ve açık IoU eşik geçme
  oranları bulunur; bunlar mAP diye adlandırılmaz.
- Gerçek bbox Precision, Recall ve AP değerleri yalnız YOLO detector tablosunda
  yer alır.
- Nitel sayfalarda seçilen görüntüdeki bütün hedef instance'lar çizilir.
- Aynı model ve referansın GT-bbox diagonalı kimlik kontrolü olarak etiketlenir.

## Taşıma Güvencesi

Birleştirme dosyaları taşınmadan önce ve sonra boyut, inode ve SHA-256 ile
doğrulandı. Kanıt:

```text
studies/teacher_reference_bias_paper/docs/MIGRATION_MANIFEST.json
```

Önceki 26 Temmuz 2026 depo geçişinin tarihsel manifestleri
`docs/migration/` altında korunur.

## Yeni Geliştirme Kuralı

1. Yeni araştırma sorusu yeni bir `studies/<study_id>/` açar.
2. Yalnız hedef sınıf/veri seti değişen aynı bildiri protokolü, mevcut paper
   study altında yeni `experiments/<id>/` olur.
3. Kod ancak en az iki bağımsız çalışma tarafından aynı sözleşmeyle
   kullanılıyorsa `src/yolo_sam/` içine taşınır.
4. Tarihsel sonuç yolları aktif runtime girdisi olamaz.
5. Tamamlanmış deney README, manifest, rapor ve otomatik QA kontrolüne sahip
   olmalıdır.
6. Boş klasör ve `.gitkeep` tutulmaz; çıktı gerektiğinde klasör oluşturulur.
