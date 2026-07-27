# Repository Architecture

## Amaç

Bu depo birden fazla uzaktan algılama deneyini barındırır. Dizin yapısının
temel kuralı şudur:

> Bir dosya yalnız bir çalışmada kullanılıyorsa o çalışmanın altında; birden
> fazla çalışmada gerçekten aynı davranışla kullanılıyorsa kökteki ortak
> katmanda bulunur.

Bu ayrım yalnız görünür düzen için değildir. Config, hazırlanmış veri, çalışma
çıktısı, rapor ve yeniden üretim komutlarının hangi bilimsel sonuca ait
olduğunu açıkça belirler.

## Kök Dizin Sözleşmesi

Kökte yalnız aşağıdaki ortak bileşenler bulunabilir:

| Dizin | Sorumluluk |
|---|---|
| `src/yolo_sam/` | Veri setinden ve araştırma sorusundan bağımsız ortak YOLO/SAM kodu |
| `tools/` | Birden fazla çalışmanın çağırdığı ortak bakım ve çalıştırma araçları |
| `datasets/` | Birden fazla çalışma tarafından kullanılan değiştirilemez ham veri kaynakları |
| `models/` | Ortak model ağırlıkları ve yerel model cache'leri |
| `external_models/` | Vendored harici model kaynak kodları |
| `docs/` | Depo mimarisi, ortak durum ve çalışma günlüğü |
| `studies/` | Her deney çalışmasının bütün sahipliğini taşıyan dizin |
| `tests/` | Yalnız ortak kütüphane davranışını sınayan testler |

Kökte `artifacts/`, `results/`, `runs/`, `data/` veya `presentation_*`
bulunmaz. Bu adlar çalışma sahipliğini gizlediği için kullanımdan
kaldırılmıştır.

## Study Sözleşmesi

Tamamlanmış bir çalışma mümkün olduğunda şu yapıyı kullanır:

```text
studies/<study_id>/
├── README.md
├── configs/
├── data/
│   └── prepared/
├── docs/
├── reports/
├── results/
├── scripts/
├── src/
└── tests/
```

- `README.md`: Araştırma sorusu, durum, kapsam, ana sonuç ve yeniden üretim
  komutları.
- `configs/`: Yalnız o çalışmaya ait protokol ve veri seti configleri.
- `data/prepared/`: Çalışmanın ürettiği train/validation/test türevleri.
- `docs/`: Deney planı, literatür, walkthrough ve yeniden üretilebilirlik eki.
- `reports/`: Markdown, DOCX, PDF ve rapora ait görseller.
- `results/`: Prediction, evaluation, detector run, audit ve manifestler.
- `scripts/`: Yalnız o çalışma için anlamlı komutlar.
- `src/`: Araştırma sorusuna veya veri eşlemesine özel Python paketi.
- `tests/`: O çalışmanın protokol ve raporlama testleri.

Bir çalışmada kullanılmayan alt dizin boş tutulmaz.

Boş dizini Git'te tutmak için `.gitkeep` veya `.keep` kullanılmaz. Gerçek bir
dosya üretildiğinde dizin doğal olarak oluşur; kullanılmayan dizin silinir.

Bir çalışmanın `configs/`, `scripts/` ve `src/` dosyaları başka bir
`studies/<study_id>/results` yolunu çalışma zamanı girdisi olarak kullanamaz.
V2 veri hazırlayıcı, v1'in doğrulanmış tam tile havuzunu yalnız başlangıç
kaynağı olarak kullanır; kaynak içerik hash ile sabitlenir ve hiçbir detector,
tahmin veya metrik sonucu aktarılmaz. Geçmiş çalışmaların geçerlilik durumunu
kaydeden `teacher_reference_bias_v1/scripts/snapshot_study_state.py` deney
girdisi okumayan belgelenmiş denetim istisnasıdır.

## Çalışmalar

### `teacher_reference_bias_v2_512`

Tamamlanmış canonical ve bildiri niteliğindeki eşlenmiş deneydir. Her veri setinde dört
alt grupta 128'er, Overall'da 512 görüntü kullanır. Aynı iSAID plane
tahminlerini hem resmi insan maskeleri hem SAM1 üretimli pseudo maskeler
karşısında ölçer; SAMRS SOTA'yı ayrı bir SAM1-pseudo referans koşulu olarak
inceler.

Durum: `completed_canonical`

### `teacher_reference_bias_v1`

İlk 4×32 protokollü eşlenmiş deneydir. V2'nin tarihsel öncülü olarak
değiştirilmeden korunur ve güncel bildiri kanıtına otomatik karıştırılmaz.

Durum: `completed_superseded`

### `isaid_vehicle_study`

İlk tarihsel çalışmadır. iSAID `Small_Vehicle + Large_Vehicle` birleşik
maskeleri üzerinde çeşitli detection/segmentation pipeline'larını karşılaştırır.
Kendi araştırma sorusu için korunur; eşlenmiş teacher-bias deneyinin kanıtı
değildir.

Durum: `historical_context_only`

### `samrs_sota_plane_study`

SAMRS SOTA plane pseudo-maskeleri üzerinde yapılan ikinci tarihsel çalışmadır.
Pseudo-reference yanlılığı araştırma sorusunu doğurması bakımından değerlidir;
ancak eski protokol eşlenmemiş olduğu için bildiri kanıtı olarak kullanılmaz.

Durum: `invalid_for_paper_evidence`

### `semantic_drone_car_study`

Semantic Drone car deneyi için hazırlanmış config ve handoff notlarını tutar.
Tamamlanmış sonuç yoktur.

Durum: `planned`

### `landcover_building_study`

Eski landcover.ai building configini ve ilgili hazırlama kodunu tarihsel olarak
tutar. Tamamlanmış ve doğrulanmış çalışma olarak sunulmaz.

Durum: `legacy_incomplete`

## Veri Sahipliği

Ham iSAID ve SAMRS kaynakları birden fazla çalışmada kullanıldığı için
`datasets/` altındadır:

```text
datasets/
├── isaid/
│   ├── raw/
│   └── downloads/
└── samrs/
    └── raw/
```

Çalışma tarafından dönüştürülen tile, label, RLE, split ve metadata dosyaları
ilgili `studies/<study_id>/data/prepared/` altında bulunur. Ham veriye
çalışma çıktısı yazılmaz.

## Model Sahipliği

Birden fazla çalışmada kullanılan YOLO başlangıç ağırlıkları:

```text
models/yolo/yolo26n.pt
models/yolo/yolo26x.pt
```

SAM ve tarihsel çalışmalarda kullanılan diğer model cache/checkpointleri de
ortak `models/` altında kalır. Eğitilmiş detector ağırlıkları ise çalışmaya
özgüdür ve ilgili `results/detectors/` altında bulunur.

## Sonuç ve Rapor Sahipliği

Her sonuç, onu üreten çalışmanın `results/` dizinine yazılır. Bir rapor yalnız
aynı çalışmanın `results/` ve `data/prepared/` kaynaklarını okuyabilir.

Tarihsel raporlar silinmez. Bilimsel geçerlilik statüsü çalışma README'sinde
ve kökteki `docs/LEGACY_STATUS.md` dosyasında açıkça yazılır.

Teacher-reference-bias çalışmasının veri seti başına tam metrik belgeleri,
tarihsel SAMRS raporunun okunaklı sayfa yapısını kullanır:

- Her tablo ayrı yatay sayfadadır.
- İlk sütun tek bir `Pipeline` adıdır.
- `Images` örnek kapsamını gösterir.
- Maske tablolarında yalnız `Avg IoU`, `Avg Dice`, `Avg Precision`,
  `Avg Recall` ve açıkça adlandırılmış `IoU ≥ 0.50/0.75/0.90` geçme oranları
  bulunur.
- Maske eşik oranları mAP olarak adlandırılmaz.
- Gerçek BBox mAP, Precision ve Recall yalnız YOLO detector tablosunda yer
  alır.
- `n`, `Sahne`, `Tekrar`, `Boundary IoU` ve uydurma `mAP proxy` sütunları
  sunum tablolarında kullanılmaz.
- Nitel örneklerin dört overlap × mask-area grubu ayrı ve okunaklı sayfalarda
  gösterilir.

## Taşıma Güvencesi

2026-07-26 tarihli mimari geçişte 63 kaynak ağaç taşınmadan önce ve sonra
dosya bazında SHA-256 ile doğrulanmıştır:

```text
docs/migration/study_layout_20260726_pre.json
docs/migration/study_layout_20260726_post.json
```

Bu manifestler dosya sayısını, toplam byte miktarını ve içerik ağaç hash'ini
saklar. Final doğrulama 63 kaydın 56'sını `verified_current`, özgün taşıma
hash'i daha önce doğrulanıp sonradan bilinçli olarak yeniden üretilen 7
değişebilir rapor/manifest ağacını
`historically_verified_then_regenerated` olarak kaydetmiştir. Üst durum
`verified_with_regenerated_mutable_outputs` değeridir.
