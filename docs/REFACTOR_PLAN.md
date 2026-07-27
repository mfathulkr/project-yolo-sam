# Repository Refactor Plan

## 1. Amaç

Depoyu rastgele büyüyen config/data/results/presentation klasörlerinden,
araştırma sorusu bazında sahipliği açık `studies/` mimarisine taşımak.
Refactor mevcut deney sonuçlarını silmeyecek, pahalı model koşularını yeniden
çalıştırmayı gerektirmeyecek ve taşınan her bilimsel artifact'i hash ile
doğrulayacaktır.

## 2. Temel İlkeler

1. **Study sahipliği:** Yalnız bir deneyde kullanılan her dosya o study
   altında bulunur.
2. **Gerçek paylaşım:** Kök `src/`, `tools/`, `datasets/` ve `models/` yalnız
   birden fazla çalışmanın aynı sözleşmeyle kullandığı bileşenleri tutar.
3. **Kayıpsız taşıma:** Dosya ağacı taşınmadan önce ve sonra dosya sayısı,
   byte miktarı ve SHA-256 tree hash ile doğrulanır.
4. **Bilimsel ayrım:** Tarihsel sonuçlar korunur; güncel canonical analize
   otomatik olarak karıştırılmaz.
5. **Tekrarlanabilir yol:** Config, prepared data, prediction, evaluation ve
   rapor yolu çalışma klasöründen çözümlenir.
6. **Doğrulama:** Unit/integration test, manifest doğrulama, rapor içerik
   kontrolü ve PDF görsel QA birlikte geçmeden refactor tamamlanmış sayılmaz.

## 3. Hedef Mimari

```text
.
├── datasets/
│   ├── isaid/
│   └── samrs/
├── docs/
│   ├── REPOSITORY_ARCHITECTURE.md
│   ├── REFACTOR_PLAN.md
│   ├── LEGACY_STATUS.md
│   ├── WORKLOG.md
│   └── migration/
├── external_models/
├── models/
│   └── yolo/
├── src/
│   └── yolo_sam/
├── studies/
│   ├── teacher_reference_bias_v2_512/
│   ├── teacher_reference_bias_v1/
│   ├── isaid_vehicle_study/
│   ├── samrs_sota_plane_study/
│   ├── semantic_drone_car_study/
│   └── landcover_building_study/
├── tests/
└── tools/
```

Kökte `artifacts/`, `data/`, `results/`, `runs/`, `configs/`, `scripts/` veya
`presentation_*` klasörü bırakılmaz.

## 4. Study Sözleşmesi

Tam bir study aşağıdaki sahiplik alanlarını kullanır:

```text
studies/<study_id>/
├── README.md
├── configs/
├── data/prepared/
├── docs/
├── reports/
├── results/
├── scripts/
├── src/
└── tests/
```

Her alt klasör zorunlu değildir. Boş klasör oluşturulmaz.

- `configs`: Protokol ve dataset configleri.
- `data/prepared`: Study tarafından üretilen tile, label, split ve metadata.
- `docs`: Deney planı, literatür, walkthrough, reproducibility appendix.
- `reports`: İnsan tarafından okunacak MD/DOCX/PDF ve report tabloları.
- `results`: Prediction, detector run, evaluation, audit ve manifestler.
- `scripts`: Study orchestration ve study-specific komutlar.
- `src`: Study’ye özel eşleme, analiz ve raporlama kodu.
- `tests`: Study configi, pipeline ve raporları doğrulayan testler.

## 5. Çalışma Sınıflandırması

| Study | Durum | Refactor kararı |
|---|---|---|
| `teacher_reference_bias_v2_512` | completed_canonical | 4×128/Overall 512 protokollü canonical çalışma; bütün sahiplik aynı study altında. |
| `teacher_reference_bias_v1` | completed_superseded | İlk 4×32 matched çalışma değiştirilemez tarihsel öncül olarak korunur. |
| `isaid_vehicle_study` | historical_context_only | Eski sunum, rapor, script, config ve sonuçlar kayıpsız korunur. |
| `samrs_sota_plane_study` | invalid_for_paper_evidence | Eski SAMRS deneyi ayrı tutulur; güncel analize girdi olmaz. |
| `semantic_drone_car_study` | planned | Hazırlık dosyaları tek study altında toplanır. |
| `landcover_building_study` | legacy_incomplete | Eksik tarihsel hazırlık ayrı study altında tutulur. |

## 6. Taşıma Matrisi

| Eski sahiplik | Yeni sahiplik |
|---|---|
| `artifacts/studies/teacher_reference_bias_v1` | `studies/teacher_reference_bias_v1/results` |
| `data/matched/{isaid_plane,samrs_sota_plane}` | `studies/teacher_reference_bias_v1/data/prepared/` |
| `configs/studies/*`, `configs/datasets/*` | `studies/teacher_reference_bias_v1/configs/` |
| `paper_teacher_reference_bias` | `studies/teacher_reference_bias_v1/reports/paper` |
| `presentation_isaid_vehicle_*` | `studies/isaid_vehicle_study/reports` |
| `presentation_samrs_sota_*` | `studies/samrs_sota_plane_study/reports` |
| Eski study config/script/results | İlgili `studies/<study_id>/` |
| Ham iSAID/SAMRS | `datasets/{isaid,samrs}/raw` |
| `yolo26n.pt`, `yolo26x.pt` | `models/yolo/` |
| `src/sam3_bbox_study` ortak parçaları | `src/yolo_sam` |

## 7. Kod Ayrıştırma Planı

### Ortak paket

`src/yolo_sam/` yalnız aşağıdakileri içerir:

- bbox/mask veri sözleşmeleri,
- ortak COCO/RLE ve metric kodu,
- SAM1/SAM2/SAM3 local wrapper'ları,
- GT-bbox ve YOLO-bbox ortak inference,
- detector evaluation çekirdeği,
- genel I/O ve hash yardımcıları.

### Teacher-reference-bias paketleri

`studies/teacher_reference_bias_v2_512/src/teacher_reference_bias/` aktif
512 görüntülü protokolün aşağıdaki çalışma kodunu içerir:

- matched protocol config çözümleme,
- pseudo-reference üretimi,
- canonical analiz,
- bootstrap ve pairing mantığı,
- study figürleri,
- tam metrik belge üretimi.

`teacher_reference_bias_v1` içindeki aynı adlı paket, tamamlanmış eski
4×32 çalışmanın tekrar üretilebilirliği için dondurulmuştur.

### Tarihsel study kodları

GroundingDINO, SegEarth, RemoteSAM text, RingMo ve eski hybrid prompt kodları
iSAID vehicle study altında tutulur. Semantic Drone ve landcover veri
adaptörleri kendi study'lerine taşınır.

## 8. Sonuç ve Manifest Planı

Aktif teacher-reference-bias v2 sonuçları:

```text
studies/teacher_reference_bias_v2_512/results/
├── analysis/
├── audits/
├── dataset_audits/
├── detectors/
├── evaluation/
├── figures/
├── predictions/
└── references/
```

Her aktif run manifesti:

- resolved config hash,
- input path ve SHA-256,
- output path ve SHA-256,
- model checkpoint/revision,
- dataset content manifest,
- seed ve prompt türü

alanlarını taşır. Taşıma sonrası path değişikliği `layout_migration` kaydıyla
eski ve yeni fingerprintleri birlikte korur.

## 9. Rapor Planı

Güncel v2 çalışma üç ayrı renkli tam metrik MD/DOCX/PDF üretir:

1. iSAID insan referansı.
2. Aynı iSAID tahminlerinin kontrollü SAM1 pseudo referansı.
3. SAMRS SOTA resmi SAM1 pseudo referansı.

V1 altı sayfalık bildiri taslağı tarihsel çalışma altında korunur; v2
sonuçlarıyla otomatik olarak yeniden yazılmaz.

Tam metrik raporların her birinde:

- detector bbox AP ve operating-point precision/recall,
- Overall,
- No Overlap × Low Mask Area,
- No Overlap × High Mask Area,
- Overlap × Low Mask Area,
- Overlap × High Mask Area,
- IoU, Dice ve piksel precision/recall,
- açıkça adlandırılmış `IoU ≥ 0.50/0.75/0.90` başarı oranları,
- overlap × mask-area nitel örnekleri,
- referans kaynağı duyarlılığı,
- Türkçe açıklama ve discussion

bulunur.

Yanıltıcı `mAP proxy` adı kullanılmaz. Success@IoU oranları COCO AP'den açıkça
ayrılır.

## 10. Doğrulama Kapıları

### Taşıma

- Pre/post manifestte 63 kaynak operasyonu.
- Her operasyonda aynı dosya sayısı.
- Aynı toplam byte.
- Aynı tree SHA-256.
- Durum `verified`.

### Kod

- Ortak unit testleri.
- Teacher study unit ve tiny end-to-end testleri.
- Import/compile kontrolü.
- `sam3_bbox_study` aktif importunun sıfır olması.
- Study dışına sonuç yazan varsayılan yolun sıfır olması.

### Veri ve sonuç

- İki dataset content manifesti.
- Altı detector run.
- Altı GT-bbox segmentation run.
- On sekiz YOLO-bbox segmentation run.
- 175.284 canonical instance-metric satırı.
- 180 aggregate satırı.
- 10.000 bootstrap koşulu.
- iSAID için 5.447 insan + 5.447 pseudo instance satırı, her koşulda tam
  eşleşen instance kimlikleri.
- SAMRS için her koşulda 3.713 pseudo-reference instance satırı.

### Rapor

- MD/DOCX/PDF ve tablo CSV'leri mevcut.
- Report manifest output hash'leri güncel.
- Üç PDF'nin her birinde bir detector, beş segmentasyon, dört nitel örnek ve
  bir discussion bölümü mevcut.
- PDF metni çıkarılabiliyor.
- DOCX zip bütünlüğü ve tablo sayıları geçerli.
- İlk, tablo, nitel örnek, karşılaştırma ve discussion sayfaları görsel QA'dan
  geçiyor.

## 11. Riskler ve Önlemler

| Risk | Önlem |
|---|---|
| Pahalı sonuç kaybı | Taşıma öncesi/sonrası hash manifesti ve immutable original arşivi |
| Eski absolute path | Aktif manifest path migration; historical original dosyaları değiştirmeme |
| Legacy sonucu canonical sanma | Study README durumu ve `LEGACY_STATUS.md` |
| Root’a yeniden sonuç yazılması | Shared CLI'larda zorunlu output/config; root boş klasörlerini kaldırma |
| Pseudo metrikleri AP sanma | Report metrik tanımları ve `Success@IoU` adı |
| PDF taşması | Render edilmiş sayfa görsel QA ve metin çıkarma kontrolü |
| Refactor sonrası import kırılması | Ortak ve study testlerinin ayrı çalıştırılması |

## 12. Tamamlanma Ölçütü

Refactor ancak şu koşullar birlikte sağlandığında tamamlanır:

1. Kök dizin sözleşmeye uyar.
2. Altı study kendi README ve sahiplik yapısına sahiptir.
3. Taşınan bilimsel içerik hash doğrulamasından geçer.
4. Canonical analiz sayıları ve ana metrikler değişmez.
5. Ortak ve study testlerinin tamamı geçer.
6. Üç tam metrik MD/DOCX/PDF üretilir.
7. Finalizer `pass` verir.
8. `WORKLOG.md` gerçek sonuçlarla güncellenir.
9. Aktif deney sözleşmesi study içindeki
   `docs/EXPERIMENT_PLAN.md` dosyasında güncel tutulur.

## 13. Durum

2026-07-27 itibarıyla refactor ve 512 görüntülü v2 çalışma tamamlanmıştır.
Tamamlanma ölçütlerinin dokuzu da sağlanmıştır:

- kök sahiplik sözleşmesi ve altı study yapısı doğrulandı,
- 63 taşıma kaydı kayıp olmadan doğrulandı,
- altı detector ve 24 segmentasyon koşulu tamamlandı,
- 175.284 instance ve 180 aggregate satırı üretildi,
- ortak, study ve integration testlerinin 88'i geçti,
- üç tam metrik MD/DOCX/PDF üretildi,
- üç rapor finalizer'dan ve görsel QA'dan geçti,
- final durum [WORKLOG.md](WORKLOG.md) içine kaydedildi,
- deney sözleşmesi ve QA checklist'i güncel tutuldu.
