# Study ve Legacy Durumu

## Amaç

Bu belge güncel teacher-reference-bias çalışmasını, önceki tarihsel
deneyleri ve henüz tamamlanmamış hazırlıkları birbirinden ayırır. “Legacy”
silinmiş veya değersiz demek değildir; güncel bildirinin canonical kanıtına
dahil edilmeyen tarihsel çalışma demektir.

## Otoritatif Çalışma

```text
studies/teacher_reference_bias_v2_512/
```

Durum: `completed_canonical`

Ana bileşenler:

```text
studies/teacher_reference_bias_v2_512/
├── README.md
├── configs/
├── data/prepared/
├── docs/
├── reports/
├── results/
├── scripts/
├── src/teacher_reference_bias/
└── tests/
```

Ana kullanıcı girişi:

```text
studies/teacher_reference_bias_v2_512/scripts/study.py
```

Bu çalışma iki veri setinde de `plane`, 1024×1024 giriş, 512 test görüntüsü,
dört eşit 128 görüntülük alt grup, SAM1/SAM2/SAM3, GT bbox ve üç YOLO seed'i
kullanır. iSAID için insan ve kontrollü SAM1-pseudo referans raporları ayrı
üretilir. Canonical analiz yalnız bu study’nin `results/analysis/` dizinini
okur.

## Teacher Reference Bias v1

```text
studies/teacher_reference_bias_v1/
```

Durum: `completed_superseded`

İlk 4×32 eşlenmiş deney ve altı sayfalık taslak tarihsel kayıt olarak
korunur. Daha geniş v2 protokolü tamamlandığı için güncel kanıt olarak v2
kullanılır.

## iSAID Vehicle Study

```text
studies/isaid_vehicle_study/
```

Durum: `historical_context_only`

İçerik:

- iSAID `Small_Vehicle + Large_Vehicle` birleşik hedefi,
- SAM2, SAM3, RemoteSAM, RingMo-SAM ve eski text/hybrid pipeline'ları,
- eski image-level union metrikleri,
- renkli DOCX/PDF ve sunum çıktıları,
- ilgili config, script, hazırlanmış veri ve pipeline sonuçları.

Bu çalışma kendi tarihsel sorusu için korunur. Güncel bildiriyle aynı hedef
sınıfa, örnekleme protokolüne veya evaluation granularity'ye sahip olmadığı
için teacher-reference-bias tablosuna eklenmez.

## SAMRS SOTA Plane Study

```text
studies/samrs_sota_plane_study/
```

Durum: `invalid_for_paper_evidence`

Bu ilk SAMRS çalışması araştırma sorusunu ortaya çıkarması bakımından
değerlidir. Ancak:

- iSAID ile eşlenmiş source-scene-safe split kullanmadı,
- bazı bbox'lar pseudo-maskeden türetildi,
- üç seed'li ortak detector protokolüne sahip değildi,
- pseudo referansı başlangıçta bağımsız ground truth gibi yorumladı.

Bu nedenle eski rapor ve sonuçlar silinmez fakat güncel bildirinin kanıtı
sayılmaz.

### Veri kimliği düzeltmesi

İlk incelemedeki “yerel veri SOTA değil” şüphesi sonraki exhaustive audit ile
yanlışlandı. Resmi SOTA-RBB arşivinde:

- 17.555 dosya,
- 615.407 instance,
- numeric class ID,
- RBox ve RHBox geometrileri

karşılaştırıldı ve kaynak resmi SAMRS SOTA-RBB olarak doğrulandı. Tarihsel
çalışmanın bilimsel sınırlılığı veri kimliği değil, eşlenmemiş protokoldür.

## Semantic Drone Car Study

```text
studies/semantic_drone_car_study/
```

Durum: `planned`

Config, veri adaptörü ve handoff notu vardır; tamamlanmış detector,
segmentation sonucu veya final rapor yoktur.

## Landcover Building Study

```text
studies/landcover_building_study/
```

Durum: `legacy_incomplete`

Eski landcover.ai building configi, hazırlama kodu ve yardımcı scriptleri
korunur. Tamamlanmış ve doğrulanmış bir study olarak sunulmaz.

## Ortak Kod

Çalışmaya özgü olmayan ortak kod:

```text
src/yolo_sam/
tools/
```

`src/sam3_bbox_study/` adı kaldırılmıştır. Bu ad eski bir deneyi çağrıştırdığı
ve artık ortak kullanılan kodun sahipliğini yanlış anlattığı için ortak paket
`yolo_sam` olarak yeniden adlandırılmıştır.

Harici model kaynakları ve ortak checkpointler:

```text
external_models/
models/
```

## Değiştirilemez Tarihsel Kayıtlar

Eski mutlak yollar aşağıdaki dosyalarda bilinçli olarak kalabilir:

- `docs/migration/*_pre.json`: taşıma öncesi konum kanıtı,
- `results/audits/**/originals/`: byte düzeyinde özgün manifest arşivleri,
- eski `ARTIFACT_MANIFEST.csv` ve QA belgeleri.

Bu dosyalar aktif runtime configi değildir. Geçmiş provenance'ı değiştirmemek
için yeniden yazılmaz.

## Yeni Geliştirme Kuralı

1. Yeni araştırma sorusu yeni bir `studies/<study_id>/` klasörü açar.
2. Config, prepared data, sonuç, rapor ve study-specific kod aynı study
   altında kalır.
3. Kod ancak en az iki çalışma tarafından aynı sözleşmeyle kullanılıyorsa
   `src/yolo_sam/` içine taşınır.
4. Yeni çalışma tarihsel study sonuçlarını canonical girdi olarak kullanmaz.
5. Her tamamlanmış çalışma kendi README, manifest ve doğrulama komutuna sahip
   olur.
