# QA Report

**Tarih:** 2026-08-11
**Durum:** OTOMATİK KALİTE KAPILARI GEÇTİ

## Düzeltme Özeti

Önceki SAM3 sonuçları, belirli-instance bbox istemi yerine PCS
visual-exemplar arayüzü ve ek aday filtresiyle üretilmişti. Bu sonuçlar geri
çekildi. Dört veri kümesindeki SAM3 GT/YOLO tahminleri `Sam3Tracker` PVS ile,
SAM2/SAM3 pseudo referansları ve bütün bağımlı metrik/raporlar ise bu yeni
tahminlerden yeniden üretildi.

## Deney Kapsamı

- iSAID Plane: 512 görüntü, 44 kaynak sahne, 5.447 instance.
- iSAID Small Vehicle: 512 görüntü, 31 kaynak sahne, 12.051 instance.
- Her hedefte dört tabaka ve tabaka başına 128 görüntü.
- Adaylar: SAM1, SAM2 ve SAM3; istemler: insan GT bbox ve seed-42 YOLO bbox.
- Referanslar: human, pseudo-SAM1, pseudo-SAM2 ve pseudo-SAM3.
- 419.952 benzersiz instance-model-prompt-reference satırı; 240 aggregate,
  36 paired effect, 12 teacher advantage, 16 ranking ve 12 agreement satırı.

## Referans ve Tahmin Bütünlüğü

- PVS ile dört SAM3 GT-bbox koşulunda 28.870/28.870 instance üretildi.
- Bu dört koşulda boş RLE, durum/alan uyuşmazlığı, bbox kimlik uyuşmazlığı ve
  maskenin bbox dışında kalması sayıları sıfırdır.
- SAM2 ve SAM3 pseudo referanslarında kayıp/yinelenen instance veya boş maske
  yoktur. iSAID Small Vehicle SAM1 pseudo referansında 19/12.051 gerçek boş
  maske vardır; bu oran `%0,158`'dir.
- Bilinen insan GT instance'taki boş pseudo referans eksik etikettir ve IoU,
  Dice, Precision ve Recall için sıfır alır. Bu nedenle Small Vehicle SAM1
  GT-bbox identity kontrolü `0,998423`, diğer dolu diagonal kontroller `1,0`dır.
- YOLO koşullarındaki boş satırlar detector `missing_bbox` kayıtlarıdır. SAM3,
  gerçekten verilen YOLO kutularında ayrıca boş maske üretmemiştir.
- Kaynak prediction, source manifest, config/run kimliği, checkpoint SHA-256,
  PVS arayüzü ve pseudo-reference RLE özdeşliği doğrulanmıştır.
- 24 kanonik inference manifestinin run kimliği benzersizdir; tamamında
  `config_hash_scope=bbox_segmentation_effective_v1`, run-local segmenter
  provenance/effective-config snapshotı, sıfır input drift ve doğrulanmış çıktı
  karması vardır. Ortak inference source-tree karması
  `185e10ce7a7a55273461eaced785cc666f98c294a8ab2ca498b5d06b01a6b663`'dir.
- 24 kanonik evaluation manifestinin run kimliği benzersizdir; tamamında sıfır
  input drift ve doğrulanmış çıktı karması vardır. Ortak evaluation source-tree
  karması
  `6b392cccd0d80cf8409531814732a0664b2dcb6c03a32868e120cc313aa6c76b`'dir.
- Üç evaluation runner'ında aynı output dizinine ikinci yazarı engelleyen
  `.evaluation_writer.lock` kapısı vardır.
- Eski PCS çıktılarının sekiz manifesti tarihsel arşivde
  `superseded_invalid_for_scientific_results` ve `must_not_enter_analysis`
  olarak işaretlenmiştir; aktif sonuç çözümlemesine dahil edilmez.

## Nihai Ana Bulgular

- İnsan referansında iki hedef ve iki bbox kaynağında sıralama
  `SAM3 > SAM1 > SAM2`dir.
- On iki pseudo-referans/bbox kombinasyonunun tamamında referansı üreten model
  birinci olmuştur. Dokuz kombinasyonda tam model sırası insan sırasından
  değişmiştir.
- YOLO-bbox self-reference paired IoU artışları Plane için SAM1/SAM2/SAM3
  sırasıyla `+0,276 / +0,279 / +0,224`; Small Vehicle için
  `+0,176 / +0,163 / +0,142`dir. Altı güven aralığının tamamı sıfırın
  üzerindedir.
- YOLO-bbox teacher advantage, Plane'de `0,127–0,139`, Small Vehicle'da
  `0,071–0,098` aralığındadır.
- İnsan-pseudo ortalama instance IoU, Plane'de SAM1/SAM2/SAM3 için
  `0,653 / 0,629 / 0,700`; Small Vehicle'da `0,658 / 0,645 / 0,698`dir.
- GT-bbox diagonal sonuçları bağımsız başarı değildir; coverage-aware identity
  control olarak işaretlenmiştir. Ana non-identical kanıt YOLO-bbox sonuçlarıdır.

## Rapor ve Yayın Çıktısı QA

- Altı kanonik ve dört multiteacher full-metric MD/DOCX/PDF güncellendi.
  On full-metric PDF'nin tamamı 14 sayfadır.
- Öğretmen karşılaştırma PDF'si 10 sayfadır.
- Her full-metric belgede detector control, Overall ve dört
  overlap × mask-area tablosu ile seçilen sahnedeki bütün hedef instance'ları
  gösteren dört nitel sayfa bulunur.
- Eski sabit-oran panel kesimi kaldırılmıştır. Kaynak 4×5 figürdeki panel
  sınırları doğrulanarak algılanır; 10 rapordaki 40 nitel sayfanın 200 paneli
  kaynakla piksel düzeyinde eşleşmiştir. Başlık kesilmesi ve komşu satır şeridi
  yoktur.
- PDF metni, sayfa sayısı ve raster görünümü; DOCX ZIP bütünlüğü, tablo/görsel
  sayısı; CSV şeması ve tüm manifest hash'leri doğrulandı.
- Karşılaştırma raporunda başarı metrikleri için yüksek, hata sayısı/oranı için
  düşük değerin iyi olduğu renk semantiği ayrı uygulanır; sıfır boş maske ve
  sıfır status mismatch artık yeşil gösterilir.
- `boundary_iou` birleşik veri küpünde şema uyumluluğu için bulunur; SAM2/SAM3
  uzantısında hesaplanmaz ve hiçbir tablo, analiz veya sonuç iddiasında
  kullanılmaz. Bu çalışmanın önceden dondurulan maske metrikleri IoU, Dice,
  Precision, Recall ve IoU eşik başarı oranlarıdır.
- `paper/assets/manifest.json` içindeki 14 tablo ve 14 figür dosyasının
  provenance/hash denetimi geçti. Overleaf citation anahtarları ve parantez
  dengesi sağlamdır.
- Repository'de `elektr.cls` ve `elksty.tex` bulunmadığı için `main.tex` yerelde
  derlenmedi; kullanıcının Overleaf şablonunda bu iki dosya mevcuttur.

## Otomatik Kontroller

```text
.venv/bin/pytest -q
234 passed, 18 warnings, 2 subtests passed
```

On sekiz uyarı `pycocotools` mask decode kodunun NumPy 2 `copy` anahtarına
henüz uyum sağlamamasından gelen `DeprecationWarning` kayıtlarıdır.

```text
.venv/bin/python studies/teacher_reference_bias_v2_512/scripts/validate_full_metric_reports.py
PASS isaid_plane_human
PASS isaid_plane_pseudo_sam1
PASS samrs_sota_plane

.venv/bin/python studies/teacher_reference_bias_small_vehicle_v1_512/scripts/validate_full_metric_reports.py
PASS isaid_small_vehicle_human
PASS isaid_small_vehicle_pseudo_sam1
PASS samrs_sota_small_vehicle

.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/validate_study.py
PASS: references, analysis, reports, rendered PDFs, paper assets, and Overleaf sources
```

Kaynak/script dizinleri ayrıca `compileall` ve Git diff whitespace denetiminden
geçirildi. Bağımsız ajan denetiminde bilimsel hesap veya aktif sonuç zinciri
hatası bulunmadı. Saptanan üç metadata kalıntısı kapatıldı: stale repair
manifesti kaldırıldı, tarihsel arşivdeki 98 dosyanın hash dizini yenilendi ve
iki legacy detector manifestindeki eski drift kaydı temizlenerek validator
testiyle güvenceye alındı.

## Yeniden Üretilebilirlik Sınırı

Ham RLE tahminleri, referanslar ve 419.952 satırlık metrik küpü yerel
`results/` dizinlerindedir. Git paketi yöntem kodunu, özet tabloları,
raporları, nitel görselleri, yayın varlıklarını ve manifestleri taşır. Ham
instance düzeyinde sıfırdan hesaplama için kanonik prediction dosyalarının da
yerel makinede bulunması gerekir.
