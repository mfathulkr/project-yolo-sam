# Worklog

## 2026-08-12 - Birleşik Paper Study

- Önceki üç teacher-reference-bias klasöründeki yaklaşık 38 GB veri ve sonuç, kopya oluşturmadan dört deneylik yeni yapıya taşındı.
- Taşınan dosyalar önce SHA-256 ile doğrulandı; eski/yeni yollar `MIGRATION_MANIFEST.json` dosyasına yazıldı.
- Dört prepared test kümesi doğrulandı: her biri 512 görüntü ve dört tabakada 128 görüntü.
- iSAID ve SAMRS için SAM1/2/3 referans küpleri tamamlandı.
- Bilinen pozitif nesnede iki boş maskenin yanlışlıkla IoU 1.0 sayılması engellendi; boş teacher referansı 0 puanlanıyor.
- SAM3 bbox koşulu PVS tracker arayüzüne düzeltildi; PCS sonuçları kanonik küpten çıkarıldı.
- Dört metric cube üretildi ve cardinality/identity kontrolleri geçti.
- 16 legacy full-metric MD/DOCX/PDF üretildi.
- Dört deney içi cross-analysis ve bir ana cross-analysis MD/DOCX/PDF üretildi.
- Nitel görseller, seçilen görüntüdeki bütün hedef instance'ları ve bütün GT kutuları içerecek biçimde yeniden üretildi.
- Bildiri için beş ana figür, beş ana tablo ve bir supplementary tabaka tablosu dört deneyden yeniden oluşturuldu.
- SAMRS published referansı insan GT olarak adlandırılmayacak biçimde config ve rapor sözleşmesi düzeltildi.
- Literatür taraması Parikh 2025/2026 biased-ruler çalışmaları ve SAMRS'nin pretraining amacıyla güncellendi.
- Plane master veri havuzları deney klasörlerine taşındı; dört deneyde de `master_config.yaml → config.yaml → prepared` zinciri kuruldu.
- CLI'a `prepare-master`, `prepare-matched` ve gerçek `--profile local_8gb` seçimi eklendi.
- 52 companion metadata dosyası ve 36 run manifesti repository-relative yollara geçirildi; özgün manifest hashleri `RUN_MANIFEST_MIGRATION_AUDIT.json` içinde korunuyor.
- Validator 36 çalışma manifestinin giriş/çıkış hashlerini, 80 maske tablosunu ve 16 detector tablosunu kaynak artifact'lara karşı strict doğruluyor.
- Bağımsız denetim bulguları kapatıldı: eski yollar, eksik Plane master zinciri, bozuk master config varsayılanları, lock dosyaları, 8 GB CLI ve Overleaf bağımlılık açıklaması düzeltildi.
- İkinci kabul denetiminde SAMRS referans/prediction cardinality'leri, model ayrışması, iki ayrı YOLO checkpoint'i ve train/validation/test kaynak-sahne ayrıklığı tekrar doğrulandı.
- Eski `teacher_reference_bias_v1` paketi ile paper study içindeki kopya arşiv kaldırıldı; rapor üreticisi artık eski raporu arşivlemek yerine deterministik olarak yeniler.
- Kök test keşfi yalnız aktif kodu çalıştıracak biçimde sadeleştirildi; master provenance regresyon testiyle birlikte tam test takımı `78 passed` sonucuna ulaştı.

## Nihai QA Sonucu

- 16 full-metric + 4 cross-analysis + 1 main PDF üretildi ve doğrulandı.
- Aktif çalışma manifestleri repository-relative ve strict hash doğrulamalıdır.
- Dört deney 512 görüntü, 4×128 tabaka ve beklenen instance sayılarını sağlıyor.
- Git/LFS yalnız kanonik seed-42 ağırlıkları ile gerekli taşınabilir sonuçları taşır; ham/prepared raster ve label ağaçları Git'e girmez.
- `validate_paper_study.py` bütün kontrollerde PASS vermektedir.
