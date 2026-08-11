# Worklog

## 2026-08-11

- Plane ve Small Vehicle için SAM2/SAM3 GT-bbox pseudo referansları, canonical frozen tahminlerden oluşturuldu.
- Aynı SAM1/SAM2/SAM3 aday tahminleri dört referansla değerlendirildi; 240 aggregate satır ve 36 paired reference-effect satırı üretildi.
- Dört yeni full-metric MD/DOCX/PDF raporu, eski canonical renkli format korunarak üretildi.
- SAM1/SAM2/SAM3 karşılaştırma raporu hazırlandı; ranking, teacher advantage, human agreement ve empty-mask audit eklendi.
- Literatür taraması remote sensing, medical imaging, imperfect reference standards ve model-generated benchmark self-bias başlıklarında tamamlandı.
- `elektr` Overleaf şablonunu koruyan section/subsection iskeleti ve BibTeX dosyası hazırlandı.
- Altı ana tablo, bir supplement tablosu ve yedi figür iki formatta üretildi; qualitative figürlerde her görüntüdeki bütün hedef instance'lar kullanıldı.

- Uçtan uca validator referans, analiz, DOCX/PDF, yayın varlığı ve Overleaf
  kaynak denetimlerinin tamamından geçti.
- Ortak ve study test paketlerinde 62 test ile 2 alt test geçti; yalnız
  `pycocotools` kaynaklı üç NumPy deprecation uyarısı kaldı.
- Yayın tablosundaki GT-diagonal 1,0 satırları yanlış başarı yorumu doğurmaması
  için açıkça `identity control` olarak işaretlendi.
- Ayrıntılı kapanış kaydı `docs/QA_REPORT.md` dosyasına yazıldı.

## Durum

Deney, rapor, literatür, bildiri iskeleti, tablo/figür ve QA işleri tamamlandı.
