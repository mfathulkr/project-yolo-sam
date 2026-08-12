# Worklog

## 2026-08-11 - Tekrar üretim sözleşmesi

- Model, veri hazırlama, detector, bbox eşleştirme, pseudo-reference, metrik,
  bootstrap ve raporlama kodları manifestlerle çaprazlanarak kritik karar
  envanteri çıkarıldı.
- `docs/REPRODUCIBILITY_FIELD_GUIDE.md` oluşturuldu. PVS ve PCS açık adları,
  işlev farkları ve SAM3'te neden `Sam3TrackerModel` kullanıldığı basit dille
  anlatıldı.
- Checkpoint revision/SHA-256 değerleri, `multimask_output=False`, logit eşiği,
  compute batch ayrımı, validation confidence seçimi, greedy YOLO-GT
  eşleştirmesi, known-positive empty ve instance-macro sözleşmeleri kaydedildi.
- Bildiri planı, supplement Table S2 önerisi ve Overleaf Methods/Reproducibility
  notları aynı sözleşmeyle güncellendi.
- Markdown link kontrolü, Python sözdizimi ve `git diff --check` geçti. Güncel
  study validator reproducibility belgesini de kapsayarak PASS verdi; genel
  test paketi `234 passed, 18 warnings, 2 subtests passed` sonucunu verdi.

## 2026-08-11 - Literatür düzeltmesi

- Önceki taramada bulunmayan arXiv:2511.00477, doğrudan `Biased Ruler`
  öncülü olarak literatür incelemesine ve BibTeX'e eklendi.
- Aynı ekibin arXiv:2605.06891 devam çalışması ile annotation style,
  sistematik label bias ve label-quality audit kaynakları eklendi.
- Bildirinin genel ilk-çalışma iddiası kaldırıldı; özgünlük yalnız kontrollü
  remote-sensing SAM cross-teacher/reference matrisi olarak tanımlandı.
- Tarama hatası ve güncel sorgu kapsamı `docs/LITERATURE_SEARCH_AUDIT.md`
  içinde şeffaf biçimde kaydedildi.

## 2026-08-11 - SAM3 düzeltmesi

- Eski SAM3 bbox inference'ının yanlış PCS visual-exemplar arayüzünü kullandığı
  saptandı; önceki SAM3 sonuçları ve tamamlandı durumu geri çekildi.
- Belirli-instance bbox istemi için `Sam3Tracker` PVS arayüzü uygulandı.
- Bilinen pozitif instance'ta boş pseudo referansın metrik değeri 0,0 olarak
  tanımlandı; boş-boş 1,0 ödülü bu deney bağlamında kaldırıldı.
- SAM3 GT/YOLO tahminleri, pseudo referanslar, çapraz metrikler, raporlar ve
  yayın varlıklarının yeniden üretimi başlatıldı.
- Dört PVS GT-bbox koşulunda toplam 28.870 instance eksiksiz üretildi; boş RLE,
  durum-alan uyuşmazlığı, bbox eşleşme hatası ve kutunun tamamen dışında maske
  sayıları `0` olarak doğrulandı.

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

SAM3 PVS düzeltmesi sonrası tahmin, referans, metrik, rapor ve yayın varlığı
yeniden üretimi tamamlandı. Üç rapor validator'ı geçti; tam test paketi
`234 passed, 18 warnings, 2 subtests passed` sonucu verdi. Karşılaştırma raporundaki hata
sayısı/oranı hücreleri için düşük-değer-iyidir renk semantiği düzeltildi.
Bağımsız son ajan denetiminde bilimsel hesap veya aktif sonuç zinciri hatası
bulunmadı. Saptanan üç metadata kalıntısı giderildi; aynı ajanla yapılan
yeniden denetimde arşiv `98/98`, legacy drift kayıtları ve aktif provenance
zinciri geçti.

## 2026-08-11 - Provenance ve nitel rapor kapanışı

- 24 inference ve 24 evaluation koşulu gerçek yeniden koşumla tamamlandı;
  benzersiz run kimlikleri, sıfır drift ve çıktı SHA-256 değerleri doğrulandı.
- Eski invalid PCS manifestleri tarihsel arşivde bilimsel kullanıma kapatıldı.
- Aynı evaluation output dizinine eşzamanlı yazmayı engelleyen kilit eklendi.
- Sabit oranla panel kesen rapor kodu kaldırıldı. On rapordaki 40 nitel sayfanın
  200 paneli doğru kaynakla piksel düzeyinde eşleşti.
- Altı kanonik ve dört multiteacher full-metric MD/DOCX/PDF yeniden üretildi;
  10 PDF'nin tamamı 14 sayfa ve manifest karmaları geçerlidir.
