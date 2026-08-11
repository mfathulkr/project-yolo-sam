# QA Report

**Tarih:** 2026-08-11
**Durum:** PASS

## Doğrulanan Deney Kapsamı

- iSAID Plane: 512 görüntü, 44 kaynak sahne, 5.447 instance.
- iSAID Small Vehicle: 512 görüntü, 31 kaynak sahne, 12.051 instance.
- Her hedefte dört tabaka ve tabaka başına 128 görüntü.
- Üç aday model: SAM1, SAM2 ve SAM3.
- İki istem kaynağı: insan GT bbox ve seed-42 YOLO bbox.
- Dört değerlendirme referansı: human, pseudo-SAM1, pseudo-SAM2 ve pseudo-SAM3.
- Toplam 419.952 benzersiz instance-model-prompt-reference metrik satırı.
- 240 aggregate satır, 36 paired reference-effect satırı, 12 teacher-advantage
  satırı, 16 ranking satırı ve 12 reference-agreement satırı.

## Referans Bütünlüğü

- SAM2/SAM3 pseudo referanslarında instance kaybı veya yinelenen instance yok.
- Pseudo referansların tamamı human GT bbox isteminden oluşturuldu.
- `empty_mask` çıktıları filtrelenmeden değerlendirmeye katıldı.
- Plane SAM3: 133/5.447 boş maske (%2,44).
- Small Vehicle SAM3: 5.345/12.051 boş maske (%44,35).
- GT bbox ve kendi pseudo referansı diagonal hücreleri tam 1,0 çıktı; bunlar
  başarı sonucu değil, tautological identity control olarak işaretlendi.
- YOLO bbox koşulunda her pseudo referansın kendi öğretmeni birinci oldu ve
  12 teacher-advantage kontrolünün tamamı pozitif kaldı.

## Rapor ve Yayın Çıktısı QA

- Dört yeni full-metric PDF'nin her biri 14 sayfa; eşlik eden Markdown ve
  DOCX dosyaları mevcut ve DOCX arşiv bütünlükleri sağlam.
- SAM1/SAM2/SAM3 karşılaştırma PDF'si 10 sayfa.
- Her full-metric belgede Overall ve dört overlap × mask-area tablosu,
  detector control tablosu ve bütün hedef instance'ları gösteren dört nitel
  örnek sayfası bulunuyor.
- PDF'ler sayfa sayfa rasterize edildi; boş/kırık sayfa bulunmadı.
- Yayın paketi 7 tabloyu CSV ve LaTeX, 7 figürü PNG ve PDF olarak içeriyor.
- Yayın varlıklarının SHA-256 değerleri `paper/assets/manifest.json` içinde.
- Overleaf dosyasındaki citation anahtarlarının tamamı `ref.bib` içinde ve
  süslü parantez dengesi sağlam.
- Kullanıcının verdiği `elektr` preamble ve son şablon bölümleri korunmuştur.
  Repository'de `elektr.cls` ile `elksty.tex` bulunmadığından yerel PDF
  derlemesi yapılmadı; `main.tex` bu iki resmi Overleaf şablon dosyasının
  bulunduğu projede derlenmelidir.

## Otomatik Testler

```text
PYTHONPATH=src .venv/bin/python -m pytest tests \
  studies/teacher_reference_bias_multiteacher_v1_512/tests -q
62 passed, 2 subtests passed, 3 warnings
```

Üç uyarı `pycocotools` mask decode kodunun NumPy 2 `copy` anahtarına henüz
uyum sağlamamasından gelen `DeprecationWarning` kayıtlarıdır; hata veya sayısal
uyuşmazlık değildir.

```text
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/validate_study.py
PASS: references, analysis, reports, rendered PDFs, paper assets, and Overleaf sources
```

Kaynak ve script dizinleri ayrıca `compileall` ile doğrulandı.

## Yeniden Üretilebilirlik Sınırı

Frozen SAM1/SAM2/SAM3 tahminleri canonical plane ve small-vehicle
çalışmalarından okunur; bu uzantı yeni model inference yapmaz. Ham RLE
referansları, 12 değerlendirme dosyası ve 220 MB'lık instance metrik küpü
yerel `results/` altında tutulur. Git paketi yöntem kodu, doğrulanmış özet
tablolar, raporlar, nitel görseller, yayın varlıkları ve manifestleri içerir.
Ham instance düzeyinde sıfırdan yeniden hesaplama için canonical prediction
dosyalarının da yerel makinede bulunması gerekir.
