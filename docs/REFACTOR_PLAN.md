# Repository Refactor Plan

## Durum

Bu plan 12 Ağustos 2026 tarihinde uygulanmıştır. Güncel mimari ve sahiplik
sözleşmesi için [REPOSITORY_ARCHITECTURE.md](REPOSITORY_ARCHITECTURE.md),
bilimsel çalışma ayrıntısı için
[`teacher_reference_bias_paper/README.md`](../studies/teacher_reference_bias_paper/README.md)
kullanılmalıdır.

## Tamamlanan İşler

- Plane, Small Vehicle ve multi-teacher parçaları tek paper study altında
  birleştirildi.
- Dört kanonik deney kendi config, veri, sonuç, analiz, figür, rapor ve
  tekrar üretim dokümanına ayrıldı.
- Human/published/SAM1/SAM2/SAM3 referans rolleri açıkça ayrıldı.
- 16 tam metrik rapor ve 4 deney içi çapraz analiz aynı formatta üretildi.
- Dört deneyi karşılaştıran ana analiz, bildiri tabloları, figürleri ve Overleaf
  iskeleti üretildi.
- Eski kökler içerik kaybı olmadan arşive taşındı ve SHA-256 ile doğrulandı.
- Aktif koddan eski çalışma yolları ve makineye özgü veri yolları kaldırıldı.
- Manifest, rapor, metrik küpü, figür kapsamı ve PDF/DOCX bütünlüğü otomatik
  validator ile denetlendi.

## Tamamlanma Kriteri

Kanonik validator ve testler başarılı olduğunda refactor tamamlanmış kabul
edilir:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
.venv/bin/pytest -q studies/teacher_reference_bias_paper/tests
```

Son doğrulama sonucu:
[`QA_REPORT.md`](../studies/teacher_reference_bias_paper/docs/QA_REPORT.md)
