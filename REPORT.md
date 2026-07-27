# Current Study Report

## Otoritatif Çalışma

Tamamlanmış canonical ve bildiri düzeyindeki çalışma:

```text
studies/teacher_reference_bias_v2_512/
```

Ana çalışma README’si:

```text
studies/teacher_reference_bias_v2_512/README.md
```

## Ana Çıktılar

```text
studies/teacher_reference_bias_v2_512/reports/full_metrics/
├── isaid_plane_human/
│   ├── isaid_plane_human_full_metric_document.md
│   ├── isaid_plane_human_full_metric_document_colored.docx
│   └── isaid_plane_human_full_metric_document_colored.pdf
├── isaid_plane_pseudo_sam1/
│   ├── isaid_plane_pseudo_sam1_full_metric_document.md
│   ├── isaid_plane_pseudo_sam1_full_metric_document_colored.docx
│   └── isaid_plane_pseudo_sam1_full_metric_document_colored.pdf
└── samrs_sota_plane/
    ├── samrs_sota_plane_full_metric_document.md
    ├── samrs_sota_plane_full_metric_document_colored.docx
    └── samrs_sota_plane_full_metric_document_colored.pdf
```

Canonical makine-okunabilir sonuç:

```text
studies/teacher_reference_bias_v2_512/results/analysis/
```

Yeniden üretilebilirlik manifesti:

```text
studies/teacher_reference_bias_v2_512/results/analysis/manifest.json
```

## Durum

- Eşlenmiş iSAID ve SAMRS SOTA protokolü 512 test görüntüsüne genişletildi.
- Her veri setinde dört alt grubun her birinde tam 128 görüntü var.
- SAM1, SAM2 ve SAM3 için GT-bbox ve üç seed'li YOLO-bbox koşulları kullanılır.
- iSAID insan referansı, aynı tahminlere ait kontrollü SAM1-pseudo referansı
  ve SAMRS resmi SAM1-pseudo referansı ayrı raporlanır.
- Üç renkli full-metric MD/DOCX/PDF belgesi üretildi ve doğrulandı.
- Altı YOLO eğitimi/testi, 24 SAM tahmin koşulu ve 24 değerlendirme
  manifesti eksiksiz tamamlandı.
- Canonical analiz 175.284 instance satırı ve 180 aggregate satırı içeriyor.
- Tarihsel çalışmalar kendi study klasörlerinde korunuyor.

Ana kontrollü bulgu: aynı iSAID tahminleri insan yerine SAM1 pseudo
referansla ölçüldüğünde GT-bbox IoU artışı SAM1/SAM2/SAM3 için sırasıyla
`+0,347 / +0,198 / +0,140` oldu ve model sırası değişti.

Refactor ve doğrulama ayrıntıları:

```text
docs/WORKLOG.md
docs/REFACTOR_PLAN.md
docs/LEGACY_STATUS.md
```
