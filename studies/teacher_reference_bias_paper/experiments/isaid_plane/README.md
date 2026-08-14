# iSAID Plane Teacher-Reference Bias

Bağımsız insan maskeleri üzerinde Plane sınıfının ana kontrol deneyidir.

- 512 test görüntüsü, 44 kaynak sahne, 5.447 instance
- 4 × 128 overlap/area tabakası
- referanslar: `human`, `pseudo_sam1`, `pseudo_sam2`, `pseudo_sam3`
- adaylar: frozen SAM1, SAM2, SAM3
- istemler: GT bbox ve seed-42 YOLO bbox

Test görüntülerinin tamamı hedef-pozitiftir ve iSAID train+validation anotasyonlarından kaynak-sahne güvenli özel split ile seçilmiştir; resmi leaderboard test protokolü değildir.

Ana ayrıntılı belgeler `reports/full_metrics/<reference>/`, deney içi karşılaştırma `reports/cross_analysis/` altındadır. `human` belgesi bağımsız model kalitesini, pseudo belgeleri aynı tahminlerin referans duyarlılığını gösterir.

Veri ve inference ayrıntısı [METHOD_AND_REPRODUCIBILITY.md](docs/METHOD_AND_REPRODUCIBILITY.md) dosyasındadır.
