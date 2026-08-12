# iSAID Small Vehicle Teacher-Reference Bias

Bağımsız insan maskeleri üzerinde yoğun ve küçük Small Vehicle sınıfının ana kontrol deneyidir.

- 512 test görüntüsü, 31 kaynak sahne, 12.051 instance
- 4 × 128 overlap/area tabakası
- referanslar: `human`, `pseudo_sam1`, `pseudo_sam2`, `pseudo_sam3`
- adaylar: frozen SAM1, SAM2, SAM3
- istemler: GT bbox ve seed-42 YOLO bbox

Ana ayrıntılı belgeler `reports/full_metrics/<reference>/`, deney içi karşılaştırma `reports/cross_analysis/` altındadır. SAM1 pseudo referansında 19 bilinen-pozitif boş maske vardır ve bunlar sıfır puanlanır.

Veri ve inference ayrıntısı [METHOD_AND_REPRODUCIBILITY.md](docs/METHOD_AND_REPRODUCIBILITY.md) dosyasındadır.
