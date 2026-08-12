# SAMRS Small Vehicle Teacher-Reference Bias

SAMRS SOTA tarafından yayımlanan SAM1-türevi Small Vehicle maskelerinin model-aile yakınlığı deneyidir. Bu deneyin temel referansı insan ground truth değildir.

- 512 test görüntüsü, 17 kaynak sahne, 7.659 instance
- 4 × 128 overlap/area tabakası
- referanslar: `published_samrs_reference`, `reproduced_pseudo_sam1`, `pseudo_sam2`, `pseudo_sam3`
- adaylar: frozen SAM1, SAM2, SAM3
- istemler: yayımlanmış horizontal detection bbox ve seed-42 YOLO bbox

Yayımlanmış referans ile yeniden üretilmiş SAM1 referansı arasındaki instance IoU `0.998338` değerindedir. Bu güçlü yakınlık bağımsız insan doğruluğu değildir.

Veri ve inference ayrıntısı [METHOD_AND_REPRODUCIBILITY.md](docs/METHOD_AND_REPRODUCIBILITY.md) dosyasındadır.
