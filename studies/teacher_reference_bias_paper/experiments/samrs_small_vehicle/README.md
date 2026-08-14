# SAMRS Small Vehicle Teacher-Reference Bias

SAMRS SOTA tarafından yayımlanan SAM1-türevi Small Vehicle maskelerinin yeniden üretilmiş sabit SAM1 checkpoint'iyle referans yakınlığı deneyidir. Bu deneyin temel referansı insan ground truth değildir ve sonuç model ailesine genellenmez.

- 512 test görüntüsü, 17 kaynak sahne, 7.659 instance
- 4 × 128 overlap/area tabakası
- referanslar: `published_samrs_reference`, `reproduced_pseudo_sam1`, `pseudo_sam2`, `pseudo_sam3`
- adaylar: frozen SAM1, SAM2, SAM3
- istemler: yayımlanmış horizontal detection bbox ve seed-42 YOLO bbox

Test görüntülerinin tamamı hedef-pozitiftir; detector sonuçları resmi benchmark AP'si değil bu deney testindeki kontroldür.

Yayımlanmış referans ile yeniden üretilmiş SAM1 referansı arasındaki nesne-ortalama IoU `0.998`'dir. Bu güçlü yakınlık bağımsız insan doğruluğu değildir.

Veri ve inference ayrıntısı [METHOD_AND_REPRODUCIBILITY.md](docs/METHOD_AND_REPRODUCIBILITY.md) dosyasındadır.
