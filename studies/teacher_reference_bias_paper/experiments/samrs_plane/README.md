# SAMRS Plane Teacher-Reference Bias

SAMRS SOTA tarafından yayımlanan SAM1-türevi Plane maskelerinin yeniden üretilmiş sabit SAM1 checkpoint'iyle referans yakınlığı deneyidir. Bu deneyin temel referansı insan ground truth değildir ve sonuç model ailesine genellenmez.

- 512 test görüntüsü, 24 kaynak sahne, 3.713 instance
- 4 × 128 overlap/area tabakası
- referanslar: `published_samrs_reference`, `reproduced_pseudo_sam1`, `pseudo_sam2`, `pseudo_sam3`
- adaylar: frozen SAM1, SAM2, SAM3
- istemler: yayımlanmış horizontal detection bbox ve seed-42 YOLO bbox

Test görüntülerinin tamamı hedef-pozitiftir; detector sonuçları resmi benchmark AP'si değil bu deney testindeki kontroldür.

Yayımlanmış referans ile yeniden üretilmiş SAM1 referansı arasındaki nesne-ortalama IoU `0.991`'dir. Bu güçlü yakınlık bağımsız insan doğruluğu değildir.

Veri ve inference ayrıntısı [METHOD_AND_REPRODUCIBILITY.md](docs/METHOD_AND_REPRODUCIBILITY.md) dosyasındadır.
