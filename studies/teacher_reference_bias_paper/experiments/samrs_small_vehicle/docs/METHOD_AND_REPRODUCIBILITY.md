# SAMRS Small Vehicle Yöntem ve Yeniden Üretim

- Kaynak: SAMRS SOTA/DOTA-v2.0; hedef kategori `small-vehicle`.
- Prepared test: 512 adet 1024×1024 görüntü, 17 kaynak sahne ve 7.659 instance.
- Area threshold: `0.0065670013427734`.
- Temel referans: yayımlanmış SAMRS SAM1-türevi maskesi; insan GT değildir.
- Segmenter GT istemi: özgün horizontal detection bbox.
- Detector validation confidence: `0.36182695627212524`.
- Detector test: bbox mAP50 `0.818990`, mAP75 `0.533639`, mAP90 `0.071970`, mAP50-95 `0.502319`.
- Yayımlanmış/reproduced SAM1 referans IoU: `0.998338`.
- Yayımlanmış referansta YOLO Avg IoU: SAM1 `0.781936`, SAM2 `0.707229`, SAM3 `0.706556`.

Matched çalıştırma ayarı `../config.yaml`, hamdan master havuz üretim ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. Bu deneyin yüksek SAM1 skoru, referansın SAM1-benzeri olmasıyla birlikte yorumlanmalıdır.
