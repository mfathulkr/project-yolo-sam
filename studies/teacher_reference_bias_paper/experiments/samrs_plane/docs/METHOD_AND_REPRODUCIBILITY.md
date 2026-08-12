# SAMRS Plane Yöntem ve Yeniden Üretim

- Kaynak: SAMRS SOTA/DOTA-v2.0; hedef kategori `plane`.
- Prepared test: 512 adet 1024×1024 görüntü, 24 kaynak sahne ve 3.713 instance.
- Area threshold: `0.011932373046875`.
- Temel referans: yayımlanmış SAMRS SAM1-türevi maskesi; insan GT değildir.
- Segmenter GT istemi: özgün horizontal detection bbox. `rhbox` alan adı oriented prompt kullanıldığı anlamına gelmez.
- Detector validation confidence: `0.7607834339141846`.
- Detector test: bbox mAP50 `0.912655`, mAP75 `0.797482`, mAP90 `0.209450`, mAP50-95 `0.665239`.
- Yayımlanmış/reproduced SAM1 referans IoU: `0.990633`.
- Yayımlanmış referansta YOLO Avg IoU: SAM1 `0.812652`, SAM2 `0.678581`, SAM3 `0.691257`.

Matched çalıştırma ayarı `../config.yaml`, hamdan master havuz üretim ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. Bu deney mutlak insan kalitesini değil SAM-türevi referans yakınlığını ölçer.
