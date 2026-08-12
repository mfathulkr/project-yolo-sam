# iSAID Plane Yöntem ve Yeniden Üretim

- Kaynak: resmi iSAID instance anotasyonu; hedef kategori `plane`.
- Prepared test: 512 adet 1024×1024 tile, 44 özgün sahne ve 5.447 Plane instance.
- Area threshold: `0.0167140960693359`.
- Temel referans: bağımsız insan maskesi.
- Detector validation confidence: `0.28115004301071167`.
- Detector test: bbox mAP50 `0.920479`, mAP75 `0.846556`, mAP90 `0.545348`, mAP50-95 `0.762251`.
- Pseudo referansların tamamı ilgili frozen modelin insan GT bbox tahmininden üretilir; boş referans yoktur.
- YOLO-bbox human Avg IoU: SAM1 `0.597033`, SAM2 `0.573660`, SAM3 `0.638357`.
- YOLO own-reference eşleşmiş fark: SAM1 `+0.276232`, SAM2 `+0.278910`, SAM3 `+0.224210`.

Matched çalıştırma ayarı `../config.yaml`, hamdan master havuz üretim ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. Sonuçların yeniden türetilmesi ortak README'deki komut sırasıyla yapılır. GT diagonal `1.0` bağımsız performans değildir.
