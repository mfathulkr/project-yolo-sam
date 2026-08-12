# iSAID Small Vehicle Yöntem ve Yeniden Üretim

- Kaynak: resmi iSAID instance anotasyonu; hedef kategori `Small_Vehicle`.
- Prepared test: 512 adet 1024×1024 tile, 31 özgün sahne ve 12.051 instance.
- Area threshold: `0.0018463134765625`.
- Temel referans: bağımsız insan maskesi.
- Detector validation confidence: `0.2740148901939392`.
- Detector test: bbox mAP50 `0.609022`, mAP75 `0.358470`, mAP90 `0.020547`, mAP50-95 `0.346237`.
- SAM1 pseudo referansındaki 19 boş maske filtrelenmez; known-positive kuralıyla 0 puanlanır.
- YOLO-bbox human Avg IoU: SAM1 `0.477966`, SAM2 `0.460845`, SAM3 `0.491179`.
- YOLO own-reference eşleşmiş fark: SAM1 `+0.175747`, SAM2 `+0.162804`, SAM3 `+0.142037`.

Matched çalıştırma ayarı `../config.yaml`, hamdan master havuz üretim ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. Yoğun sahnelerde detector kaçırmaları mask skoruna boş tahmin olarak yansır; detector yanlış pozitifleri ayrı bbox metriklerinde ölçülür.
