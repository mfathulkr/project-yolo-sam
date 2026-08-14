# iSAID Plane Yöntem ve Yeniden Üretim

- Kaynak: resmi iSAID train+validation instance anotasyonu; hedef kategori `plane`. Kaynak sahneler özel train/validation/test split'ine yeniden ayrılmıştır; resmi leaderboard testi değildir.
- Hazırlanmış test: 512 adet 1024×1024 görüntü, 44 özgün sahne ve 5.447 uçak nesnesi.
- Alan eşiği: görüntü alanının `%1.671`'i; görüntüdeki bütün uçak maskelerinin toplamıdır, tek nesne büyüklüğü değildir.
- Temel referans: bağımsız insan maskesi.
- Detector doğrulama confidence değeri: `0.281`.
- Detector testi: bbox mAP50 `0.920`, mAP75 `0.847`, mAP90 `0.545`, mAP50-95 `0.762`. Testteki 512 görüntünün tamamı hedef-pozitiftir; bu resmi benchmark AP'si değildir.
- Pseudo referansların tamamı ilgili frozen modelin insan GT bbox tahmininden üretilir; boş referans yoktur.
- İnsan etiketinde YOLO-bbox Avg IoU: SAM1 `0.597`, SAM2 `0.574`, SAM3 `0.638`.
- İnsan etiketinden modelin kendi etiketine geçince puan değişimi: SAM1 `+0.276`, SAM2 `+0.279`, SAM3 `+0.224`.
- Modelin kendi etiketindeki IoU ile diğer iki SAM etiketindeki ortalama IoU arasındaki ek puan: SAM1 `+0.128`, SAM2 `+0.124`, SAM3 `+0.141`.

Tam hassasiyetli eşik ve sonuçlar analiz/config dosyalarında saklanır. Çalıştırma ayarı `../config.yaml`, ham veri hazırlama ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. GT bbox ile kendi maskesine göre ölçülen `1.000` değeri bağımsız performans değildir.
