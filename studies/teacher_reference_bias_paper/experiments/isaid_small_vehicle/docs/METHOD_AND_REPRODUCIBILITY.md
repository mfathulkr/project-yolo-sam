# iSAID Small Vehicle Yöntem ve Yeniden Üretim

- Kaynak: resmi iSAID train+validation instance anotasyonu; hedef kategori `Small_Vehicle`. Kaynak sahneler özel train/validation/test split'ine yeniden ayrılmıştır; resmi leaderboard testi değildir.
- Hazırlanmış test: 512 adet 1024×1024 görüntü, 31 özgün sahne ve 12.051 küçük araç nesnesi.
- Alan eşiği: görüntü alanının `%0.185`'i; görüntüdeki bütün küçük araç maskelerinin toplamıdır, tek nesne büyüklüğü değildir.
- Temel referans: bağımsız insan maskesi.
- Detector doğrulama confidence değeri: `0.274`.
- Detector testi: bbox mAP50 `0.609`, mAP75 `0.358`, mAP90 `0.021`, mAP50-95 `0.346`. Testteki 512 görüntünün tamamı hedef-pozitiftir; bu resmi benchmark AP'si değildir.
- SAM1 pseudo referansındaki 19 boş maske filtrelenmez; known-positive kuralıyla 0 puanlanır.
- İnsan etiketinde YOLO-bbox Avg IoU: SAM1 `0.478`, SAM2 `0.461`, SAM3 `0.491`.
- İnsan etiketinden modelin kendi etiketine geçince puan değişimi: SAM1 `+0.176`, SAM2 `+0.163`, SAM3 `+0.142`.
- Modelin kendi etiketindeki IoU ile diğer iki SAM etiketindeki ortalama IoU arasındaki ek puan: SAM1 `+0.098`, SAM2 `+0.074`, SAM3 `+0.075`.
- Detector eğitimi tarihsel olarak resume edilmiştir. Final `best.pt` ve değerlendirme hash'leri korunur; resume başlangıç checkpoint byte'ları korunmadığı için eğitim trajectory'si ilk adımdan birebir tekrar garanti edilmez.

Tam hassasiyetli eşik ve sonuçlar analiz/config dosyalarında saklanır. Çalıştırma ayarı `../config.yaml`, ham veri hazırlama ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. Yoğun sahnelerde detector kaçırmaları maske skoruna boş tahmin olarak yansır; detector yanlış pozitifleri ayrı bbox metriklerinde ölçülür.
