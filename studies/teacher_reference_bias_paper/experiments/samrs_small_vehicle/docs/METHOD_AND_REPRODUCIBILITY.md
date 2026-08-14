# SAMRS Small Vehicle Yöntem ve Yeniden Üretim

- Kaynak: SAMRS SOTA/DOTA-v2.0; hedef kategori `small-vehicle`.
- Hazırlanmış test: 512 adet 1024×1024 görüntü, 17 kaynak sahne ve 7.659 küçük araç nesnesi.
- Alan eşiği: görüntü alanının `%0.657`'si; görüntüdeki bütün küçük araç maskelerinin toplamıdır.
- Temel referans: yayımlanmış SAMRS SAM1-türevi maskesi; insan GT değildir.
- Segmenter GT istemi: özgün horizontal detection bbox.
- Detector doğrulama confidence değeri: `0.362`.
- Detector testi: bbox mAP50 `0.819`, mAP75 `0.534`, mAP90 `0.072`, mAP50-95 `0.502`. Test yalnız hedef-pozitif 512 görüntüdür ve resmi benchmark değildir.
- Yayımlanmış ve yeniden üretilmiş SAM1 referansları arasındaki IoU: `0.998`.
- Yayımlanmış referansta YOLO Avg IoU: SAM1 `0.782`, SAM2 `0.707`, SAM3 `0.707`.
- Modelin kendi etiketindeki IoU ile diğer iki SAM etiketindeki ortalama IoU arasındaki ek puan: SAM1 `+0.081`, SAM2 `+0.043`, SAM3 `+0.060`.
- Detector eğitimi tarihsel olarak resume edilmiştir. Final `best.pt` ve değerlendirme hash'leri korunur; resume başlangıç checkpoint byte'ları korunmadığı için eğitim trajectory'si ilk adımdan birebir tekrar garanti edilmez.

Tam hassasiyetli eşik ve sonuçlar analiz/config dosyalarında saklanır. Çalıştırma ayarı `../config.yaml`, ham veri hazırlama ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. Bu deneyin yüksek SAM1 skoru, referansın SAM1-benzeri olmasıyla birlikte yorumlanmalıdır.
