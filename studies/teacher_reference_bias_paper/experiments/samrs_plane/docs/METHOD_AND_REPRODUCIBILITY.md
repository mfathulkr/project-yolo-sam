# SAMRS Plane Yöntem ve Yeniden Üretim

- Kaynak: SAMRS SOTA/DOTA-v2.0; hedef kategori `plane`.
- Hazırlanmış test: 512 adet 1024×1024 görüntü, 24 kaynak sahne ve 3.713 uçak nesnesi.
- Alan eşiği: görüntü alanının `%1.193`'ü; görüntüdeki bütün uçak maskelerinin toplamıdır.
- Temel referans: yayımlanmış SAMRS SAM1-türevi maskesi; insan GT değildir.
- Segmenter GT istemi: özgün horizontal detection bbox. `rhbox` alan adı oriented prompt kullanıldığı anlamına gelmez.
- Detector doğrulama confidence değeri: `0.761`.
- Detector testi: bbox mAP50 `0.913`, mAP75 `0.797`, mAP90 `0.209`, mAP50-95 `0.665`. Test yalnız hedef-pozitif 512 görüntüdür ve resmi benchmark değildir.
- Yayımlanmış ve yeniden üretilmiş SAM1 referansları arasındaki IoU: `0.991`.
- Yayımlanmış referansta YOLO Avg IoU: SAM1 `0.813`, SAM2 `0.679`, SAM3 `0.691`.
- Modelin kendi etiketindeki IoU ile diğer iki SAM etiketindeki ortalama IoU arasındaki ek puan: SAM1 `+0.133`, SAM2 `+0.122`, SAM3 `+0.132`.

Tam hassasiyetli eşik ve sonuçlar analiz/config dosyalarında saklanır. Çalıştırma ayarı `../config.yaml`, ham veri hazırlama ayarı `../master_config.yaml`, ortak protokol `../../../configs/protocol.yaml` dosyasındadır. Bu deney mutlak insan kalitesini değil SAM-türevi referans yakınlığını ölçer.
