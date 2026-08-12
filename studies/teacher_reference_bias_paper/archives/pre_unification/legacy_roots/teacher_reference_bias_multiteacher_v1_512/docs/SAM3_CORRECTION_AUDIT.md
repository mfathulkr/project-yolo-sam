# SAM3 BBox Düzeltme ve Yeniden Üretim Denetimi

## Sorun

İlk SAM3 bbox uygulaması `Sam3Model` PCS görsel örnek arayüzünü kullanıyordu. Bu arayüzde kutu, SAM1/SAM2'deki gibi yalnız kutudaki belirli nesneyi segmentleyen bir istem değildir; kutudaki kavrama benzeyen nesneleri arayan visual-exemplar girdisidir. Ayrıca 0,5 çıktı olasılığı filtresi küçük nesne adaylarını eleyerek çok sayıda boş maske oluşturuyordu.

En belirgin anomali iSAID Small Vehicle GT-box koşulundaydı: insan GT bbox verilmesine rağmen 12.051 instance'ın 5.345'i boş dönmüştü. Aynı kutularda SAM2'nin boş maske üretmemesi ve boşluğun küçük nesne alanında yoğunlaşması, sorunun veri etiketinden çok SAM3 arayüz/protokol uyumsuzluğu olduğunu gösterdi.

## Düzeltme

- SAM3 bbox inference, aynı checkpoint'in belirli-instance segmentasyonu için sunulan `Sam3TrackerModel` PVS arayüzüne geçirildi.
- Her giriş bbox'ı için tam bir çıktı maskesi zorunlu kılındı.
- Maske eşiği olasılık 0,5 filtresi yerine modelin logit uzayında 0,0 olarak donduruldu.
- A6000 profili için kutu batch boyutu 128 olarak protokole yazıldı. 16 ve 128 batch ile aynı 128 kutuda üretilen maskeler piksel düzeyinde birebir aynı bulundu; en büyük IoU-skor farkı `3,6e-7` idi.
- 8 GB yerel profil için kutu batch boyutu 8 olarak ayrı tanımlandı.

## Metrik Politikası

Bu çalışmadaki her instance insan anotasyonuyla varlığı bilinen bir nesnedir. Öğretmen GT bbox verilmesine rağmen boş maske üretirse bu gerçek negatif değil, eksik pseudo etikettir. Bu nedenle tahmin de boş olsa bile IoU, Dice, Precision ve Recall 0,0 kabul edilir. Generic boş-görüntü davranışı instance deneyine taşınmaz.

Pseudo referans üreticisi artık RLE alanını doğrudan hesaplar ve şu tutarsızlıkları reddeder:

- `status=ok` fakat gerçek maske alanı sıfır;
- `status=empty_mask` fakat gerçek maske alanı sıfırdan büyük;
- eksik veya yinelenen instance kimliği;
- GT bbox dışında bir öğretmen istemi;
- beklenmeyen model/checkpoint kimliği.

## Yeniden Üretilen Kapsam

SAM3 için aşağıdaki koşulların GT-box ve seed-42 YOLO-box tahminleri yeniden üretilmiştir:

- iSAID Plane;
- SAMRS SOTA Plane;
- iSAID Small Vehicle;
- SAMRS SOTA Small Vehicle.

Ardından insan, SAM1 pseudo, SAM2 pseudo ve SAM3 pseudo referanslarına karşı bütün SAM1/SAM2/SAM3 değerlendirmeleri; aggregate tablolar; nitel görseller; full-metric Markdown/DOCX/PDF belgeleri; öğretmen karşılaştırma raporu ve bildiri varlıkları yeniden oluşturulmuştur.

## Kalite Kapıları

Final koşulda aşağıdakiler zorunludur:

- tahmin satır sayısı, prepared COCO hedef instance sayısıyla birebir aynı;
- instance kimlikleri benzersiz ve referans kümesiyle tam eşleşmiş;
- RLE boyutu görüntü boyutuyla uyumlu;
- durum ile gerçek maske alanı arasında sıfır uyuşmazlık;
- dolu maskenin kendi giriş bbox'ıyla en az bir piksel kesişmesi;
- SAM3 manifestinde `inference_interface=sam3_tracker_pvs`, `mask_threshold=0.0` ve `box_batch_size=128`;
- GT-box pseudo referanslarında teacher prompt türü yalnız `gt_bbox`;
- rapor tablolarının Overall için 512, her tabaka için 128 görüntüyü ve bu görüntülerdeki bütün target instance'ları kapsaması;
- test, compile, rapor hash, DOCX arşiv ve PDF raster kalite kontrollerinin geçmesi.

Yeniden üretim tamamlandı. Dört SAM3 GT-bbox koşulunda 28.870/28.870 instance
ve sıfır boş RLE doğrulandı. Nihai sayılar ile dosya hash'leri
`results/analysis`, report manifestleri ve `docs/QA_REPORT.md` içindedir.

## Provenance ve Sunum Kapanışı

- Dört veri seti × üç model × iki bbox kaynağından oluşan 24 inference koşulu
  gerçek model çalıştırmasıyla yeniden üretildi; her run kendi immutable
  provenance ve effective-config snapshotını taşır.
- 24 kanonik değerlendirme de yeniden çalıştırıldı ve aynı çıktı dizinine
  eşzamanlı yazmayı engelleyen lock ile korundu.
- Eski PCS manifestleri tarihsel arşivde açıkça geçersiz işaretlendi; geçici
  provenance-yamalama betiği gerçek yeniden koşumdan sonra kaldırıldı.
- Altı kanonik ve dört multiteacher full-metric rapor yeniden üretildi.
  40 nitel sayfadaki 200 panel doğru kaynak satır/sütunla piksel düzeyinde
  eşleşti; komşu satır taşması ve kesik başlık kalmadı.
