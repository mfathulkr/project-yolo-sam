# QA Checklist

## Veri

- [x] Plane ve small-vehicle protokolleri `study_id` dışında aynı hash'e sahip.
- [x] Ham iSAID provenance audit geçti.
- [x] Ham SAMRS provenance audit geçti.
- [x] iSAID test kümesi 512 görüntü ve dört grupta 128'er görüntü.
- [x] SAMRS test kümesi 512 görüntü ve dört grupta 128'er görüntü.
- [x] Her iki veri setinde train/validation/test kaynak sahne kesişimleri sıfır.
- [x] iSAID maskeleri resmi poligonlardan kanonik COCO RLE olarak doğrulandı.
- [x] SAMRS pseudo maskeleri decode/area/bbox kontrollerinden geçti.

## Modeller

- [x] SAM1 checkpoint ve revision hashleri doğrulandı.
- [x] SAM2 checkpoint ve revision hashleri doğrulandı.
- [x] SAM3 checkpoint ve config tree hashleri doğrulandı.
- [x] İki sabit seed 42 YOLO eğitimi tamamlandı.
- [x] Validation confidence eşikleri testten önce donduruldu.
- [x] İki YOLO test bbox değerlendirmesi tamamlandı.

## Segmentasyon

- [x] İki veri setinde üçer GT-bbox segmenter koşulu tamamlandı.
- [x] iSAID SAM1 pseudo referansı donduruldu.
- [x] İki veri setinde üçer YOLO-bbox koşulu tamamlandı.
- [x] İnsan ve pseudo evaluation manifestleri tamamlandı.
- [x] Eksik veya yinelenen instance bulunmadığı doğrulandı.

## Rapor

- [x] iSAID insan MD/DOCX/PDF üretildi.
- [x] iSAID SAM1 pseudo MD/DOCX/PDF üretildi.
- [x] SAMRS SOTA MD/DOCX/PDF üretildi.
- [x] Her raporda Overall 512 ve alt tablolar 128 görüntü.
- [x] Her segmentasyon tablosunda yalnız 6 canonical pipeline var.
- [x] Yalnız gerçek bbox mAP ve tanımlı maske metrikleri var.
- [x] RemoteSAM, RingMoSAM, mask mAP proxy ve Boundary IoU yok.
- [x] Hücre renkleri ve PDF sayfaları görsel olarak kontrol edildi.
- [x] Rapor input/output hashleri doğrulandı.

## Taşınabilirlik

- [x] Yalnız seed 42 `best.pt` ağırlıkları manifestte kayıtlı.
- [x] Canonical sonuç arşivi ağırlık, log, cache ve raster görüntü içermiyor.
- [x] Prepared metadata arşivi veri seti görüntüsü içermiyor.
- [x] Dört LFS varlığının SHA-256 doğrulaması strict modda geçti.
