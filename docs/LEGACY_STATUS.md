# Study ve Legacy Durumu

## Kanonik Çalışma

```text
studies/teacher_reference_bias_paper/
```

Durum: `completed_canonical_paper_study`

Bu çalışma dört eşlenmiş deneyi içerir: iSAID Plane, iSAID Small Vehicle,
SAMRS Plane ve SAMRS Small Vehicle. Dört deney de 512 görüntü, 4×128 tabaka,
seed 42 YOLO, frozen SAM1/SAM2/SAM3 ve GT/YOLO bbox koşullarını kullanır.
iSAID deneyleri bağımsız insan referansına sahiptir. SAMRS yayımlanmış
referansları SAM1 kökenli olduğu için destekleyici provenance/affinity
kontrolüdür.

Ana giriş:

```text
studies/teacher_reference_bias_paper/scripts/study.py
```

## Kaldırılan Teacher-Bias Kopyaları

Önceki Plane, Small Vehicle, multi-teacher ve `teacher_reference_bias_v1`
paketleri kanonik dört deney doğrulandıktan sonra kaldırıldı. Aktif kod,
config, rapor ve dokümantasyon bu eski yollara bağlı değildir. Taşıma geçmişi
Git geçmişinde ve `teacher_reference_bias_paper/docs/MIGRATION_MANIFEST.json`
dosyasında kayıtlıdır.

## iSAID Vehicle Study

```text
studies/isaid_vehicle_study/
```

Durum: `historical_context_only`

Birleşik `Small_Vehicle + Large_Vehicle`, eski RemoteSAM/RingMo-SAM ve
text/hybrid karşılaştırmalarını korur. Hedef, granularity ve örnekleme
protokolü farklı olduğu için teacher-reference-bias bildirisinin kanıtı
değildir.

## SAMRS SOTA Plane Study

```text
studies/samrs_sota_plane_study/
```

Durum: `historical_context_only`

Araştırma sorusunu ortaya çıkaran ilk SAMRS çalışmasıdır. Eski eşlenmemiş
protokolü nedeniyle güncel kanıt olarak kullanılmaz; kanonik SAMRS tekrarları
paper study içindedir.

## Diğer Çalışmalar

- `semantic_drone_car_study`: planlandı, tamamlanmış sonuç yok.
- `landcover_building_study`: eksik tarihsel hazırlık.

## Kopya Politikası

Kanonik çalışma içinde `archives/` veya ayrı bir eski teacher-bias paketi
tutulmaz. Türetilmiş eski çıktılar Git geçmişinden alınabilir; aktif sonuçların
tek sahibi `studies/teacher_reference_bias_paper/` dizinidir.
