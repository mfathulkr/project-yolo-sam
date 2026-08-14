# QA Report

Durum: **completed**

PASS: 37 | FAIL: 0

| Kontrol | Durum | Ayrıntı |
| --- | --- | --- |
| repo_root_contract | PASS | script ve kanonik path modülü aynı repository kökünü çözdü |
| isaid_plane:prepared | PASS | 512 görüntü, 5447 instance, 44 kaynak sahne |
| isaid_plane:predictions | PASS | 6 prediction kümesi; 4879 eşleşen, 568 kaçan ve 397 eşleşmeyen detection bağımsız doğrulandı |
| isaid_plane:run_manifests | PASS | 9 strict ve taşınabilir run manifest |
| isaid_plane:references | PASS | 5447 native referans ve üç GT-bbox pseudo zinciri RLE düzeyinde doğrulandı |
| isaid_plane:analysis | PASS | 130728 nesne-metrik ve 30 ek-IoU istatistik satırı |
| isaid_plane:figures | PASS | 4 nitel + 2 analiz figürü |
| isaid_plane:full_reports | PASS | 4 full-metric MD/DOCX/PDF |
| isaid_plane:cross_report | PASS | 18 sayfa |
| isaid_small_vehicle:prepared | PASS | 512 görüntü, 12051 instance, 31 kaynak sahne |
| isaid_small_vehicle:predictions | PASS | 6 prediction kümesi; 8633 eşleşen, 3418 kaçan ve 7712 eşleşmeyen detection bağımsız doğrulandı |
| isaid_small_vehicle:run_manifests | PASS | 9 strict ve taşınabilir run manifest |
| isaid_small_vehicle:references | PASS | 12051 native referans ve üç GT-bbox pseudo zinciri RLE düzeyinde doğrulandı |
| isaid_small_vehicle:analysis | PASS | 289224 nesne-metrik ve 30 ek-IoU istatistik satırı |
| isaid_small_vehicle:figures | PASS | 4 nitel + 2 analiz figürü |
| isaid_small_vehicle:full_reports | PASS | 4 full-metric MD/DOCX/PDF |
| isaid_small_vehicle:cross_report | PASS | 18 sayfa |
| samrs_plane:prepared | PASS | 512 görüntü, 3713 instance, 24 kaynak sahne |
| samrs_plane:predictions | PASS | 6 prediction kümesi; 3130 eşleşen, 583 kaçan ve 282 eşleşmeyen detection bağımsız doğrulandı |
| samrs_plane:run_manifests | PASS | 9 strict ve taşınabilir run manifest |
| samrs_plane:references | PASS | 3713 native referans ve üç GT-bbox pseudo zinciri RLE düzeyinde doğrulandı |
| samrs_plane:analysis | PASS | 89112 nesne-metrik ve 30 ek-IoU istatistik satırı |
| samrs_plane:figures | PASS | 4 nitel + 2 analiz figürü |
| samrs_plane:full_reports | PASS | 4 full-metric MD/DOCX/PDF |
| samrs_plane:cross_report | PASS | 18 sayfa |
| samrs_small_vehicle:prepared | PASS | 512 görüntü, 7659 instance, 17 kaynak sahne |
| samrs_small_vehicle:predictions | PASS | 6 prediction kümesi; 6310 eşleşen, 1349 kaçan ve 2453 eşleşmeyen detection bağımsız doğrulandı |
| samrs_small_vehicle:run_manifests | PASS | 9 strict ve taşınabilir run manifest |
| samrs_small_vehicle:references | PASS | 7659 native referans ve üç GT-bbox pseudo zinciri RLE düzeyinde doğrulandı |
| samrs_small_vehicle:analysis | PASS | 183816 nesne-metrik ve 30 ek-IoU istatistik satırı |
| samrs_small_vehicle:figures | PASS | 4 nitel + 2 analiz figürü |
| samrs_small_vehicle:full_reports | PASS | 4 full-metric MD/DOCX/PDF |
| samrs_small_vehicle:cross_report | PASS | 18 sayfa |
| paper_outputs | PASS | 4 açıklamalı dergi-boyutlu figür, 7 tablo, main report ve Overleaf iskeleti |
| canonical_segmenter_provenance | PASS | SAM1/SAM2/SAM3 checkpoint ve protocol hash'leri doğrulandı |
| deep_scientific_audit | PASS | split leakage yok; cross-experiment bağımlılık ve audit sınırlamaları kayıtlı |
| active_paths | PASS | 183 aktif metin/kod dosyası |
