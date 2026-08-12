# Teacher-Reference Bias Handoff

## Ne Yapıldı?

Önceki Plane, Small Vehicle ve multi-teacher klasörleri dört deneylik tek paper study altında birleştirildi. Veri, detector, dondurulmuş SAM tahminleri ve eski raporlar kaybolmadan hash doğrulamalı taşındı. Yeni kanonik deneyler:

1. `experiments/isaid_plane`
2. `experiments/isaid_small_vehicle`
3. `experiments/samrs_plane`
4. `experiments/samrs_small_vehicle`

Her deneyde dört referans, üç model, iki bbox kaynağı ve beş rapor tabakası vardır.

Her deney kendi `master_config.yaml`, matched `config.yaml`, seed-42 detector checkpoint'i, dondurulmuş prediction'ları, referansları, analizleri ve raporlarıyla tek başına izlenebilir. Aktif 36 run manifesti repository-relative yollarla strict hash doğrulamasından geçer.

## En Önemli Sonuç

iSAID insan kontrolünde, aynı modelin kendi pseudo referansına karşı YOLO-bbox Avg IoU artışı:

| Deney | SAM1 | SAM2 | SAM3 |
| --- | ---: | ---: | ---: |
| iSAID Plane | +0.276 | +0.279 | +0.224 |
| iSAID Small Vehicle | +0.176 | +0.163 | +0.142 |

Bütün `%95` kaynak-sahne kümeli güven aralıkları sıfırın üzerindedir. Bu, özdeş GT diagonalinden farklıdır çünkü teacher referansı GT bbox, aday maskesi YOLO bbox kullanır.

SAMRS yayımlanmış referans ile yeniden üretilmiş SAM1 referansı arasındaki Avg IoU:

- Plane: `0.990633`
- Small Vehicle: `0.998338`

Bu sonuç yayımlanmış SAMRS maskelerinin SAM1'e çok yakın olduğunu destekler; insan doğruluğu göstermez.

## Hangi PDF'ler Gösterilecek?

- Ana özet: `analysis/main_cross_analysis_colored.pdf`
- Deney içi karşılaştırma: `experiments/<id>/reports/cross_analysis/*_colored.pdf`
- Ayrıntılı metrik: `experiments/<id>/reports/full_metrics/<reference>/*_colored.pdf`

İnsan kontrollü ana sunum için önce iSAID `human` full-metric PDF'leri, sonra iSAID cross-analysis PDF'leri gösterilmelidir. SAMRS belgeleri destekleyici referans-affinity bulgusu olarak sunulmalıdır.

## Kritik Uyarılar

- GT-bbox pseudo diagonal `1.0` başarı değildir.
- SAMRS temel etiketi insan GT değildir.
- “Pseudo etiket işe yaramaz” sonucu çıkarılamaz.
- Maske IoU eşik oranları mAP değildir.
- Detector bbox AP ile mask IoU aynı metrik değildir.
- iSAID ve SAMRS skorları tek bir ortalama olarak birleştirilmez.
- Small Vehicle SAM1 pseudo referansında 19 boş maske vardır ve bunlar 0 puanlanır.

## Sıradaki İnsan İşi

Deneysel artifact'lar hazırlandı. Kullanıcının yapacağı ana iş, `paper_writing/PAPER_STRUCTURE.md` ve Overleaf yorumlarındaki fikirleri kendi akademik diliyle yazmak, yazar/kurum bilgilerini doldurmak ve bildirinin sayfa sınırına göre figür/tablo seçmektir.

## Kontrol Komutu

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
```

Ham veriden yeniden hazırlama ve 8 GB VRAM komutları `docs/REPRODUCIBILITY.md` içindedir. Overleaf'in `elektr.cls` ve `elksty.tex` dosyaları dergi şablon projesinden sağlanır; repo bunları yeniden dağıtmaz.
