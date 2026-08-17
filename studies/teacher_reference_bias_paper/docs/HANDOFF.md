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

iSAID insan kontrolünde aynı YOLO-bbox tahmini üç SAM referansıyla ölçüldü. Aşağıdaki değer, modelin kendi ürettiği etiketteki IoU'sundan diğer iki SAM etiketindeki ortalama IoU çıkarılarak hesaplanan **ek IoU** değeridir:

| Deney | SAM1 | SAM2 | SAM3 |
| --- | ---: | ---: | ---: |
| iSAID Plane | +0.128 | +0.124 | +0.141 |
| iSAID Small Vehicle | +0.098 | +0.074 | +0.075 |

Her satırdaki pozitif değer, modelin kendi ürettiği maske referans olarak kullanıldığında daha yüksek puan aldığını gösterir. Kaynak-sahne kümeli güven aralıkları bütün iSAID koşullarında sıfırın üzerindedir; ayrıntıları analiz CSV'lerinde saklanır. Bu karşılaştırmalar ilk sonuçlar görüldükten sonra geliştirildiği ve çoklu karşılaştırma düzeltmesi uygulanmadığı için destekleyici bulgu olarak yorumlanmalıdır.

Önceki `pseudo − human` artışları Plane için `+0.276/+0.279/+0.224`, Small Vehicle için `+0.176/+0.163/+0.142` değerindedir; bunlar yararlı betimleyici etkidir fakat tek başına doğrudan affinity testi değildir.

SAMRS yayımlanmış referans ile yeniden üretilmiş SAM1 referansı arasındaki Avg IoU:

- Plane: `0.991`
- Small Vehicle: `0.998`

Bu sonuç yayımlanmış SAMRS maskelerinin SAM1'e çok yakın olduğunu destekler; insan doğruluğu göstermez.

## Hangi PDF'ler Gösterilecek?

- Ana YOLO-bbox özeti: `analysis/main_cross_analysis_colored.pdf`
- Aynı formatta GT-bbox özeti: `analysis/main_cross_analysis_gt_bbox_colored.pdf`
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
- iSAID ve SAMRS farklı anotasyon ürünleri olsa da DOTA kökenli görüntüleri kısmen paylaşır; dört deney bağımsız tekrar değildir.
- Detector testleri yalnız hedef-pozitif 512 görüntüdür; AP değerleri resmi benchmark değildir.
- Maske ortalamasına eşleşmeyen detector yanlış pozitifleri eklenmez; bu nedenle sonuç tam uçtan uca COCO mask AP değildir.
- Small Vehicle SAM1 pseudo referansında 19 boş maske vardır ve bunlar 0 puanlanır.
- Small Vehicle detectorlerinin tarihsel resume başlangıç byte'ları yoktur; final ağırlık ve değerlendirme provenance'ı korunmuştur.
- Aynı-üretici etkisi yalnız aynı dondurulmuş checkpoint için ölçülmüştür; farklı checkpoint/seed veya model ailesi genellemesi test edilmemiştir.
- GT-bbox pseudo üretimi insan/yayımlanmış kutu lokalizasyonunu kullanır; ölçülen fark yalnız maske sınırı değildir ve deney tam otomatik etiketleme hattı değildir.

## Sıradaki İnsan İşi

Deneysel artifact'lar hazırlandı. Kullanıcının yapacağı ana iş, `paper_writing/PAPER_STRUCTURE.md` ve Overleaf yorumlarındaki fikirleri kendi akademik diliyle yazmak, yazar/kurum bilgilerini doldurmak ve bildirinin sayfa sınırına göre figür/tablo seçmektir.

Bilimsel veri bağımlılığı, ortak görüntüler ve exploratory insan–SAMRS anlaşması `docs/DEEP_SCIENTIFIC_AUDIT.md` içinde kayıtlıdır; bildiri sınırlılıkları yazılırken bu dosya atlanmamalıdır.

## Kontrol Komutu

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
```

Ham veriden yeniden hazırlama ve 8 GB VRAM komutları `docs/REPRODUCIBILITY.md` içindedir. Overleaf'in `elektr.cls` ve `elksty.tex` dosyaları dergi şablon projesinden sağlanır; repo bunları yeniden dağıtmaz.
