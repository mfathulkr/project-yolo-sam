# Altı Sayfalık Bildiri Yapısı

## Önerilen Başlık

**When the teacher becomes the test: Reference affinity in SAM-generated remote-sensing segmentation benchmarks**

`Bias` kelimesi kullanılabilir; ancak genel biased-ruler olgusunda ilk çalışma iddiası kurulmayacak. Özgünlük, remote sensing'de üç SAM öğretmenini üç SAM adayıyla çaprazlayan, insan kontrollü ve tabakalı deney tasarımıdır.

## Ana Fikir

Görüntü, instance, bbox istemi ve aday tahmini sabitken yalnız değerlendirme maskesini değiştirmek skor ve model sıralamasını değiştirebilir. Bu çalışma aynı dondurulmuş üretici checkpoint'in kendi referansındaki avantajını ölçer; model ailesi düzeyinde genelleme yapmaz.

## Araştırma Soruları

- RQ1: SAM pseudo referansları iSAID insan referansına göre skoru ne kadar değiştirir?
- RQ2: Referansı üreten model kendi referansında sistematik avantaj kazanır mı?
- RQ3: Etki GT-bbox identity control dışında, non-identical YOLO-bbox koşulunda da sürer mi?
- RQ4: Plane/Small Vehicle ve dört overlap × area tabakasında yön tutarlı mı?
- RQ5: SAMRS yayımlanmış maskeleri yeniden üretilmiş SAM1 maskelerine ne kadar yakındır?

## Sayfa Planı

| Bölüm | Yaklaşık alan | İçerik |
| --- | ---: | --- |
| Abstract | 0,30 | Problem, dört deney, ana iSAID deltas, dengeli sonuç |
| Introduction | 0,75 | Etiket maliyeti, ölçüm bağımsızlığı sorunu, katkılar |
| Literature Review | 0,65 | SAMRS, biased ruler, imperfect/circular reference |
| Materials and Methods | 1,35 | Dört deney, promptlar, referanslar, metrik/statistik |
| Results | 1,45 | Detector control, kendi-etiketi etkisi, referansa bağlı sıralama, tabakalar ve SAMRS bütünlüğü |
| Discussion | 0,85 | Anlam, eğitim/test ayrımı, öneriler, sınırlılıklar |
| Conclusion | 0,20 | Üç kısa sonuç cümlesi |

## Abstract Notları

1. Remote-sensing piksel anotasyonu pahalıdır ve SAM ölçeklenebilir etiket üretir.
2. Aynı dondurulmuş checkpoint'in ürettiği maskeyi test referansı yapmak ölçümü bağımlı kılabilir.
3. Dört 512-görüntülük deneyde SAM1/2/3 tahminlerini temel ve üç model referansıyla çaprazladık.
4. iSAID YOLO-bbox sonuçlarında modelin kendi ürettiği etikete göre IoU'su, diğer iki SAM etiketindeki ortalama IoU'sundan Plane'de `0.124–0.141`, Small Vehicle'da `0.074–0.098` daha yüksektir. Güven aralıkları ayrıntılı analiz dosyalarında saklanır.
5. SAMRS published/reproduced SAM1 anlaşması `0.991/0.998`; bu maskelerin SAM1-benzeri olduğunu, insan doğruluğunu değil gösterir.
6. Model üretimli maskeler eğitim için kullanılabilir; bağımsız değerlendirme için insan audit'i ve reference-affinity denetimi gerekir.

## Introduction

- Küçük, yoğun ve yönlü remote-sensing nesneleri; piksel etiketi maliyeti.
- SAMRS gibi veri motorlarının faydası ve asıl pretraining motivasyonu.
- Problem: referans cetveli aday modelden bağımsız değilse hata korelasyonu agreement skorunu yükseltebilir.
- En yakın öncül: Parikh et al. 2025, aynı nnU-Net tahminlerini silver ve uzman-düzeltilmiş gold maskelerle ölçerek Biased Ruler etkisini gösterdi.
- Özgün katkılar: üç etiket üreticisi × üç değerlendirilen model matrisi; her modelin kendi etiketinde kazandığı ek IoU; GT bbox matematiksel eşitlik kontrolü ile YOLO bbox ana değerlendirmesinin ayrılması; iki sınıf, iki DOTA kökenli anotasyon ürünü ve dört sahne grubu.

## Literature Review

1. Remote sensing model-generated masks: iSAID, SAMRS, SAMST, ReSAM, SOPSeg.
   Remote SAMsing, model ağırlığını değiştirmeden tile ölçeği, multi-pass,
   contextual padding ve tile merge'in etkisini gösteren preprocessing bağlamı
   olarak kullanılacak; teacher-reference bias kanıtı gibi sunulmayacak.
   SAMIX ve Boxes2Pixels, pseudo maskeyi eğitim sinyali olarak seçen/düzelten
   fakat bağımsız test referansı saymayan güncel karşı örnekler olarak anılacak.
2. Imperfect reference standards: Parikh 2025/2026, annotation style, leave-one-out consensus, medical silver/gold standards.
3. Machine-generated benchmark self-affinity: evaluator/benchmark producer dependence.

Training pseudo label ile evaluation pseudo reference kesin ayrılacak.

## Materials and Methods

### Datasets

- Dört deney ve Table 1.
- Her deney 512 görüntü, dört eşit 128 tabaka.
- Kaynak sahne güvenli özel split; iSAID resmi leaderboard test protokolü değildir. Area eşikleri model sonuçlarından önce donduruldu.
- Dört test hedef-pozitiftir; iSAID/SAMRS ve Plane/Small Vehicle testleri kısmen aynı DOTA görüntülerini içerir. Bunlar bağımsız dört replikasyon olarak sunulmayacak.

### Models and prompts

- SAM1 ViT-H, SAM2.1 Hiera-Large, SAM3 local frozen.
- GT bbox ve seed-42 YOLO bbox.
- SAM3 PVS/PCS farkı açıkça yazılacak; kullanılan yol tracker PVS.
- `multimask_output=False`, `mask_threshold=0.0`, orijinal 1024 çözünürlüğe post-process.
- Detector confidence validation'da seçildi; testte ayarlanmadı.

### References

- iSAID: human + SAM1/2/3 pseudo.
- SAMRS: published SAM-derived + reproduced SAM1 + SAM2/3 pseudo.
- Empty mask filtrelenmedi.
- GT diagonal identity control; ana kanıt YOLO-bbox.

### Metrics

- Instance-macro IoU ana metrik.
- Dice, precision, recall ve threshold success supplementary full reports.
- Temel referanstan modelin kendi etiketine geçildiğindeki IoU değişimi.
- Ana okunabilir metrik: `Kendi Etiketiyle IoU − Diğer İki SAM Etiketindeki Ortalama IoU`. Pozitif değer, modelin kendi ürettiği etikette ek puan aldığını gösterir.
- Kaynak-sahne kümeli 10.000 bootstrap güven aralıkları ve ikincil istatistiksel kontrol hesaplanır; kalabalık ana tablolara basılmaz, ayrıntılı analiz CSV'lerinde tutulur.
- Kontrastlar sonuçlar görüldükten sonra geliştirilmiştir; preregistered/confirmatory değildir, çoklu karşılaştırma düzeltmesi yoktur ve farklı checkpoint/seed genellemesi yapılmaz.
- Ranking, teacher advantage, reference agreement, empty rate.

## Results Sırası

1. Detector control: dört detectorün bbox AP/precision/recall bağlamı.
2. iSAID human-reference baseline.
3. GT diagonal identity control; başarı diye sunulmayacak.
4. Figure 2: yalnız insan kontrollü iSAID'da aynı dondurulmuş tahminin diğer iki SAM etiketindeki ortalama IoU'su ile kendi modelinin etiketindeki IoU'su. Bağlı iki nokta ve yanındaki `Ek IoU` ana bulguyu doğrudan gösterir.
5. Figure 3: dört deneyde değerlendirme referansı değiştiğinde en yüksek skoru alan model. Tam 3 × 4 matris yerine yalnız karar açısından gerekli kazanan model ve skoru gösterilir.
6. Table 4: temel referans IoU'sundan kendi etiketteki IoU'ya geçiş.
7. Figure 4: iSAID'da aynı ek IoU'nun dört overlap × area grubundaki yönü ve yüzde 95 güven aralığı. Tam sayısal döküm yalnız supplementary Table S1'de tutulur.
8. Table 5: SAMRS published/reproduced SAM1 bütünlüğü. Yalnız iki değer bulunduğu için ayrıca grafik üretilmez.
9. Nitel örnekler supplementary full-metric belgelerde.

## Discussion

- Yüksek pseudo agreement bağımsız accuracy değildir.
- Olası mekanizma: ortak boundary, over/under-segmentation ve empty-mask davranışı.
- SAMRS'nin pretraining yararlılığı reddedilmez; bağımsız benchmark olarak kullanım riski tartışılır.
- Parikh biased-ruler bulgusu genel öncüldür; bizim katkımız aynı dondurulmuş producer-checkpoint affinity ve model ranking analizidir.
- Öneri: küçük kör insan audit seti, teacher provenance, empty rate, cross-teacher matrix, leave-one-model-out consensus.
- Sınırlılıklar: iki sınıf, ortak DOTA kökenli iki anotasyon ürünü ve deneyler arası görüntü bağımlılığı; hedef-pozitif seçilmiş detector testi; tek detector seed; insan/yayın lokalizasyonu kullanan GT-bbox teacher prompt; tam otomatik etiketleme olmaması; aynı checkpoint dışına genellenememe; post-hoc exploratory kontrastlar ve çoklu karşılaştırma düzeltmesi olmaması; SAMRS insan maskesi yok; eşleşmeyen detector yanlış pozitifleri mask ortalamasında yok; boundary IoU kanonik küpte yok; iki resumed Small Vehicle detector eğitiminin başlangıç checkpoint byte'ları korunmamış.

## Ana Figür ve Tablolar

- `assets/figures/figure_1_study_design.pdf`
- `assets/figures/figure_2_isaid_own_label_comparison.pdf`
- `assets/figures/figure_3_reference_dependent_model_selection.pdf`
- `assets/figures/figure_4_stratified_own_label_extra_iou.pdf`
- `assets/tables/table_1_experimental_design.tex`
- `assets/tables/table_2_baseline_reference_results.tex`
- `assets/tables/table_3_direct_teacher_affinity.tex`
- `assets/tables/table_4_raw_own_reference_effect.tex`
- `assets/tables/table_5_samrs_reference_integrity.tex`
- `assets/tables/table_6_detector_control.tex`
- `assets/tables/table_s1_stratified_reference_effect.tex`

Ana metin için önerilen seçim dört figür ile Table 1, 2, 4, 5 ve 6'dır; toplam
dokuz görsel öğe olur. Figure 2 kullanılırsa aynı sayıları yineleyen Table 3 ana
metne eklenmeyecek, yalnız kesin değer gerektiren supplementary/çalışma çıktısı
olarak tutulacaktır. Benzer biçimde Figure 4'ün ayrıntılı sayıları Table S1'de
kalacaktır. Böylece derginin ana metinde en fazla 10 figür+tablo sınırı ve aynı
veriyi tablo ile figürde tekrarlamama ilkesi korunur.

Figürler 16 cm nihai genişlikte, beyaz zeminde ve vektör PDF olarak üretilir.
Grafik içi tipografi derginin istediği Times ailesine metrik uyumlu Liberation
Serif ile hazırlanır; renklerin yanında marker, doğrudan etiket ve panel harfi
kullanıldığı için gri baskıda da anlam korunur. PNG dosyaları yalnız görsel QA
içindir; Overleaf'e PDF dosyaları eklenmelidir.

## Figure Caption Taslakları

- **Figure 1. Controlled cross-reference evaluation protocol.** The image,
  target instance, bounding-box prompt, and frozen model prediction remain
  fixed while only the evaluation reference mask changes. Extra IoU is the
  model's IoU on its own generated labels minus its mean IoU on labels
  generated by the other two SAM models.
- **Figure 2. Own-label score increase on the human-controlled iSAID
  experiments.** Open markers show each frozen model's mean IoU on labels from
  the other two SAM models; filled markers show IoU on labels generated by the
  evaluated model itself. Numbers report the paired difference under YOLO-box
  prompting.
- **Figure 3. Reference-dependent model selection.** Each cell reports the
  frozen model with the highest mean instance IoU for one evaluation
  reference. Human iSAID references select SAM3, whereas each SAM-generated
  reference selects its producer; the SAM-derived published SAMRS reference
  selects SAM1.
- **Figure 4. Own-label score increase across iSAID scene strata.** Markers
  report extra IoU under YOLO-box prompting and horizontal bars report 95\%
  confidence intervals from source-scene clustered bootstrap resampling. The
  effect remains positive across overlap and mask-area groups.

## Kullanılmayacak İddialar

- “Otomatik referans yanlılığını ilk kez keşfettik.”
- “SAMRS işe yaramaz.”
- “Pseudo etiketle eğitim yapmak leakage'dir.”
- “SAM1 insan ground truth'ta en iyi modeldir.”
- “GT diagonal 1.0 olduğu için model kusursuzdur.”
