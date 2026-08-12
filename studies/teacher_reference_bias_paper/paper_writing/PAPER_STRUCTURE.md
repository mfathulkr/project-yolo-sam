# Altı Sayfalık Bildiri Yapısı

## Önerilen Başlık

**When the teacher becomes the test: Reference affinity in SAM-generated remote-sensing segmentation benchmarks**

`Bias` kelimesi kullanılabilir; ancak genel biased-ruler olgusunda ilk çalışma iddiası kurulmayacak. Özgünlük, remote sensing'de üç SAM öğretmenini üç SAM adayıyla çaprazlayan, insan kontrollü ve tabakalı deney tasarımıdır.

## Ana Fikir

Görüntü, instance, bbox istemi ve aday tahmini sabitken yalnız değerlendirme maskesini değiştirmek skor ve model sıralamasını değiştirebilir. Aynı model ailesinin ürettiği referans ortak sınır/hata stilini ödüllendirebilir.

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
| Results | 1,45 | Detector control, matris, paired effect, tabakalar, SAMRS integrity |
| Discussion | 0,85 | Anlam, eğitim/test ayrımı, öneriler, sınırlılıklar |
| Conclusion | 0,20 | Üç kısa sonuç cümlesi |

## Abstract Notları

1. Remote-sensing piksel anotasyonu pahalıdır ve SAM ölçeklenebilir etiket üretir.
2. Aynı model ailesinden maskeyi test referansı yapmak ölçümü bağımlı kılabilir.
3. Dört 512-görüntülük deneyde SAM1/2/3 tahminlerini temel ve üç model referansıyla çaprazladık.
4. iSAID YOLO-bbox own-reference artışları Plane'de `+0.224–+0.279`, Small Vehicle'da `+0.142–+0.176`; bütün `%95` GA sıfırın üstünde.
5. SAMRS published/reproduced SAM1 anlaşması `0.991/0.998`; bu maskelerin SAM1-benzeri olduğunu, insan doğruluğunu değil gösterir.
6. Model üretimli maskeler eğitim için kullanılabilir; bağımsız değerlendirme için insan audit'i ve reference-affinity denetimi gerekir.

## Introduction

- Küçük, yoğun ve yönlü remote-sensing nesneleri; piksel etiketi maliyeti.
- SAMRS gibi veri motorlarının faydası ve asıl pretraining motivasyonu.
- Problem: referans cetveli aday modelden bağımsız değilse hata korelasyonu agreement skorunu yükseltebilir.
- En yakın öncül: Parikh et al. 2025, aynı nnU-Net tahminlerini silver ve uzman-düzeltilmiş gold maskelerle ölçerek Biased Ruler etkisini gösterdi.
- Özgün katkılar: üç öğretmen × üç aday matrisi; GT identity ile YOLO non-identical kontrol ayrımı; iki sınıf/two dataset family/four strata.

## Literature Review

1. Remote sensing model-generated masks: iSAID, SAMRS, SAMST, ReSAM, SOPSeg.
2. Imperfect reference standards: Parikh 2025/2026, annotation style, leave-one-out consensus, medical silver/gold standards.
3. Machine-generated benchmark self-affinity: evaluator/benchmark producer dependence.

Training pseudo label ile evaluation pseudo reference kesin ayrılacak.

## Materials and Methods

### Datasets

- Dört deney ve Table 1.
- Her deney 512 görüntü, dört eşit 128 tabaka.
- Kaynak sahne güvenli split; area eşikleri model sonuçlarından önce donduruldu.

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
- Paired delta ve scene-clustered 10.000 bootstrap `%95` GA.
- Ranking, teacher advantage, reference agreement, empty rate.

## Results Sırası

1. Detector control: dört detectorün bbox AP/precision/recall bağlamı.
2. iSAID human-reference baseline.
3. GT diagonal identity control; başarı diye sunulmayacak.
4. Figure 2: dört YOLO model-reference matrisi.
5. Figure 3/Table 3: iSAID paired own-reference etkisi.
6. Figure 5/Table S1: dört tabakada etkinin yönü.
7. Figure 4/Table 4: SAMRS published/reproduced SAM1 integrity.
8. Nitel örnekler supplementary full-metric belgelerde.

## Discussion

- Yüksek pseudo agreement bağımsız accuracy değildir.
- Olası mekanizma: ortak boundary, over/under-segmentation ve empty-mask davranışı.
- SAMRS'nin pretraining yararlılığı reddedilmez; bağımsız benchmark olarak kullanım riski tartışılır.
- Parikh biased-ruler bulgusu genel öncüldür; bizim katkımız teacher-family affinity ve model ranking'dir.
- Öneri: küçük kör insan audit seti, teacher provenance, empty rate, cross-teacher matrix, leave-one-model-out consensus.
- Sınırlılıklar: iki sınıf, iki veri ailesi, tek detector seed, GT-bbox teacher prompt, SAMRS insan maskesi yok, boundary IoU kanonik küpte yok.

## Ana Figür ve Tablolar

- `assets/figures/figure_1_study_design.pdf`
- `assets/figures/figure_2_model_reference_iou_matrix.pdf`
- `assets/figures/figure_3_isaid_reference_effect_with_ci.pdf`
- `assets/figures/figure_4_samrs_reference_integrity.pdf`
- `assets/figures/figure_5_stratified_self_reference_effect.pdf`
- `assets/tables/table_1_experimental_design.tex`
- `assets/tables/table_2_baseline_reference_results.tex`
- `assets/tables/table_3_own_reference_effect.tex`
- `assets/tables/table_4_samrs_reference_integrity.tex`
- `assets/tables/table_5_detector_control.tex`
- `assets/tables/table_s1_stratified_reference_effect.tex`

## Kullanılmayacak İddialar

- “Otomatik referans yanlılığını ilk kez keşfettik.”
- “SAMRS işe yaramaz.”
- “Pseudo etiketle eğitim yapmak leakage'dir.”
- “SAM1 insan ground truth'ta en iyi modeldir.”
- “GT diagonal 1.0 olduğu için model kusursuzdur.”
