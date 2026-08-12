# Bildiri Tablo ve Figür Planı

## Ana Metin İçin Önerilen Paket

### Figure 1: Study design

Dosya: `paper/assets/figures/figure_1_study_design.pdf`

Kullanım: Methods başında. Yalnız referansın değiştiğini, tahminlerin sabit olduğunu tek bakışta anlatır.

### Table 1: Experimental design

Dosya: `paper/assets/tables/table_1_experimental_design.tex`

Kullanım: Datasets and targets. İki target için görüntü, sahne, instance ve deney çaprazını verir.

### Figure 2: Model-reference IoU matrix

Dosya: `paper/assets/figures/figure_2_model_reference_iou_matrix.pdf`

Kullanım: Ana sonuç figürü. Her dataset ve bbox koşulunda satır aday modeli, sütun referansı gösterir. Diagonal hücrelerin GT koşulunda identity control olduğu caption'da yazılmalıdır.

Önerilen caption:

> Mean instance IoU under human and SAM-generated references. GT-box diagonal cells are identity controls because the evaluated mask is identical to the reference-generating output. YOLO-box panels use non-identical candidate predictions and expose persistent teacher affinity.

### Figure 3: Paired pseudo-minus-human effect

Dosya: `paper/assets/figures/figure_3_reference_effect_with_ci.pdf`

Kullanım: Ana istatistik figürü. Eşleştirilmiş farkları source-scene-clustered yüzde 95 CI ile gösterir.

### Table 3: Self-reference effect

Dosya: `paper/assets/tables/table_3_self_reference_effect.tex`

Kullanım: Figure 3'ün kesin sayısal karşılığı. Altı model/target/prompt kombinasyonunun human ve own-pseudo IoU değerleri ile CI bilgisi.

### Figure 4: Reference integrity

Dosya: `paper/assets/figures/figure_4_reference_integrity.pdf`

Kullanım: Human agreement ve empty rate'i birlikte verir. SAM3 Small Vehicle failure mode'unu açıklar.

### Figure 5: Stratified effect

Dosya: `paper/assets/figures/figure_5_stratified_self_reference_effect.pdf`

Kullanım: Overlap/no-overlap ve low/high mask area duyarlılık analizi. Yer yetmezse supplement'e taşınabilir, fakat bulgunun tabakalar boyunca sürdüğünü gösterdiği için değerlidir.

## Supplement / Appendix

- `table_2_model_reference_iou.tex`: Figure 2'nin bütün sayıları.
- `table_4_teacher_advantage.tex`: Öğretmen ile diğer iki model ortalaması arasındaki fark.
- `table_5_reference_integrity.tex`: Human agreement ve exact empty counts.
- `table_6_detector_control.tex`: YOLO bbox AP ve detection precision/recall.
- `table_s1_stratified_self_reference_effect.tex`: Bütün tabaka sayıları.
- Önerilen `Table S2 - Reproducibility contract`: model revision/checkpoint
  hash'i, SAM arayüzü, mask threshold, multimask ayarı, detector seed,
  validation confidence seçimi, bbox eşleştirme IoU'su, bootstrap ve boş maske
  politikasını tek tabloda özetlemeli. Bu tablo yazım aşamasında
  `docs/REPRODUCIBILITY_FIELD_GUIDE.md` içindeki dondurulmuş değerlerden
  üretilmeli; elle yeni değer eklenmemeli.
- `figure_q_isaid_plane_reference_examples.pdf`: Plane için dört tabakadan median-difficulty nitel örnekler.
- `figure_q_isaid_small_vehicle_reference_examples.pdf`: Small Vehicle için dört tabakadan median-difficulty nitel örnekler.

Nitel figürlerin her satırında görüntüdeki bütün hedef instance'lar kullanılmıştır; tek nesne seçilmemiştir. İlk sütunda bütün GT kutuları görünür. Örnekler ortalama insan-referans anlaşmasının tabaka medyanına en yakın görüntülerden deterministik seçilir; görsel olarak en iyi veya en kötü örnek cherry-pick edilmez.

## Altı Sayfaya Sığdırma

Ana metinde en fazla dört görsel öğe önerilir:

1. Figure 1 study design.
2. Figure 2 model-reference matrix.
3. Figure 3 paired effect.
4. Figure 4 reference integrity veya Table 3.

Figure 5 ve qualitative örnekler supplement'e alınabilir. Table 1 çok küçük olduğu için yöntem metnine gömülebilir.

## Renk ve Erişilebilirlik

- SAM1: mavi.
- SAM2: yeşil.
- SAM3: turuncu.
- İnsan referansı: mor/pembe.
- Heatmap'lerde sayılar hücrelerin içinde ayrıca yazılıdır; yorum yalnız renge bağlı değildir.
- PDF figürler vektör metin içerir; PNG dosyaları hızlı önizleme içindir.

## Reproducibility

Tablo ve figürler şu komutla yeniden üretilir:

```bash
.venv/bin/python studies/teacher_reference_bias_multiteacher_v1_512/scripts/generate_paper_assets.py
```

`paper/assets/manifest.json`, her çıktının byte boyutunu ve SHA-256 hash'ini içerir.
Bilimsel koşunun tamamını belirleyen model, veri, detector, pseudo-reference,
metrik ve istatistik kararları
[`REPRODUCIBILITY_FIELD_GUIDE.md`](REPRODUCIBILITY_FIELD_GUIDE.md) içinde
toplanmıştır.
