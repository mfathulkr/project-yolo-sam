# Bildiri Yapısı ve Yazım Planı

## Önerilen Ana Başlık

**When the teacher becomes the test: Self-reference bias in SAM-generated remote-sensing segmentation benchmarks**

Alternatif:

**Do SAM-generated masks fairly evaluate SAM models? A controlled study of reference bias in remote sensing**

İlk başlık daha güçlü ve akılda kalıcıdır; ikinci başlık daha temkinli ve açıklayıcıdır.

## Tek Cümlelik Ana Fikir

Görüntüler, nesneler, promptlar, aday tahminler ve metrikler sabitken yalnız değerlendirme maskesini insan referansından SAM1/SAM2/SAM3 pseudo referansına çevirmek, referansı üreten modeli avantajlı gösterir ve bazı koşullarda insan referansındaki model sıralamasını değiştirir.

## Araştırma Soruları

- **RQ1:** SAM pseudo referansları, insan referansına göre segmentasyon skorlarını ne kadar değiştirir?
- **RQ2:** Her pseudo referans kendi üreticisini sistematik olarak avantajlı hâle getirir mi?
- **RQ3:** Etki yalnız özdeş GT-bbox maskelerinde mi görülür, yoksa YOLO-bbox tahminlerinde de sürer mi?
- **RQ4:** Etki Plane ve Small Vehicle sınıflarında, overlap ve mask-area tabakalarında tutarlı mıdır?
- **RQ5:** İnsan referansıyla düşük anlaşma ve boş pseudo maskeler apparent başarıyı nasıl etkiler?

## Hipotezler

- **H1:** Her SAM öğretmeni kendi pseudo referansında diğer modellerden daha yüksek IoU alır.
- **H2:** Kendi-pseudo eksi insan IoU farkı YOLO-bbox koşulunda da pozitiftir.
- **H3:** Referans seçimi en az bir veri seti/istem koşulunda model sıralamasını değiştirir.
- **H4:** Düşük insan anlaşması veya yüksek boş maske oranı bulunan referansta apparent self-advantage daha büyüktür.

## 6 Sayfalık Yerleşim Önerisi

| Bölüm | Yaklaşık alan | Amaç |
| --- | ---: | --- |
| Abstract + keywords | 0,35 sayfa | Soru, kontrollü tasarım, ana sayılar, uyarı |
| Introduction | 0,80 sayfa | Maliyet motivasyonu, sorun, boşluk, katkılar |
| Literature Review | 0,70 sayfa | SAMRS, imperfect reference, medical ve LLM self-bias |
| Materials and Methods | 1,35 sayfa | Veri, eşleştirilmiş protokol, referanslar, metrik/statistik |
| Results | 1,45 sayfa | Matris, YOLO self-advantage, tabakalar, empty-mask denetimi |
| Discussion | 0,85 sayfa | Anlam, eğitim/test ayrımı, öneriler, sınırlılıklar |
| Conclusion | 0,25 sayfa | Tek sonuç ve pratik öneri |
| Kaynakça | Dergi düzenine bağlı | Ana gövde dışında veya kalan alan |

## Bölüm Bazında Yazılacaklar

### Abstract

Beş cümlelik iskelet:

1. Pixel-level remote-sensing annotation pahalı olduğu için SAM maskeleri ölçeklenebilir etiket kaynağıdır.
2. Ancak aynı model ailesinin ürettiği maskeler bağımsız test referansı olarak kullanılırsa self-reference bias oluşabilir.
3. İki iSAID hedefinde 512'şer görüntü, 17.498 instance, üç aday SAM, dört referans ve iki bbox istemiyle yalnız referansı değiştiren kontrollü deney yaptık.
4. YOLO-bbox paired self-reference artışının Plane'de `+0,224–+0,279`, Small
   Vehicle'da `+0,142–+0,176` olduğunu; 12/12 pseudo koşulda öğretmenin birinci
   geldiğini ve 9/12 koşulda tam sıralamanın değiştiğini özetle.
5. Pseudo etiketler pretraining için yararlı olabilir, fakat bağımsız model değerlendirmesi için insan audit'i, provenance, empty-mask raporu ve leave-one-out/heterojen referans gerekir.

Abstract 300 kelimeyi geçmemeli ve kaynak içermemelidir.

### Introduction

Paragraf 1: Remote-sensing instance segmentationın önemi, küçük/yoğun/yönlü nesneler ve piksel etiketi maliyeti.

Paragraf 2: SAM ve SAMRS gibi veri motorlarının annotation ölçekleme faydası. SAMRS'nin esas kullanımının pretraining olduğunu dengeli şekilde söyle.

Paragraf 3: Problem tanımı. Test referansı adaydan bağımsız değilse ortak hata biçimi agreement skorunu artırabilir. Bunun klasik train/test leakage'den farklı bir measurement-dependence sorunu olduğunu belirt.

Paragraf 4: En doğrudan öncülü açıkça ver. Parikh, Das ve Feragen 2025 aynı
nnU-Net tahminlerini otomatik silver ve uzman-düzeltilmiş gold maskelerle
ölçerek `Biased Ruler` etkisini göstermiştir; Mayıs 2026 devam çalışması bozuk
referansın farkı gizleyebildiğini veya tersine çevirebildiğini gösterir. Bu
genel olgu üzerinde ilk olma iddiası kurma. Literatür boşluğunu yalnız
remote-sensing SAM1/SAM2/SAM3 cross-teacher matrisi, model ranking ve GT/YOLO
bbox kontrolleri olarak tanımla.

Paragraf 5: Katkılar, üç madde:

1. Human/SAM1/SAM2/SAM3 referanslarını aynı dondurulmuş tahminlerle çaprazlayan protokol.
2. GT-bbox identity control ile YOLO-bbox non-identical generalization kontrolü ve source-scene bootstrap.
3. Plane/Small Vehicle ile overlap/area tabakaları, ranking ve empty-mask audit.

### Literature Review

Alt başlık 1: Model-generated segmentation annotations in remote sensing.
iSAID, SAMRS, SOPSeg, SAMST ve ReSAM. SAMST/ReSAM ile pseudo maskenin
refinement sonrası eğitimde kullanılması, bağımsız test referansı kullanımından
ayrılacak.

Alt başlık 2: Imperfect and circular reference standards. Önce Parikh 2025
`Biased Ruler` ve Parikh 2026 label-bias audit; ardından Nichyporuk annotation
style, Vorontsov sistematik/rastgele etiket hatası, Commowick leave-one-out,
SparseGT, AI-collaborative labeling ve Hu et al.'ın tek gold standard
gerektirmeyen çoklu uzman testi.

Alt başlık 3: Self-bias in machine-generated benchmarks. SILENCER ve translation benchmark örneği.

Bu bölümde eğitim pseudo etiketi ile değerlendirme pseudo referansını kesin ayır. “Pseudo etiket kötüdür” sonucu çıkarma.

### Materials and Methods

#### Datasets and targets

- iSAID insan maskeleri.
- Plane: 512 görüntü, 44 sahne, 5.447 instance.
- Small Vehicle: 512 görüntü, 31 sahne, 12.051 instance.
- Her hedef için dört tabaka ve tabaka başına 128 görüntü.
- Overlap ölçütünün bbox kesişimine, low/high ayrımının hedef instance maske alanları toplamı için testten önce dondurulan veri setine özgü eşiğe dayandığını canonical protokolden birebir aktar. Bu alan toplamının mask union olmadığını ve örtüşen pikselin instance başına ayrı sayılabileceğini belirt.

#### Models and prompts

- SAM1 `facebook/sam-vit-huge`.
- SAM2 `facebook/sam2.1-hiera-large`.
- SAM3 local frozen checkpoint.
- PVS'yi ilk kullanımda aç: Promptable Visual Segmentation, uzamsal kutu
  istemiyle belirli bir nesne örneğini segmentler. PCS'yi de aç: Promptable
  Concept Segmentation, metin/görsel örneğe uyan bütün kavram örneklerini arar.
- SAM3 için `Sam3TrackerProcessor` + `Sam3TrackerModel`,
  `multimask_output=False`, `mask_threshold=0.0` ve bbox başına tek çıktı
  kullanıldığını yaz. PCS visual-exemplar arayüzünün neden eşdeğer olmadığını
  bir cümlede açıkla ve SAM3 makalesine atıf ver.
- SAM1/SAM2 için de `multimask_output=False`, `mask_threshold=0.0`, özgün
  boyuta post-process ve frozen checkpoint kullanıldığını belirt.
- İnsan GT bbox ve seed 42 YOLO bbox.
- YOLO confidence eşiğinin testte seçilmediğini; validation'da bbox IoU 0,50
  kuralıyla maksimum F1 noktasında dondurulduğunu belirt.
- YOLO-GT eşleştirmesinin confidence sıralı greedy, bire bir ve bbox IoU
  `>=0.50` olduğunu; eşleşmeyen GT'nin sıfır maske aldığını yaz.
- YOLO detector sonuçlarını control table olarak ver; mask IoU ile bbox AP'yi
  karıştırma.

#### Reference construction

- Human iSAID masks.
- Her SAM'in aynı GT bbox ile ürettiği frozen mask.
- Hiçbir boş maskenin filtrelenmediğini yaz.
- GT diagonal = kapsama-duyarlı identity control; dolu kendi-referansı 1,0,
  bilinen nesnedeki boş kendi-referansı 0,0 alır. Ana kanıt YOLO koşuludur.

#### Metrics and statistics

- Instance-macro IoU ana metrik.
- Dice, precision, recall ve threshold success full raporlarda tamamlayıcı.
- Scene-clustered 10.000 bootstrap, yüzde 95 CI.
- Paired delta, ranking change, teacher advantage, human agreement, empty rate.

#### Reproducibility

- Sabit seed 42 detector training/inference ve bootstrap için; tek detector
  seed'inin sınırlılık olduğunu ayrıca yaz.
- Kesin model revision ve checkpoint SHA-256 değerleri; SAM3 PVS arayüzü;
  exact canonical environment sürümleri.
- Source-scene-safe split, dondurulmuş area/overlap eşikleri ve 4 x 128 seçim.
- Pseudo referansların yalnız GT bbox frozen prediction RLE'sinden birebir
  üretildiği; boş maskelerin filtrelenmediği.
- `known_positive_instance`: bilinen nesnedeki boş pseudo referansın boş
  tahminle eşleşse bile sıfır puanlanması.
- Instance-macro ortalama, kaynak-sahne kümeli 10.000 bootstrap ve image-union
  metriğinin yalnız ikincil olması.
- Effective config, input/output fingerprint, run lock ve report/paper
  manifestlerinin rolü.
- Ana yöntem metninde kısa anlat; tam operasyonel sözleşmeyi
  `docs/REPRODUCIBILITY_FIELD_GUIDE.md` olarak supplementary material içinde
  göster.

### Results

Sonuç sırası önemlidir:

1. **Detector control:** Plane detector çok güçlü, Small Vehicle detector daha zayıf. Böylece mask sonuçlarının prompt kalitesi bağlamını ver.
2. **Human-reference baseline:** İnsan referansında her hedef/istem için model sıralamasını yaz.
3. **Identity control:** GT diagonalın kapsama-duyarlı değerini ver; bunu başarı
   olarak sunma.
4. **Main finding:** YOLO koşulunda her pseudo referans kendi öğretmenini
   birinci yaptı. Paired self-reference artışları Plane'de SAM1/SAM2/SAM3 için
   `+0,276 / +0,279 / +0,224`, Small Vehicle'da
   `+0,176 / +0,163 / +0,142` oldu; bütün %95 CI'lar sıfırın üzerinde kaldı.
5. **Reference-dependent ranking:** Hangi ranking'lerin insan sırasından değiştiğini ver.
6. **Stratified robustness:** Dört overlap/area tabakasında self-reference delta değerlerinin pozitif kaldığını figürle göster.
7. **Reference integrity:** İnsan anlaşması, gerçek RLE empty-mask oranları ve
   kaynak-manifest soy zinciri.
8. **Qualitative evidence:** Tüm hedef instance'ları içeren seçilmiş median-difficulty örnekler.

Sonuç bölümünde neden olduğuna dair uzun açıklama yapma; yorum Discussion'a bırakılmalı.

### Discussion

Paragraf 1: Sonuçların anlamı. Pseudo referans agreement ölçer, mutlak accuracy kanıtlamaz.

Paragraf 2: Olası mekanizma. Aynı model/ailenin sınır, boş maske, over/under-segmentation ve küçük nesne hata biçimleri korelasyonludur.

Paragraf 3: SAMRS için dengeli yorum. Büyük ölçekli pretraining kaynağı olarak yararlı olabilir; bağımsız benchmark olarak doğrudan kullanım dikkat gerektirir.

Paragraf 4: Tıp ve LLM literatürüyle bağlantı. Parikh'in Biased Ruler deneyi
bizim en yakın öncülümüzdür. Farkı net kur: onlar demografik fairness gap ve
tek nnU-Net silver referansı inceler; biz model üreticisi affinity'sini üç SAM
öğretmeniyle, model ranking üzerinden ve remote sensing'de çaprazlarız.
Imperfect reference, circular consensus ve self-bias aynı bağımsızlık ilkesine
işaret eder.

Paragraf 5: Pratik protokol:

- küçük kör insan audit alt kümesi;
- teacher provenance;
- empty/failure rate;
- cross-teacher matrix;
- consensus kullanılacaksa leave-one-model-out;
- nihai testte insan veya bağımsız expert consensus.

Paragraf 6: Sınırlılıklar. İki sınıf, aynı iSAID kaynak ailesi, tek detector seed, GT-box tabanlı pseudo üretim, human inter-rater maskeleri yok, BIoU yeni dört referans küpünde hesaplanmadı.

Özgünlük sınırı: “segmentasyonda otomatik referans yanlılığını ilk gösteren
çalışma” denmeyecek. Güvenli iddia, `to our knowledge` ile remote-sensing
instance segmentationda üç SAM pseudo-reference üreticisini üç SAM adayıyla,
human kontrol altında çaprazlayan ilk kontrollü çalışma olmasıdır.

### Conclusion

Üç cümle yeterlidir:

1. Referans üreticisi model sıralamasını değiştirebilir.
2. Etki farklı promptta da sürdüğü için yalnız özdeş maske tautology'si değildir.
3. Model üretimli maskeler eğitimde kullanılabilir; değerlendirmede bağımsız insan doğrulaması ve reference-bias audit'i gerekir.

## Ana Sayılar

- Sabit deney kapsamı: 1.024 görüntü ve 17.498 instance.
- İnsan referansı sıralaması iki hedef ve iki istemde
  `SAM3 > SAM1 > SAM2`dir.
- 12/12 pseudo koşulda referansı üreten model birinci; 9/12 koşulda tam model
  sırası insan referansına göre farklıdır.
- YOLO teacher advantage Plane'de `0,127–0,139`, Small Vehicle'da
  `0,071–0,098` aralığındadır.
- İnsan-pseudo agreement Plane'de SAM1/SAM2/SAM3 için
  `0,653 / 0,629 / 0,700`, Small Vehicle'da
  `0,658 / 0,645 / 0,698`dir.
- Empty pseudo: Plane SAM1/SAM2/SAM3 `0/0/0`; Small Vehicle `19/0/0`.

## Yazımda Kullanılmaması Gereken İfadeler

- “SAMRS bir boka yaramıyor.”
- “Pseudo etiket kullanmak overfit'tir.”
- “AI ile etiketlenmiş bütün veri leakage içerir.”
- “SAM1 en iyi modeldir.”
- “İnsan etiketi yüzde 100 doğrudur.”

Bunların yerine ölçülen koşulu ve referansı açıkça söyle.
