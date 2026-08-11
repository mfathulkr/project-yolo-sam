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
4. Her öğretmen kendi pseudo referansında birinci oldu; YOLO isteminde öğretmen avantajı Plane için 0,113–0,138, Small Vehicle için 0,179–0,299 IoU oldu; SAM3 Small Vehicle referanslarının yüzde 44,4'ü boştu.
5. Pseudo etiketler pretraining için yararlı olabilir, fakat bağımsız model değerlendirmesi için insan audit'i, provenance, empty-mask raporu ve leave-one-out/heterojen referans gerekir.

Abstract 300 kelimeyi geçmemeli ve kaynak içermemelidir.

### Introduction

Paragraf 1: Remote-sensing instance segmentationın önemi, küçük/yoğun/yönlü nesneler ve piksel etiketi maliyeti.

Paragraf 2: SAM ve SAMRS gibi veri motorlarının annotation ölçekleme faydası. SAMRS'nin esas kullanımının pretraining olduğunu dengeli şekilde söyle.

Paragraf 3: Problem tanımı. Test referansı adaydan bağımsız değilse ortak hata biçimi agreement skorunu artırabilir. Bunun klasik train/test leakage'den farklı bir measurement-dependence sorunu olduğunu belirt.

Paragraf 4: Literatür boşluğu. Tıpta imperfect reference/circular consensus, LLM'lerde self-bias çalışılmış olsa da remote-sensing SAM maskelerinde çok öğretmenli kontrollü çapraz matris bulunmadığını ihtiyatlı söyle.

Paragraf 5: Katkılar, üç madde:

1. Human/SAM1/SAM2/SAM3 referanslarını aynı dondurulmuş tahminlerle çaprazlayan protokol.
2. GT-bbox identity control ile YOLO-bbox non-identical generalization kontrolü ve source-scene bootstrap.
3. Plane/Small Vehicle ile overlap/area tabakaları, ranking ve empty-mask audit.

### Literature Review

Alt başlık 1: Model-generated segmentation annotations in remote sensing.
iSAID, SAMRS, SOPSeg, SAMST ve ReSAM. SAMST/ReSAM ile pseudo maskenin
refinement sonrası eğitimde kullanılması, bağımsız test referansı kullanımından
ayrılacak.

Alt başlık 2: Imperfect and circular reference standards. Commowick
leave-one-out, medical AI reference bias, SparseGT, AI-collaborative labeling
ve Hu et al.'ın tek gold standard gerektirmeyen çoklu uzman testi.

Alt başlık 3: Self-bias in machine-generated benchmarks. SILENCER ve translation benchmark örneği.

Bu bölümde eğitim pseudo etiketi ile değerlendirme pseudo referansını kesin ayır. “Pseudo etiket kötüdür” sonucu çıkarma.

### Materials and Methods

#### Datasets and targets

- iSAID insan maskeleri.
- Plane: 512 görüntü, 44 sahne, 5.447 instance.
- Small Vehicle: 512 görüntü, 31 sahne, 12.051 instance.
- Her hedef için dört tabaka ve tabaka başına 128 görüntü.
- Overlap ölçütünün bbox kesişimine, low/high ayrımının hedef mask alanı içindeki medyan tabaka ayrımına dayandığını canonical protokolden birebir aktar.

#### Models and prompts

- SAM1 `facebook/sam-vit-huge`.
- SAM2 `facebook/sam2.1-hiera-large`.
- SAM3 local frozen checkpoint.
- İnsan GT bbox ve seed 42 YOLO bbox.
- YOLO detector sonuçlarını control table olarak ver; mask IoU ile bbox AP'yi karıştırma.

#### Reference construction

- Human iSAID masks.
- Her SAM'in aynı GT bbox ile ürettiği frozen mask.
- Hiçbir boş maskenin filtrelenmediğini yaz.
- GT diagonal = identity control; ana kanıt YOLO koşulu.

#### Metrics and statistics

- Instance-macro IoU ana metrik.
- Dice, precision, recall ve threshold success full raporlarda tamamlayıcı.
- Scene-clustered 10.000 bootstrap, yüzde 95 CI.
- Paired delta, ranking change, teacher advantage, human agreement, empty rate.

#### Reproducibility

- Sabit seed 42 yalnız detector training/inference için.
- Model/checkpoint sürümleri ve manifest hash'leri.
- Kod ve rapor üretim komutları.

### Results

Sonuç sırası önemlidir:

1. **Detector control:** Plane detector çok güçlü, Small Vehicle detector daha zayıf. Böylece mask sonuçlarının prompt kalitesi bağlamını ver.
2. **Human-reference baseline:** İnsan referansında her hedef/istem için model sıralamasını yaz.
3. **Identity control:** GT diagonal 1,0; bunu başarı olarak sunma.
4. **Main finding:** YOLO koşulunda her model kendi pseudo referansında birinci ve teacher advantage pozitif.
5. **Reference-dependent ranking:** Hangi ranking'lerin insan sırasından değiştiğini ver.
6. **Stratified robustness:** Dört overlap/area tabakasında self-reference delta değerlerinin pozitif kaldığını figürle göster.
7. **Reference integrity:** İnsan anlaşması ve empty-mask oranları. SAM3 Small Vehicle yüzde 44,4 boş maske ana failure case.
8. **Qualitative evidence:** Tüm hedef instance'ları içeren seçilmiş median-difficulty örnekler.

Sonuç bölümünde neden olduğuna dair uzun açıklama yapma; yorum Discussion'a bırakılmalı.

### Discussion

Paragraf 1: Sonuçların anlamı. Pseudo referans agreement ölçer, mutlak accuracy kanıtlamaz.

Paragraf 2: Olası mekanizma. Aynı model/ailenin sınır, boş maske, over/under-segmentation ve küçük nesne hata biçimleri korelasyonludur.

Paragraf 3: SAMRS için dengeli yorum. Büyük ölçekli pretraining kaynağı olarak yararlı olabilir; bağımsız benchmark olarak doğrudan kullanım dikkat gerektirir.

Paragraf 4: Tıp ve LLM literatürüyle bağlantı. Imperfect reference, circular consensus ve self-bias aynı bağımsızlık ilkesine işaret eder.

Paragraf 5: Pratik protokol:

- küçük kör insan audit alt kümesi;
- teacher provenance;
- empty/failure rate;
- cross-teacher matrix;
- consensus kullanılacaksa leave-one-model-out;
- nihai testte insan veya bağımsız expert consensus.

Paragraf 6: Sınırlılıklar. İki sınıf, aynı iSAID kaynak ailesi, tek detector seed, GT-box tabanlı pseudo üretim, human inter-rater maskeleri yok, BIoU yeni dört referans küpünde hesaplanmadı.

### Conclusion

Üç cümle yeterlidir:

1. Referans üreticisi model sıralamasını değiştirebilir.
2. Etki farklı promptta da sürdüğü için yalnız özdeş maske tautology'si değildir.
3. Model üretimli maskeler eğitimde kullanılabilir; değerlendirmede bağımsız insan doğrulaması ve reference-bias audit'i gerekir.

## Ana Sayılar

- Toplam: 1.024 görüntü ve 17.498 instance.
- Plane YOLO teacher advantage: SAM1 0,138; SAM2 0,119; SAM3 0,113.
- Small Vehicle YOLO teacher advantage: SAM1 0,210; SAM2 0,179; SAM3 0,299.
- Human agreement, Plane: SAM1 0,653; SAM2 0,629; SAM3 0,655.
- Human agreement, Small Vehicle: SAM1 0,658; SAM2 0,645; SAM3 0,370.
- SAM3 empty rate: Plane yüzde 2,4; Small Vehicle yüzde 44,4.

## Yazımda Kullanılmaması Gereken İfadeler

- “SAMRS bir boka yaramıyor.”
- “Pseudo etiket kullanmak overfit'tir.”
- “AI ile etiketlenmiş bütün veri leakage içerir.”
- “SAM1 en iyi modeldir.”
- “İnsan etiketi yüzde 100 doğrudur.”

Bunların yerine ölçülen koşulu ve referansı açıkça söyle.
