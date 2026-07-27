# Teacher-Generated Segmentation Referanslarında Değerlendirme Yanlılığı: Literatür İncelemesi

## 1. İncelemenin Amacı

Bu doküman aşağıdaki araştırma sorusunun literatürdeki yerini belirlemek için
hazırlanmıştır:

> Bir segmentasyon modeli tarafından otomatik üretilen maskeler test referansı
> olarak kullanıldığında, aynı model veya aynı model ailesi olduğundan daha
> başarılı görünebilir mi?

Çalışmamızın odağı pseudo-label kullanımının eğitimde yararlı olup olmadığı
değildir. İki ayrı kullanım birbirine karıştırılmamalıdır:

1. **Weak supervision / pretraining:** Otomatik maskeler eğitim verisini
   büyütmek için kullanılır ve sonuç bağımsız insan etiketli test kümesinde
   ölçülür.
2. **Benchmark evaluation:** Otomatik maskeler doğrulanmış ground truth gibi
   kullanılır ve modeller doğrudan bu maskelere benzerlikleriyle sıralanır.

İlk kullanım geçerli ve yararlı olabilir. İkinci kullanım, referansı üreten
modeli veya ona benzeyen yöntemleri sistematik olarak kayırabilir.

## 2. İncelenen Birincil Kaynaklar

| Kaynak | Konuyla ilişkisi | Bu çalışmada nasıl kullanılacak? |
|---|---|---|
| [SAMRS, NeurIPS Datasets and Benchmarks 2023](https://arxiv.org/abs/2305.02034) | DOTA, DIOR, FAIR1M ve HRSC2016 detection annotation'larını SAM ile maskeye dönüştürür. | SAMRS SOTA maskelerinin üretim kaynağı ve amaçlanan kullanımını belirlemek için. |
| [SAMRS resmi kod deposu](https://github.com/ViTAE-Transformer/SAMRS) | Üretim kodu ve yayınlanan veri bağlantılarını içerir. | SOTA-RBB üretiminde `sam_vit_h_4b8939.pth`, `vit_h` ve RHBox kullanıldığını doğrulamak için. |
| [Segment Anything, ICCV 2023](https://arxiv.org/abs/2304.02643) | SAM1 modelini ve promptable segmentation yaklaşımını tanımlar. | SAM1'in öğretmen model ailesini tanımlamak için. |
| [iSAID, CVPR Workshops 2019](https://arxiv.org/abs/1905.12886) | DOTA görüntüleri üzerinde insan tarafından bağımsız instance segmentation annotation'ları sunar. | Bağımsız insan referanslı kontrol veri setini tanımlamak için. |
| [On the Limits of Pseudo Ground Truth in Visual Camera Re-localisation, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Brachmann_On_the_Limits_of_Pseudo_Ground_Truth_in_Visual_Camera_ICCV_2021_paper.html) | Pseudo-ground-truth üreten referans algoritmasının benzer yöntem ailelerini kayırabildiğini gösterir. | Bizim teacher-reference affinity hipotezimizin en yakın metodolojik öncülü olarak. |
| [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning, IJCNN 2020](https://arxiv.org/abs/1908.02983) | Yanlış pseudo-label'ların eğitim sırasında kendini pekiştirebildiğini gösterir. | Eğitimde pseudo-label yanlılığı ile değerlendirmedeki referans yanlılığını ayırmak için. |
| [Validation of Image Segmentation by Estimating Rater Bias and Variance, 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC3227147/) | Tek bir segmentasyonu mutlak gerçek kabul etmenin sorunlarını, rater bias ve variance kavramlarını tartışır. | Referans segmentasyonun da ölçüm hatası ve sistematik stile sahip olabileceğini temellendirmek için. |
| [SOPSeg, 2025 preprint](https://arxiv.org/abs/2509.03002) | Küçük uzaktan algılama nesneleri için crop/magnification, oriented prompt ve edge-aware decoder kullanır. | Model değişiminden çok girdi ölçeği, prompt geometrisi ve sınır iyileştirmenin etkisini açıklamak için. |
| [Boxes2Pixels, CVPR Workshops 2026](https://openaccess.thecvf.com/content/CVPR2026W/AI4RWC/html/Lendering_Boxes2Pixels_Learning_Defect_Segmentation_from_Noisy_SAM_Masks_CVPRW_2026_paper.html) | SAM maskelerini ground truth değil, gürültülü öğretmen çıktısı olarak ele alır ve insan etiketli benchmark'ta değerlendirir. | Güncel iyi uygulama örneği olarak. |

## 3. SAMRS Tam Olarak Ne Yapar?

SAMRS çalışmasının temel amacı pahalı pixel-level remote sensing
annotation'larını daha ucuza üretmek ve büyük ölçekli segmentation pretraining
verisi sağlamaktır. Çalışma:

- mevcut detection veri setlerindeki kategori ve bbox bilgisini alır;
- bbox'ları SAM prompt'u olarak kullanır;
- SAM çıktısını instance ve semantic segmentation label'ına dönüştürür;
- üretilen veriyi downstream modellerin ön eğitimi için kullanır;
- downstream faydayı iSAID ve Potsdam gibi ayrı veri setlerindeki fine-tuning
  sonuçlarıyla inceler.

Bu nedenle SAMRS'nin varlığı kendi başına metodolojik hata değildir. Temel
bilimsel ayrım şudur:

- **SAMRS'yi eğitim verisi olarak kullanmak:** Makul bir weak-supervision
  yaklaşımıdır.
- **SAMRS maskelerini bağımsız ground truth kabul edip SAM1, SAM2 ve SAM3'ü
  sıralamak:** Referans kaynağı açıklanmadan yapılırsa geçerlilik sorunu doğurur.

### 3.1 SOTA-RBB üretim ayrıntısı

Resmi kod incelemesinde SOTA-RBB için:

- model türü `vit_h`;
- checkpoint `sam_vit_h_4b8939.pth`;
- detection annotation kaynağı DOTA;
- prompt geometrisi rotated bbox'ın minimum çevreleyen horizontal kutusu,
  yani RHBox;
- çıktı maskesi SAM1 tarafından üretilen pseudo-mask

olarak doğrulanmıştır.

Bizim SAMRS GT-bbox değerlendirmemizde SAM1 de aynı model ailesini ve aynı
RHBox geometrisini kullanır. Dolayısıyla SAM1 prediction'ı ile SAMRS referansı
bağımsız iki gözlem değildir.

### 3.2 SAMRS makalesinin kendi kalite kontrolü

SAMRS, prompt seçimini insan maskesi bulunan 124 HRSC2016 test görüntüsünde
inceler. Makaledeki HBox sonucu instance ortalamalı mIoU için yaklaşık
`89.97`, pixel ağırlıklı mIoU için yaklaşık `79.40` olarak raporlanır. Bu
önemli bir bulgudur:

- SAM maskeleri yüksek kaliteli olabilir;
- fakat mükemmel değildir;
- instance ve pixel ağırlıklı ölçümler belirgin biçimde ayrılabilir;
- bu maskeler yine de insan ground truth ile aynı şey değildir.

## 4. En Yakın Metodolojik Önceki Çalışma

Brachmann ve arkadaşlarının ICCV 2021 çalışması, büyük ölçekli kamera
re-localisation benchmark'larında SfM veya SLAM gibi algoritmalarla üretilen
pseudo-ground-truth pozlarını inceler. Temel sonuçları şudur:

- değerlendirme sonucu kullanılan referans algoritmasına göre değişebilir;
- referans algoritmasına yöntemsel olarak benzeyen modeller kayrılabilir;
- benchmark gerçekte "fiziksel gerçeği" değil, referans algoritmasının
  çıktısını yeniden üretme derecesini ölçebilir;
- yöntem sıralaması referans değiştiğinde değişebilir.

Bizim çalışma aynı problemi segmentation alanında, özellikle remote sensing
promptable segmentation için ölçmektedir:

| Kamera re-localisation çalışması | Bizim segmentation çalışmamız |
|---|---|
| SfM/SLAM pseudo pose | SAM1 pseudo-mask |
| Kamera poz hatası | Instance mask IoU/Dice/Boundary IoU |
| Referans algoritmasına benzer yöntem | SAM1 veya aynı prompt/model ailesi |
| Farklı pseudo referanslarda sıralama | Human ve SAM1 referansında sıralama |

Dolayısıyla genel fikir tamamen bilinmeyen değildir. Çalışmamızın yeniliği,
bu değerlendirme yanlılığını remote sensing segmentation'da kontrollü,
instance-level ve aynı prediction'ları iki referansa karşı ölçen bir protokolle
nicelleştirmesidir.

## 5. Confirmation Bias ile Farkı

Pseudo-label literatüründeki **confirmation bias**, modelin kendi hatalı
tahminlerini eğitim etiketi olarak yeniden görmesi ve bu hataları öğrenerek
pekiştirmesidir. Bizim ana problemimiz farklıdır:

- confirmation bias bir **training dynamics** problemidir;
- teacher-reference affinity bir **measurement validity** problemidir.

İkisi aynı kökten gelir: model çıktısının bağımsız gerçek gibi kullanılması.
Ancak bildiride terimler birbirinin yerine kullanılmamalıdır. Ana terim olarak
**teacher-reference affinity**, **generator-reference affinity** veya
**pseudo-reference evaluation bias** tercih edilmelidir.

## 6. SOPSeg'in Gösterdiği Teknik Sonuç

SOPSeg, küçük uzaktan algılama nesnelerinde SAM'in temel sorununun yalnızca
decoder olmadığını nicel olarak gösterir. iSAID ablation tablosundaki ortalama
IoU değişimleri:

| Aşama | mIoU | Önceki aşamaya göre kazanç |
|---|---:|---:|
| Bbox prompt ile fine-tuned SAM decoder baseline | 71.65 | - |
| + Region-Adaptive Magnification (RAM) | 79.40 | +7.75 |
| + Oriented Prompt Mechanism (OPM) | 81.69 | +2.29 |
| + Edge-aware Enhanced Decoder (EDE) | 82.96 | +1.27 |

Makaledeki metin RAM kazancını `+7.84` olarak yuvarlanmış/alternatif
hesaplamayla ifade eder; tabloda görülen fark `79.40 - 71.65 = 7.75`'tir.
Bildiri veya sunumda doğrudan tablo değerleri kullanılmalıdır.

Bu ablation'ın yorumu:

1. En büyük katkı nesneyi uygun context ile crop edip büyütmekten gelir.
2. Rotated/oriented nesnenin yön bilgisini prompt'a taşımak ikinci büyük
   katkıdır.
3. Decoder'a edge prediction ve progressive refinement eklemek yararlıdır,
   ancak toplam kazancın daha küçük kısmını oluşturur.

Bu sonuç, "klasik görüntü işleme geri geliyor" ifadesini kısmen destekler.
Ancak kullanılan işlemler yalnızca klasik filtreler değildir. Sistem:

- geometriye duyarlı crop ve yeniden ölçekleme;
- position embedding interpolation;
- bbox ile üç yön noktası birleştiren prompt tasarımı;
- öğrenilen edge token;
- çok ölçekli progressive refinement;
- mask, edge ve IoU için multi-task loss

kullanır. Doğru ifade, **model seçiminin yanında scale-aware preprocessing,
geometry-aware prompting ve boundary-aware learning belirleyicidir** şeklinde
olmalıdır.

## 7. Bizim Çalışmanın Literatürdeki Boşluğu

İncelenen kaynaklarda aşağıdaki protokolün remote sensing segmentation için
birlikte uygulandığı doğrudan bir çalışma bulunmamıştır:

1. Aynı görüntü ve aynı instance için sabit prediction üretme.
2. Prediction'ı bağımsız insan maskesine ve teacher-generated pseudo-mask'e
   karşı ayrı ayrı ölçme.
3. Sadece referansı değiştirerek metric inflation hesaplama.
4. SAM1, SAM2 ve SAM3 sıralamasının referansla değişip değişmediğini ölçme.
5. Kaynak sahne düzeyinde bootstrap confidence interval üretme.
6. Overlap ve mask area strata'larında etkinin değişimini inceleme.
7. Aynı kaynak görüntülerdeki bağımsız iSAID ve SAMRS annotation'larını
   eşleyerek doğrudan referans kalitesini ölçme.

Bu boşluk bildirinin asıl savunulabilir katkısıdır.

## 8. Bildiride Kurulabilecek ve Kurulmaması Gereken İddialar

### 8.1 Savunulabilir iddialar

- SAM1-generated maskeler, SAM1'i bağımsız insan referansına göre olduğundan
  daha başarılı gösterebilir.
- Teacher-generated test referansı model sıralamasını değiştirebilir.
- Aynı prediction üzerinde referansı değiştirmek, veri seti zorluğu ve
  inference farkını kontrol eder.
- SAMRS SOTA maskeleri pretraining için yararlı olabilir; bu, SAM1
  değerlendirmesi için bağımsız ground truth oldukları anlamına gelmez.
- Pseudo-mask benchmark'ları generator provenance'ını açıkça raporlamalı ve
  mümkünse insan etiketli audit subset sağlamalıdır.

### 8.2 Kurulmaması gereken iddialar

- "SAMRS işe yaramaz."
- "SAMRS makalesi hatalı veya hilelidir."
- "SAM1 gerçekten mükemmeldir."
- "SAM1 bütün remote sensing görevlerinde SAM2 ve SAM3'ten üstündür."
- "Pseudo-label ile eğitim her zaman overfit üretir."
- "Gözlenen bütün veri seti farkı yalnızca label leakage'den kaynaklanır."

## 9. Önerilen Terminoloji

| Terim | Kullanım |
|---|---|
| `pseudo-label` | Eğitim veya referans için model tarafından üretilen etiket |
| `pseudo-reference` | Değerlendirmede referans olarak kullanılan model etiketi |
| `teacher model` | Pseudo etiketi üreten model |
| `teacher-reference affinity` | Teacher ile referans arasındaki ortak üretim stilinin ölçüme etkisi |
| `reference inflation` | Aynı prediction'ın pseudo ve human referanstaki skor farkı |
| `ranking reversal` | Model sıralamasının referans türü değişince değişmesi |
| `independent human reference` | Değerlendirilen modelden bağımsız insan annotation'ı |
| `data leakage` | Bu çalışma için yalnızca genel benzetme olarak kullanılmalı; doğrudan training-test örnek sızıntısı iddiası kurulmadıkça ana teknik terim yapılmamalı |

## 10. Araştırma Sonucu

Literatür, hocanın öne sürdüğü temel endişeyi desteklemektedir: pseudo-ground
truth, onu üreten yönteme benzeyen modelleri kayırabilir. Bu genel problem
başka alanlarda bilinmektedir; dolayısıyla bildirinin katkısı "bu olgu ilk kez
keşfedildi" değildir.

Savunulabilir katkı şudur:

> Remote sensing promptable instance segmentation için teacher-generated
> referans yanlılığını, iki veri seti, üç SAM nesli, sabit prediction'ların
> dual-reference değerlendirmesi, source-scene bootstrap ve zorluk strata'ları
> ile kontrollü olarak nicelleştiren yeniden üretilebilir bir çalışma.

Bu çerçeve, mevcut iSAID ve SAMRS sonuçlarını tek başına karşılaştırmaktan daha
güçlüdür; çünkü veri seti zorluğu ile referans kaynağı etkisini birbirinden
ayırır.
