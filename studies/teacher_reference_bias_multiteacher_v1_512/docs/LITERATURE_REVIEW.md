# Güncel Literatür İncelemesi: Model Üretimli Referanslar ve Değerlendirme Yanlılığı

## 1. Kapsam ve Tarama Yöntemi

Tarama tarihi: **11 Ağustos 2026**.

Amaç, şu soruya doğrudan veya metodolojik olarak yakın çalışmaları bulmaktır:

> Bir yapay zekâ modelinin ürettiği etiketler değerlendirme referansı olarak kullanıldığında, aynı model veya benzer model ailesi bağımsız doğruluğundan daha başarılı görünebilir mi?

Arama eksenleri:

- remote sensing + SAM + pseudo mask + benchmark;
- segmentation + pseudo ground truth + evaluation bias;
- medical imaging + imperfect reference standard + automated annotation;
- model-generated benchmark + self-bias;
- leave-one-out consensus + circular evaluation;
- SAM-generated annotation + training/evaluation ayrımı.

Öncelik sırası hakemli makale sayfaları, resmi konferans depoları, PubMed/PMC, OpenReview ve arXiv olmuştur. Bloglar ve ikincil özetler ana kanıt olarak kullanılmamıştır. Arama sonucunda birebir aynı remote-sensing deney matrisini kullanan bir çalışma bulunmamıştır; aşağıdaki çalışmalar kavramsal, metodolojik veya alan bağlamı açısından en yakın kaynaklardır.

## 2. Ana Sonuç

Çalışmanın temel fikri tamamen bilinmeyen bir olgu değildir. Tıp literatüründeki **imperfect reference standard bias**, çoklu uzman segmentasyonundaki **circular evaluation** ve LLM benchmark literatüründeki **self-bias** aynı genel problemi farklı biçimlerde tanımlar: değerlendirme cetveli bağımsız değilse ölçülen skor gerçek performanstan sapabilir.

Buna karşılık, literatür taramasında şu kontrollü kombinasyon bulunmamıştır:

- gerçek remote-sensing instance segmentation verisi;
- aynı görüntü ve aynı aday tahminleri sabit tutma;
- insan, SAM1, SAM2 ve SAM3 referanslarını çapraz karşılaştırma;
- hem GT bbox hem YOLO bbox istemleri;
- overlap ve mask-area tabakaları;
- model sıralaması, eşleştirilmiş etki ve boş maske denetimi.

Dolayısıyla bildiri “model üretimli etiket yanlılığı ilk kez keşfedildi” iddiası kurmamalıdır. Daha savunulabilir özgünlük, **yerleşik bir ölçüm geçerliliği problemini remote-sensing SAM pseudo maskelerinde kontrollü ve çok öğretmenli olarak göstermektir**.

## 3. Terimler ve Ayrımlar

### 3.1 Pseudo etiket yararlılığı

Pseudo maskeler eğitim verisini büyütmek, pretraining yapmak veya uzman iş yükünü azaltmak için kullanılabilir. Bu kullanımın yararlı olup olmadığı, bağımsız ve tercihen insan denetimli test referansında ölçülür.

### 3.2 Pseudo referans geçerliliği

Model üretimli maske test cetveli olarak kullanılıyorsa artık soru etiketin “yeterince iyi görünmesi” değil, aday modelleri tarafsız sıralayıp sıralamadığıdır. Aynı ortalama hata oranına sahip iki referans, hata örüntüleri farklı olduğu için model sıralamasını farklı değiştirebilir.

### 3.3 Self-reference / teacher affinity

Bir model kendi ürettiği maskeye veya aynı mimari ailenin maskesine benzer hatalar yapar. Bu hata korelasyonu, model insan maskesine daha yakın olmasa bile pseudo referansa daha yakın görünmesine neden olur.

### 3.4 Klasik data leakage ile farkı

Buradaki problem yalnız klasik train/test kaçağı değildir. Görüntüler train setine sızmasa bile test referansının aday modelle ortak üretim mekanizmasına sahip olması ölçümü bağımlı hâle getirir. “Reference dependence”, “self-reference bias” veya “imperfect reference standard bias” daha doğru terimlerdir.

## 4. Remote-Sensing Literatürü

### 4.1 iSAID: bağımsız insan referansı

[iSAID](https://arxiv.org/abs/1905.12886), 2.806 yüksek çözünürlüklü görüntüde 15 sınıf ve 655.451 instance içerir. Makale maskelerin profesyonel annotator'larca sıfırdan çizildiğini, uzmanlarca çapraz kontrol edildiğini ve DOTA kutu etiketlerindeki eksik/hatalı örneklerden bağımsız olarak yeniden oluşturulduğunu belirtir. Bu özellik iSAID'ı çalışmamızdaki insan referansı için uygun kılar.

Makalenin vurguladığı güçlükler bizim örneklemimizle uyumludur: yüksek instance yoğunluğu, küçük nesneler, yön çeşitliliği, geniş ölçek aralığı ve karmaşık bağlam. Bu nedenle iSAID sonuçlarının SAMRS/SOTA'ya göre düşük olması tek başına model başarısızlığı değildir; referans üretim biçimi ve veri güçlüğü birlikte etkilidir.

### 4.2 SAMRS/SOTA: SAM1 ile otomatik maske üretimi

[SAMRS](https://arxiv.org/abs/2305.02034), HRSC2016, DOTA-v2.0, DIOR ve FAIR1M-2.0 detection verilerindeki kutuları SAM'e prompt olarak vererek 105.090 görüntü ve 1.668.241 instance maskesi üretir. SOTA alt kümesi DOTA-v2.0'dan türetilmiş 17.480 adet 1024×1024 görüntüdür.

Önemli ayrıntılar:

- DOTA ve DIOR için horizontal bbox prompt kullanılır.
- Prompt seçimi, insan piksel maskesi bulunan 124 HRSC2016 test görüntüsündeki ablation ile yapılır.
- H-box, bu küçük doğrulamada instance-macro mIoU açısından en iyi prompttur.
- Makale SAM'in bütün bölgeleri yakalamayabileceğini açıkça kabul eder.
- İstatistiklerde yalnız geçerli maskeler sayılır.
- SAMRS özellikle semantic/instance segmentation pretraining ve sınırlı downstream etikette fine-tuning için önerilir.

Bu nedenle SAMRS'nin kendi makalesi “SOTA maskeleri kusursuz bağımsız ground truth'tur” iddiası kurmaz. Ancak geniş çaplı model karşılaştırmasında aynı SAM1 maskelerini test referansı yapmak, SAM1'e veya benzer hata biçimlerine sahip modellere avantaj verebilir. Bizim bildiri motivasyonumuz bu kullanım riskidir; SAMRS'nin pretraining amacıyla yararlılığını reddetmez.

### 4.3 SAM, SAM2 ve SAM3

[Segment Anything](https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html) SAM1'i ve 11 milyon görüntüde bir milyardan fazla maskeli SA-1B veri motorunu tanıtır. [SAM2](https://arxiv.org/abs/2408.00714), görüntü ve video segmentasyonu için memory tabanlı bir yapı ve daha büyük video veri motoru getirir. [SAM3](https://arxiv.org/abs/2511.16719), kavram promptlarıyla detection, segmentation ve tracking'i birleştirir.

Üç model aynı çıktı uzayında maske üretse de mimarileri ve eğitim veri motorları aynı değildir. Bu nedenle yalnız SAM1 pseudo referansını test etmek, sorunun tek bir modele özgü olup olmadığını göstermez. Bizim SAM1/SAM2/SAM3 referans matrisi bu boşluğu hedefler.

### 4.4 SOPSeg ve domain-specific iyileştirme

[SOPSeg](https://arxiv.org/abs/2509.03002), remote-sensing small-object segmentation için SAM'in 1/16 feature çözünürlüğünün ince detayı kaybettiğini savunur. Region-adaptive magnification, edge prediction, progressive refinement ve oriented bbox'a özel prompt mekanizması kullanır.

SOPSeg bu bildirinin ana yanlılık sorusunu araştırmaz; fakat iki noktada bağlam sağlar:

1. Remote-sensing başarısı yalnız “hangi SAM sürümü” ile açıklanamaz; crop/büyütme, geometri ve sınır işleme önemlidir.
2. Pseudo referansın sınır stili, modeli değerlendirirken ciddi fark yaratabilir. Edge-aware bir model, kaba öğretmen maskesine göre haksız yere cezalandırılabilir.

### 4.5 “Segment Anything, from Space?”

[Osco ve arkadaşları](https://arxiv.org/abs/2304.13000), SAM'in overhead görüntülere sıfır atış aktarımını inceler ve bazı hedeflerde iyi genelleme görülse de overhead görüntülerin özgün nesne/ölçek özelliklerinde failure case'ler bulunduğunu raporlar. Bu çalışma pseudo referans yanlılığını ölçmez; insan referansıyla domain shift'in varlığını destekler.

### 4.6 Güncel pseudo-label refinement çalışmaları: SAMST ve ReSAM

[SAMST](https://arxiv.org/abs/2507.11994), IGARSS 2025'te Potsdam üzerinde
semi-supervised semantic segmentation için SAM pseudo etiketlerini doğrudan
kullanmaz. Threshold filter, connected-region prompt generation ve label
stitching/refinement aşamalarından geçirir; ardından gerçek etiketli ve pseudo
etiketli verilerle self-training yapar. Bu tasarım, remote-sensing literatürünün
ham SAM maskesini otomatik olarak gold standard kabul etmediğini gösterir.

[ReSAM](https://openaccess.thecvf.com/content/CVPR2026/html/Subhani_ReSAM_Refine_Requery_and_Reinforce_Self-Prompting_Point-Supervised_Segmentation_for_Remote_CVPR_2026_paper.html),
CVPR 2026'da bir nokta isteminden coarse pseudo maske üretir, maskeden yeniden
box prompt çıkarıp SAM'i sorgular ve cross-augmentation embedding alignment ile
hata yayılımını azaltır. WHU, HRSID ve NWPU VHR-10 üzerindeki sonuçları pseudo
etiketin weak supervision için yararlı olabileceğini destekler. Ancak ne SAMST
ne ReSAM model-üretimli maskeyi bağımsız test referansı olarak savunur; ikisi de
asıl olarak pseudo etiketi iyileştirme ve eğitim sinyali olarak kullanma
problemine odaklanır.

## 5. Tıbbi Görüntüleme: En Yakın Analojiler

### 5.1 Sistematik pseudo-GT hatasının öğrenilmesi

[Valabregue ve arkadaşları, “Unraveling Systematic Biases in Brain Segmentation”](https://openreview.net/forum?id=B3xO0c2Q3h), otomatik araçlardan türetilen beyin MR pseudo-ground-truth etiketlerindeki sistematik anatomik hataların modellerce öğrenilip yeniden üretilebildiğini gösterir. Putamen sınırında claustrum parçalarının sistematik olarak eklenmesi örneğini kullanır ve yüksek kaliteli uzman doğrulamasının gerekliliğini vurgular.

Bu, sağlık alanındaki en yakın motivasyon çalışmasıdır. Farkı şudur: onların odağı eğitim etiketindeki sistematik hatanın öğrenciye aktarılmasıdır; bizim odağımız aynı model ailesinin ürettiği maskenin değerlendirme referansı olduğunda skor ve sıralamanın değişmesidir.

### 5.2 Model üretimli referansın performans ölçümünü bozması

[Chavoshi ve arkadaşları, 2026](https://pubs.rsna.org/doi/10.1148/ryai.250477), LLM ile raporlardan çıkarılan tanı etiketleri referans olarak kullanıldığında model duyarlılık/özgüllük tahminlerinin sistematik biçimde bozulabileceğini simülasyonla gösterir. 10.000 vakalık deneyde etki prevalansa bağlıdır; düşük prevalansta referansın küçük specificity hatası bile kusursuz bir modelin apparent sensitivity değerini ciddi düşürebilir.

Bu çalışma segmentasyon değil sınıflandırmadır ve etiketleyici ile aday model bağımsız simüle edilir. Yine de temel mesaj aynıdır: model performansı kusurlu referansın davranışından ayrı yorumlanamaz.

### 5.3 Imperfect reference standard bias

[Diagnostic test evaluation methodology sistematik incelemesi](https://pmc.ncbi.nlm.nih.gov/articles/PMC6788703/), kusursuz gold standard bulunmadığında kullanılan alternatif referansların ölçüm yanlılığı ürettiğini ve correction, multiple imperfect references, latent class veya consensus gibi yöntemlerin gerektiğini gösterir. Bu kavram tıbbi tanı literatüründe uzun süredir yerleşiktir.

Bildiri için çıkarım: “AI etiketi kötüdür” yerine “referans standardının hatası ve adayla bağımlılığı raporlanmalıdır” denmelidir.

### 5.4 Circular consensus ve leave-one-out

[Commowick ve arkadaşlarının MS lesion veri seti](https://www.sciencedirect.com/science/article/pii/S1053811921008624), 53 hasta ve yedi uzman segmentasyonu kullanır. Bir uzmanı, kendi maskesinin de katıldığı consensus'a göre değerlendirmeyi circular evaluation olarak tanımlar ve her uzman için onu consensus dışarıda bırakan leave-one-out stratejisi uygular.

Bu metodolojik paralel çok güçlüdür: GT-bbox diagonal SAM hücrelerini ana başarı sonucu olarak vermememizin nedeni aynıdır. Referansın üretimine katılan aday, kendi katkısını içeren cetvelle tarafsız ölçülemez.

### 5.5 AI-assisted annotation ve automation bias

[Dreizin ve arkadaşları](https://pmc.ncbi.nlm.nih.gov/articles/PMC10362988/), 57 CT üzerinde nnU-Net çıktılarının uzman tarafından düzeltilmesiyle annotation süresini azaltır. AI-collaborative labeling, uzman etiketlemeye göre 8,7 kat daha hızlıdır; ancak çalışma edited AI labels için automation bias riskini açıkça ele alır ve kör bağımsız uzman kalite denetimi uygular.

Bu sonuç dengeli argüman için önemlidir: AI destekli etiketleme yararlı olabilir, fakat bağımsız kalite kontrolü olmadan “ground truth” kabul edilmemelidir.

### 5.6 SparseGT: model destekli referans nasıl doğrulanır?

[Li ve arkadaşları](https://pubmed.ncbi.nlm.nih.gov/33588116/), seyrek uzman çizimlerinden otomatik doldurulan pseudo-GT üretir. Yaklaşık 500 CT çalışmasında, pseudo-GT değişkenliğinin uzmanlar arası doğal değişkenlikten istatistiksel olarak ayırt edilemediği nesne-özel koşulları belirler ve değerlendirme hatasını insan GT'ye karşı ayrıca ölçer.

Bu, pseudo referans kullanımını reddetmek yerine nasıl doğrulanabileceğini gösteren iyi örnektir: küçük ama bağımsız insan seti, nesne-özel doğrulama ve ölçüm hatası raporu gerekir.

### 5.7 Çoklu model consensus ve collective bias

[SEG](https://pmc.ncbi.nlm.nih.gov/articles/PMC9980141/), insan GT bulunmadığında hücre/nucleus segmentasyonlarını çoklu model ensemble'ından türetilen pseudo referansla değerlendirir. Yazarlar ensemble'ın collective bias riskini kabul eder, model ablation ile ağırlıkları düzenler ve yöntemi insan etiketli küçük bir alt kümede doğrular.

Bizim çalışma açısından öneri: tek öğretmen yerine heterojen öğretmen consensus'u düşünülebilir; fakat consensus içindeki modeli kendi katkısını içeren referansla değerlendirmemek için leave-one-model-out gerekir.

### 5.8 Belirsiz, küçük veya boş referanslar

[USE-Evaluator](https://arxiv.org/abs/2209.13008), belirsiz, küçük veya boş tıbbi segmentasyon referanslarında Dice/IoU gibi metriklerin davranışını inceler. Küçük ve boş maskelerin ayrı sınıflandırma mantığıyla ele alınmasını ve yalnız overlap ortalamasına güvenilmemesini önerir.

SAM3 Small Vehicle pseudo referansındaki yüzde 44,4 boş maske oranı nedeniyle bu kaynak doğrudan ilgilidir. Boş oranını ana tabloya koymak zorunludur.

### 5.9 SAM pseudo maskeleriyle eğitim

[Medical Image Segmentation with SAM-generated Annotations](https://arxiv.org/abs/2409.20253), Medical Segmentation Decathlon CT görevlerinde SAM bbox promptlarıyla pseudo etiket üretip U-Net eğitir ve bazı koşullarda fully supervised modele yakın sonuç bildirir. [Push the Boundary of SAM](https://arxiv.org/abs/2308.00883), SAM pseudo etiketlerini kalite değerlendirme ve uncertainty tabanlı correction ile iyileştirir. [Nakai ve Hotta](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/html/Nakai_Unsupervised_Nuclei_Segmentation_by_Improving_Pseudo_Labels_from_Segment_Anything_ICCVW_2025_paper.html), nucleus pseudo maskelerinde eksik ve yanlış bölgeleri üç U-Net ve majority vote ile düzeltir.

Bu çalışmalar pseudo etiketlerin eğitimde yararlı olabileceğini gösterir. Aynı zamanda ham SAM çıktısının doğrudan güvenilir kabul edilmediğini; correction, consensus veya bağımsız insan GT değerlendirmesi gerektiğini de gösterir.

### 5.10 Tek bir gold standard olmadan çoklu uzman değerlendirmesi

[Hu ve arkadaşları, 2025](https://pubmed.ncbi.nlm.nih.gov/41132782/), bir AI
segmentasyon cihazını agregat bir uzman maskesine karşı tek skorla ölçmek yerine
AI-insan ve insan-insan overlap dağılımlarını karşılaştıran istatistiksel bir
protokol önerir. Çalışma FDA araştırmacıları tarafından yayımlanmıştır ve ortak
bir gold standard gerektirmeden AI'nın çoklu uzman paneliyle aynı agreement
düzeyinde olup olmadığını sınamayı amaçlar.

Bizim veri setinde birden fazla bağımsız insan maskesi bulunmadığı için bu
protokol doğrudan uygulanamaz. Yine de sonraki veri toplama için güçlü bir
öneridir: tek bir SAM consensus'u üretmek yerine küçük bir çoklu-uzman alt
kümeyle insan-insan değişkenliği ölçülebilir ve model skorları bu dağılımla
karşılaştırılabilir.

## 6. Model Üretimli Benchmark ve Self-Bias Literatürü

### 6.1 SILENCER

[SILENCER](https://openreview.net/forum?id=dbioYc7qav), modelin kendi ürettiği benchmarkta şişmiş performans göstermesini açıkça `self-bias` olarak tanımlar. Çoklu ve heterojen generator kullanarak bias-neutralizing ensemble oluşturur; insan benchmarkıyla Pearson korelasyonunu ortalama 0,655'ten 0,833'e yükseltir.

Alan LLM olsa da bizim model–referans matrisimizin kavramsal karşılığıdır: her üretici kendi test cetvelinde avantaj kazanabilir; heterojen referans ve insan benchmarkı bu etkiyi görünür kılar.

### 6.2 LLM-generated translation benchmark self-bias

[Deconstructing Self-Bias in LLM-generated Translation Benchmarks](https://arxiv.org/abs/2509.26600), test girdisini/etiketini üreten LLM'nin kendi benchmarkında sistematik olarak avantaj kazandığını ve veri üretimi ile evaluator tarafındaki self-bias birleştiğinde etkinin büyüdüğünü gösterir.

Bizde evaluator matematiksel IoU olduğu için judge-model yanlılığı yoktur; yanlılık referans maskenin üretiminden gelir. Bu ayrım makalede açık yazılmalıdır.

### 6.3 Modelin kendi üretimini tercih etmesi

[Pride and Prejudice](https://aclanthology.org/2024.acl-long.826/), altı LLM ve farklı görevlerde modellerin kendi ürettikleri çıktıları tercih etmesini ölçer. Bu çalışma benchmark üretiminden çok self-evaluation davranışına odaklanır; “aynı üretim dağılımına yakınlık” mekanizmasına destek verir.

## 7. Literatürden Çıkan Ortak İlkeler

1. **Etiket üreticisi mutlaka raporlanmalıdır.** Model, sürüm, checkpoint, prompt ve post-processing belirtilmelidir.
2. **Eğitim ve test referansı ayrılmalıdır.** Pseudo etiket eğitimde kullanılabilir; nihai performans bağımsız testte ölçülmelidir.
3. **Self-diagonal ana sonuç değildir.** Üreticinin kendi referansında IoU 1,0 alması identity control'dür.
4. **Küçük bağımsız insan audit seti gereklidir.** Pseudo referansın insanla anlaşması ve model sıralamasını koruyup korumadığı ölçülmelidir.
5. **Tek öğretmen yerine heterojen kaynaklar düşünülebilir.** Ancak consensus için leave-one-model-out ve collective-bias denetimi gerekir.
6. **Boş/geçersiz çıktılar saklanmalıdır.** Geçersiz maskeleri sessizce silmek sonuçları iyimserleştirir.
7. **Ortalama tek başına yeterli değildir.** Instance-macro skor, tabaka analizi, güven aralığı, empty rate ve ranking birlikte verilmelidir.

## 8. Bildirinin Güvenli Özgünlük İddiası

Önerilen cümle:

> To our knowledge, this is the first controlled remote-sensing instance-segmentation study that holds images, object instances, prompts, candidate predictions, and metric computation fixed while crossing three SAM-family pseudo-reference generators with three evaluated SAM models and an independent human reference.

Bu cümlede `to our knowledge` korunmalıdır. “AI-labeled datasets are useless” veya “teacher bias has never been studied” denmemelidir.

## 9. Savunulabilir Sonuç Dili

Kullanılabilir:

- “Model-generated references inflated agreement with the generating model.”
- “Reference choice changed model ranking relative to the human reference.”
- “The effect persisted under YOLO-box prompting, where predictions were not identical to GT-box teacher masks.”
- “Pseudo labels may remain useful for pretraining; our results concern their use as independent evaluation references.”

Kaçınılmalı:

- “SAMRS has no value.”
- “Human labels are perfect.”
- “All pseudo-label datasets leak test data.”
- “SAM3 is universally worse.”
- “High pseudo-reference IoU proves segmentation accuracy.”

## 10. Kaynak Durumu ve Kanıt Gücü

| Kaynak | Alan | Hakem durumu | Bizim soruya yakınlık | Ana kullanım |
| --- | --- | --- | --- | --- |
| iSAID | Remote sensing | CVPRW/arXiv dataset paper | Doğrudan veri kaynağı | İnsan referansı ve zorluk |
| SAMRS | Remote sensing | NeurIPS Datasets & Benchmarks | Doğrudan pseudo veri kaynağı | SOTA üretim yöntemi ve amaç |
| SOPSeg | Remote sensing | ArXiv preprint | Dolaylı | Small-object/preprocess bağlamı |
| SAMST | Remote sensing | IGARSS 2025 | Yakın eğitim kullanımı | Pseudo-label filtering/refinement |
| ReSAM | Remote sensing | CVPR 2026 | Yakın eğitim kullanımı | Hata yayılımını azaltan self-prompting |
| Valabregue et al. | Medical segmentation | MIDL 2024 short paper | Çok yakın | Sistematik pseudo-GT hatası |
| Chavoshi et al. | Medical AI evaluation | Radiology: AI 2026 | Çok yakın kavram | Model üretimli kusurlu referans |
| Commowick et al. | Medical segmentation | NeuroImage 2021/2022 | Çok yakın yöntem | Circularity ve leave-one-out |
| Dreizin et al. | Medical annotation | Frontiers in Radiology 2023 | Yakın | HITL, kör QA, automation bias |
| SparseGT | Medical segmentation | Medical Image Analysis 2021 | Yakın | Pseudo-GT doğrulama protokolü |
| USE-Evaluator | Medical segmentation | Neuroinformatics 2023 | Yakın | Küçük/boş referans metrikleri |
| Hu et al. | Medical segmentation evaluation | Journal of Medical Imaging 2025 | Çok yakın yöntem | Tek gold standard olmadan çoklu uzman testi |
| SILENCER | LLM benchmark | NeurIPS 2025 | Çok yakın kavram | Self-bias ve heterojen generator |
| Translation self-bias | LLM benchmark | ArXiv 2025 | Yakın | Generator kaynaklı benchmark bias |
| Pride and Prejudice | LLM evaluation | ACL 2024 | Dolaylı | Kendi üretimini tercih etme |

## 11. Taramanın Sınırları

- Arama İngilizce yayınlar ve erişilebilir birincil kaynaklarla sınırlıdır.
- “Teacher bias” terimi semi-supervised learning'de farklı anlamlarda da kullanıldığı için yalnız terim araması yeterli değildir; `pseudo ground truth`, `imperfect reference`, `self-bias`, `circular evaluation` ve `model-generated benchmark` birlikte aranmıştır.
- 11 Ağustos 2026 sonrasında çıkan yayınlar kapsanmaz.
- Birebir aynı deney bulunamaması, böyle bir çalışmanın kesinlikle mevcut olmadığının kanıtı değildir; bu nedenle özgünlük iddiası ihtiyatlı kurulmalıdır.
