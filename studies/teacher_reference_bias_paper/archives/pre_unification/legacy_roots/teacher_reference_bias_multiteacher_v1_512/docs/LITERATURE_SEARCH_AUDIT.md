# Literatür Tarama Denetimi

**Tarih:** 11 Ağustos 2026
**Amaç:** Model üretimli segmentasyon maskelerinin eğitim etiketi veya
değerlendirme referansı olarak kullanılmasında oluşan sistematik yanlılığı ve
teacher-reference affinity sorununu taramak.

## Düzeltme Kaydı

İlk literatür taraması Parikh, Das ve Feragen'in
[arXiv:2511.00477](https://arxiv.org/abs/2511.00477) çalışmasını bulmamış,
kullanıcı tarafından daha sonra verilen kaynak da literatür dosyasına ve
BibTeX'e aktarılmamıştır. Bu önemli bir tarama ve kayıt hatasıdır.

Kök neden, aramanın `SAM`, `pseudo ground truth`, `self-reference` ve
`imperfect reference` terimlerine fazla bağlanmasıdır. İlgili makale problemi
`label bias`, `age-related disparity` ve `Biased Ruler` terimleriyle kurar.
Güncel taramada fairness ve annotation-style terminolojisi ayrı bir arama
ekseni olarak eklenmiştir.

## Arama Soru Aileleri

1. `segmentation` + `label bias` + `automated labels` + `validation`
2. `biased ruler` + `image segmentation`
3. `silver standard` + `gold standard` + `segmentation evaluation`
4. `annotation style` + `segmentation` + `model ranking`
5. `machine-generated reference` + `segmentation benchmark`
6. `same model` veya `teacher` + `pseudo label` + `evaluation bias`
7. `SAMRS` + `SAM/SAM2/SAM3` + `benchmark/evaluation`
8. `circular evaluation`, `leave-one-out consensus` ve `imperfect reference`

Anahtar kelime aramasına ek olarak Parikh 2025'in kaynakları, aynı yazarların
sonraki çalışmaları ve yakın yayınların ileri/geri kavram zinciri taranmıştır.

## Dahil Etme Ölçütleri

- Birincil akademik kaynak veya resmi yayın sayfası olması;
- segmentasyon etiketi/referansı, model üretimli etiket veya annotation style
  ile doğrudan ilişkili olması;
- eğitim etkisi ile değerlendirme etkisinin ayrılabilmesi;
- bizim deney tasarımımızı değiştirecek yöntem veya özgünlük bilgisi vermesi.

Blog, haber ve ikincil özetler keşif için görülebilir; kanıt veya BibTeX kaynağı
olarak kullanılmaz.

## En Yüksek Öncelikli Bulgular

| Öncelik | Kaynak | Durum | Bizim çalışma için anlamı |
| --- | --- | --- | --- |
| 1 | [Parikh et al. 2025](https://arxiv.org/abs/2511.00477) | ArXiv v1, ISBI 2026'ya gönderilmiş | Aynı tahmini gold/silver referansla ölçen en yakın biased-ruler öncülü |
| 2 | [Parikh et al. 2026](https://arxiv.org/abs/2605.06891) | ArXiv preprint | Bozuk maskenin farkı büyütme, gizleme veya tersine çevirme mekanizması |
| 3 | [Nichyporuk et al. 2022](https://arxiv.org/abs/2210.17398) | MELBA 2022 | Annotation style'ın label-space shift ve görünen generalization farkı oluşturması |
| 4 | [Vorontsov ve Kadoury 2021](https://arxiv.org/abs/2107.02189) | ArXiv | Sistematik bias'ın rastgele label noise'dan daha zararlı olması |
| 5 | [Lad ve Mueller 2023](https://arxiv.org/abs/2307.05080) | ICML DMLR Workshop | Temiz referans yokken model tabanlı etiket kalite önceliklendirme |
| 6 | [SAMRS](https://arxiv.org/abs/2305.02034) | NeurIPS Datasets and Benchmarks 2023 | SAM1 maskelerinin esas kullanım amacı pretraining; bağımsız gold iddiası yok |

## Parikh 2025 Deneyinin Özeti

- Veri: 1.506 MAMA-MIA meme DCE-MRI vakası.
- Silver referans: dış veriyle eğitilmiş nnU-Net otomatik maskesi.
- Gold referans: otomatik başlangıç maskesinin 16 uzman tarafından düzeltilmiş
  son hâli.
- En yakın deney: tek M-BASELINE tahmin kümesini silver ve gold maskelere karşı
  ayrı ayrı ölçme.
- Fairness gap: gold `0,0399`, silver `0,0559`; görünen fark yaklaşık `%40`
  büyür.
- Ayrı training deneyi: silver etiketle eğitim gold referanstaki farkı
  `0,0399`dan `0,0661`e çıkarır; `%66` amplification.

İlk madde bizim değerlendirme sorumuza, ikinci madde pseudo-label training
sorusuna karşılık gelir. Bunlar aynı deney değildir.

## Bizim Özgünlük Sınırımız

Artık savunulamayacak iddialar:

- segmentasyonda otomatik referans yanlılığını ilk kez gösterme;
- biased-ruler etkisini ilk kez tanımlama;
- model üretimli etiketlerin değerlendirmeyi bozabileceğini ilk kez bulma.

Savunulabilir dar iddia:

> To our knowledge, this is the first controlled remote-sensing instance-
> segmentation study that holds images, instances, prompts, predictions, and
> metrics fixed while crossing three SAM-family reference generators with
> three evaluated SAM models and an independent human reference.

Bu iddia yine de gönderim öncesi güncel Scopus/Web of Science/Google Scholar
aramasıyla doğrulanmalıdır; web taraması yokluk kanıtı değildir.

## Tasarıma Etkisi

- Human ve pseudo referansların aynı dondurulmuş tahmin üzerinde eşleştirilmesi
  korunur; bu tasarım Parikh 2025 ile uyumludur.
- Ana kanıt GT-diagonal değildir; YOLO-bbox non-identical sonuçlarıdır.
- Teacher affinity, genel biased-ruler etkisinden ayrı tanımlanır.
- IoU/Dice yanında over/under-segmentation yönü ve boundary davranışı sonraki
  deney uzantısı olarak önerilir.
- Sonuç dili pseudo etiketlerin eğitim yararını reddetmez; yalnız bağımsız test
  referansı geçerliliğini sınar.

## Kalan Literatür Riski

Arama sonucunda aynı remote-sensing SAM1/SAM2/SAM3 cross-teacher matrisini
raporlayan bir çalışma bulunmamıştır. Ancak bu sonuç kesin yokluk kanıtı
değildir. Gönderimden hemen önce başlık/özet benzerliği, atıf zinciri ve yeni
2026 yayınları yeniden taranmalıdır.
