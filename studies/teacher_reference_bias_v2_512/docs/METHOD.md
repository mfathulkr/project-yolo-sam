# Yöntem

## Amaç

Çalışma, model veya model ailesi tarafından üretilmiş pseudo maskelerin
bağımsız test etiketi olarak kullanılmasının ölçülen segmentasyon başarısını
nasıl değiştirdiğini inceler.

## Kontrollü Karşılaştırma

iSAID için aynı:

- 512 test görüntüsü,
- 5.447 uçak örneği,
- GT ve YOLO bbox istemleri,
- SAM1, SAM2 ve SAM3 tahminleri

iki farklı referansa karşı değerlendirilir:

1. Resmi insan çizimli iSAID maskesi
2. SAM1 GT-bbox tahmininden dondurulan pseudo maske

Bu eşlemede görüntü, nesne, bbox ve tahmin değişmez. Yalnız değerlendirme
referansı değişir.

SAMRS SOTA-RBB deneyi aynı model ve bbox koşullarını yayımlanan SAM1 kaynaklı
pseudo maskeler üzerinde tekrarlar. Bu sonuç bağımsız insan ground truth
başarısı olarak yorumlanmaz.

GT-bbox istemi iSAID için resmi insan instance anotasyonundaki kutudan,
SAMRS için yayımlanan özgün detection anotasyonundan gelir. Hiçbir ana deney
kutusu pseudo maskeden yeniden türetilmez.

## Split ve Örnekleme

Split birimi tile değil kaynak sahnedir. Train, validation ve test arasında
kaynak sahne kesişimi sıfır olmalıdır.

Test kümesi dört dengeli gruptan oluşur:

| Grup | Görüntü |
|---|---:|
| No Overlap × Low Mask Area | 128 |
| No Overlap × High Mask Area | 128 |
| Overlap × Low Mask Area | 128 |
| Overlap × High Mask Area | 128 |
| Overall | 512 |

`No Overlap`, görüntüdeki hiçbir GT bbox çiftinin kesişmemesidir
(`max pair bbox IoU = 0`). `Overlap`, en az bir GT bbox çiftinin IoU
değerinin `0,001` veya üstünde olmasıdır. Bu iki sınır arasında kalan adaylar
test kümesine alınmaz. `High/Low Mask Area`, görüntüdeki toplam referans uçak
maskesi alanının her veri seti için v1 aday havuzundan dondurulan eşiğin
üstünde veya altında olmasına göre belirlenir. Eşik test sonucuna bakılarak
seçilmez.

Strata tanımı nedeniyle testteki 512 görüntünün tamamı en az bir uçak içerir.
Detector mAP gerçek COCO bbox hesabıdır; ancak negatif arka plan görüntülerini
içeren resmi tam veri seti benchmark'ı değil, bu dengeli pozitif test alt
kümesinin sonucudur.

## Detector

Her veri setinde YOLO26x üç bağımsız seed ile baştan eğitilir:

- 42
- 123
- 2026

İki veri setinde de aynı `yolo26x` başlangıç ağırlığı, 1024×1024 giriş,
100 epok üst sınırı, batch 12 ve patience 30 kullanılır.

iSAID eğitim bölümü 1.571, SAMRS eğitim bölümü 2.191 görüntüdür. Aynı epok
ve hiperparametre ayarları bu nedenle aynı optimizasyon adımı sayısı anlamına
gelmez. Veri setleri arasındaki detector skor farkı bağlamsal olarak
raporlanır; yalnız referans kaynağının nedensel etkisi sayılmaz. Ana
referans-kaynağı karşılaştırması, aynı iSAID tahminlerini insan ve SAM1
pseudo maskesine karşı ölçen eşlenmiş deneydir.

Validation kümesinde seçilen confidence eşiği test değerlendirmesinden önce
sabitlenir. BBox mAP50, mAP75, mAP90 ve mAP50-95 gerçek COCO bbox AP
ölçümleridir. BBox Precision ve Recall ayrıca 0,50/0,75/0,90 bbox IoU
eşiklerinde verilir.

## Segmentasyon

SAM1, SAM2 ve SAM3 bu iki veri setinde yeniden eğitilmez veya ince ayar
görmez. Dondurulmuş model ağırlıklarına yalnız GT veya YOLO bbox istemi
verilir.

Her uçak örneği ayrı değerlendirilir. Bir instance için:

- `IoU = TP / (TP + FP + FN)`
- `Dice = 2TP / (2TP + FP + FN)`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`

Ortalama önce her uçak için hesaplanan skorların aritmetik ortalamasıdır.
Nesneler eşit ağırlıklıdır; büyük uçaklar daha çok piksele sahip olduğu için
sonucu tek başına baskılamaz.

`IoU ≥ 0.50/0.75/0.90`, ilgili eşiği geçen uçak örneklerinin oranıdır. Bunlar
COCO mask AP değildir. Confidence sırasındaki bütün tahmin maskelerini ve
yanlış pozitifleri kullanan uçtan uca COCO segmentation AP bu raporun
eşlenmiş instance-metrik protokolünden farklı bir değerlendirmedir. Ayrı
çalıştırılmadığı için bu eşik oranları AP veya mAP diye yeniden adlandırılmaz.

YOLO'nun kaçırdığı bir gerçek uçak, YOLO-bbox maske değerlendirmesinde boş
tahmin olarak kalır ve o instance'ın maske skorları sıfır olur. Hiçbir gerçek
uçakla eşleşmeyen yanlış pozitif YOLO kutuları ise sahte bir referans instance
oluşturularak maske ortalamasına eklenmez; yanlış pozitif etkisi detector
Precision, Recall ve mAP tablosunda ölçülür. Bu nedenle segmentation
tablolarındaki IoU eşik oranları mask AP değildir.

## Raporlar

Üç bağımsız full metric document oluşturulur:

1. iSAID insan referansı
2. iSAID SAM1 pseudo referansı
3. SAMRS SOTA SAM1 pseudo referansı

Her raporda yalnız SAM1/SAM2/SAM3 × GT/YOLO bbox koşulları bulunur.
RemoteSAM, RingMoSAM ve başka eski pipeline'lar bu deney matrisine dahil
değildir.
