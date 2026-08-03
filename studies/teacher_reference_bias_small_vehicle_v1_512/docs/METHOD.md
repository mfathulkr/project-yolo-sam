# Yöntem

## Amaç

Çalışma, model veya model ailesi tarafından üretilmiş pseudo maskelerin
bağımsız test etiketi olarak kullanılmasının ölçülen segmentasyon başarısını
nasıl değiştirdiğini inceler.

## Kontrollü Karşılaştırma

iSAID için aynı:

- 512 test görüntüsü,
- hazırlanmış COCO anotasyonundaki bütün küçük araç örnekleri,
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
test kümesine alınmaz. `High/Low Mask Area`, görüntüdeki toplam referans küçük araç
maskesi alanının her veri seti için v1 aday havuzundan dondurulan eşiğin
üstünde veya altında olmasına göre belirlenir. Eşik test sonucuna bakılarak
seçilmez.

Dondurulan alan oranı eşikleri iSAID için `0,0018463134765625`, SAMRS için
`0,0065670013427734` değeridir. iSAID eşiği, model çalıştırılmadan önce resmi
test aday havuzunda dört grubun da en az 128 görüntü içermesini sağlayacak
şekilde annotation alan dağılımından seçilmiştir. Bu eşikler model veya metrik
sonucuna göre değiştirilmez.

Doğrulanan final iSAID splitleri train/validation/test için sırasıyla
5.930/1.353/512 görüntü ve 359.927/71.275/12.051 instance içerir. SAMRS
splitleri 7.824/1.567/512 görüntü ve 304.414/49.792/7.659 instance içerir.
iSAID test görüntüleri 31, SAMRS test görüntüleri 17 kaynak sahneden gelir.

Strata tanımı nedeniyle testteki 512 görüntünün tamamı en az bir küçük araç içerir.
Detector mAP gerçek COCO bbox hesabıdır; ancak negatif arka plan görüntülerini
içeren resmi tam veri seti benchmark'ı değil, bu dengeli pozitif test alt
kümesinin sonucudur.

## Detector

Her veri setinde YOLO26x sabit seed `42` ile baştan eğitilir. Bu çalışma seed
varyansını ölçmez; iki veri setinde aynı deterministik seed'i kullanır.

İki veri setinde de aynı `yolo26x` başlangıç ağırlığı, 1024×1024 giriş,
100 epok üst sınırı, batch 12 ve patience 30 kullanılır.

Ultralytics detection trainer normalde validation batch'ini otomatik olarak
eğitim batch'inin iki katına çıkarır. Small-vehicle sahnelerindeki yoğun nesne
sayısı bu davranışta çok büyük eşleştirme matrisleri ürettiği için validation
batch'i de 12 olarak tutulur. Bu yalnız verinin hesaplama parçalarına
bölünmesini değiştirir; eğitim batch'i, ağırlık güncellemeleri, görüntü sırası,
etiketler, confidence seçimi ve COCO metrikleri değişmez.

Eğitim splitlerinin görüntü ve nesne sayıları prepared metadata'dan okunur.
Split büyüklükleri farklı olduğunda aynı epok ve hiperparametre ayarları aynı
optimizasyon adımı sayısı anlamına gelmez. Veri setleri arasındaki detector
skor farkı bağlamsal raporlanır; yalnız referans kaynağının nedensel etkisi
sayılmaz. Ana referans-kaynağı karşılaştırması, aynı iSAID tahminlerini insan
ve SAM1 pseudo maskesine karşı ölçen eşlenmiş deneydir.

Validation kümesinde seçilen confidence eşiği test değerlendirmesinden önce
sabitlenir. BBox mAP50, mAP75, mAP90 ve mAP50-95 gerçek COCO bbox AP
ölçümleridir. BBox Precision ve Recall ayrıca 0,50/0,75/0,90 bbox IoU
eşiklerinde verilir.

## Segmentasyon

SAM1, SAM2 ve SAM3 bu iki veri setinde yeniden eğitilmez veya ince ayar
görmez. Dondurulmuş model ağırlıklarına yalnız GT veya YOLO bbox istemi
verilir.

Her küçük araç örneği ayrı değerlendirilir. Bir instance için:

- `IoU = TP / (TP + FP + FN)`
- `Dice = 2TP / (2TP + FP + FN)`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`

Ortalama önce her küçük araç için hesaplanan skorların aritmetik ortalamasıdır.
Nesneler eşit ağırlıklıdır; büyük nesneler daha çok piksele sahip olduğu için
sonucu tek başına baskılamaz.

Bir görüntüde birden fazla küçük araç varsa görüntüdeki bütün instance'lar koşuya
dahildir. GT-bbox koşulunda her nesne kendi kutusuyla ayrı istem olarak
çalıştırılır; örneğin 20 GT nesnesi bulunan bir görüntü her model için 20
tahmin kaydı üretir. Nitel rapor sayfalarında bu ayrı instance maskeleri
sahnenin tamamını inceleyebilmek için birleştirilerek gösterilir. Bu
görselleştirme tabloların instance-level hesap mantığını değiştirmez.

Dense sahnelerde GPU belleğini aşmamak için SAM1 ve SAM2 kutu istemleri en
fazla 16 kutuluk hesap parçalarına ayrılır. Instance listesi, kutu sırası ve
çıktı sayısı korunur; bu yalnız hesaplama batch'idir ve deney koşulu değildir.

`IoU ≥ 0.50/0.75/0.90`, ilgili eşiği geçen küçük araç örneklerinin oranıdır. Bunlar
COCO mask AP değildir. Confidence sırasındaki bütün tahmin maskelerini ve
yanlış pozitifleri kullanan uçtan uca COCO segmentation AP bu raporun
eşlenmiş instance-metrik protokolünden farklı bir değerlendirmedir. Ayrı
çalıştırılmadığı için bu eşik oranları AP veya mAP diye yeniden adlandırılmaz.

YOLO'nun kaçırdığı bir gerçek küçük araç, YOLO-bbox maske değerlendirmesinde boş
tahmin olarak kalır ve o instance'ın maske skorları sıfır olur. Hiçbir gerçek
küçük araçla eşleşmeyen yanlış pozitif YOLO kutuları ise sahte bir referans instance
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
