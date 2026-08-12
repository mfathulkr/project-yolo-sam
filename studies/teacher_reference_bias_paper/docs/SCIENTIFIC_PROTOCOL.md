# Bilimsel Protokol ve Metrik Sözleşmesi

## 1. Amaç

Ana deney değişkeni model değil, değerlendirme referansıdır. Aynı SAM1/2/3 tahminleri insan, yayımlanmış SAMRS ve model üretimli pseudo maskelere karşı ayrı ayrı ölçülür. Böylece skor değişiminin tahmin değişiminden değil referans değişiminden geldiği bilinir.

## 2. Deney Birimleri

- Dört deneyde 512'şer test görüntüsü vardır.
- Her deney `No Overlap/Overlap × Low/High Mask Area` biçiminde dört tabakaya ayrılır.
- Her tabaka tam 128 görüntüdür; Overall 512 görüntüdür.
- Ana metrik birimi nesne instance'ıdır, görüntü değildir.
- Kaynak sahne kimliği korunur; bootstrap aynı büyük sahneden gelen crop'ları bağımsız saymaz.

## 3. Overlap ve Mask Area

`Overlap`, aynı görüntüde en az bir hedef bbox çiftinin IoU değerinin `0.001` veya üstünde olmasıdır. Bu eşik yalnız “kutular hiç kesişiyor mu?” ayrımını yapar; yüksek kaliteli detection eşiği değildir.

`Low/High Mask Area`, temel veri seti etiketlerindeki bütün hedef instance alanlarının görüntü alanına oranıyla belirlenir. Eşik her veri seti/sınıf için testten önce dondurulmuştur. Referans SAM1/2/3'e çevrildiğinde tabaka üyeliği yeniden hesaplanmaz.

## 4. Referanslar

### iSAID

- `human`: profesyonel insan instance maskeleri; bağımsız kontrol.
- `pseudo_sam1/2/3`: ilgili modele insan GT bbox verilerek üretilmiş instance maskesi.

### SAMRS

- `published_samrs_reference`: SAMRS veri setiyle yayımlanmış SAM1-türevi maske; insan GT değildir.
- `reproduced_pseudo_sam1`: bu çalışmanın dondurulmuş güncel SAM1 checkpoint'i ve yayımlanmış detection bbox'ı ile yeniden üretilmiş maske.
- `pseudo_sam2/3`: aynı bbox ile SAM2/3 tarafından üretilmiş maske.

SAMRS published ve reproduced SAM1 ayrı referanslardır. Plane'de aralarındaki instance-macro IoU `0.990633`, Small Vehicle'da `0.998338` çıkmıştır; yakınlık aynı dosya oldukları anlamına gelmez.

## 5. Tahmin Koşulları

Her model iki bbox kaynağıyla çalışır:

- `gt_bbox`: veri setinin özgün horizontal detection kutusu;
- `yolo_bbox`: seed 42 detector tahmini, confidence sıralı greedy bire bir eşleştirme ve bbox IoU `>=0.50` kabul kuralı.

YOLO'nun kaçırdığı GT instance boş tahmin ve sıfır maske skoru alır. Eşleşmeyen yanlış pozitif YOLO kutusu maske ortalamasına sahte bir GT instance olarak eklenmez; detector precision, recall ve bbox AP'de cezalandırılır.

## 6. TP, FP, FN ve Maske Metrikleri

Bir instance için:

- `TP`: tahmin ve referansta nesne olan piksel;
- `FP`: yalnız tahminde nesne olan piksel;
- `FN`: yalnız referansta nesne olan piksel.

Metrikler:

- `IoU = TP / (TP + FP + FN)`;
- `Dice = 2TP / (2TP + FP + FN)`;
- `Precision = TP / (TP + FP)`;
- `Recall = TP / (TP + FN)`.

Her metrik önce instance başına hesaplanır, sonra instance'lar eşit ağırlıkla ortalanır. Bu `instance-macro` değerlendirmedir; büyük maskeler sonucu piksel sayısıyla perdelemez.

`IoU >= 0.50/0.75/0.90` sütunları, eşiği geçen instance oranıdır. Bunlar confidence sıralı precision-recall eğrisi kullanmadığı için AP veya mAP değildir.

## 7. Detector Metrikleri

- `BBox Precision@t = TP / (TP + FP)`;
- `BBox Recall@t = TP / (TP + FN)`;
- `BBox APt`, confidence eşiği taranırken oluşan precision-recall eğrisinin alanıdır;
- `BBox mAP50-95`, bbox IoU 0.50:0.05:0.95 eşiklerindeki AP ortalamasıdır.

Buradaki IoU, iki bbox'ın geometrik örtüşmesidir; maske IoU değildir.

## 8. Bilinen Pozitifte Boş Referans

Veri setinin temel anotasyonu nesnenin var olduğunu söylerken pseudo öğretmen boş maske döndürürse bu eksik pseudo etikettir. Aday tahmin de boş olsa bile iki boş maskeye IoU `1.0` verilmez; bütün maske metrikleri `0` kabul edilir. iSAID Small Vehicle SAM1 pseudo referansında `19/12.051` boş maske vardır; diğer kanonik referanslarda boşluk yoktur.

## 9. Identity Control

Bir modelin GT-bbox tahmini doğrudan kendi pseudo referansına kopyalandığı için dolu maskelerde IoU matematiksel olarak `1.0` olur. Bu hücre:

- modelin insan doğruluğu değildir;
- bağımsız benchmark sonucu değildir;
- yalnız RLE kopyası, instance kapsaması ve metrik kodu kontrolüdür.

Ana bulgu YOLO-bbox koşulundaki eşleşmiş farktır. Burada teacher referansı GT bbox'tan, aday tahmin YOLO bbox'tan geldiği için iki maske özdeş değildir.

## 10. İstatistik

- Ana etki: aynı instance için `pseudo IoU - temel referans IoU`.
- Güven aralığı: 10.000 tekrar, kaynak-sahne kümeli bootstrap, `%95`.
- Ek sonuçlar: model sıralaması, teacher advantage, referanslar arası instance IoU ve boş maske oranı.
- Farklı veri setleri tek bir global IoU ortalamasında birleştirilmez.

## 11. Geçerli İddia Sınırı

Çalışma “pseudo etiketler yararsızdır” demez. Gösterdiği şey, model üretimli etiket test cetveli olduğunda cetvel ile aday arasındaki ortak hata stilinin skoru ve model sıralamasını değiştirebilmesidir. Eğitim yararlılığı bağımsız insan testinde ayrı ölçülmelidir.
