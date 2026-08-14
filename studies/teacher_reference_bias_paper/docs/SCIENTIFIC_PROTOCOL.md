# Bilimsel Protokol ve Metrik Sözleşmesi

## 1. Amaç

Ana deney değişkeni model değil, değerlendirme referansıdır. Aynı SAM1/2/3 tahminleri insan, yayımlanmış SAMRS ve model üretimli pseudo maskelere karşı ayrı ayrı ölçülür. Böylece skor değişiminin tahmin değişiminden değil referans değişiminden geldiği bilinir.

## 2. Deney Birimleri

- Dört deneyde 512'şer test görüntüsü vardır.
- Her deney `No Overlap/Overlap × Low/High Mask Area` biçiminde dört tabakaya ayrılır.
- Her tabaka tam 128 görüntüdür; Overall 512 görüntüdür.
- Ana metrik birimi nesne instance'ıdır, görüntü değildir.
- Dört test kümesi hedef-pozitif olacak biçimde seçilmiştir; her görüntüde en az bir hedef instance vardır.
- Kaynak sahne kimliği korunur; bootstrap aynı büyük sahneden gelen crop'ları bağımsız saymaz.
- iSAID ve SAMRS ayrı anotasyon ürünleridir, fakat ikisi de DOTA kökenli görüntüler içerir. Test görüntülerindeki kısmi örtüşme nedeniyle dört deney bağımsız replikasyon sayılmaz.

## 3. Overlap ve Mask Area

`Overlap`, aynı görüntüde en az bir hedef bbox çiftinin IoU değerinin `0.001` veya üstünde olmasıdır. Bu eşik yalnız “kutular hiç kesişiyor mu?” ayrımını yapar; yüksek kaliteli detection eşiği değildir.

`Low/High Mask Area`, tek bir nesnenin büyüklüğü değil, görüntüdeki bütün hedef maskelerin toplam alanının görüntü alanına oranıdır. Yüzde olarak eşikler iSAID Plane `%1.671`, iSAID Small Vehicle `%0.185`, SAMRS Plane `%1.193`, SAMRS Small Vehicle `%0.657`'dir. Tam hassasiyetli değerler deney config dosyalarında saklanır. Referans SAM1/2/3'e çevrildiğinde grup üyeliği yeniden hesaplanmaz.

iSAID'da resmi train ve validation anotasyon sürümlerindeki kaynak sahneler, kaynak-sahne güvenli özel train/validation/test bölünmesine yeniden ayrılmıştır; resmi iSAID test leaderboard protokolü kullanılmaz. SAMRS yayıncı split'inde bulunan kaynak-sahne çakışmaları da aynı nedenle özel bölünmede giderilmiştir.

## 4. Referanslar

### iSAID

- `human`: profesyonel insan instance maskeleri; bağımsız kontrol.
- `pseudo_sam1/2/3`: ilgili modele insan GT bbox verilerek üretilmiş instance maskesi.

### SAMRS

- `published_samrs_reference`: SAMRS veri setiyle yayımlanmış SAM1-türevi maske; insan GT değildir.
- `reproduced_pseudo_sam1`: bu çalışmanın dondurulmuş güncel SAM1 checkpoint'i ve yayımlanmış detection bbox'ı ile yeniden üretilmiş maske.
- `pseudo_sam2/3`: aynı bbox ile SAM2/3 tarafından üretilmiş maske.

SAMRS yayımlanmış ve yeniden üretilmiş SAM1 maskeleri ayrı referanslardır. Aralarındaki nesne-ortalama IoU Plane'de `0.991`, Small Vehicle'da `0.998`'dir; yakınlık aynı dosya oldukları anlamına gelmez.

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
Her deneyde detector yalnız tek hedef sınıf içerir; dolayısıyla sınıflar
üzerindeki ortalama olan mAP, bu düzenekte o tek sınıfın AP değerine eşittir.

Detector testlerinin tamamı hedef-pozitif 512 görüntüden oluşur. Bu nedenle AP/precision/recall değerleri resmi veri seti benchmark'ı değil, mask deneyinin seçilmiş test kümesindeki detector kontrolüdür.

## 8. Bilinen Pozitifte Boş Referans

Veri setinin temel anotasyonu nesnenin var olduğunu söylerken pseudo öğretmen boş maske döndürürse bu eksik pseudo etikettir. Aday tahmin de boş olsa bile iki boş maskeye IoU `1.0` verilmez; bütün maske metrikleri `0` kabul edilir. iSAID Small Vehicle SAM1 pseudo referansında `19/12.051` boş maske vardır; diğer kanonik referanslarda boşluk yoktur.

## 9. Identity Control

Bir modelin GT-bbox tahmini doğrudan kendi pseudo referansına kopyalandığı için dolu maskelerde IoU matematiksel olarak `1.0` olur. Bu hücre:

- modelin insan doğruluğu değildir;
- bağımsız benchmark sonucu değildir;
- yalnız RLE kopyası, instance kapsaması ve metrik kodu kontrolüdür.

Ana bulgu YOLO-bbox koşulundaki eşleşmiş farktır. Burada teacher referansı GT bbox'tan, aday tahmin YOLO bbox'tan geldiği için iki maske özdeş değildir.

## 10. İstatistik

- Temel referans değişimi: aynı nesne için modelin kendi etiketindeki IoU eksi insan/yayımlanmış temel etiketteki IoU. Bu değer yalnız referans değişince puanın ne kadar değiştiğini gösterir.
- Ana ek IoU: aynı dondurulmuş modelin kendi ürettiği etiketteki IoU'su eksi diğer iki SAM etiketindeki ortalama IoU'su. Pozitif değer, modelin kendi etiketinde daha yüksek puan aldığını gösterir.
- İkincil istatistiksel kontrol: modelin diğer modellere göre avantajının kendi etiketinde, temel etikete kıyasla ne kadar değiştiği de hesaplanır. Bu kontrol teknik analiz CSV'sinde tutulur ve ana okuyucu tablosuna basılmaz.
- Güven aralığı: 10.000 tekrar, kaynak-sahne kümeli bootstrap, `%95`.
- Ek sonuçlar: model sıralaması, teacher advantage, referanslar arası instance IoU ve boş maske oranı.
- Nitel örnek seçimi: model ve referans skorlarından bağımsız, tabaka içi
  `mask_area_ratio` medyanına en yakın görüntü; dört referansta aynı görüntüler
  ve seçilen görüntüdeki bütün hedef instance'lar.
- Farklı veri setleri tek bir global IoU ortalamasında birleştirilmez.

Bu kontrastlar ilk sonuçlar görüldükten sonra geliştirilmiştir; preregistered
confirmatory test değildir ve çoklu karşılaştırma düzeltmesi uygulanmamıştır.
Bu nedenle etki büyüklükleri ve güven aralıkları exploratory kanıt olarak
yorumlanır. Aynı-model sonucu yalnız kullanılan dondurulmuş SAM1/2/3
checkpoint'leri için geçerlidir; farklı eğitim seed'i/checkpoint'i ve model
ailesi düzeyinde genelleme test edilmemiştir.

GT-bbox doğrudan kontrastları identity kontrol etkisini içerir. Bilimsel ana yorum, öğretmen referansının GT bbox ile; değerlendirilen adayın ise eşleşmiş YOLO bbox ile üretildiği non-identical koşula dayanır.

## 11. Deneyler Arası Bağımlılık ve Exploratory Audit

Piksel olarak birebir aynı test görüntüsü sayıları `docs/DEEP_SCIENTIFIC_AUDIT.md` içinde kayıtlıdır. Örneğin iSAID Plane ile SAMRS Plane arasında 19, iSAID Small Vehicle ile SAMRS Small Vehicle arasında 6 ortak test görüntüsü vardır. Bu nedenle sonuçlar deney içinde eşleşmiş olarak yorumlanır; dört deneylik pooled p-değeri veya bağımsız tekrar iddiası kurulmaz.

Ortak görüntülerde insan ve yayımlanmış SAMRS maskeleri arasındaki post-hoc eşleştirme yalnız exploratory'dir. Alt küme temsili seçilmediği ve anotasyon kapsamı farklı olduğu için ana hipotez testi veya benchmark sonucu değildir.

## 12. Geçerli İddia Sınırı

Çalışma “pseudo etiketler yararsızdır” demez. Sonuçlar, kullanılan dondurulmuş checkpoint'ler ve seçilmiş test kapsamı içinde, model üretimli test cetveli ile aday arasındaki bağımlılığın skoru ve model sıralamasını değiştirebildiğini destekler. Pseudo üretimi insan/yayın kutusu lokalizasyonunu kullandığı için deney tam otomatik etiketleme hattı değildir ve yalnız maske sınırı etkisini izole etmez. Eğitim yararlılığı bağımsız insan testinde ayrı ölçülmelidir.
