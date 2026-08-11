# Yöntem ve Deney Sınırları

## Araştırma Sorusu

Bir segmentasyon modeli tarafından oluşturulan maskeler bağımsız insan referansı yerine değerlendirme referansı olarak kullanıldığında, referansı üreten model ve aynı model ailesinden adaylar yapay olarak avantaj kazanıyor mu?

Bu soru pseudo etiketlerin eğitimde yararlı olup olmadığından farklıdır. Çalışma, pseudo etiketli bir eğitim kümesinin faydasını değil, model üretilmiş bir test referansının ölçüm geçerliliğini inceler.

## Kontrollü Tasarım

Değişmeyen girdiler:

- iSAID görüntüleri ve seçilen instance kimlikleri;
- `overlap/no_overlap × low/high mask area` tabakaları;
- insan GT bbox istemleri;
- seed 42 YOLO bbox istemleri;
- SAM1, SAM2 ve SAM3'ün dondurulmuş tahmin dosyaları;
- eşleştirme ve metrik kodu.

Değişen tek değerlendirme girdisi referans maskedir:

1. İnsan tarafından çizilmiş iSAID maskesi.
2. SAM1'in insan GT bbox ile ürettiği pseudo maske.
3. SAM2'nin aynı insan GT bbox ile ürettiği pseudo maske.
4. SAM3'ün aynı insan GT bbox ile ürettiği pseudo maske.

Bu nedenle skor değişimi yeni bir inference çalıştırmasının veya farklı bir örnek seçiminin sonucu değildir.

## Deney Birimi

Temel deney birimi instance'dır. Her hedef nesne için tahmin ve referans maskesi ayrı karşılaştırılır. Avg IoU, instance IoU değerlerinin aritmetik ortalamasıdır; büyük nesneler daha fazla piksele sahip oldukları için daha yüksek ağırlık almaz.

Görüntü ve kaynak sahne bilgisi korunur. Güven aralıklarında aynı kaynak sahneden gelen crop'lar bağımsız örnekler gibi tekrar tekrar sayılmaz.

## Metrikler

Her instance için:

- `IoU = TP / (TP + FP + FN)`
- `Dice = 2TP / (2TP + FP + FN)`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`

Buradaki TP, FP ve FN piksel sayılarıdır. `IoU >= 0.50`, `0.75` ve `0.90` başarı oranları, ilgili eşikten geçen instance oranını gösterir; COCO mAP değildir. YOLO detector tablosundaki bbox AP değerleri ise kutu eşleştirmesi ve güven skorları üzerinden hesaplanan gerçek detection metrikleridir.

Ana bildiri analizi Avg IoU kullanır. Full-metric belgeler aynı örneklerde Avg Dice, Avg Precision, Avg Recall ve eşik oranlarını da gösterir.

## İstatistik

- Her pseudo referans ile insan referansı arasındaki skor farkı instance bazında eşleştirilir.
- Yüzde 95 güven aralığı 10.000 tekrar ile source-scene-clustered bootstrap üzerinden hesaplanır.
- Her pseudo referans için model sıralaması çıkarılır ve insan referansındaki sıralamayla karşılaştırılır.
- `teacher advantage`, pseudo referansı üreten modelin skoru ile diğer iki modelin ortalama skoru arasındaki farktır.
- GT bbox diagonal hücreleri aynı maskenin kendisiyle karşılaştırılmasıdır ve zorunlu olarak IoU 1,0 verir. Bunlar performans sonucu değil `identity control` olarak raporlanır.
- YOLO bbox diagonal hücreleri özdeş değildir; fakat öğretmen ve adayın model ailesi aynı olduğundan korelasyonlu hata avantajını ölçebilir.

## Boş Maskeler

`empty_mask` durumu başarısız inference olarak silinmez; modelin dondurulmuş çıktısı olarak değerlendirmeye dahil edilir. Boş referans ve boş tahmin çifti IoU 1,0 üretebildiği için boş maske sayısı ve oranı ayrıca raporlanır.

En kritik durum SAM3 Small Vehicle'dır: 12.051 pseudo referansın 5.345'i, yani yaklaşık yüzde 44,4'ü boştur. Bu nedenle yalnız ortalama IoU'ya bakılarak SAM3 pseudo referansının kalitesi yorumlanamaz.

## Geçerli Yorum

Deneyin desteklediği sav:

> Model üretilmiş bir segmentasyon referansı, referans üreticisiyle hata biçimi benzer olan adayları sistematik olarak avantajlı gösterebilir ve bağımsız insan referansındaki model sıralamasını değiştirebilir.

Deneyin desteklemediği savlar:

- Bütün pseudo etiketler yararsızdır.
- SAMRS eğitim için değersizdir.
- İnsan maskeleri hatasız mutlak gerçektir.
- Gözlenen büyüklük bütün sınıflara, veri setlerine ve prompt türlerine aynen genellenir.

## Eğitim ile Değerlendirme Ayrımı

Pseudo etiket eğitimde ek veri, ön eğitim veya weak supervision sağlayabilir. Bu kullanımda başarı, bağımsız insan etiketli validation/test kümesinde ölçülmelidir. Aynı pseudo etiket üreticisinin çıktısını hem öğrenme hedefi hem test referansı yapmak veya üreticiyi kendi çıktısıyla değerlendirmek ölçüm bağımsızlığını bozar.

SAMRS makalesi veri setini özellikle segmentation pre-training için konumlandırır. Bizim bulgumuz bu kullanım amacıyla çelişmez; SAMRS/SOTA maskelerini bağımsız bir model karşılaştırma cetveli gibi yorumlamanın riskini gösterir.
