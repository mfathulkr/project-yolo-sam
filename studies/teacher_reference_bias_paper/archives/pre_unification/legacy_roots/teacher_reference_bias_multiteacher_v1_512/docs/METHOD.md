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
- GT bbox diagonal hücreleri aynı tahmin maskesini kendi pseudo referansıyla karşılaştıran `identity control` hücreleridir. Referans doluysa skor 1,0; bilinen GT nesnesine rağmen öğretmen boş maske üretmişse bu eksik pseudo etikettir ve skor 0,0'dır. Dolayısıyla diagonal ortalama aynı zamanda öğretmenin referans kapsamasını gösterir; bağımsız segmentasyon başarısı değildir.
- YOLO bbox diagonal hücreleri özdeş değildir; fakat öğretmen ve adayın model ailesi aynı olduğundan korelasyonlu hata avantajını ölçebilir.

## Boş Maskeler

Bu çalışmadaki her satır, insan anotasyonuyla varlığı bilinen bir hedef nesneyi temsil eder. Bu nedenle öğretmenin GT bbox verilmesine rağmen boş maske üretmesi gerçek bir negatif örnek değil, eksik pseudo etikettir. Böyle bir referansta tahmin de boş olsa bile IoU, Dice, Precision ve Recall 0,0 kabul edilir. Boş çıktı silinmez; başarısızlık ve referans kapsama kaybı olarak ayrıca raporlanır.

Genel amaçlı maske metrik fonksiyonu, gerçekten hedef nesne bulunmayan görüntüler için boş-boş çiftini ayrı bir bağlamda 1,0 sayabilir. Instance deneyinde bu davranış açık `known_positive_instance` politikasıyla kapatılmıştır. Raporlanan boş oranı, yalnız RLE içindeki gerçek piksel alanından hesaplanır ve kayıt durumuyla uyuşması zorunludur.

## SAM3 BBox Arayüzü

`PVS`, Promptable Visual Segmentation demektir: kutu, nokta veya maske gibi
uzamsal bir istemle belirli bir nesne örneği segmentlenir. `PCS`, Promptable
Concept Segmentation demektir: metin veya görsel örnekle aynı kavrama uyan
bütün nesneler aranır. Dolayısıyla PCS içindeki bir görsel örnek kutusu,
SAM1/SAM2 tarzı "yalnız bu kutudaki nesneyi segmentle" kutusu değildir.

Bu deneyde `Sam3TrackerProcessor` ve `Sam3TrackerModel` ile PVS kullanılır.
`multimask_output=False` olduğundan her GT veya YOLO kutusu için tek maske
üretilir. Maske logit eşiği 0,0'dır; kutu sayısı ile maske sayısı eşit değilse
koşu hata verir. PVS ve PCS'nin tam akışı, checkpoint hash'leri ve yanlış
alternatiflerin etkisi [tekrar üretim rehberinde](REPRODUCIBILITY_FIELD_GUIDE.md)
açıklanmıştır.

Önceki PCS uygulamasında küçük nesne adaylarına uygulanan 0,5 çıktı olasılığı filtresi çok sayıda boş maske yaratmıştı. Bu çıktılar geçersiz sayılarak SAM3 GT/YOLO tahminleri, pseudo referanslar, metrikler ve raporlar PVS ile yeniden üretilmiştir.

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

## Tekrar Üretim Sözleşmesi

Model arayüzü, checkpoint, örnekleme, detector confidence seçimi, YOLO-GT
eşleştirmesi, pseudo-reference soy zinciri, boş maske politikası, metrik
granülaritesi, bootstrap ve raporlama kararlarının tam listesi
[Kritik Uygulama Kararları ve Tekrar Üretim Rehberi](REPRODUCIBILITY_FIELD_GUIDE.md)
dosyasındadır. Bildirideki yöntem bu belgeyle birlikte okunmalıdır.
