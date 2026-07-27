# Teacher Reference Bias v2 Deney Planı

## 1. Araştırma Sorusu

Bir segmentasyon modeliyle otomatik üretilen maskeler bağımsız test etiketi
olarak kullanıldığında:

1. Aynı model ailesinin ölçülen başarısı ne kadar yükselir?
2. İnsan referansına göre model sıralaması değişir mi?
3. GT bbox yerine eğitilmiş YOLO bbox kullanmak sonuçlara ne kadar ek kayıp
   getirir?
4. Bu etkiler overlap ve toplam maske alanı koşullarında nasıl değişir?

Çalışma, pseudo etiketlerin eğitim veya ön-etiketleme amacıyla kullanılamaz
olduğunu iddia etmez. İncelenen risk, model üretimli referansın bağımsız test
ground truth'u gibi yorumlanmasıdır.

## 2. Ana Hipotezler

- **H1 - Skor enflasyonu:** Aynı iSAID tahminleri SAM1 pseudo referansına
  karşı, insan referansına kıyasla daha yüksek IoU alacaktır.
- **H2 - Üretici yakınlığı:** Skor artışı, pseudo referansı üreten SAM1 için
  SAM2 ve SAM3'e göre daha büyük olacaktır.
- **H3 - Sıralama duyarlılığı:** Referans kaynağı yalnız mutlak skoru değil,
  gözlenen model sıralamasını da değiştirebilir.
- **H4 - Detector kaybı:** YOLO bbox koşulu, GT bbox koşuluna göre daha düşük
  maske kalitesi gösterecektir.
- **H5 - Sahne koşulu:** Overlap ve düşük/yüksek toplam maske alanı grupları
  aynı modeller için farklı zorluk düzeyleri oluşturacaktır.

## 3. Veri Setleri ve Hedef

### 3.1 iSAID Plane

- Hedef sınıf: `plane`
- Birincil referans: resmi insan çizimli instance maskeleri
- Kontrollü ikinci referans: aynı GT bbox'ların SAM1 ViT-H modeline verilmesi
  ile dondurulan pseudo maskeler
- GT bbox kaynağı: resmi insan instance anotasyonu
- Veri seti: https://captain-whu.github.io/iSAID/
- Makale: https://arxiv.org/abs/1905.12886

### 3.2 SAMRS SOTA Plane

- Hedef sınıf: `plane`
- Referans: yayımlanan SAM1 ViT-H kaynaklı pseudo maskeler
- GT bbox kaynağı: yayımlanan özgün detection anotasyonu
- Bbox pseudo maskeden yeniden türetilmez
- Veri seti ve kod: https://github.com/ViTAE-Transformer/SAMRS
- Makale: https://arxiv.org/abs/2305.02034

SAMRS sonucu insan çizimli bağımsız ground truth başarısı olarak sunulmaz.
Bu koşul, model üretimli referansın davranışını dış veri setinde gösteren
ikinci pseudo-reference deneyidir.

## 4. Deney Birimi ve Split

- Görüntü girişi: `1024×1024`
- Ana değerlendirme birimi: ayrı uçak örneği
- Split birimi: kaynak sahne
- Aynı kaynak sahnenin tile'ları train, validation ve test arasında
  paylaştırılmaz
- Detector ağırlıkları veya test tahminleri veri setleri arasında
  paylaşılmaz
- iSAID eğitim bölümü 1.571, SAMRS eğitim bölümü 2.191 görüntüdür. Aynı
  epok ve hiperparametre ayarları aynı optimizasyon adımı sayısı anlamına
  gelmediği için veri setleri arası detector farkı kontrollü referans etkisi
  olarak yorumlanmaz.

Her veri setinin test kümesi:

| Alt grup | Görüntü |
|---|---:|
| No Overlap × Low Mask Area | 128 |
| No Overlap × High Mask Area | 128 |
| Overlap × Low Mask Area | 128 |
| Overlap × High Mask Area | 128 |
| Overall | 512 |

Alt gruplar birbirini dışlar. Overall, dört grubun birleşimidir.
Strata tanımı nedeniyle testteki bütün görüntüler en az bir uçak içerir.
Detector sonucu bu dengeli pozitif test alt kümesinin COCO bbox
değerlendirmesidir; negatif görüntüler dâhil resmi tam benchmark olarak
sunulmaz.

### 4.1 Overlap Tanımı

- `No Overlap`: görüntüdeki bütün GT bbox çiftlerinde IoU tam `0`
- `Overlap`: en az bir GT bbox çiftinde IoU `≥ 0,001`
- İki sınır arasındaki belirsiz adaylar test kümesine alınmaz

### 4.2 Mask Area Tanımı

Low/High ayrımı, görüntüdeki toplam referans uçak maskesi alanının görüntü
alanına oranına göre yapılır. Veri setine özgü eşik aday havuzundan test
sonuçlarına bakılmadan önce dondurulur. Bu eşik model sonucuna göre
değiştirilmez.

## 5. Model ve Koşul Matrisi

Segmenterler:

- SAM1 ViT-H
- SAM2.1 Hiera Large
- SAM3

SAM modelleri iSAID veya SAMRS üzerinde yeniden eğitilmez ve ince ayar
görmez. Yalnız bbox istemi kullanılır.

Her veri setindeki koşullar:

| Model | GT bbox | YOLO bbox |
|---|---:|---:|
| SAM1 | 1 sabit koşul | 3 detector seed |
| SAM2 | 1 sabit koşul | 3 detector seed |
| SAM3 | 1 sabit koşul | 3 detector seed |

Her veri setinde 3 GT-bbox ve 9 YOLO-bbox tahmin koşulu vardır.

### 5.1 Detector Protokolü

- Mimari: YOLO26x
- Seed: `42`, `123`, `2026`
- Başlangıç ağırlığı: aynı `models/yolo/yolo26x.pt`
- Giriş: `1024×1024`
- Üst sınır: 100 epok
- Batch: 12
- Patience: 30
- Deterministik çalışma: açık
- Confidence seçimi: yalnız validation kümesinde bbox IoU 0,50 kabul eşiğinde
  maksimum F1
- Seçilen confidence: testten önce dondurulur ve hem detector testi hem
  YOLO-bbox SAM çıkarımı için aynen kullanılır

## 6. Referans Karşılaştırmaları

### 6.1 Kontrollü iSAID Karşılaştırması

Her iSAID uçak örneğinde aşağıdakiler sabit kalır:

- görüntü
- instance kimliği
- GT veya YOLO bbox istemi
- SAM tahmini
- detector seed

Yalnız değerlendirme referansı değişir:

1. İnsan maskesi
2. SAM1 GT-bbox pseudo maskesi

Bu eşlenmiş tasarım veri seti, bbox veya tahmin farkını ortadan kaldırır.
Ölçülen fark doğrudan referans kaynağına duyarlılıktır.

Kontrollü pseudo referans doğrudan SAM1 GT-bbox tahmininden üretildiği için
SAM1 GT-bbox satırı bir kimlik kontrolüdür. Bu satır bağımsız model başarısı
olarak yorumlanmaz.

### 6.2 SAMRS Karşılaştırması

SAMRS resmi maskeleri ayrı bir üretim hattında SAM1 ViT-H ile oluşturulmuştur.
Bu nedenle SAM1 sonucu kontrollü iSAID kimlik kontrolü kadar birebir olmak
zorunda değildir; yine de üretici model ailesine yakınlığı ölçer.

## 7. Metrik Sözleşmesi

### 7.1 Maske Metrikleri

Her uçak örneği için:

- `IoU = TP / (TP + FP + FN)`
- `Dice = 2TP / (2TP + FP + FN)`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`

Önce her uçak ayrı hesaplanır, sonra uçaklar eşit ağırlıkla ortalanır. Büyük
nesneler daha çok piksele sahip oldukları için küçük nesnelerin sonuçlarını
perdeleyemez.

Ek başarı oranları:

- `IoU ≥ 0.50`
- `IoU ≥ 0.75`
- `IoU ≥ 0.90`

Bu oranlar ilgili eşiği geçen uçak örneklerinin payıdır; COCO AP veya mAP
değildir.

YOLO'nun kaçırdığı bir gerçek uçak boş maske olarak değerlendirilir ve o
instance için maske skorları sıfır olur. Hiçbir GT ile eşleşmeyen yanlış
pozitif kutular sahte bir GT instance oluşturarak maske ortalamasına eklenmez;
etkileri detector tablosunda ölçülür.

### 7.2 Detector Metrikleri

- BBox mAP50
- BBox mAP75
- BBox mAP90
- BBox mAP50-95
- BBox Precision@0.50 / Recall@0.50
- BBox Precision@0.75 / Recall@0.75
- BBox Precision@0.90 / Recall@0.90

AP değerleri gerçek COCO bbox değerlendirmesidir. Precision/Recall,
validation'da seçilen sabit confidence noktasında ölçülür.

### 7.3 Raporlanmayan Metrikler

- Uydurma `mAP proxy`
- `Pred/GT Area`
- Sunum kapsamı dışında bırakılan `Boundary IoU`
- Confidence-sıralı ayrı bir COCO segmentation değerlendirmesi

Gerçek COCO mask AP ayrı bir uçtan uca protokol gerektirir; instance IoU eşik
oranları mask AP diye yeniden adlandırılmaz.

## 8. İstatistiksel Analiz

- GT-bbox koşulları tek sabit çalışma olarak raporlanır
- YOLO-bbox koşulları üç seed ortalaması ± standart sapması olarak raporlanır
- IoU belirsizliği kaynak sahne kümeli 10.000 bootstrap ile hesaplanır
- Model çiftleri aynı instance üzerinde eşlenir
- İnsan-pseudo enflasyonu aynı instance ve tahmin üzerinde hesaplanır
- Model sıralamaları insan ve pseudo referansta ayrı karşılaştırılır
- Yakın nokta tahminleri tek başına istatistiksel üstünlük sayılmaz

## 9. Çıktılar

Üç bağımsız full-metric belge üretilecektir:

1. `isaid_plane_human`
2. `isaid_plane_pseudo_sam1`
3. `samrs_sota_plane`

Her biri aşağıdaki dosyaları içerir:

- Markdown
- renkli DOCX
- renkli PDF
- detector CSV
- Overall ve dört alt grup CSV'si
- dört nitel örnek görseli
- giriş/çıkış SHA-256 manifesti

Her PDF'de:

- metric ve TP/FP/FN açıklaması
- gerçek YOLO bbox tablosu
- Overall + dört alt grup segmentasyon tablosu
- yalnız SAM1/SAM2/SAM3 × GT/YOLO bbox satırları
- görünür GT bbox içeren dört nitel örnek
- dinamik Findings/Discussion

Başarı hücreleri `0,0-1,0` aralığında kırmızı-sarı-yeşil ölçekle
renklendirilir.

## 10. Kalite Kapıları

Deney ancak aşağıdakilerin tamamı sağlandığında bitmiş sayılır:

- Her veri setinde 512 test görüntüsü ve tam 4×128 alt grup
- iSAID 5.447, SAMRS 3.713 test instance'ı
- Train/validation/test kaynak sahne kesişimi sıfır
- Altı detector eğitimi ve altı gerçek COCO bbox testi
- 24 tahmin manifesti
- 24 değerlendirme manifesti
- 175.284 benzersiz canonical instance-metric satırı
- 180 aggregate satırı
- Eksik/tekrarlı instance anahtarı sıfır
- Bütün maske metrikleri sonlu ve `[0,1]` içinde
- Validation confidence ile test/SAM confidence birebir aynı
- Üç MD/DOCX/PDF ve rapor hashleri geçerli
- DOCX zip bütünlüğü ve PDF metin çıkarma kontrolü
- İlk sayfa, detector, Overall, dört nitel örnek ve Discussion görsel QA
- Ortak/study testleri ve repository layout doğrulaması

## 11. İddia Sınırı

Bu çalışma:

- SAMRS'in eğitim verisi olarak yararsız olduğunu söylemez
- SAM1'in insan etiketlerinde kusursuz olduğunu söylemez
- IoU eşik oranlarını mask mAP olarak sunmaz
- Farklı veri setlerindeki ham skorları tek başına adil model karşılaştırması
  saymaz
- Farklı eğitim görüntüsü ve optimizasyon adımı sayılarından doğan detector
  farkını yalnız pseudo etiket kaynağına bağlamaz

Savunulan temel sonuç şudur:

> Test referansı değerlendirilmekte olan model veya aynı model ailesiyle
> üretilirse, ölçülen başarı bağımsız insan etiketine göre yapay biçimde
> yükselebilir ve görünen model sıralaması referans üreticisine duyarlı hâle
> gelebilir.
