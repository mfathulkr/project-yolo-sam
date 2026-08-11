# WORKLOG

## Amaç

Bu dosya, refactor ve teacher-reference bias deneyinin ilerlemesini teknik ayrıntıya boğmadan izlemek için tutulur. Her ana aşamadan sonra:

- Ne yapıldığı,
- Ne doğrulandığı,
- Hangi sorun bulunduğu,
- Bir sonraki adımın ne olduğu

kısa ve anlaşılır biçimde eklenir.

Bu kayıt sonuç dosyalarının yerine geçmez. Ayrıntılı teknik kararlar planlarda, makine tarafından okunabilir çalışma bilgileri run manifestlerinde tutulur.

## Güncel Durum

**Tarih:** 2026-08-11
**Aktif aşama:** Tamamlandı
**Genel durum:** Canonical iSAID Plane ve Small Vehicle deneyleri SAM2 ve SAM3
pseudo referanslarıyla genişletildi. Aynı frozen SAM1/SAM2/SAM3 tahminleri
human, pseudo-SAM1, pseudo-SAM2 ve pseudo-SAM3 referanslarında çapraz ölçüldü.
Dört yeni full-metric rapor, öğretmen karşılaştırma raporu, güncel literatür
incelemesi, `elektr` Overleaf iskeleti ve yayın tablo/figür paketi tamamlandı.
Uçtan uca validator ile 62 test ve 2 alt test geçti.

## Son Çalışmalar

### 2026-08-11 - Çok öğretmenli pseudo referans uzantısı tamamlandı

Yapılanlar:

- iSAID Plane ve Small Vehicle için SAM2/SAM3 GT-bbox pseudo referansları
  oluşturuldu; mevcut SAM1 referansı ve human referansla ortak matrise alındı.
- Dört adet 14 sayfalık full-metric PDF ve 10 sayfalık SAM1/SAM2/SAM3
  karşılaştırma PDF'si üretildi.
- Remote sensing, medical imaging, imperfect reference standards, circular
  evaluation ve model-generated benchmark self-bias literatürü incelendi.
- Kullanıcının `elektr` şablonunu koruyan Overleaf bölüm planı ile 7 tablo ve
  7 figürden oluşan yayın paketi hazırlandı.

Doğrulananlar:

- 1.024 görüntü ve 17.498 instance üzerinde 419.952 çapraz metrik satırı var.
- Her pseudo referans YOLO bbox koşulunda kendi öğretmenini birinci sıraladı.
- Small Vehicle SAM3 pseudo referansındaki 5.345/12.051 boş maske gizlenmedi;
  rapor ve ana figürlerde açıkça gösterildi.
- Dört full-metric PDF sayfa/render, DOCX bütünlük, hash/manifest ve metin
  kontrollerinden geçti; 62 test ve 2 alt test başarılı oldu.

Basit sonuç:

> Pseudo maskenin üreticisi değiştiğinde aynı sabit tahminlerin skorları ve
> model sırası değişiyor. Bu etki yalnız SAM1'e özgü değil; her üç öğretmen
> kendi referansında avantaj kazanıyor. Pseudo etiket eğitimde yararlı olabilir,
> fakat bağımsız test ground truth'u gibi kullanılması ölçüm yanlılığı yaratıyor.

### 2026-07-27 - Nitel görseller bütün instance'ları gösterecek biçimde düzeltildi

Yapılanlar:

- Full metric belgelerdeki dört nitel sayfa tek bir seçilmiş instance yerine
  seçilen görüntüdeki bütün GT uçak kutularını gösterir hale getirildi.
- SAM1, SAM2 ve SAM3 tarafından her kutu için ayrı üretilen maskeler nitel
  panelde birleştirilerek sahne düzeyinde TP/FP/FN görünümü oluşturuldu.
- Görselde instance sayısı ve yalnız nitel inceleme amacı taşıyan union IoU
  açıkça yazıldı; ana tablolar instance-level bırakıldı.

Doğrulananlar:

- Eski iSAID insan raporunun 12. sayfasındaki `P2249_0014` görüntüsünde 20
  GT uçak bulunduğu ve SAM1/SAM2/SAM3 prediction dosyalarının her birinde 20
  tahmin kaydının eksiksiz olduğu doğrulandı.
- Bu görüntünün toplam uçak maskesi 3.380 piksel (`%0,322`), dondurulmuş
  high/low eşiği `%1,671` ve 10 bbox-overlap çifti bulunduğu için
  `Overlap / Low Mask Area` etiketi doğrudur.
- Yenilenen iSAID insan raporunun 12. sayfasında seçilen sahnedeki 13
  instance'ın bütün kutuları ve birleşik model maskeleri görünür durumdadır.
- iSAID insan, iSAID SAM1 pseudo ve SAMRS rapor validator'ları yeniden geçti.

Basit sonuç:

> Model koşularında nesne kaybı yoktu; hata, nitel görselin yalnız bir
> instance'ı göstermesi nedeniyle deney kapsamını eksik anlatmasıydı.
> Rapor artık her seçilmiş sahnedeki bütün hedefleri gösteriyor.

### 2026-07-27 - Canonical v2 deney ve refactor kapatıldı

Yapılanlar:

- iSAID ve SAMRS test kümeleri dört overlap × mask-area grubunda tam 128'er,
  Overall'da 512 görüntü olacak biçimde yeniden hazırlandı.
- İki veri setinde üç seed'li altı YOLO26x detector eğitimi ve gerçek bbox
  değerlendirmesi tamamlandı.
- SAM1, SAM2 ve SAM3 için GT-bbox ve YOLO-bbox koşulları çalıştırıldı; iSAID
  tahminleri insan ve kontrollü SAM1 pseudo referansına ayrı ayrı ölçüldü.
- iSAID insan, iSAID SAM1 pseudo ve SAMRS resmi SAM1 pseudo referansı için
  üç ayrı renkli full metric Markdown, DOCX ve PDF üretildi.
- Eski config, sonuç, rapor ve kodlar sahip study klasörlerine taşındı; kökte
  yalnız ortak kod, model, veri, araç ve dokümantasyon bırakıldı.

Doğrulananlar:

- iSAID test kümesinde 512 görüntü ve 5.447 plane instance; SAMRS test
  kümesinde 512 görüntü ve 3.713 plane instance bulunuyor.
- Train/validation/test kaynak sahne kesişimleri sıfırdır.
- Canonical analiz 175.284 instance satırı ve 180 aggregate satırı içeriyor.
- Ortak kodda 54, study unit paketinde 32 ve integration paketinde 2 olmak
  üzere 88 test geçti.
- Üç rapor validator, DOCX bütünlük, PDF metin ve sayfa sayfa görsel QA
  kontrollerinden geçti.
- 63 taşıma kaydının 56'sı güncel hedef hash'iyle birebir aynı; sonradan
  bilinçli olarak yeniden üretilen 7 değişebilir tarihsel ağaç, özgün taşıma
  doğrulaması korunarak ayrıca işaretlendi.

Ana sonuç:

> Aynı iSAID tahminleri insan yerine SAM1 pseudo referansla ölçüldüğünde
> GT-bbox Overall IoU, SAM1/SAM2/SAM3 için sırasıyla
> `+0,347 / +0,198 / +0,140` arttı. İnsan referansındaki
> `SAM3 > SAM1 > SAM2` sırası pseudo referansta `SAM1 > SAM2 > SAM3`
> oldu. Bu, pseudo referansın teacher model ve yakın mimarileri sistematik
> biçimde avantajlı gösterebildiğini doğrudan ortaya koyuyor.

Bulunan ve yönetilen sorun:

- Yedi SAM3 tahmininde model dolu maske üretmesine rağmen maske prompt
  kutusuyla kesişmedi. Görsel ve kayıt denetimi bunun dosya bozulması değil,
  modelin komşu nesneyi seçtiği gerçek başarısızlık olduğunu gösterdi.
  Sonuçlar yapay biçimde iyileştirilmedi; bu hatalar metriklerde korundu.

Basit sonuç:

> Hocanın istediği iki veri setinde birebir eşlenmiş deney, insan/pseudo
> ayrımı ve üç okunaklı full metric belge tamamlandı. Eski yüksek SAMRS
> skoru artık bağımsız insan başarısı olarak değil, SAM1 üretimli referans
> üzerindeki teacher-reference etkisi olarak doğru biçimde yorumlanıyor.

Bir sonraki adım:

- Zorunlu deney veya refactor işi kalmadı. Bildiri metni canonical v2
  sonuçları temel alınarak güncellenebilir.

### 2026-07-26 - Tam metrik raporlar eski okunaklı biçime getirildi

Yapılanlar:

- Teacher-reference-bias çalışmasının iSAID ve SAMRS tam metrik Markdown,
  DOCX ve PDF belgeleri tarihsel
  `samrs_sota_plane_full_metric_document_colored.pdf` sayfa düzenine göre
  yeniden üretildi.
- Her maske tablosu `Pipeline`, `Images`, `Avg IoU`, `Avg Dice`,
  `Avg Precision`, `Avg Recall` ve `IoU ≥ 0.50/0.75/0.90` sütunlarına
  indirildi.
- `n`, `Sahne`, `Tekrar`, `Boundary IoU` ve yanıltıcı `mAP proxy`
  sütunları sunum tablolarından kaldırıldı.
- Gerçek `BBox mAP50/75/90/50-95` ile bbox Precision/Recall değerleri
  yalnız YOLO detector tablosunda tutuldu. Maske eşik geçme oranları mAP
  olarak adlandırılmadı.
- Dört overlap × mask-area nitel örneği veri seti başına dört ayrı okunaklı
  sayfaya dönüştürüldü. GT bbox, referans, SAM1, SAM2 ve SAM3 panelleri aynı
  sayfada gösteriliyor.
- iSAID açıklaması netleştirildi: birincil sonuçlar insan maskesine karşıdır.
  SAM1, resmi GT bbox prompt'larıyla ayrıca pseudo referans üretir; bu ikinci
  referans yalnız teacher-bias kontrolünde kullanılır.
- iSAID ve Semantic Drone configlerinin Landcover sonucuna olan çalışma
  zamanı bağımlılığı kaldırıldı. Eğitim süresi tahmini study dışı dosya
  okumadan, configteki açık sabit değeri kullanır.
- Boş klasörler ve `.gitkeep/.keep` dosyaları kaldırıldı. Study config,
  script ve kaynak kodlarında yabancı study sonucu bağımlılığını engelleyen
  repository layout testi eklendi.

Doğrulananlar:

- Kök test paketi `50 / 50`, teacher-reference-bias test paketi `44 / 44`;
  toplam `94 / 94` test geçti.
- Tarihsel iSAID ve SAMRS doğrulayıcıları yeniden geçti; eski rapor ve
  sonuçlar korunuyor.
- Yeni iSAID PDF'si `19`, SAMRS PDF'si `13` sayfadır. iSAID/SAMRS DOCX
  dosyaları sırasıyla `12 / 7` tablo ve `4 / 4` nitel görsel içerir; ZIP
  bütünlükleri sağlamdır.
- Rapor CSV'lerinde yasaklı sütun veya `mAP proxy` bulunmuyor. Her YOLO bbox
  satırı üç bağımsız eğitimin ortalaması ± standart sapmasıdır.
- PDF detector, Overall ve nitel örnek sayfaları raster görüntü olarak
  incelendi; metin, tablo ve görsellerde kesilme veya üst üste binme yoktur.
- Repo içinde harici model Git iç yapıları ve sanal ortam hariç boş dizin ya
  da placeholder dosya kalmadı.

Basit sonuç:

> Yeni raporlar eski alışılmış tam metrik belgenin okunaklı görünümünü
> koruyor; ancak eski belgedeki yanıltıcı proxy adları yerine gerçek ölçümün
> ne olduğunu açıkça söylüyor. iSAID insan referansı ile kontrollü SAM1
> pseudo referansı birbirine karıştırılmıyor.

Bir sonraki adım:

- Zorunlu iş kalmadı. Final reproducibility manifesti güncel rapor, kod ve
  doküman hash'leriyle `completed` durumunda yeniden üretildi.

### 2026-07-26 - Study mimarisi ve iki tam metrik raporu tamamlandı

Yapılanlar:

- Repository, kökte yalnız ortak kod/veri/model altyapısı kalacak şekilde
  yeniden düzenlendi. Aktif teacher-reference-bias çalışması ile tarihsel
  iSAID vehicle, SAMRS SOTA plane, semantic drone car ve landcover building
  çalışmaları ayrı `studies/<study>/` köklerine alındı.
- `src/sam3_bbox_study` adı kaldırılarak ortak paket `src/yolo_sam` yapıldı.
  Teacher-reference-bias'a özgü analiz, raporlama ve pseudo-reference kodu
  yalnız ilgili study altına taşındı.
- Eski `presentation_*`, kök `configs`, `data`, `results`, `runs` ve `scripts`
  dizinleri kaldırıldı; içerikleri silinmeden ilgili study klasörlerine
  ayrıştırıldı. Kök YOLO ağırlıkları `models/yolo/` altına alındı.
- 63 taşıma işlemi öncesi ve sonrası SHA-256 manifestleriyle doğrulandı.
  Taşıma nedeniyle değişmesi gereken 86 metadata dosyası arşivli audit
  zincirine alındı; bunların 26'sındaki bağımlı fingerprint sırası ayrıca
  eski, ara ve güncel hash'lerle onarıldı.
- Teacher-reference-bias sonuçlarından iSAID ve SAMRS SOTA için eski rapor
  biçimiyle uyumlu iki ayrı Markdown, renkli DOCX ve renkli PDF üretildi.
  Her raporda detector metrikleri, GT/YOLO bbox ile SAM1/2/3 sonuçları,
  Overall ve dört overlap-mask-area tabakası, başarı eşikleri, görseller ve
  açıklamalı tartışma bulunuyor.

Doğrulananlar:

- Ortak testlerin `44 / 44`, study testlerinin `43 / 43`, toplamda
  `87 / 87` tanesi geçti.
- Teacher finalizer bütün çalışma zincirini temiz geçti ve 629 dosyalık
  `results/reproducibility_manifest.json` üretti.
- Tarihsel iSAID doğrulayıcısı geçti. Tarihsel SAMRS doğrulayıcısı 512
  görüntüyü, dört tabakada `128 + 128 + 128 + 128` örneği, 11 pipeline'ı ve
  5.632 per-image metrik satırını doğruladı.
- Canonical sonuçlar değişmedi: `41.580` metrik satırı, `180` aggregate satırı
  ve `6` detector satırı korunuyor. SAM1/SAM2/SAM3 referans enflasyonu
  sırasıyla `0,349895 / 0,225371 / 0,183522` kaldı.
- Bildiri PDF'si 6, iSAID tam metrik PDF'si 17 ve SAMRS tam metrik PDF'si 12
  sayfadır. DOCX arşivleri sağlamdır ve sırasıyla 4, 12 ve 7 tablo içerir.
  Tam metrik raporlarında yanlış `mAP proxy`, `IoU >=` veya `IoU = 0`
  etiketleri bulunmuyor.

Basit sonuç:

> Değerli deney çıktıları silinmeden repository artık study bazında
> izlenebilir durumdadır. Aktif bildiri çalışması ile eski deneyler birbirine
> karışmıyor; iki veri setinin yeni tam metrik raporları aynı doğrulanmış
> canonical sonuç zincirinden üretiliyor.

Bir sonraki adım:

- Bu refactor ve rapor üretimi kapsamında zorunlu iş kalmadı.

### 2026-07-26 - Eşlenmiş deney ve final bildiri tamamlandı

Yapılanlar:

- SAMRS seed `2026` detector eğitimi `100 / 100` epok tamamlandı; validation
  eşiği donduruldu ve test detector değerlendirmesi çalıştırıldı.
- SAM1, SAM2 ve SAM3 için bütün GT-bbox ve üç seed'li YOLO-bbox çıkarımları
  iki veri setinde tamamlandı.
- Finalizer'ın yakaladığı dokuz eski iSAID dual-reference hash uyuşmazlığı
  incelendi. Aynı prediction dosyaları güncel ve doğru kimlikli SAM1 pseudo
  referansına karşı yeniden değerlendirildi; bütün yeni manifestlerde
  başlangıç/bitiş hash'i aynı ve `input_drift=[]`.
- Beş legacy detector manifesti özgün kopyaları arşivlenerek onarıldı; altıncı
  modern manifest değiştirilmedi. Altı satırlık repair audit `pass` oldu.
- Legacy onarım aracı yarım kalan doğrulanmış onarımdan güvenle devam edecek
  şekilde sağlamlaştırıldı. Detector actual-args karşılaştırmasında koşuya
  özgü çıktı yolu `save_dir` doğru biçimde izinli alan yapıldı.
- Kanonik analiz, figürler, Markdown, DOCX, altı sayfalık PDF,
  `REPRODUCIBILITY_APPENDIX.md` ve 513 dosyalık final reproducibility manifest
  yeniden üretildi.

Doğrulananlar:

- Kanonik tablo `41.580`, aggregate tablo `180`, detector tablosu `6` satırdır.
- Prediction audit `24` matched ve `18` unmatched koşulu kapsar.
- Altı YOLO eğitiminin tamamı `100` epoktur ve final core metrikleri sonludur.
- Final detector ortalamaları iSAID için AP50 `0,936`, AP75 `0,862`,
  AP50-95 `0,795`; SAMRS için AP50 `0,953`, AP75 `0,871`, AP50-95
  `0,725` olmuştur.
- Aynı `1.033` tahmin yalnız referans değiştirilerek ölçüldüğünde IoU
  enflasyonu SAM1 için `0,350` (%95 GA `[0,313, 0,378]`), SAM2 için `0,225`
  (`[0,188, 0,255]`) ve SAM3 için `0,184` (`[0,142, 0,216]`) olmuştur.
- YOLO-bbox insan iSAID IoU ortalamaları SAM1/SAM2/SAM3 için sırasıyla
  `0,607 / 0,596 / 0,626`; SAM1 pseudo referansında
  `0,881 / 0,760 / 0,726` olmuştur.
- SAMRS SAM1 pseudo referansında üç-seed YOLO-bbox IoU değerleri
  `0,869 / 0,707 / 0,591` olmuştur.
- PDF tam `6` A4 sayfadır; altı sayfa tek tek incelenmiş, metin/tablo taşması
  veya kesilme bulunmamıştır. DOCX arşiv doğrulaması ve `84 / 84` test
  geçmiştir.

Basit sonuç:

> Aynı görüntü, bbox ve tahmin sabitken yalnız insan referansı SAM1 üretimli
> referansla değiştirildiğinde bütün modellerin skoru yükseliyor; en büyük
> yükseliş referansı üreten SAM1'de görülüyor. Pseudo-maskeler eğitim için
> yararlı olabilir, fakat bağımsız insan ground truth'u ile aynı statüde
> benchmark referansı kabul edilmemelidir.

Bir sonraki adım:

- Bu çalışma kapsamındaki zorunlu iş kalmadı. Pseudo-label training utility
  deneyi ayrı bir gelecek çalışma olarak tutulmaktadır.

### 2026-07-26 - İki-seed ara analiz ve veri bütünlüğü yeniden doğrulandı

Yapılanlar:

- SAMRS seed `42` ve `123` sonuçlarıyla canonical analiz ara modda yeniden
  derlendi.
- iSAID ve SAMRS prepared veri kökleri COCO, YOLO etiketi, source-scene split,
  sınıf, görüntü boyutu ve strata bütünlüğü açısından tekrar doğrulandı.
- Bildiri sınırlılıklarına aynı `100` epokun farklı train görüntü sayıları
  nedeniyle aynı optimizer adımı sayısı anlamına gelmediği açıkça eklendi.

Doğrulananlar:

- Ara canonical tablo `37.455` instance satırı içeriyor.
- SAMRS YOLO-bbox özeti iki seed'i doğru birleştiriyor ve seed standart
  sapmaları sonlu.
- Prepared veri denetimleri iki veri setinde de `PASS`; iSAID testinde
  `1.045`, SAMRS testinde `1.375` instance bulunuyor.
- Devam eden seed `2026` eğitiminin başlangıçta kaydedilmiş dört girdi hash'i
  denetimden sonra da mevcut dosyalarla birebir aynı.

Basit sonuç:

> Üçüncü seed beklenirken veri veya analiz şeması değişmedi. Detector
> karşılaştırmasında ayar ve epok eşitliği korunuyor; fakat train kümesi
> büyüklüğünden doğan optimizer adımı farkı artık sonuçların sınırı olarak
> açıkça raporlanıyor.

Bir sonraki adım:

- Seed `2026` ile tam `41.580` satırlık final matrisi üretmek.

### 2026-07-26 - SAMRS seed 123 uçtan uca tamamlandı

Yapılanlar:

- SAMRS plane YOLO detector seed `123` ile `100` epok eğitildi.
- Confidence eşiği yalnız validation split'inde bbox IoU `0,50` için F1
  eniyilemesiyle `0,79512` olarak donduruldu.
- Aynı eşik test detector değerlendirmesinde ve SAM1, SAM2, SAM3 YOLO-bbox
  çıkarımlarında değiştirilmeden kullanıldı.

Doğrulananlar:

- Test bbox AP50 `0,9548`, AP75 `0,8663`, AP90 `0,2702` ve AP50-95
  `0,7260` oldu.
- Test P@IoU50 `0,9595`, R@IoU50 `0,8793`; `1.375` GT instance'ın `1.209`
  tanesi detector kutusuyla eşleşti.
- Üç prediction ve üç evaluation manifestinin tamamı `completed`; giriş
  değişimi listelerinin tamamı boştur.

Basit sonuç:

> İkinci bağımsız SAMRS seed'i de detector'dan segmentasyona kadar aynı
> protokolle eksiksiz tamamlandı. AP50 ve AP75 seed `42` ile yakınken,
> confidence eşiği ve çok sıkı AP90 değeri seed'e bağlı değişkenliğin neden
> üç koşuyla raporlanması gerektiğini gösteriyor.

Bir sonraki adım:

- Seed `2026` eğitimini ve post-processing matrisini tamamlamak; ardından
  üç-seed ortalaması, güven aralıkları, nihai bildiri ve artifact kalite
  kapısını üretmek.

### 2026-07-26 - Ara bildiri görsel kalite kontrolü yapıldı

Yapılanlar:

- Tamamlanmış mevcut koşullarla açıkça `KISMİ TASLAK` işaretli canonical
  analiz, görseller, DOCX ve altı sayfalık PDF yeniden üretildi.
- PDF'nin altı sayfası raster görüntüye çevrilip tek tek görsel olarak
  incelendi.

Bulunan ve düzeltilen sorun:

- Beşinci sayfadaki detector ve YOLO-bbox tablolarında uzun SAMRS etiketleri
  hücre dışına taşıyordu.
- PDF'ye özel kontrollü satır kırmaları eklendi; tablo etiketleri artık hücre
  içinde ve okunaklıdır.

Doğrulananlar:

- Ara PDF tam `6` A4 sayfadır.
- Altı sayfada başlık, metin, tablo ve görsellerde kesilme veya üst üste binme
  görülmedi.
- DOCX ZIP bütünlük testi hatasız geçti.

Basit sonuç:

> Final sonuçlar henüz yazılmadı; ancak belge üretim ve sayfa düzeni zinciri
> gerçek tablo boyutlarıyla doğrulandı. Üç seed tamamlandığında aynı zincir
> frozen sonuçlarla yeniden çalıştırılacak.

Bir sonraki adım:

- Kalan iki SAMRS seed'ini tamamlamak ve ara taslağı final frozen çıktıyla
  değiştirmek.

### 2026-07-26 - SAMRS seed 42 uçtan uca tamamlandı

Yapılanlar:

- SAMRS plane YOLO detector seed `42` ile `100` epok eğitildi.
- Confidence eşiği yalnız validation split'inde bbox IoU `0,50` için F1
  eniyilemesiyle `0,58326` olarak donduruldu.
- Aynı eşik değiştirilmeden test detector değerlendirmesinde ve SAM1, SAM2,
  SAM3 YOLO-bbox inference koşullarında kullanıldı.

Doğrulananlar:

- Test bbox AP50 `0,9464`, AP75 `0,8653`, AP90 `0,2569`, AP50-95 `0,7182`
  oldu.
- Test P@IoU50 `0,9259`, R@IoU50 `0,8996`; `1.375` GT instance'ın `1.237`
  tanesi detector kutusuyla eşleşti, `138` tanesi açıkça `missing_bbox`
  olarak sıfır segmentasyon skoru aldı.
- SAM1, SAM2 ve SAM3 YOLO-bbox pseudo-reference ortalama IoU değerleri
  sırasıyla `0,8704`, `0,7069` ve `0,5879` oldu.
- Üç prediction ve üç evaluation manifesti `completed`; giriş değişimi sayısı
  sıfırdır.

Basit sonuç:

> İlk SAMRS seed'i detector'dan segmentasyona kadar eksiksiz çalıştı. Düşük
> AP90, kutuların nesneyi bulsa bile çok sıkı geometrik eşleşmede zorlandığını;
> instance-level segmentasyonun ise kaçırılan `138` nesneyi gizlemeden sıfır
> olarak hesaba kattığını gösteriyor.

Bir sonraki adım:

- Seed `123` ve `2026` eğitimlerini ve aynı post-processing matrisini
  tamamlamak; seed ortalaması ve varyansını yalnız üçü bittikten sonra
  raporlamak.

### 2026-07-26 - Düzeltilmiş GT-bbox ve ortak referans denetimi tamamlandı

Yapılanlar:

- iSAID ve SAMRS için SAM1, SAM2 ve SAM3 GT-bbox tahminleri aynı frozen
  prediction dosyalarından 10.000 bootstrap örneğiyle yeniden değerlendirildi.
- iSAID'de aynı tahminler hem bağımsız insan maskesine hem kontrollü SAM1
  pseudo-maskesine karşı ölçüldü.
- SAMRS test tile'ları aynı DOTA görüntülerindeki iSAID insan annotation'larına
  yeniden eşlendi.

Doğrulananlar:

- Altı evaluation manifestinin tamamı `completed` ve giriş değişimi sayısı
  sıfırdır.
- SAMRS testindeki `126 / 128` görüntü kaynak iSAID görüntüsüne piksel düzeyinde
  tam eşleşti.
- `1.033` eşleşen tile-instance görünümü `770` benzersiz insan etiketli uçağı
  ve `35` bağımsız source scene'i kapsıyor.
- Aynı tahminde pseudo eksi human IoU enflasyonu SAM1 için `0,3499`
  (`%95 GA [0,3134, 0,3776]`), SAM2 için `0,2254`
  (`[0,1876, 0,2545]`) ve SAM3 için `0,1835`
  (`[0,1423, 0,2157]`) oldu.
- Kayıpsız insan maskesi ve düzeltilmiş global bbox-mask eşlemesinden sonra
  iSAID human GT-bbox ortalama IoU değerleri SAM1 `0,6613`, SAM2 `0,6502`,
  SAM3 `0,6672` oldu.

Basit sonuç:

> Görüntü, bbox ve model tahmini sabit tutulup yalnız değerlendirme referansı
> değiştirildiğinde üç modelin de skoru yükseliyor; en büyük yükselme referansı
> üreten SAM1'de görülüyor. Bu, veri setleri arası zorluk farkından bağımsız
> doğrudan öğretmen-referans yakınlığı kanıtıdır.

Bir sonraki adım:

- Üç SAMRS YOLO seed'inin detector ve YOLO-bbox matrisini tamamlamak.
- Ardından canonical analiz, görseller, altı sayfalık bildiri ve final kalite
  kapısını üretmek.

### 2026-07-26 - Pseudo-reference öğretmen kimliği kesinleştirildi

Bulunan sorun:

- Kontrollü SAM1 pseudo-reference üreticisi `model_version` alanında
  `"sam1"` metnini arıyordu.
- Canonical prediction sözleşmesinde bu alan model adı değil, sabitlenmiş
  gerçek checkpoint revision'ıdır. Bu nedenle doğru SAM1 çıktısı yanlışlıkla
  reddediliyordu.

Yapılanlar:

- Üretici artık öğretmeni açıkça `model_id + checkpoint revision` çiftiyle
  doğruluyor.
- Beklenen kimlik study config'inden CLI'ya taşınıyor ve çalışma manifestine
  kaydediliyor.
- Yanlış model kimliği ile yanlış revision'ı ayrı ayrı reddeden testler
  eklendi.

Doğrulananlar:

- Hedefli pseudo-reference testleri `3/3 PASS` verdi.
- Unit ve integration testlerinin tamamı `82/82 PASS` verdi.

Basit sonuç:

> Pseudo-reference yalnız dondurulmuş ve config'de ilan edilmiş SAM1
> checkpoint'inden üretilebilir. Model adı veya revision değişirse çalışma
> sessizce devam etmek yerine hata verir.

Bir sonraki adım:

- Pseudo-reference'ı yeniden üretmek ve iki veri setindeki altı GT-bbox
  değerlendirmesini 10.000 bootstrap örneğiyle tamamlamak.

### 2026-07-26 - iSAID üç-seed YOLO matrisi tamamlandı

Yapılanlar:

- iSAID plane için seed `42`, `123` ve `2026` detector validation/test
  değerlendirmeleri tamamlandı.
- Her seed için SAM1, SAM2 ve SAM3 YOLO-bbox tahminleri üretildi ve aynı
  tahminler insan ile SAM1 pseudo referansına karşı değerlendirildi.
- Toplam `3 model x 3 detector seed = 9` iSAID YOLO-bbox segmentation koşulu
  tamamlandı.

Doğrulananlar:

- Her confidence eşiği yalnız ilgili validation split'inde bbox IoU `0,50`
  için F1 eniyilemesiyle seçildi ve testte değiştirilmeden kullanıldı.
- Üç bağımsız test koşusunda ortalama bbox AP50 `0,9361 +/- 0,0039`, AP75
  `0,8625 +/- 0,0054`, AP90 `0,6225 +/- 0,0078` ve AP50-95
  `0,7954 +/- 0,0011` oldu; `+/-` değerleri seed'ler arası örnek standart
  sapmadır.
- Bütün yeni detector/segmentation/evaluation manifestleri başlangıç ve bitiş
  input hash'lerini, boş `input_drift` listesini ve çıktı hash'lerini içeriyor.

Basit sonuç:

> iSAID detector sonucu üç seed arasında kararlıdır. Segmentasyon için seed
> ortalaması ve seed varyansı canonical analiz tamamlandığında birlikte
> raporlanacaktır.

Bir sonraki adım:

- İki veri setindeki altı GT-bbox koşulunu düzeltilmiş referanslarla yeniden
  üretmek; parity ve prompt-geometri denetimlerini tamamlamak.

### 2026-07-26 - Detector değerlendirme belleği sınırlandı

Bulunan sorun:

- Ultralytics'e tüm COCO görüntü yolları tek Python listesi olarak verildiğinde
  `batch=12` argümanına rağmen liste tek büyük tensöre dönüştürülüyordu.
- iSAID validation'daki `336` görüntü tek seferde GPU'ya alınarak yaklaşık
  `35,7 GiB` bellek ayırıyor ve ilk konvolüsyonda değerlendirmeyi durduruyordu.
- Sunucudaki NVML driver/library uyuşmazlığı, native PyTorch allocator'da
  ayrıca bir iç doğrulama hatasına yol açıyordu.

Yapılanlar:

- Detector değerlendirme yolları kod içinde açıkça en fazla `12` görüntülük
  dilimlere ayrıldı; her dilim `stream=True` ile tüketiliyor.
- `study.py` alt süreçleri için
  `PYTORCH_ALLOC_CONF=backend:cudaMallocAsync` güvenli varsayılanı eklendi.
- CUDA görünürlüğü ve PyTorch allocator seçimi yeni run manifestlerinde
  kaydedilmeye başlandı.
- Dilim sayısını, dilim üst sınırını ve akışlı çıkarımı doğrulayan birim
  testleri eklendi; allocator override davranışıyla birlikte ilgili hedefli
  testler `9/9 PASS` verdi.
- Doğru proje paket yolu ile unit ve integration testlerinin tamamı yeniden
  çalıştırıldı ve `80/80 PASS` verdi.

Doğrulananlar:

- iSAID seed `2026` validation ve test detector değerlendirmesi aynı frozen
  modelle başarıyla tamamlandı.
- Bağımsız `128` görüntülük test split'inde bbox AP50 `0,9384`, AP75
  `0,8685`, AP90 `0,6151` ve AP50-95 `0,7951` oldu.
- Validation'da seçilen sabit confidence eşiği `0,26796` testte değiştirilmeden
  kullanıldı.

Basit sonuç:

> Bu değişiklik model veya deney protokolünü değiştirmedi; yalnızca aynı
> görüntüleri sabit bellekle değerlendirdi. Eksik iSAID SAM1/2/3 YOLO-bbox
> koşuları GPU2 üzerinde yeniden başlatıldı.

Bir sonraki adım:

- Dört GPU'daki eğitim ve değerlendirme kuyruklarını tamamlamak; ardından
  istatistik, görsel, bildiri ve final bütünlük kapılarını çalıştırmak.

### 2026-07-26 - Üç iSAID YOLO eğitimi tamamlandı

Yapılanlar:

- iSAID plane detector'ı seed `42`, `123` ve `2026` ile aynı frozen protokol
  altında ayrı ayrı `100` epok eğitildi.
- Seed `123` tamamlanınca boşalan GPU üzerinde son SAMRS seed'i olan `2026`
  yeni start/finish fingerprint şemasıyla başlatıldı.
- iSAID seed `2026` tamamlanınca validation threshold, test detector ve
  YOLO-bbox SAM değerlendirme işçisi otomatik devreye girdi.

Doğrulananlar:

- Final eğitim satırındaki AP50-95 değerleri seed `42`, `123`, `2026` için
  sırasıyla `0,80883`, `0,80911`, `0,80910` oldu.
- Final AP50 değerleri sırasıyla `0,94821`, `0,95365`, `0,95585` oldu.
- Üç koşulun da final precision, recall, AP50 ve AP50-95 değerleri sonludur.
- SAMRS seed `2026`, scoped detector content manifestinin başlangıç hash'i ve
  boş `input_drift` listesiyle başladı.

Basit sonuç:

> iSAID YOLO eğitimi üç seed arasında oldukça kararlı görünüyor. Bunlar
> validation eğitim özetleridir; final detector sonucu, validation'da seçilen
> confidence eşiğiyle bağımsız test split'inde ayrıca raporlanacaktır.

Bir sonraki adım:

- iSAID detector test ve YOLO-bbox SAM1/2/3 sonuçlarını tamamlamak; üç SAMRS
  detector eğitiminin bitmesini beklemek.

### 2026-07-26 - Run input drift ve detector provenance kapısı eklendi

Bulunan sorun:

- Eski manifest bitirme kodu başlangıç input hash'inin üzerine bitişte yeniden
  hesaplanan hash'i yazabiliyordu. Bu davranış süreç sırasında değişen girdiyi
  gizleyebilirdi.
- Bazı uzun YOLO eğitimleri yeni fingerprint alanları eklenmeden önce
  başlatılmıştı.
- Full prepared content manifesti, detector'ın kullanmadığı segmentation
  maskelerini ve test split'ini de içerdiği için detector girdisini gereğinden
  geniş tanımlıyordu.

Yapılanlar:

- Başlangıç input fingerprint'i immutable yapıldı; bitiş fingerprint'i ve
  `input_drift` ayrı alanlara taşındı.
- Finalizer artık manifestleri değiştirmiyor ve input drift'i doğrudan hata
  kabul ediyor.
- Finalizer altı `args.yaml` dosyasını da okuyor; dataset yolu, output yolu ve
  seed dışındaki gerçek Ultralytics eğitim argümanlarının birebir aynı
  olmasını zorunlu tutuyor.
- Detector için yalnız train/validation görüntüleri, YOLO bbox label'ları ve
  `data.yaml` dosyasını kapsayan scoped content manifesti eklendi.
- Eski pahalı detector run'ları için özgün manifesti byte düzeyinde arşivleyen
  ve yalnız `train_detector` aşamasını onarabilen sınırlı bir provenance
  repair aracı eklendi.
- Repair arşivleri ve geçersiz eski iSAID sonuç arşivi aktif run manifest
  taramasından yalıtıldı.

Doğrulananlar:

- iSAID detector ağacı `4.043` dosya ve
  `9e87f350...e399252` tree hash'i içeriyor.
- SAMRS detector ağacı `5.977` dosya ve
  `92fd411b...6fc77e1` tree hash'i içeriyor.
- İki scoped manifestte de test görüntüsü ve COCO segmentation annotation'ı
  yoktur.
- Manifest, scoped content ve finalizer hedefli testleri `16/16 PASS` verdi.
- Güncel unit ve integration testlerinin tamamı `76/76 PASS` verdi.

Basit sonuç:

> Maske RLE düzeltmesi detector eğitim verisini değiştirmedi. Artık bu durum
> varsayım olarak değil, detector'ın gerçekten tükettiği görüntü ve bbox label
> ağacının ayrı SHA-256 kaydıyla gösteriliyor.

Bir sonraki adım:

- Altı detector eğitimi bitince legacy detector manifest repair auditini bir
  kez üretmek; diğer bütün aşamaları yeni start/finish fingerprint şemasıyla
  yeniden çalıştırmak.

### 2026-07-26 - Eski iSAID sonuçları geçersiz olarak arşivlendi

Bulunan sorun:

- Kayıpsız RLE öncesindeki contour tabanlı insan referansı ile düzeltilmiş
  insan referansı, aynı `1.045` test instance'ında ortalama yalnızca `0,864`
  IoU ve medyan `0,888` IoU ile örtüşüyor.
- Eski referansta bir test maskesi tamamen boştu. Düzeltilmiş referansta boş
  maske yoktur.
- `601 / 1.045` instance'ın eski-yeni referans IoU'su `0,90` altında,
  `954 / 1.045` instance'ınki `0,95` altındadır.
- Düzeltilmiş maskelerin ortalama alanı eski contour maskelerinden yaklaşık
  `194,5` piksel daha büyüktür.

Yapılanlar:

- Düzeltme öncesindeki bütün mevcut iSAID GT-bbox tahminleri ve metrikleri,
  kaynak ve arşiv SHA-256 değerleriyle ayrı bir audit klasörüne kopyalandı.
- Arşiv açıkça
  `superseded_invalid_for_scientific_results` olarak etiketlendi.
- Final kalite kapısı artık RLE sensitivity auditini, eski metrik arşivini,
  eski/yeni test COCO hash'lerini ve arşivlenen her dosyanın hash'ini
  doğruluyor.
- Arşiv altındaki eski run manifestleri, aktif manifest backfill ve
  fingerprint taramasından yalıtıldı; böylece finalizer geçmiş kanıtı
  değiştiremiyor.

Doğrulananlar:

- Gerçek migration, sensitivity ve `33` dosyalık eski sonuç arşivi birlikte
  doğrulandığında hata sayısı `0` çıktı.
- Unit ve integration testlerinin tamamı `72/72 PASS` verdi.

Basit sonuç:

> Önceki iSAID metrikleri final bilimsel sonuç değildir ve bildiride
> kullanılamaz. SAM1, SAM2 ve SAM3 için iSAID GT-bbox ve YOLO-bbox metrikleri
> kayıpsız insan referansı üzerinde baştan üretilecektir.

Bir sonraki adım:

- Devam eden eğitimleri tamamlamak ve bütün iSAID değerlendirmelerini
  düzeltilmiş RLE referansıyla yeniden çalıştırmak.

### 2026-07-26 - iSAID maskeleri kayıpsız RLE'ye geçirildi

Bulunan sorun:

- iSAID hazırlama kodu doğru raster maskeyi önce OpenCV ile oluşturuyor, sonra
  COCO polygonuna çevirmek için contour çıkarıyordu.
- OpenCV ve `pycocotools` polygon rasterizasyon kuralları farklı olduğu için
  train'de `2`, testte `1` çok ince sınır maskesi boş decode ediliyordu.
- Diğer maskelerde de COCO'dan decode edilen alan ile hazırlama sırasında
  kaydedilen raster alan birebir aynı değildi.

Yapılanlar:

- Hazırlama kodu insan maskesini contour polygonuna çevirmek yerine doğrudan
  kayıpsız compressed COCO RLE olarak yazacak şekilde düzeltildi.
- Mevcut train/validation/test maskeleri resmi ham iSAID polygonları,
  dondurulmuş tile koordinatları ve OpenCV rasterizasyonuyla yeniden kuruldu.
- Düzeltme öncesi üç COCO dosyası ayrı audit klasöründe saklandı; kaynak ve
  önce/sonra SHA-256 değerleri migration manifestine yazıldı.
- Yeni prepared validator boş/bozuk maskeyi ve decoded area ile COCO `area`
  uyuşmazlığını engelliyor.

Doğrulananlar:

- Train `10.019`, validation `1.961`, test `1.045` instance'ın tamamında
  düzeltme sonrası boş maske ve alan uyuşmazlığı `0`.
- Görüntü, bbox, YOLO label, metadata, split ve instance sayıları değişmedi.
- iSAID ve SAMRS prepared veri doğrulamaları yeni kalite kapısıyla `PASS`
  verdi.
- Ham iSAID'deki `10.601` plane bbox'ının insan polygon envelope'u olduğu,
  yalnız inclusive piksel genişliği/yüksekliği nedeniyle `+1` kullandığı
  exhaustive olarak doğrulandı.

Basit sonuç:

> iSAID GT bbox'ı SAM1 pseudo-maskeden gelmez; resmi insan annotation'ının
> bbox'ıdır. SAMRS GT bbox'ı ise özgün DOTA RHBox'ıdır. Bu provenance farkı
> cross-dataset mutlak GT skorlarını betimleyici yapar; aynı SAMRS tahminini
> insan ve pseudo referansla ölçen ana nedensel kontrolü etkilemez.

### 2026-07-26 - YOLO gerçek eğitim çizelgeleri karşılaştırıldı

Doğrulananlar:

- Bütün koşullarda model, batch `12`, `100` epok, image size `1024`,
  optimizer ayarı, `lr0/lrf`, momentum, weight decay ve deterministic mod
  aynıdır.
- Her seed yalnız seed değeriyle ayrılır; temel paket sürümleri aynıdır.
- iSAID ve SAMRS'nin bir epoktaki batch sayıları farklı olduğu için ilk üç
  warm-up epokunda kaydedilen ortalama öğrenme oranı çok küçük farklıdır.
- Dördüncü epoktan itibaren iki veri setindeki üç parameter group öğrenme
  oranı çizelgesi birebir aynıdır.

Basit sonuç:

> Detector'lara elle farklı hyperparameter verilmedi. İlk üç epoktaki küçük
> fark, aynı epok-temelli warm-up kuralının farklı büyüklükteki veri
> kümelerine uygulanmasının doğal sonucudur ve bildiri sınırlılıklarında açık
> biçimde raporlanacaktır.

### 2026-07-26 - Kanonik analiz matrisi içerik düzeyinde kilitlendi

Yapılanlar:

- Final kalite kapısına iki veri seti, üç model, GT/YOLO bbox, üç detector
  seed'i ve uygun human/pseudo referanslarından oluşan tam koşul matrisi
  denetimi eklendi.
- Her koşulun dondurulmuş test COCO dosyasındaki instance sayısı kadar satır
  içermesi zorunlu hale getirildi.
- `overall` ve dört zorluk stratumunun her koşulda bulunması; mask, detector
  ve güven aralığı metriklerinin sonlu ve geçerli aralıklarda olması
  denetleniyor.
- Detector analizi yalnız `test` split'inden gelmeli ve confidence eşiğinin
  kaynağı `validation` olmalı.
- Eksiksiz sentetik matrisin geçtiğini ve tek instance satırı çıkarıldığında
  finalizer'ın durduğunu gösteren regresyon testi eklendi.

Doğrulananlar:

- Bütün birim ve entegrasyon testleri `68/68 PASS` verdi.
- Python derleme, `pip check` ve `git diff --check` başarılı oldu.

Basit sonuç:

> Dosyanın mevcut olması artık yeterli değil. Bir model, seed, referans,
> stratum veya instance satırı sessizce kaybolursa bildiri üretilmiş olsa bile
> çalışma final olarak kabul edilmeyecek.

### 2026-07-26 - SAM2 görüntü wrapper'ı yükleme raporuyla doğrulandı

Doğrulananlar:

- Resmi `facebook/sam2.1-hiera-large` snapshot'ı `sam2_video` metadata'sı
  taşıdığı için `Sam2Model` kurulurken bir uyumluluk uyarısı veriyor.
- Transformers görüntü modeli, video checkpoint'indeki ortak vision encoder,
  prompt encoder ve mask decoder ağırlıklarını kullanıyor; video belleği
  bileşenleri görüntü inference'ına dahil edilmiyor.
- Pinli revision yerel snapshot'tan `Sam2Model` olarak yüklenip ayrıntılı
  loading info incelendi.
- `216.924.865` çekirdek parametre için eksik, beklenmeyen, boyutu uyuşmayan
  ağırlık ve yükleme hatası sayılarının tamamı `0` çıktı.

Basit sonuç:

> Görülen uyarı yanlış checkpoint veya eksik SAM2 ağırlığı anlamına gelmiyor.
> Kullandığımız wrapper, resmi SAM2.1 Large checkpoint'inin görüntü
> segmentasyonu için gerekli alt modelini eksiksiz yüklüyor.

### 2026-07-26 - YOLO eğitim-sağlığı kalite kapısı eklendi

Yapılanlar:

- Altı detector eğitiminin `results.csv` geçmişini tek tabloda toplayan
  `training_health_audit.csv` çıktısı eklendi.
- Her koşul için tamamlanan epok sayısı, final precision, recall, AP50 ve
  AP50-95 değerleri kaydediliyor.
- Eğitim boyunca görülen bütün sonlu olmayan hücreler ile validation-loss
  alanlarındaki geçici `NaN` değerleri ayrı sayılıyor.
- Finalizer artık iki veri seti ve üç seed'den oluşan altı koşulun her birinde
  `100` epok, sonlu final metrikler ve `[0, 1]` aralığı arıyor.
- CSV boolean alanının metin olarak okunup yanlışlıkla doğru kabul edilmesini
  önleyen katı ayrıştırma eklendi.

Doğrulananlar:

- Geçici `NaN` içeren fakat final metrikleri sonlu kalan örnek eğitim geçmişi
  için regresyon testi geçti.
- SAM3 global bbox eşleme, reporting ve final fingerprint testlerinden oluşan
  hedefli paket `13/13 PASS` verdi.
- `scripts`, `src` ve `tests` ağaçlarının derlenmesi ile `git diff --check`
  başarılı oldu.

Basit sonuç:

> Yalnız son checkpoint'in varlığı artık yeterli değil. Her YOLO eğitiminin
> tam bütçeyle bittiği ve raporlanan final değerlerin sayısal olarak geçerli
> olduğu final belge üretilmeden önce otomatik denetlenecek.

Bir sonraki adım:

- Devam eden detector eğitimlerini tamamlamak; düzeltme sonrası SAM3 ve bütün
  YOLO-bbox sonuçlarını üretmek; gerçek altı koşulun eğitim-sağlığı tablosunu
  incelemek.

### 2026-07-26 - SAM3 çoklu bbox eşleme hatası bulundu ve düzeltildi

Bulunan sorun:

- SAM3 bazı bbox prompt'ları için maske üretmediğinde çıktı sayısı giriş bbox
  sayısından az olabiliyordu.
- Eski sıralı eşleme, eksik kalan erken bir bbox nedeniyle sonraki maskeleri
  yanlış instance'lara kaydırabiliyordu.
- Geometrik denetimde iSAID SAM3 GT-bbox çıktılarındaki `995` non-empty
  maskenin `79` tanesinin kendi prompt bbox'ıyla bbox IoU değerinin
  `0,05` altında olduğu; bunların `85` tanesinde maske alanının yüzde 10'undan
  azının prompt bbox içinde kaldığı görüldü. Örneklerin çoğu çok nesneli
  görüntülerde ardışık kayma biçimindeydi.
- Piksel düzeyinde hiç prompt kesişimi olmayan non-empty maske sayısı iSAID'de
  `70`, SAMRS'de `55` idi; SAMRS'de ayrıca `63` maskenin alanının yüzde
  10'undan azı kendi prompt bbox'ı içindeydi.

Yapılanlar:

- SAM3 çıktıları ile giriş bbox'ları artık bütün eşleşmeleri birlikte
  değerlendiren global birebir IoU atamasıyla eşleniyor.
- Prompt bbox ile hiç örtüşmeyen bir çıktı ilgili instance'a atanmayıp boş
  maske olarak kaydediliyor.
- Erken bbox çıktısının eksik olması ve sıfır IoU durumları için regression
  testleri eklendi.
- Canonical prediction auditine
  `nonempty_masks_without_prompt_overlap` kalite ölçüsü eklendi; final çalışma
  bu sayı sıfır değilse kabul edilmeyecek.
- Düzeltme öncesi maske hash'leri ayrı parity snapshot'ında saklandı. Final
  kalite kapısı SAM1/SAM2 hash'lerinin değişmemesini, iki veri setindeki SAM3
  hash'lerinin ise bu bilinçli düzeltme nedeniyle değişmesini zorunlu tutuyor.

Basit sonuç:

> Önceki düşük SAM3 skorlarının bir kısmı model performansı değil, çoklu
> çıktının yanlış uçağa bağlanmasından kaynaklanıyordu. SAM3 sonuçları final
> bildiride kullanılmadan önce iki veri setinde de baştan üretilecek.

Bir sonraki adım:

- SAMRS seed 42 eğitimi bittiğinde GT-bbox koşullarını düzeltme sonrası yeniden
  çalıştırmak, SAM3'ü ikinci kez çalıştırarak deterministik pariteyi kanıtlamak
  ve bütün dual-reference sonuçlarını yeniden hesaplamak.

### 2026-07-26 - Benzersiz nesne duyarlılık analizi tamamlandı

Yapılanlar:

- Ortak insan referanslı denetim yeniden çalıştırıldı ve
  `unique_human_object_sensitivity.csv` üretildi.
- `pixel-exact` görüntü sayısı yalnızca başarıyla eşlenmiş görüntüler arasından
  hesaplanacak şekilde kesinleştirildi.
- PDF yöntem tablosundaki açıklamasız yıldız işareti kaldırıldı.

Doğrulananlar:

- `128` SAMRS test görüntüsünün `126` tanesi iSAID kaynak görüntüleriyle piksel
  düzeyinde birebir aynıdır.
- `1.033` eşleşmiş crop görünümü `770` benzersiz insan anotasyonlu uçağa ve
  `35` kaynak sahneye aittir.
- Benzersiz nesne ağırlıklı ortalama IoU enflasyonu SAM1, SAM2 ve SAM3 için
  sırasıyla `0,349`, `0,227` ve `0,172` olarak ölçüldü.
- İnsan ve pseudo referanslarındaki model sırası aynıdır; Spearman ve Kendall
  korelasyonları `1,0`'dır.
- Ortak denetim, analiz ve final fingerprint kapılarını kapsayan hedefli
  testler `12/12 PASS` verdi.

Basit sonuç:

> Aynı uçağın örtüşen crop'larda tekrar görünmesi ana sonucu üretmiyor.
> Tekrarlar tek nesne ağırlığına indirildiğinde de SAM1 kendi ürettiği
> referansa karşı en fazla skor artışını almaya devam ediyor.

Bir sonraki adım:

- Devam eden altı YOLO eğitimini, validation tabanlı eşik seçimini ve tüm
  YOLO-bbox SAM1/SAM2/SAM3 değerlendirmelerini tamamlamak.
- Ardından final analiz, figür, reproducibility eki, altı sayfalık bildiri ve
  tam kalite denetimini çalıştırmak.

### 2026-07-26 - Tekrarlanan crop görünümleri için duyarlılık denetimi eklendi

Yapılanlar:

- Ortak iSAID-SAMRS denetimindeki `1.033` eşleşmenin `770` benzersiz insan
  anotasyonlu uçağa ait olduğu belirlendi.
- SAMRS'nin örtüşen tile'larında aynı uçağın birden fazla crop görünümü
  bulunabildiği için manifest artık hem tile-instance hem benzersiz nesne
  sayısını ayrı kaydediyor.
- Her uçağın farklı görünümlerini önce kendi içinde ortalayan, sonra nesneleri
  eşit ağırlıklandıran ayrı bir duyarlılık analizi eklendi.
- Bildiri ve reproducibility eki bu ayrımı açıkça yazacak şekilde güncellendi.
- Sıralama analizine Kendall tau ve açık `SAM1 teacher advantage change`
  sütunu eklendi.

Doğrulananlar:

- Kaynak sahne düzeyindeki bootstrap zaten aynı sahneden gelen crop'ları
  bağımsız saymıyordu.
- Benzersiz nesne ağırlıklı ön hesapta IoU enflasyonu SAM1/SAM2/SAM3 için
  sırasıyla yaklaşık `0,349 / 0,227 / 0,172` kaldı; ana sonuç değişmedi.
- İlgili yeni testler dahil hedefli test paketi `17/17 PASS` verdi.

Basit sonuç:

> Bulguyu `1.033 bağımsız uçak` diye sunmayacağız. Doğru ifade, 770 benzersiz
> uçağın 1.033 crop görünümüdür; hem sahne kümeli hem benzersiz nesne ağırlıklı
> analiz aynı teacher-reference bias sonucunu vermektedir.

Bir sonraki adım:

- Eğitim ve tüm downstream koşullar tamamlandıktan sonra bu duyarlılık
  sonuçlarını final canonical analiz ve altı sayfalık bildiriye işlemek.

### 2026-07-26 - Ortak görüntü eşlemesi kesinleştirildi

Yapılanlar:

- SAMRS tile'larının iSAID kaynak görüntülerine kabul edilmesi için hem
  `template_score >= 0,995` hem de piksel düzeyinde birebir eşitlik zorunlu
  hale getirildi.
- Yüksek şablon skoru aldığı halde piksel içeriği farklı bir görüntü artık
  `pixel_mismatch` olarak reddediliyor.
- Final kalite kapısı, kabul edilen `126` görüntünün tamamının piksel düzeyinde
  birebir aynı olmasını zorunlu tutacak şekilde genişletildi.

Doğrulananlar:

- Önceki ortak-reference denetiminde kabul edilen `126 / 128` görüntünün tamamı
  zaten piksel düzeyinde birebir aynıydı; değişiklik mevcut bilimsel sonucu
  değiştirmiyor, kod ile raporlanan kabul kuralını eşitliyor.

Basit sonuç:

> Aynı görüntü üzerinde human ve SAM1-pseudo referanslarını karşılaştırdığımız
> ana deney artık yalnız gerçekten aynı piksel içeriğine sahip örnekleri kabul
> ediyor.

Bir sonraki adım:

- Devam eden detector eğitimlerini ve onları izleyen segmentasyon,
  dual-reference analiz, bildiri ve final QA zincirini tamamlamak.

### 2026-07-26 - Model pinleme öncesi maske paritesi donduruldu

Yapılanlar:

- SAM1, SAM2 ve SAM3 GT-bbox tahminlerinin yalnız `instance_id`, çalışma
  durumu ve maske RLE içeriğini hash'leyen iki aşamalı bir parite denetimi
  eklendi.
- Mevcut altı GT-bbox koşulu yeniden inference öncesinde baseline olarak
  donduruldu.
- Final kalite kapısı, pinlenmiş model revision'larıyla yeniden üretilen
  maskeler baseline ile birebir aynı değilse çalışmayı kabul etmeyecek şekilde
  genişletildi.
- `scripts/study.py prediction-parity --mode snapshot|verify` komutu eklendi.

Doğrulananlar:

- iSAID'de model başına `1.045`, SAMRS'de model başına `1.375` instance
  baseline kaydına alındı.
- Eski SAM3 çalışmasında iSAID'de `995/1.045`, SAMRS'de `1.363/1.375`
  instance `ok` durumundadır. Başarısız kalan instance'lar da parite hash'ine
  dahil edilmiştir.
- Runtime ve `model_version` metadata değişiminin maske hash'ini değiştirmediği,
  tek bir RLE değişikliğinin ise denetimi bozduğu birim testlerle doğrulandı.
- Eşlenmiş model karşılaştırmasındaki Wilcoxon testi instance'ları bağımsız
  saymak yerine kaynak-sahne ortalama farkları üzerinde çalışacak biçimde
  düzeltildi; Holm çoklu-test düzeltmesi korunmuştur.
- Canonical analiz her prediction dosyası için başarılı, boş maske, eksik
  bbox, inference hatası, duplicate ID ve RLE alan tutarlılığı sayılarını
  `prediction_status_audit.csv` dosyasında toplayacak şekilde genişletildi.
- Final kalite kapısı beklenen `24` matched ve `18` unmatched prediction
  koşulunu; duplicate, durum-alan uyuşmazlığı ve inference hatası bulunmamasını
  zorunlu tutar.
- Mevcut GT çıktılarında SAM3'ün iSAID'deki `50`, SAMRS'deki `12` başarısız
  çıktısının inference hatası değil, tutarlı biçimde kodlanmış boş maske olduğu
  doğrulandı.
- Seçilmiş test setlerinde `No Overlap` görüntülerinin tamamında maksimum çift
  bbox IoU değeri tam `0`; `Overlap` görüntülerinin tamamında değer dondurulmuş
  `0,001` eşiğinin üzerindedir. İki veri setinde fiilî strata semantiği aynıdır.
- Ham split kökeninin birebir olmadığı açık bir sınırlılık olarak kaydedildi:
  iSAID resmi insan etiketli validation havuzunu test için korurken SAMRS tüm
  kaynakları `source_scene_id` düzeyinde grouped `70/15/15` ayırır. Her iki
  tarafta source-scene leakage sıfır ve final test örnekleme kuralı aynıdır;
  ancak cross-dataset mutlak skor farkı yalnız referans etkisi diye
  yorumlanmayacaktır. Nedensel ana kanıt aynı görüntü/tahmin dual-reference
  denetimidir.
- Model provenance denetimi checkpoint hash'lerine ek olarak SAM1/SAM2/SAM3
  processor ve config dosyalarının tekil SHA-256 değerlerini ve birleşik tree
  hash'ini de donduracak şekilde genişletildi. Güncel model denetimi yeniden
  çalıştırıldı ve üç model için `PASS` verdi.
- Henüz başlayacak detector ve segmentasyon stage'leri prepared dataset content
  manifestini çalışma başlangıcında doğrudan input olarak kaydedecek şekilde
  güncellendi. Segmentasyon stage'leri ayrıca model provenance auditini
  başlangıç manifestinde hash'ler; yalnız final aşamasındaki backfill'e
  güvenmez.
- Güncel tam test paketi `59/59 PASS`; compile kontrolü ve
  `git diff --check` geçmiştir.

Bir sonraki adım:

- İlk GPU boşaldığında altı GT-bbox inference koşulunu pinlenmiş revision ve
  checkpoint hash'leriyle yeniden üretmek.
- `prediction-parity --mode verify` ile altı koşulda birebir maske
  değişmezliğini doğrulamak.
- Devam eden altı YOLO eğitimini tamamlayıp validation eşiği, test AP ve
  YOLO-bbox segmentation matrisini üretmek.

### 2026-07-26 - Bildiri üretimi ve dört GPU'lu frozen eğitim akışı kuruldu

Yapılanlar:

- Canonical analiz dosyalarından aynı anda Markdown, DOCX ve tam altı A4
  sayfalık PDF üreten bildiri aracı eklendi.
- Bildiri üretimi `scripts/study.py paper` komutuna bağlandı.
- Bütün deney matrisi, analiz, figür ve belge çıktısını tek seferde denetleyen
  `scripts/study.py finalize` kalite kapısı eklendi.
- Final modda üç detector seed'i ve bütün YOLO-bbox sonuçları bulunmuyorsa
  aracın hata ile durması; yalnız `--allow-partial` ile açıkça işaretli taslak
  üretmesi sağlandı.
- Kapak, ilgili çalışmalar, tartışma ve benchmark kontrol listesi sayfaları
  görsel olarak düzenlendi.
- Ortak insan-referans grafiği Türkçeleştirildi; heatmap renk skalasının son
  sütunu kapatması düzeltildi.
- Refactor ve deney planlarında kalan eski “SAMRS class ID 4 = ARJ21”
  teşhisi kaldırıldı; exhaustive resmi detection audit sonucu yazıldı.
- Boş fiziksel GPU 1 ve GPU 2, `CUDA_VISIBLE_DEVICES` ile ayrı ayrı izole
  edilerek NVML uyarısına rağmen gerçek CUDA tensor testiyle doğrulandı.
- iSAID seed `123` ve `2026` eğitimleri de başlatıldı. Böylece iSAID
  seed `42/123/2026` ile SAMRS seed `42`, dört ayrı RTX A6000 üzerinde aynı
  frozen `1024×1024`, `batch=12` protokolüyle paralel çalışıyor.

Doğrulananlar:

- Kısmi bildiri PDF'i tam `6` sayfa ve A4 boyutundadır.
- Ana CLI içinde `paper` komutu görünür ve çalışır.
- Yeni iSAID eğitimlerinin ilk batch'leri yaklaşık `31,8 GB` GPU belleğiyle
  batch düşürmeden başlamıştır.
- Devam eden dört eğitim aynı model, optimizer çözümleme ve veri hazırlama
  sözleşmesini kullanmaktadır.
- Boş detector/YOLO sonuçlarında bile canonical CSV başlıklarının sabit kaldığı
  unit testle güvence altına alınmıştır.
- Run manifestleri yeni stage'lerde giriş ve çıkış dosyalarının SHA-256
  hash'lerini ve byte boyutlarını otomatik kaydeder.
- Final kalite kapısının eksik detector ve YOLO-bbox koşullarını, eksik
  10.000-bootstrap analizini ve kısmi bildiriyi kabul etmeden doğru biçimde
  durduğu doğrulandı.
- Önceki sabit `0.20` YOLO confidence kullanımının deney planındaki
  validation-based threshold sözleşmesini karşılamadığı bulundu ve
  düzeltildi. Her seed için eşik artık yalnız validation setinde bbox IoU 0.50
  altında F1'i en yüksek yapan noktadan seçiliyor; test ve YOLO-bbox SAM
  inference aynı dondurulmuş eşiği kullanıyor.
- Final kalite kapısı test eşiğinin validation artifact'iyle birebir aynı
  olduğunu ayrıca doğruluyor.
- Altı GT-bbox değerlendirmesi aynı prediction dosyalarından yeniden
  hesaplandı; artık her biri completed evaluation manifesti ve giriş/çıkış
  SHA-256 kayıtları taşıyor.
- Kontrollü iSAID SAM1 pseudo-referansı da completed run manifesti, kaynak
  prediction hash'i ve çıktı hash'iyle yeniden üretildi.
- Hazırlanmış iki veri setindeki bütün image, YOLO label, COCO ve metadata
  dosyaları deterministic content manifestlerine alındı. iSAID manifesti
  `4.307` dosya ve yaklaşık `2,46 GB`; SAMRS manifesti `6.242` dosya ve
  yaklaşık `4,73 GB` içeriği donduruyor.
- SAM1 Hugging Face revision'ı `87aecf0...`, SAM2.1 revision'ı
  `665f8e2...` olarak pinlendi. SAM1, SAM2 ve yerel SAM3 için yüklenen
  `model.safetensors` SHA-256 değerleri protokole eklendi ve model provenance
  audit'i `PASS` verdi.
- Wrapper'lar artık SAM1/SAM2 modellerini yalnız pinli revision ile yüklüyor;
  prediction `model_version` alanı SAM1/SAM2 revision'ını, SAM3 checkpoint
  SHA-256 değerini taşıyor.
- Sentetik bir prediction JSONL dosyasını ortak runner'dan dual-reference
  instance ve image-union evaluator'a kadar taşıyan küçük uçtan uca
  entegrasyon testi eklendi.
- Eski ve canonical evaluator'ın ortak non-empty mask IoU, Dice, precision ve
  recall hesaplarının birebir aynı olduğu parity testiyle doğrulandı.
- O aşamadaki tam test paketi `55/55 PASS`; `git diff --check` ve
  `pip check` de geçmiştir.

Bir sonraki adım:

- Eğitimleri batch ve tamamlanma manifestleri bakımından izlemek.
- Her tamamlanan seed için detector prediction ve COCO bbox AP üretmek.
- Dondurulmuş detector çıktılarıyla SAM1/SAM2/SAM3 YOLO-bbox inference ve
  dual-reference evaluation çalıştırmak.

### 2026-07-26 - Aynı görüntülerde bağımsız insan referansı doğrulandı

Yapılanlar:

- SAMRS SOTA test döşemeleri özgün iSAID kaynak görüntülerine template matching ile geri eşlendi.
- Eşleme yalnız yüksek template score ve piksel düzeyinde birebir görüntü kontrolü birlikte sağlandığında kabul edildi.
- SAMRS plane instance'ları ile iSAID insan plane annotation'ları one-to-one bbox IoU eşlemesiyle birleştirildi.
- Aynı SAM1, SAM2 ve SAM3 prediction'ları hem SAM1 pseudo-mask hem bağımsız iSAID insan maskesi üzerinde ölçüldü.
- İki sınır döşemesindeki sağ/alt padding durumu eşleyicide desteklenip üç unit testle doğrulandı.
- Konunun önceki çalışmalarla ilişkisi ve bildirinin iddia sınırları `docs/LITERATURE_REVIEW.md` içinde birincil kaynaklarla belgelendi.
- `scripts/study.py` prepare, validation, detector, inference, dual-reference evaluation, analysis, figure ve shared-reference audit aşamalarını tek CLI altında birleştirecek şekilde genişletildi.
- Ana README aktif matched çalışmayı ve tek CLI komutlarını gösterecek şekilde yenilendi.

Doğrulananlar:

- SAMRS test görüntüsü: `128`.
- iSAID insan etiketli kaynağa piksel düzeyinde eşlenen görüntü: `126`.
- İki eşlenemeyen kaynak, iSAID'in erişilebilir train/validation annotation'larında bulunmayan `P5949` ve `P5155` sahneleridir.
- Eşlenen SAMRS plane instance: `1.033 / 1.375`.
- Eşlenen source scene: `35`.
- SAM1 pseudo-mask ile bağımsız insan maskesi arasındaki ortalama IoU: `0.647`.
- Planlanan `10.000` tekrarlı source-scene bootstrap ile model sonuçları:

| Model | Human IoU | SAM1 pseudo IoU | Reference inflation | %95 GA |
|---|---:|---:|---:|---:|
| SAM1 | 0.648 | 0.998 | +0.350 | [0.313, 0.378] |
| SAM2 | 0.580 | 0.806 | +0.225 | [0.188, 0.255] |
| SAM3 | 0.518 | 0.694 | +0.176 | [0.140, 0.202] |

- Güncel tam unit test paketi: `47/47 PASS`.
- Ortak human-reference şekli üretildi ve görsel QA'dan geçti.

Bilimsel yorum:

- Bu analiz farklı veri setlerindeki iki ayrı test sonucu karşılaştırmıyor; aynı görüntü, aynı instance, aynı bbox ve aynı prediction üzerinde yalnız referansı değiştiriyor.
- Üç modelin skoru da SAM1 pseudo referansında yükseliyor, fakat artış SAM1 için en büyük.
- SAM1'in pseudo referansta yaklaşık `0.998`, bağımsız insan referansında yaklaşık `0.648` IoU alması teacher-reference affinity için doğrudan kanıttır.
- İlk analizde human ve pseudo referans model sıralaması değişmedi; bu nedenle mevcut kanıt güçlü metric inflation gösterir, fakat ranking reversal iddiası göstermez.
- SAMRS training/pretraining verisi olarak yararlı olabilir. Bulgular, onu üreten SAM1'i ölçmek için bağımsız ground truth kabul etmenin sakıncalı olduğunu gösterir.

Bir sonraki adım:

- YOLO eğitimleri tamamlandığında üç seed'li detector ve YOLO-bbox segmentasyon sonuçlarını üretmek.
- Final canonical analizi `10.000` bootstrap ile yeniden üretmek.
- Altı sayfalık bildiri ve reproducibility appendix çıktısını tamamlamak.

### 2026-07-26 - SAMRS GT-bbox karşılaştırması tamamlandı

Yapılanlar:

- SAM1, SAM2 ve SAM3 aynı 128 SAMRS SOTA test görüntüsünde aynı original detection bbox'larıyla çalıştırıldı.
- Bütün modeller aynı SAM1 pseudo mask referansına karşı, instance seviyesinde ve aynı evaluator ile ölçüldü.
- COCO dosyasındaki referans türünün `human` veya `pseudo_sam1` olarak açıkça belirtilmesi zorunlu hale getirildi.
- Evaluator sözleşmesindeki isim değişikliği testlere yansıtıldı ve tam test paketi yeniden çalıştırıldı.
- iSAID ve SAMRS seed `42` detector eğitimleri aynı `1024×1024`, batch `12` ve frozen hyperparameter protokolüyle ayrı GPU'larda başlatıldı.

Doğrulananlar:

- Değerlendirilen SAMRS plane instance sayısı: `1.375`.
- Tam unit test paketi: `42/42 PASS`.
- SAMRS GT-bbox genel instance metrikleri:

| Model | Referans | Avg IoU | Avg Dice | Precision | Recall | Boundary IoU | Success@0.50 |
|---|---|---:|---:|---:|---:|---:|---:|
| SAM1 | SAM1 pseudo | 0.997 | 0.998 | 0.999 | 0.998 | 0.997 | 0.999 |
| SAM2 | SAM1 pseudo | 0.791 | 0.877 | 0.816 | 0.964 | 0.790 | 0.971 |
| SAM3 | SAM1 pseudo | 0.635 | 0.739 | 0.668 | 0.889 | 0.633 | 0.755 |

Bilimsel yorum:

- SAM1'in yaklaşık kusursuz skoru, SAM1'in bağımsız human ground truth'a göre kusursuz olduğunu göstermez.
- Referans maskeleri SAM1 ailesi ve aynı bbox-prompt prosedürüyle üretildiği için SAM1, kendi karar sınırlarına benzeyen etikete karşı ölçülmektedir.
- SAM2 ve SAM3 aynı bbox'ı kullandığı halde skorların kademeli düşmesi, generator-reference yakınlığının ölçülen sonucu ciddi biçimde değiştirebildiğini doğrudan gösterir.
- Bu sonuç tek başına yeterli değildir; iSAID human reference, iSAID kontrollü SAM1 pseudo reference ve detector kaynaklı koşullarla birlikte istatistiksel olarak raporlanacaktır.

Basit sonuç:

> AI ile üretilmiş bir maskeyi bağımsız ground truth gibi kullanmak, etiketi üreten modeli yapay biçimde neredeyse kusursuz gösterebilir. Bu deneyde ölçülen şey yalnızca segmentasyon kalitesi değil, model ile referans üreticisi arasındaki benzerliktir.

Bir sonraki adım:

- İki veri kümesindeki üç YOLO seed eğitimini tamamlamak.
- Her seed için tek bir dondurulmuş detection dosyası ve gerçek COCO bbox AP metrikleri üretmek.
- Aynı detection dosyasını SAM1/2/3'e vererek YOLO-bbox segmentasyon sonuçlarını çıkarmak.

### 2026-07-26 - SAMRS SOTA veri kimliği kesin olarak doğrulandı

Yapılanlar:

- İndirilen resmi `rdetlabels.zip` içindeki 17.555 rotated-detection etiketi veri köküne eklendi.
- Pickle kayıtları ile orijinal detection etiketlerini dosya, instance sırası, numeric class ID, RBox ve RHBox geometrisi düzeyinde karşılaştıran exhaustive audit yazıldı.
- Audit'in yanlış pickle sınıf adını yalnızca tam detection eşleşmesi varsa uyarıya indirmesi; tek bir geometri farkında ise hata vermesi unit testlerle doğrulandı.
- SAMRS config'i doğrulanan arşiv hash'i ve sürüm kimliğiyle güncellendi.
- Eski resmi train/valid listelerini kullanmak yerine bütün source scene'leri yeniden ayıracak matched SAMRS hazırlayıcı eklendi.

Doğrulananlar:

- Görüntü, pickle ve original detection label stem kümeleri birebir aynı: `17.555` dosya.
- Dosya başına instance sayısı farkı: `0`.
- Numeric class ID farkı: `0`.
- RBox geometri farkı: `0`.
- RHBox geometri farkı: `0`.
- Toplam doğrulanan instance: `615.407`.
- Authoritative DOTA 2.0 mapping içinde class ID `4`, gerçekten `plane`.
- İndirilen arşiv SHA-256: `bbae2fb7f81b09dae3146dde2df406db1641716a8d7b3204dbafcabf8f00c706`.

Önceki kayıt için düzeltme:

- Aşağıdaki eski kayıtlarda yer alan “yerel veri resmi SOTA değil” ve “ID 4 yalnızca ARJ21” sonuçları artık geçerli değildir.
- Bu yanlış ilk sonuç, pickle içindeki `category` metin alanına güvenilmesinden kaynaklandı.
- Pickle içindeki metin alanları FAIR1M adları taşısa da numeric label ve tüm kutu geometrileri resmi SOTA/DOTA detection dosyalarıyla eksiksiz eşleşmektedir.
- Yayımlanan train/valid listelerindeki `535` ortak source scene sorunu gerçektir; matched hazırlayıcı bu listeleri kullanmayıp source scene seviyesinde yeniden split edecektir.

Basit sonuç:

> Yerel veri resmi SAMRS SOTA kaynağı olarak kullanılabilir. Güvenilir sınıf kaynağı pickle metni değil, exhaustive olarak eşleştirilen orijinal detection anotasyonudur. Yüksek eski skorlar henüz geçerli sayılmaz; yeni source-scene-safe split ve eşlenmiş protokol ile yeniden ölçülecektir.

Bir sonraki adım:

- SAMRS SOTA'yı source-scene ayrık train/validation/test olarak hazırlamak.
- Testte her `overlap × mask area` stratum'unda tam 32 görüntü olduğunu bağımsız QA ile doğrulamak.
- Aynı SAM1/2/3 ve detector protokolünü çalıştırmak.

### 2026-07-26 - SAMRS SOTA matched corpus QA'dan geçti

Yapılanlar:

- 17.555 görüntünün tamamı source scene grupları korunarak yeniden train, validation ve test havuzlarına ayrıldı.
- Plane dışı görüntüler detector eğitimi için dondurulmuş `0.25` negative ratio ile örneklendi.
- Original RHBox görüntü sınırını aştığında maskeden kutu çıkarmak yerine detection kutusu geometrik olarak görüntüye kırpıldı.
- Kırpma öncesi kutu ve `bbox_was_clipped` bilgisi anotasyonda saklandı.
- Prepared corpus bağımsız COCO, YOLO, provenance, image-size, source-scene ve strata QA aracından geçirildi.

Doğrulananlar:

- Train: `2.581` görüntü, `12.734` plane instance, `505` source scene.
- Validation: `407` görüntü, `3.556` plane instance, `111` source scene.
- Test: `128` görüntü, `1.375` plane instance, `42` source scene.
- Train, validation ve test source scene kümelerinin kesişimi sıfır.
- Testteki dört `overlap × mask area` stratum'unun her birinde tam `32` görüntü var.
- Bütün bbox'lar original detection annotation provenance'ına sahip; mask-derived bbox yok.
- Prepared dataset QA sonucu: `PASS`.
- Güncel unit test sayısı `41`; tamamı geçti.

Basit sonuç:

> İki veri seti artık aynı plane sınıfı, aynı 1024×1024 giriş, aynı 128 test görüntüsü, aynı 4×32 strata düzeni ve source-scene-safe split kuralıyla karşılaştırılabilir durumdadır.

Bir sonraki adım:

- SAMRS testinde SAM1/2/3 GT-bbox inference ve pseudo-reference değerlendirmelerini çalıştırmak.
- SAMRS detector seed'lerini iSAID ile aynı frozen ayarlarda eğitmek.

### 2026-07-26 - iSAID matched corpus ve GT-bbox sonuçları tamamlandı

Yapılanlar:

- Test büyüklüğü protocol config içinde stratum başına `32` olarak donduruldu; toplam test büyüklüğü `128` görüntü oldu.
- Prepared dataset için COCO, metadata, görüntü, YOLO label, bbox provenance, source-scene split ve strata bütünlüğünü denetleyen bağımsız QA aracı eklendi.
- Detector inference ve gerçek COCO bbox AP hesabı SAM inference'dan ayrıldı. Her YOLO seed'i yalnızca bir dondurulmuş detection dosyası üretecek; SAM1/2/3 aynı dosyayı kullanacak.
- SAM1, SAM2 ve SAM3 gerçek ağırlıklarıyla tek örnek smoke testleri çalıştırıldı.
- Üç modelin tamamı aynı 128 test görüntüsünde original human bbox kullanılarak çalıştırıldı.
- Üç model için instance-level ve image-union human-reference metrikleri üretildi.
- Unit test sayısı `37`ye çıktı ve tümü geçti.
- YOLO seed `42` eğitimi tek GPU üzerinde frozen batch `12` ile başlatıldı.

Doğrulananlar:

- Train: `1.685` görüntü, `10.019` plane instance, `362` source scene.
- Validation: `336` görüntü, `1.961` plane instance, `78` source scene.
- Test: `128` görüntü, `1.045` plane instance, `45` source scene.
- Train, validation ve test source scene kümelerinin kesişimi sıfır.
- Testteki dört `overlap × mask area` stratum'unun her birinde tam `32` görüntü var.
- Prepared COCO bbox'larının tamamı original human annotation provenance'ına sahip; mask-derived bbox yok.
- GT-bbox genel instance metrikleri:

| Model | Avg IoU | Avg Dice | Precision | Recall | Boundary IoU | Success@0.50 |
|---|---:|---:|---:|---:|---:|---:|
| SAM1 | 0.616 | 0.751 | 0.634 | 0.959 | 0.612 | 0.836 |
| SAM2 | 0.598 | 0.735 | 0.613 | 0.963 | 0.594 | 0.785 |
| SAM3 | 0.579 | 0.688 | 0.596 | 0.838 | 0.577 | 0.767 |

Bulunan sorunlar:

- Sistem dört RTX A6000 kartını CUDA ile görüyor; ancak NVML driver/library sürümleri uyuşmadığı için NCCL çoklu GPU eğitimi başlatılamıyor.
- GPU 1 ve GPU 2'de ayrı eğitim denemeleri aynı NVML allocator sorununa takıldı. Ultralytics'in batch'i otomatik `3`e indirdiği deneme bilimsel protokolü bozacağı için durduruldu.
- Seed `42`, GPU 0 üzerinde doğru batch `12` ile sağlıklı ilerliyor. Diğer seed'ler aynı ayarla bu eğitimden sonra sırayla çalıştırılacak.
- SAM2 checkpoint'i yüklenirken Transformers `sam2_video` ile `sam2` model tipi uyumluluk uyarısı veriyor. Inference ve çıktı boyutu doğru olsa da wrapper/model sınıfı final QA'da ayrıca doğrulanacak.

Basit sonuç:

> İlk kez source-scene leakage olmadan, aynı plane sınıfı, aynı 128 görüntü, aynı original bbox ve aynı instance-level evaluator ile SAM1/2/3 karşılaştırması elde edildi. Bunlar eski eşlenmemiş raporların yerine geçecek ilk geçerli iSAID sonuçlarıdır.

Bir sonraki adım:

- Seed `42` YOLO eğitimini tamamlayıp gerçek COCO bbox metriklerini çıkarmak.
- Seed `123` ve `2026` eğitimlerini aynı frozen ayarlarla tamamlamak.
- Dondurulmuş YOLO bbox'larla SAM1/2/3 sonuçlarını üretmek.
- Doğrulanmış SAMRS kaynağını çözmek veya kontrollü pseudo-reference deneyi için bilimsel olarak savunulabilir fallback'i uygulamak.

### 2026-07-26 - Ortak deney çekirdeği ve iSAID matched corpusu

Yapılanlar:

- Dataset, image, instance, mask reference, bbox source ve prediction için ortak veri sözleşmeleri eklendi.
- Ana deneyde maskeden türetilmiş bbox kullanımı kod seviyesinde yasaklandı.
- Source scene gruplarını tek split'te tutan deterministik split üreticisi eklendi.
- Ortak ve dondurulmuş `teacher_reference_bias_v1` protocol config'i oluşturuldu.
- `scripts/study.py preflight` komutu eklendi.
- iSAID ve SAMRS aynı preflight komutunda denetlendi.
- iSAID tile bbox üretimi mask tight box yerine original human bbox'ın geometrik clipping işlemiyle düzeltildi.
- SAMRS adapter'ında pseudo mask tight box kullanımı kaldırıldı; original `rhbox` zorunlu yapıldı.
- SAM1, SAM2 ve SAM3 wrapper'ları birleşik maskeye ek olarak instance maskelerini koruyacak şekilde genişletildi.
- Ortak batch bbox segmenter, prediction JSONL şeması ve dual-reference evaluator eklendi.
- IoU, Dice, pixel precision, pixel recall, Boundary IoU ve `Success@IoU50/75/90` testlerle doğrulandı.
- Reference inflation, model ranking karşılaştırması ve source-scene clustered bootstrap eklendi.
- YOLO seed eğitimleri için idempotent run manifest altyapısı eklendi.
- iSAID resmi validation sahnelerini final test havuzu yapan, resmi train sahnelerini train/validation ayıran matched plane hazırlığı başlatıldı.

Doğrulananlar:

- iSAID human-reference audit'ini geçti.
- iSAID'de numeric category ID split'e göre değişiyor: `plane` train'de `4`, validation'da `14`.
- Sınıf seçimi split içindeki category name-to-ID mapping üzerinden yapılmalı.
- Ortak study preflight beklenen biçimde başarısız oluyor; neden SAMRS verisinin audit fail olması ve config'te `unverified` işaretlenmesi.
- CUDA tensor oluşturma testi geçti ve sistem dört NVIDIA RTX A6000 görüyor; `nvidia-smi` tarafında ayrı bir NVML driver/library uyarısı var.
- Güncel unit test sayısı 31 ve tamamı geçiyor.

Bulunan sorunlar:

- Eski wrapper'lar bir görüntüdeki bütün bbox maskelerini birleştiriyor, instance bilgisini kaybediyordu.
- Eski iSAID tile bbox'ları human maskeden yeniden tight box olarak türetiliyordu.
- Gerçek SOTA arşivi hâlâ doğrulanmış biçimde mevcut değil.

Basit sonuç:

> Yeni altyapı aynı bbox ve prediction'ı modelden bağımsız biçimde izleyebiliyor. Human ve pseudo reference artık aynı prediction üzerinde paired olarak ölçülebilecek; veri doğrulanmazsa GPU aşaması otomatik duracak.

Bir sonraki adım:

- Devam eden iSAID plane corpus hazırlığını tamamlayıp split ve strata QA yapmak.
- GT bbox + SAM1/2/3 inference çalıştırmak.
- YOLO detector eğitimlerini üç seed ile başlatmak.

### 2026-07-26 - Repo donduruldu ve provenance audit'i otomatikleştirildi

Yapılanlar:

- Refactor öncesi önemli config, script, kaynak kod ve sunum dosyalarını kapsayan hash'li repo manifesti üretildi.
- Manifest 178 dosya, Git commit kimliği ve dirty worktree durumunu kaydediyor.
- Dataset profile, category mapping, instance alanları ve source-scene split'ini kontrol eden audit aracı yazıldı.
- Audit davranışı üç sentetik unit test ile doğrulandı.
- Araç gerçek yerel `data/samrs_raw/sota` klasöründe çalıştırıldı.

Doğrulananlar:

- Yerel veride 17.555 görüntü, 17.555 pickle anotasyonu ve 615.407 instance var.
- Yerel kategori listesi resmi SAMRS SOTA profiliyle uyuşmuyor.
- Kategori ID `4`, `plane` değil `ARJ21`.
- Train ve validation listeleri arasında 535 ortak source scene var.

Bulunan sorunlar:

- Bu veri klasörü SOTA adıyla kullanılırsa audit artık hata koduyla duruyor.
- Mevcut split dosya seviyesinde ayrılmış olsa da source-scene bağımsızlığı sağlamıyor.

Basit sonuç:

> Eski yüksek skorların kaynağı artık otomatik ve tekrar üretilebilir biçimde gösterildi. Yanlış veri profili, yanlış hedef sınıf ve split leakage düzeltilmeden yeni model çalıştırılmayacak.

Üretilen kayıtlar:

- `artifacts/legacy/pre_refactor_20260726/repo_manifest.json`
- `artifacts/audits/local_samrs_claimed_sota/samrs_sota_audit.json`
- `artifacts/audits/local_samrs_claimed_sota/samrs_sota_audit.md`

Bir sonraki adım:

- Canonical dataset/image/instance sözleşmesini eklemek.
- Source scene gruplarını ayıran ve kesişimi hata kabul eden split üreticisini tamamlamak.

### 2026-07-26 - Kapsamlı planlar oluşturuldu

> **Tarihsel düzeltme:** Bu ilk kayıttaki SAMRS veri kimliği ve
> `category 4 = ARJ21` teşhisi daha sonra resmi detection anotasyonlarıyla
> yapılan exhaustive denetimde yanlışlanmıştır. Güncel doğrulanmış açıklama
> yukarıdaki “SAMRS SOTA veri kimliği kesin olarak doğrulandı” kaydındadır.
> Eski deney yine de eşlenmemiş protokol, source-scene leakage ve mask-derived
> bbox nedeniyle final kanıt olarak kullanılmamaktadır.

Yapılanlar:

- Repo refactoru için ayrıntılı uygulama planı yazıldı.
- iSAID ve SAMRS deneylerini birebir eşitleyen bilimsel deney planı yazıldı.
- Ana bildiri sorusu teacher-reference bias olarak netleştirildi.
- Ana pipeline kapsamı SAM1, SAM2 ve SAM3 ile GT bbox ve YOLO bbox koşullarına indirildi.
- Human mask ve SAM1 pseudo mask için aynı prediction'ı kullanan dual-reference değerlendirme tasarlandı.
- Instance-level metriklerin birincil, image-level union metriklerinin ikincil olması kararlaştırıldı.
- `mAP proxy` adının yanlış olduğu ve `Success@IoU50/75/90` olarak düzeltilmesi gerektiği kaydedildi.

Doğrulanan önemli sorunlar:

- Eski iSAID ve SAMRS deneyleri birebir aynı protokolü kullanmıyor.
- Yerel `samrs_raw/sota` verisindeki sınıflar resmi SOTA sınıflarıyla uyuşmuyor.
- Mevcut `plane = category 4` eşlemesi tüm uçaklar yerine yalnızca `ARJ21` sınıfını seçiyor.
- Bazı bbox'lar orijinal detection anotasyonu yerine pseudo maskeden türetiliyor.
- SAMRS split'inde aynı source scene'in farklı tile'ları train ve evaluation arasında bulunuyor.
- iSAID plane için stratum başına 128 bağımsız örnek gerçekçi görünmüyor; ortak sayı audit sonrası yaklaşık 32-40 bandında dondurulmalı.

Basit sonuç:

> Önceki yüksek "SAMRS plane" skorlarını bildiride kullanamayız. Önce doğru SOTA verisini doğrulamalı, aynı plane sınıfını seçmeli, source-scene leakage'i kapatmalı ve iki deneyi aynı ayarlarla yeniden çalıştırmalıyız.

Bir sonraki adım:

- Mevcut repo, config, script ve sonuçları silmeden envanterleyip legacy durum manifesti oluşturmak.
- Veri kimliği ve sınıf eşlemesi hatalarını deney başlamadan yakalayan audit aracını yazmak.

## Karar Kaydı

| Tarih | Karar | Gerekçe |
|---|---|---|
| 2026-07-26 | Ana hedef sınıf plane olacak | İki veri setinde ortak, boxy değil ve remote-sensing bağlamına uygun |
| 2026-07-26 | Ana karşılaştırma SAM1, SAM2 ve SAM3 ile sınırlandırılacak | Detection + bbox-prompted segmentation sorusunu temiz tutmak |
| 2026-07-26 | GT bbox ve YOLO bbox ayrı ana deneyler olacak | Segmenter kalitesi ile detector etkisini ayırmak |
| 2026-07-26 | Human reference birincil olacak | Pseudo reference bağımsız ground truth kabul edilmeyecek |
| 2026-07-26 | Pseudo reference yalnızca bias ölçümü için kullanılacak | Generator-reference ilişkisini doğrudan ölçmek |
| 2026-07-26 | Split source scene seviyesinde yapılacak | Sibling tile leakage'i önlemek |
| 2026-07-26 | Ana metrik instance-level olacak | Büyük maskelerin küçük nesneleri perdelemesini önlemek |
| 2026-07-26 | Maskeden bbox türetmek ana deneyde yasak olacak | Dairesel ve yapay kolaylaştırılmış prompt'u önlemek |
| 2026-07-26 | Eski sonuçlar silinmeyecek, geçerlilik etiketiyle korunacak | Araştırma geçmişi ve izlenebilirlik |

## Engel Durumu

Final çalışma için açık engel yoktur. Aşağıdaki maddeler çalışma sırasında
çözülmüş veya kontrollü biçimde yönetilmiş altyapı notlarıdır.

### 1. Human-pseudo instance eşlemesi

Çözüldü. SAMRS test görüntülerinin `126 / 128` tanesi piksel düzeyinde iSAID kaynak görüntülerine eşlendi ve `1.033` plane instance bağımsız insan maskesiyle birleştirildi. Eşlenemeyen iki sahne erişilebilir iSAID train/validation annotation'larında yoktur ve analizden açıkça dışlanmıştır.

### 2. GPU sistem yazılımı

NVML driver/library uyuşmazlığı doğrudan çoklu GPU NCCL kullanımını engelliyor. Fiziksel GPU 3, `CUDA_VISIBLE_DEVICES=3` ile süreç içinde logical GPU 0 olarak görünür hale getirilerek native allocator ve frozen `batch=12` ile doğrulandı. iSAID fiziksel GPU 0, SAMRS fiziksel GPU 3 üzerinde paralel eğitiliyor.

## Kilometre Taşları

- [x] Refactor planını yaz
- [x] Eşlenmiş deney planını yaz
- [x] WORKLOG oluştur
- [x] Mevcut repo ve artifact envanterini dondur
- [x] Dataset provenance audit aracını tamamla
- [x] Canonical veri sözleşmesini tamamla
- [x] Source-scene-safe split üretimini tamamla
- [x] Ortak config ve config diff kontrolünü tamamla
- [x] SAM1/2/3 ortak bbox runner'ını tamamla
- [x] Dual-reference evaluator'ı tamamla
- [x] Unit ve integration testlerini tamamla
- [x] Doğrulanmış iSAID matched deneyi çalıştır
- [x] Doğrulanmış SAMRS matched deneyi çalıştır
- [x] İstatistiksel analiz ve görselleri üret
- [x] Altı sayfalık bildiri taslağını üret
- [x] Final tekrar üretilebilirlik ve artifact QA yap

## Güncelleme Şablonu

Yeni kayıtlar aşağıdaki biçimde eklenir:

```markdown
### YYYY-MM-DD - Kısa aşama adı

Yapılanlar:

- ...

Doğrulananlar:

- ...

Bulunan sorunlar:

- ...

Basit sonuç:

> ...

Bir sonraki adım:

- ...
```

### 2026-07-26 - 512 görüntülü v2 matched çalışma

Yapılanlar:

- `teacher_reference_bias_v1` değiştirilmeden
  `studies/teacher_reference_bias_v2_512` oluşturuldu.
- iSAID ve SAMRS test protokolü her overlap × mask-area grubunda 128,
  Overall'da 512 görüntü olacak şekilde genişletildi.
- iSAID insan, iSAID SAM1 pseudo ve SAMRS SOTA pseudo sonuçları için üç ayrı
  full metric document tanımlandı.
- Eski geniş iSAID havuzundaki kayıplı COCO maskeleri resmi insan
  poligonlarından yeniden rasterize edildi.

Doğrulananlar:

- İki ham veri setinin provenance preflight denetimi geçti.
- iSAID: 512 test görüntüsü, 5.447 plane instance, dört grupta 128'er görüntü.
- SAMRS: 512 test görüntüsü, 3.713 plane instance, dört grupta 128'er görüntü.
- Hazırlanan iki veri setinin COCO maskesi, bbox, label, çözünürlük,
  referans türü ve kaynak-sahne ayrıklığı kontrolleri geçti.
- SAM1, SAM2 ve SAM3 checkpoint/revision hashleri dondurulmuş protokolle
  eşleşti.

Bulunan sorunlar:

- Eski v1 `test_pool` içindeki bazı iSAID COCO maskeleri decode edilen alanla
  uyuşmuyordu. V2 hazırlayıcı resmi kaynak poligonları kanonik RLE olarak
  yeniden kuracak biçimde düzeltildi; tekrar doğrulama geçti.

Basit sonuç:

> 32/grup eski çalışma relabel edilmedi. 128/grup protokol için yeni,
> sahne-güvenli split üretildi ve split değiştiği için altı YOLO detector
> eğitimi baştan başlatıldı.

Bir sonraki adım:

- Altı YOLO eğitimini ve gerçek bbox değerlendirmelerini tamamlamak.
- SAM1/SAM2/SAM3 GT ve YOLO bbox çıkarımlarını çalıştırmak.
- Üç ayrı full metric MD/DOCX/PDF raporunu üretip görsel QA yapmak.

### 2026-07-27 - V2 uygulama ve eğitim denetimi

Yapılanlar:

- Full metric rapor üreticisinin üç bağımsız belgeyi ayrı referans
  kapsamlarıyla üretmesi sağlandı: iSAID insan, iSAID SAM1 pseudo ve SAMRS
  SAM1 pseudo.
- Her belgede yalnız SAM1/SAM2/SAM3 × GT/YOLO bbox olmak üzere altı
  segmentation pipeline bırakıldı.
- Detector tablosu gerçek COCO bbox AP ve sabit confidence noktasındaki bbox
  precision/recall değerleriyle sınırlandırıldı.
- iSAID için üç detector seed'i ve SAMRS için ilk detector seed'i dört
  fiziksel GPU'da paralel başlatıldı.

Doğrulananlar:

- V2 çalışma kodunun 31 birim testi geçti.
- Depo yerleşiminin 6 yapısal testi geçti.
- Rapor validator'ı her segmentation belgesinde tam 5 tablo, her tabloda 6
  pipeline ve sırasıyla 512/128 görüntü bulunmasını zorunlu kılıyor.
- Eğitimlerin sonuç CSV'leri düzenli güncelleniyor; ilk dört süreçte son
  detector metrikleri sonlu ve checkpoint'ler üretiliyor.

Bulunan sorunlar:

- Proje sanal ortamında `pytest` komut girişi yoktu. Test kodu ortak test
  ortamından, deney bağımlılıkları ise proje sanal ortamından yüklenerek
  bağımlılık değiştirmeden doğrulandı.
- `nvidia-smi` NVML sürüm uyuşmazlığı nedeniyle çalışmıyor; eğitim süreçleri
  daha önce doğrulanan `CUDA_VISIBLE_DEVICES=<fiziksel GPU>` yöntemiyle CUDA
  üzerinde sorunsuz ilerliyor.

Basit sonuç:

> Veri, kapsam ve rapor sözleşmesi kilitlendi. Gerçek sonuçlar henüz
> yazılmadı; aktif eğitimler ve takip eden çıkarımlar tamamlanınca tablolar
> yalnız üretilmiş metric dosyalarından doldurulacak.

Bir sonraki adım:

- Aktif eğitimleri tamamlamak ve SAMRS'in kalan iki seed'ini çalıştırmak.
- Validation'da confidence seçip testte gerçek detector metriklerini ölçmek.
- Bütün SAM koşullarını çıkarıp üç raporu üretmek ve sayfa sayfa doğrulamak.

### 2026-07-27 - V2 ara sonuç ve kuyruk güvenliği denetimi

Yapılanlar:

- iSAID için üç YOLO26x eğitimi, validation/test detector değerlendirmesi ve
  SAM1/SAM2/SAM3 × GT/YOLO bbox çıkarımlarının tamamı bitirildi.
- Aynı iSAID tahminleri hem resmi insan maskesine hem kontrollü SAM1 pseudo
  maskesine karşı değerlendirildi.
- SAMRS seed 42 detector eğitimi/testi ve üç GT-bbox segmenter koşulu
  tamamlandı; seed 123 ve 2026 eğitimleri sürüyor.
- Son tahmin geldiğinde iki sürecin aynı evaluation dosyasını eşzamanlı
  yazmasını önlemek için rapor sonlandırma kuyruğu, 24 evaluation manifestinin
  tamamlanmasını bekleyecek biçimde ayrıştırıldı.

Doğrulananlar:

- iSAID testindeki her segmenter koşulu tam 5.447 instance içeriyor; dual
  reference dosyalarında insan ve pseudo satırları birlikte tam 10.894 kayıt.
- SAMRS GT-bbox koşullarının her biri tam 3.713 instance içeriyor.
- İlk ara sonuçta SAMRS pseudo referansına karşı GT-bbox IoU,
  SAM1/SAM2/SAM3 için sırasıyla yaklaşık `0,991 / 0,781 / 0,612`;
  iSAID insan referansında `0,653 / 0,629 / 0,655`.
- SAMRS seed 42 test detector sonucu gerçek bbox AP50 `0,913`, AP75 `0,797`,
  AP90 `0,209` ve AP50-95 `0,665`.

Bulunan sorunlar:

- `.gitkeep` dosyaları silinmiş olmasına rağmen `.gitignore` içinde kalan iki
  ölü istisna vardı; boş klasör sözleşmesiyle uyumlu olacak biçimde kaldırıldı.
- NVML uyarısı devam ediyor fakat çalışan CUDA eğitim ve çıkarım süreçlerinde
  hata veya eksik manifest gözlenmedi.

Basit sonuç:

> İnsan referansında modeller birbirine yakınken SAM1 üretimli referansta
> SAM1 açık biçimde avantaj kazanıyor. Nihai yorum, kalan iki SAMRS seed'i ve
> üç raporun bütünlük kontrolü tamamlandıktan sonra dondurulacak.

Bir sonraki adım:

- SAMRS seed 123 ve 2026 eğitimlerini, detector testlerini ve YOLO-bbox
  segmentasyonlarını tamamlamak.
- Canonical analizi derleyip üç MD/DOCX/PDF raporunu üretmek.
- Tabloları, hash manifestlerini ve PDF sayfalarını bağımsız olarak
  doğrulamak.

### 2026-07-27 - Yerel inference aktarım paketi

Yapılanlar:

- Canonical v2 çalışmasının altı eğitilmiş YOLO26x `best.pt` ağırlığı, mevcut
  beklenen yolları korunarak Git LFS kapsamına alındı.
- Tahmin, değerlendirme, analiz ve audit çıktıları ağırlıkları tekrar
  içermeyen tek canonical bundle'a dönüştürüldü.
- Prepared split anotasyonları, metadata ve YOLO label dosyaları görüntüsüz
  ayrı bir bundle'a dönüştürüldü.
- RTX 4060 8 GB için detector batch `1` ve SAM `float16` kullanan yerel
  inference profili eklendi; canonical `float32` protokol değiştirilmedi.
- LFS durumunu ve SHA-256 değerlerini kontrol eden, bundle'ları güvenli açan,
  SAM modellerini indiren ve 512+512 private test görüntüsünü aktarım için
  paketleyen yönetim aracı eklendi.

Gönderilmeyenler:

- iSAID/DOTA ve SAMRS üçüncü taraf görüntüleri public GitHub deposuna
  yüklenmedi.
- Gated SAM3 ve resmi SAM1/SAM2 checkpoint'leri yeniden dağıtılmadı; pinned
  Hugging Face kaynaklarından indirme adımı belgelendi.
- Canonical v2 için gerekmeyen tarihsel checkpoint ve raw dataset kopyaları
  aktarım paketine alınmadı.

Basit sonuç:

> Yerel bilgisayar detector eğitimini tekrarlamadan altı YOLO ağırlığını ve
> bütün canonical makine çıktılarını GitHub'dan çekebilir. Gerçek görüntüler
> lisans nedeniyle private aktarılır; 8 GB GPU profili modelleri sırayla ve
> düşük bellekle çalıştırır.

### 2026-08-03 - Sabit seed 42 plane raporları ve small-vehicle deney başlangıcı

Yapılanlar:

- Bildiri kapsamındaki detector protokolü iki veri setinde de yalnız sabit
  seed `42` kullanacak biçimde donduruldu.
- Üç plane full-metric raporu yalnız seed 42 detector ve YOLO-bbox sonuçlarıyla
  yeniden üretildi; MD, renkli DOCX, renkli PDF, tablo ve hash manifestleri
  doğrulandı.
- Small-vehicle çalışması plane protokolünün hedef-sınıf eşleniği olarak ayrı
  `teacher_reference_bias_small_vehicle_v1_512` study klasöründe oluşturuldu.
- SAMRS SOTA `small-vehicle` final split'i kaynak sahne sızıntısı olmadan 512
  görüntü ve tam `4×128` strata ile hazırlandı. Testte 7.659 küçük araç örneği
  bulunuyor.
- Small-vehicle kodunda plane protokolünden sapma kontrol edildi; protokol
  dosyaları çalışma kimliği dışında aynı ve 38 test başarılı.
- SAMRS için SAM1 ve SAM2 GT-bbox çıkarımları tamamlandı; SAM2 değerlendirme
  dosyaları yazıldı. SAM3 çıkarımı, SAM1 değerlendirmesi ve seed 42 YOLO26x
  eğitimi sürüyor.
- iSAID `Small_Vehicle` master corpus hazırlığı başlatıldı; tamamlanınca alan
  eşiği dondurulup final `4×128` split oluşturulacak.

Kalite notu:

- Kopyalanmış dokümanlardaki eski plane ve üç-seed ifadeleri temizleniyor.
- Small-vehicle QA checklist'i tamamlanmamış işleri yanlışlıkla bitmiş
  göstermeyecek biçimde sıfırlandı.
- Rapor scope metni eğitim görüntüsü sayılarını prepared metadata'dan dinamik
  okuyacak şekilde değiştirildi; eski plane sayılarını small-vehicle raporuna
  taşıyan sabit metin kaldırıldı.

Basit sonuç:

> Plane raporları artık istenen tek seed protokolünü gösteriyor. Small-vehicle
> deneyi aynı 1024 çözünürlük, 512 test görüntüsü, 4×128 strata, üç SAM modeli,
> GT/YOLO bbox ve aynı metrik sözleşmesiyle yürütülüyor.

Bir sonraki adım:

- iSAID final split'ini hazırlayıp veri QA kapılarını geçirmek.
- İki seed 42 detector eğitimini ve bütün GT/YOLO bbox çıkarımlarını bitirmek.
- iSAID insan, iSAID SAM1 pseudo ve SAMRS pseudo için üç full-metric raporu
  üretip sayfa, metrik, hash ve görsel QA kontrollerini tamamlamak.

### 2026-08-03 - Small-vehicle veri kapısı ve dense-sahne dayanıklılığı

Yapılanlar:

- iSAID `Small_Vehicle` master havuzu resmi insan polygonlarından tamamlandı.
- Model sonuçlarına bakılmadan sabitlenen `0,0018463134765625` alan eşiğiyle
  31 kaynak sahneden tam 512 test görüntüsü ve `4×128` strata oluşturuldu.
- Final iSAID train/validation/test splitleri 5.930/1.353/512 görüntü ve
  359.927/71.275/12.051 instance içeriyor; kaynak sahne kesişimleri sıfır.
- SAMRS final splitinin 7.824/1.567/512 görüntü ve
  304.414/49.792/7.659 instance içerdiği yeniden doğrulandı.
- iSAID SAM2 GT-bbox çıkarımı bütün 12.051 instance ile tamamlandı; SAM1 ve
  SAM3 GT-bbox çıkarımları başlatıldı.
- Çok yoğun bir iSAID sahnesinde SAM1'in tüm kutuları tek GPU batch'ine
  alması OOM üretti. Ortak SAM1/SAM2 wrapper'ı kutu sırasını ve instance
  sayısını koruyarak en fazla 16 kutuluk hesap parçaları kullanacak şekilde
  düzeltildi; ilgili birim testleri geçti ve SAM1 koşusu temizden başlatıldı.
- SAMRS ve iSAID YOLO26x eğitimleri sabit seed 42 ile ayrı GPU'larda sürüyor.
  iSAID eğitiminin ilk başlatmasındaki fiziksel GPU eşleme sorunu
  `CUDA_VISIBLE_DEVICES` ile izole edilerek aynı protokol değiştirilmeden
  giderildi.

Basit sonuç:

> İki veri seti de 512 görüntü ve dört eşit alt grup sözleşmesini karşılıyor.
> Dense sahnelerde hiçbir küçük araç atlanmıyor; GPU belleği yalnız hesaplama
> batch'leri küçültülerek kontrol ediliyor.

### 2026-08-03 - Kesinti sonrası tek-seed koşularının güvenli devamı

Yapılanlar:

- Detector komutuna yarım kalmış bir `last.pt` kontrol noktasını aynı çalışma
  dizininde sürdüren açık `--resume` seçeneği eklendi; `--resume` ile
  `--force` birlikte kullanılamıyor.
- SAMRS SOTA small-vehicle YOLO26x eğitiminin seed `42` kontrol noktası
  doğrulandı ve eğitim 6. epoktan devam ettirildi.
- iSAID small-vehicle eğitiminin ilk epoku tamamlanmadan kesilmiş eski koşusu
  temizlenerek aynı protokolle seed `42` eğitimine baştan başlandı.
- iSAID için SAM1, SAM2 ve SAM3 GT-bbox tahminleri 512 görüntüdeki 12.051
  instance'ın tamamıyla üretildi. İnsan ve SAM1-pseudo referanslarına karşı
  çift değerlendirmeler RAM sınırı gözetilerek sıraya alındı.
- Değerlendirme sırasında görüntü veya instance örneklemesi yapılmıyor;
  yalnız yüzde 95 güven aralıkları için sabit seed `42` ile 10.000 bootstrap
  tekrarından yararlanılıyor.

Basit sonuç:

> Hesaplama kesintisi deney protokolünü değiştirmedi ve tamamlanmış çıktılar
> kaybedilmedi. İki detector da tek seed `42` ile ilerliyor; segmentasyon
> değerlendirmeleri bütün 12.051 iSAID küçük araç örneğini kapsıyor.

### 2026-08-03 - Yoğun detector validation için bellek güvenliği

Yapılanlar:

- Ultralytics detection trainer'ın validation batch'ini varsayılan olarak
  eğitim batch'inin iki katına (`12 → 24`) çıkardığı doğrulandı.
- Yoğun small-vehicle batch'lerinde bu davranışın çok büyük
  `TaskAlignedAssigner` matrisleri ürettiği ve validation'ı CPU fallback'e
  taşıdığı ölçüldü.
- Eğitim batch'ini ve optimizasyonu değiştirmeden validation batch'ini 12'de
  tutan `DenseInstanceDetectionTrainer` eklendi.
- Düzeltmenin yalnız validation batch'ini yarıya indirdiğini ve train batch'ini
  değiştirmediğini doğrulayan iki birim test eklendi; small-vehicle paketinin
  40 testi de geçti.
- iSAID 1. epok, SAMRS 7. epok checkpoint sınırında tamamlandıktan sonra iki
  koşu da yarım optimizer adımı kullanmadan yeni trainer ile sırasıyla 2. ve
  8. epoktan sürdürüldü.
- iSAID SAM1/SAM2/SAM3 GT-bbox çift değerlendirmelerinin her birinde 24.102
  instance-metrik ve 1.024 görüntü-union satırı bulunduğu, NaN veya eksik
  instance olmadığı doğrulandı.

Basit sonuç:

> Validation artık aynı 1.353/1.567 görüntüyü daha küçük hesaplama parçalarında
> işler. Model eğitimi, etiketler ve COCO metrikleri değişmez; yoğun sahnelerin
> geçici RAM kullanımı ve validation süresi düşer.

### 2026-08-03 - Tek-seed taşınabilir paket ve bağımsız eğitim süreçleri

Yapılanlar:

- iSAID ve SAMRS small-vehicle detector eğitimleri farklı fiziksel GPU'larda,
  terminal veya Codex oturumu kapansa da devam eden bağımsız süreçlere alındı.
- SAMRS sekizinci epok doğrulaması `Precision 0,797`, `Recall 0,767`,
  `mAP50 0,779` ve `mAP50-95 0,400` ile tamamlandıktan sonra dokuzuncu epok
  kontrol noktasından sürdürüldü.
- Plane canonical sonuç paketi yeniden üretildi. Eski seed 123/2026 dosyaları,
  checkpoint kopyaları, eğitim logları ve veri seti görüntüleri arşivden
  çıkarıldı.
- Plane manifesti artık yalnız iki seed 42 `best.pt`, canonical sonuç arşivi
  ve görüntüsüz metadata arşivini listeliyor; dört varlığın SHA-256 kontrolü
  geçti.
- Hem plane hem small-vehicle paket üreticisine eski seed ve log sızıntısını
  engelleyen filtreler ve birim testleri eklendi.

Basit sonuç:

> Plane çalışmasının taşınabilir kopyası artık raporlarla aynı tek-seed
> sözleşmesine sahip. Small-vehicle eğitimleri kullanıcı oturumundan bağımsız
> biçimde ilerliyor ve tamamlanınca aynı paket kapılarından geçecek.

Ek güvence:

- İki dataset worker'ı detector manifesti tamamlanınca gerçek detector testi,
  SAM1/SAM2/SAM3 YOLO-bbox çıkarımları ve değerlendirmeleri otomatik
  çalıştıracak biçimde başlatıldı.
- Ayrı finalizer iki worker'ı bekliyor; analiz, figür, üç full-metric belge,
  rapor validator'ı, bundle üretimi ve hash kontrolünü sırasıyla çalıştıracak.
- Worker ve finalizer durum dosyaları orchestration kaydıdır; canonical
  bilimsel sonuç arşivinden açıkça dışlanır.

### 2026-08-04 - Small-vehicle eşlenmiş deney tamamlandı

Yapılanlar:

- iSAID ve SAMRS SOTA small-vehicle YOLO26x eğitimleri yalnız sabit seed `42`
  ile tamamlandı. iSAID 100 epoch çalıştı; SAMRS patience 30 ile 69. epoch'ta
  erken durdu.
- Validation üzerinde seçilen confidence eşikleri donduruldu ve 512 görüntülü
  test kümelerinde gerçek COCO bbox metrikleri hesaplandı. iSAID/SAMRS
  mAP50-95 değerleri sırasıyla `0,346 / 0,502` oldu.
- İki veri setinde SAM1, SAM2 ve SAM3 için GT-bbox ve YOLO-bbox çıkarımları
  tamamlandı. iSAID tahminleri hem insan hem SAM1 pseudo referansına karşı,
  SAMRS tahminleri yayımlanan SAM1 pseudo referansına karşı değerlendirildi.
- Kanonik analizde 190.566 instance satırı ve 90 aggregate satırı üretildi.
  Her pipeline 512 görüntüyü, her alt grup 128 görüntüyü kapsıyor; eksik,
  yinelenen veya sonlu olmayan metrik bulunmadı.
- iSAID insan, iSAID SAM1 pseudo ve SAMRS SOTA pseudo için üç full-metric
  belge MD, renkli DOCX ve renkli PDF olarak üretildi. PDF'ler 14/14/13
  sayfadır ve bütün sayfalar görsel olarak incelendi.
- Nitel görsellerin tek instance seçmediği doğrulandı: seçilen sahnedeki tüm
  GT kutuları ayrı istem olarak işleniyor, tüm referans ve tahmin maskeleri
  birleşik görünümde sunuluyor.
- Rapor validator'ı tek başına çalışabilecek biçimde kaynak yolunu kuruyor;
  resume edilmiş detector koşularında base model ile resume checkpoint
  provenance'ını ayrı doğruluyor.
- Canonical sonuç paketinden raster eğitim/validation görselleri çıkarıldı.
  Sonuç ve metadata paketleri ağırlık, log, cache ve özel veri seti görüntüsü
  içermiyor; dört LFS varlığının SHA-256 strict kontrolü geçti.

Final Overall instance IoU:

| Referans ve bbox | SAM1 | SAM2 | SAM3 |
|---|---:|---:|---:|
| iSAID insan, GT bbox | 0,658 | 0,645 | 0,370 |
| iSAID SAM1 pseudo, GT bbox | 1,000 | 0,749 | 0,419 |
| SAMRS SAM1 pseudo, GT bbox | 0,998 | 0,846 | 0,685 |
| iSAID insan, YOLO bbox | 0,478 | 0,461 | 0,299 |
| iSAID SAM1 pseudo, YOLO bbox | 0,655 | 0,550 | 0,341 |
| SAMRS SAM1 pseudo, YOLO bbox | 0,782 | 0,707 | 0,560 |

Basit sonuç:

> Aynı iSAID GT-bbox tahminleri insan yerine SAM1 pseudo referansla
> ölçüldüğünde SAM1/SAM2/SAM3 IoU değerleri `+0,342 / +0,103 / +0,049`
> değişti. Model sırası small-vehicle sınıfında aynı kaldı; ancak referansı
> üreten SAM1'in skor artışı açık biçimde daha büyüktü. Bu, plane deneyindeki
> teacher-reference bias bulgusunu ikinci sınıfta mutlak skor enflasyonu
> açısından tekrarlar.
