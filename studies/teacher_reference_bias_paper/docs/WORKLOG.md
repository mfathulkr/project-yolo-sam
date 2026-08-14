# Worklog

## 2026-08-14 - Dergi Figürlerinin Yeniden Tasarımı

- TÜBİTAK Turkish Journal of Electrical Engineering & Computer Sciences resmî
  [final manuscript kılavuzu](https://journals.tubitak.gov.tr/elektrik/styleguide.html),
  [Overleaf şablonu](https://www.overleaf.com/latex/templates/turkish-journal-of-electrical-engineering-and-computer-science/qbbtnbyqvtmm)
  ve güncel yayımlanmış bilgisayarlı görü makalesi örnekleri incelendi.
- Eski beş figür; aynı veriyi tabloyla yinelediği, dört ısı haritasını aynı anda
  okumayı gerektirdiği ve kategorik sahne gruplarını çizgi grafiğiyle bağladığı
  için kaldırıldı.
- Dört yeni ana figür üretildi: kontrollü protokol akışı, iSAID kendi-etiketi
  bağlı nokta karşılaştırması, referansa göre kazanan model ızgarası ve dört
  sahne grubunda yüzde 95 güven aralıklı ek-IoU nokta grafiği.
- Yalnız iki SAMRS bütünlük değeri ayrıca grafiğe çevrilmedi; Table 5'te
  bırakıldı. Figure 2 ile aynı değerleri taşıyan Table 3 ana metin alternatifi
  olarak işaretlendi. Önerilen dört figür + beş tablo seçimi derginin en fazla
  10 figür+tablo sınırı içinde dokuz görsel öğedir.
- Figürler beyaz zeminli, renk-körü uyumlu, marker ve doğrudan metinle yedekli,
  16×20 cm sınırları içinde vektör PDF olarak üretildi. Times uyumlu Liberation
  Serif fontları PDF'lere gömüldü; PNG kopyaları yalnız QA amacıyla tutuldu.
- Validator figür dosya kümesini, açıklayıcı metin katmanını ve PDF sayfa
  boyutunu denetleyecek biçimde genişletildi. Sonuç: 37/37 çalışma kontrolü,
  79 test ve 2 alt test geçti; bir `pycocotools` deprecation uyarısı kaldı.
- Kullanıcının önceki talebine uygun olarak commit atılmadı.

## 2026-08-14 - Rapor ve Tablo Sadeleştirmesi

- Dört deney çapraz raporu ile ana çapraz rapordaki teknik ara tablolar okuyucu odaklı biçimde yeniden tasarlandı.
- Ana tablo artık yalnız modelin kendi etiketindeki IoU'sunu, diğer iki SAM etiketindeki ortalama IoU'yu ve aradaki ek IoU'yu gösteriyor.
- Güven aralıkları ve ikincil istatistiksel kontroller hesaplanmaya devam ediyor; kalabalık rapor tablolarından çıkarılıp analiz CSV'lerinde tutuluyor.
- Temel referanstan kendi etikete puan değişimi, referans maskeler arası benzerlik ve boş maske denetimi sade adlarla ayrı tablolara ayrıldı.
- Bütün görünür rapor ve bildiri tabloları en fazla üç ondalığa indirildi; tablo üstlerine neyi gösterdiklerini açıklayan kısa notlar eklendi.
- Dört çapraz rapor, ana rapor, bildiri tabloları ve figürleri yeniden üretildi. 37 otomatik çalışma denetimi, 79 test, 2 alt test ve 21 rapor PDF'sinin metin/ondalık taraması geçti.

## 2026-08-12 - Derin Bilimsel Denetim ve Doğrudan Affinity Testi

- Testlerin ötesinde split, görüntü hash'i, anotasyon kapsamı, provenance ve istatistik iddiaları yeniden denetlendi.
- Dört deney içinde train/validation/test kaynak-sahne kesişiminin sıfır olduğu tekrar doğrulandı.
- Dört testin de yalnız hedef-pozitif 512 görüntüden oluştuğu belgelendi; detector AP'nin resmi benchmark olmadığı raporlara eklendi.
- iSAID ve SAMRS'nin DOTA kökenli görüntüleri kısmen paylaştığı ölçüldü. Dört deneyin bağımsız replikasyon olduğu dili kaldırıldı.
- Piksel olarak aynı iSAID–SAMRS test görüntülerinde post-hoc human/published-mask anlaşması ve eşleşmeyen instance sayıları exploratory audit olarak üretildi.
- Yalnız insan etiketinden pseudo etikete geçişteki artışın yeterli olmadığı görüldü. Bu nedenle her modelin kendi etiketindeki IoU'su diğer iki SAM etiketindeki ortalama IoU ile karşılaştırıldı; güven aralıkları 10.000 kaynak-sahne bootstrap ile üretildi. İkincil istatistiksel kontrol ayrıntılı CSV'de tutuldu.
- iSAID Plane ve Small Vehicle'da iki doğrudan kontrastın bütün SAM1/2/3 YOLO-bbox güven aralıkları sıfırın üzerinde çıktı.
- `image_union` hesaplandığına ilişkin yanlış config/Overleaf ifadesi kaldırıldı; kanonik değerlendirme yalnız instance-macro olarak tanımlandı.
- Full-metric belgelerde GT-vs-YOLO farkının tam uçtan uca mask AP olmadığı ve detector yanlış pozitiflerinin mask ortalamasına girmediği açıklandı.
- SAMRS Small Vehicle tarihsel COCO description/supercategory hatası işlevsiz metadata errata olarak kaydedildi; immutable run hash'leri korunurken yeniden üretim kodu düzeltildi.
- Güncel checkpoint hash denetimi için taşınabilir `provenance/segmenter_provenance.json` üretim zinciri eklendi.
- Dört kanonik metric-cube ham prediction/reference dosyalarından yeniden
  hesaplandı; yeniden üretim öncesi ve sonrası CSV SHA-256 değerleri dört
  deneyde de birebir aynı kaldı.
- Detector AP50/AP75/AP90/AP50-95 ile validation-threshold
  precision/recall/TP/FP/FN değerleri rapor kodundan bağımsız ikinci bir
  uygulamayla yeniden hesaplandı; dört deneyde en büyük mutlak fark `0.0`
  çıktı.
- YOLO confidence filtresi ve greedy bbox eşlemesi bağımsız yeniden kurularak
  173.220 model×bbox prediction satırı, bütün native referanslar ve 12 pseudo
  referans zinciri instance düzeyinde semantik olarak denetlendi.
- Taşıma sonrası pseudo manifestlerinde eski kalan GT-inference manifest
  hash'leri düzeltildi. 12 pseudo JSONL dosyasının hash'i değişmedi; yalnız
  provenance manifestleri güncellendi.
- İddia sınırı aynı dondurulmuş checkpoint yakınlığına daraltıldı. Doğrudan
  kontrastların post-hoc exploratory olduğu, preregistered/confirmatory
  olmadığı ve çoklu karşılaştırma düzeltmesi içermediği bütün ana belgelerde
  açıklandı.
- GT/published bbox ile pseudo üretiminin tam otomatik olmadığı ve yalnız maske
  sınırı stilini izole etmediği rapor, protokol, handoff ve Overleaf notlarına
  işlendi.
- Beş bildiri figürü görsel olarak tekrar denetlendi; yinelenen çizim, üst üste
  binen etiket ve başlığa çarpan bar etiketi düzeltildi. 26 PDF'nin toplam 303
  sayfası raster ve text-layer düzeyinde tarandı; boş sayfa, eksik sayfa veya
  Unicode replacement karakteri bulunmadı.
- Bağımsız salt-okunur ajan denetiminin geçerli bilimsel bulguları işlendi:
  checkpoint düzeyi genelleme sınırı, post-hoc analiz, GT-bbox lokalizasyon
  confound'u ve SAMRS için exact-original-reproduction iddiasından kaçınma.
- Bildiri Table 3, metin ve Figure 3 ile aynı bilimsel kapsama çekildi: ana
  direct-affinity tablosunda yalnız bağımsız insan kontrolü bulunan iki iSAID
  deneyi bırakıldı; SAMRS sonuçları ayrı reference-integrity kanıtıdır.
- Güncel literatür tekrar tarandı; SAMIX ve Boxes2Pixels, pseudo maskeyi eğitim
  için seçen/düzelten fakat bağımsız test ground truth'u saymayan yakın 2026
  çalışmaları olarak literatür ve BibTeX'e eklendi.
- Kullanıcının isteğine uygun olarak bu denetim sırasında commit atılmadı.

## 2026-08-12 - Birleşik Paper Study

- Önceki üç teacher-reference-bias klasöründeki yaklaşık 38 GB veri ve sonuç, kopya oluşturmadan dört deneylik yeni yapıya taşındı.
- Taşınan dosyalar önce SHA-256 ile doğrulandı; eski/yeni yollar `MIGRATION_MANIFEST.json` dosyasına yazıldı.
- Dört prepared test kümesi doğrulandı: her biri 512 görüntü ve dört tabakada 128 görüntü.
- iSAID ve SAMRS için SAM1/2/3 referans küpleri tamamlandı.
- Bilinen pozitif nesnede hem aday hem referans boş olduğunda bunun yanlışlıkla
  IoU 1.0 sayılması engellendi; boş teacher referansı 0 puanlanıyor.
- SAM3 bbox koşulu PVS tracker arayüzüne düzeltildi; PCS sonuçları kanonik küpten çıkarıldı.
- Dört metric cube üretildi ve cardinality/identity kontrolleri geçti.
- 16 legacy full-metric MD/DOCX/PDF üretildi.
- Dört deney içi cross-analysis ve bir ana cross-analysis MD/DOCX/PDF üretildi.
- Nitel görseller, seçilen görüntüdeki bütün hedef instance'ları ve bütün GT kutuları içerecek biçimde yeniden üretildi.
- Nitel örneklerin referansın kendi IoU sonucuna göre seçilmesinin sonuç-bağımlı
  bir görsel seçim oluşturduğu saptandı. Seçim model/referans skorlarından
  bağımsız tabaka medyanı yöntemine çevrildi ve deney içindeki dört referans için
  aynı dört görüntü zorunlu kılındı.
- Bildiri için beş ana figür, beş ana tablo ve bir supplementary tabaka tablosu dört deneyden yeniden oluşturuldu.
- SAMRS published referansı insan GT olarak adlandırılmayacak biçimde config ve rapor sözleşmesi düzeltildi.
- Literatür taraması Parikh 2025/2026 biased-ruler çalışmaları ve SAMRS'nin pretraining amacıyla güncellendi.
- Plane master veri havuzları deney klasörlerine taşındı; dört deneyde de `master_config.yaml → config.yaml → prepared` zinciri kuruldu.
- CLI'a `prepare-master`, `prepare-matched` ve gerçek `--profile local_8gb` seçimi eklendi.
- 52 companion metadata dosyası ve 36 run manifesti repository-relative yollara geçirildi; özgün manifest hashleri `RUN_MANIFEST_MIGRATION_AUDIT.json` içinde korunuyor.
- Validator 36 çalışma manifestinin giriş/çıkış hashlerini, 80 maske tablosunu ve 16 detector tablosunu kaynak artifact'lara karşı strict doğruluyor.
- Bağımsız denetim bulguları kapatıldı: eski yollar, eksik Plane master zinciri, bozuk master config varsayılanları, lock dosyaları, 8 GB CLI ve Overleaf bağımlılık açıklaması düzeltildi.
- İkinci kabul denetiminde SAMRS referans/prediction cardinality'leri, model ayrışması, iki ayrı YOLO checkpoint'i ve train/validation/test kaynak-sahne ayrıklığı tekrar doğrulandı.
- Eski `teacher_reference_bias_v1` paketi ile paper study içindeki kopya arşiv kaldırıldı; rapor üreticisi artık eski raporu arşivlemek yerine deterministik olarak yeniler.
- Kök test keşfi yalnız aktif kodu çalıştıracak biçimde sadeleştirildi; master provenance regresyon testiyle birlikte son tam test takımı `79 passed, 2 subtests passed` sonucuna ulaştı.

## Nihai QA Sonucu

- 16 full-metric + 4 cross-analysis + 1 main PDF üretildi ve doğrulandı.
- Aktif çalışma manifestleri repository-relative ve strict hash doğrulamalıdır.
- Dört deney 512 görüntü, 4×128 tabaka ve beklenen instance sayılarını sağlıyor.
- Git/LFS yalnız kanonik seed-42 ağırlıkları ile gerekli taşınabilir sonuçları taşır; ham/prepared raster ve label ağaçları Git'e girmez.
- `validate_paper_study.py` bütün kontrollerde PASS vermektedir.
