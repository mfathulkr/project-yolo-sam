# iSAID Vehicle YOLO/SAM Study Walkthrough

Bu not, final raporun arkasındaki mantığı sakin sakin okumak için hazırlandı. Amaç sadece sonuç tablosunu görmek değil; hangi soru soruldu, hangi veri neden seçildi, hangi pipeline neyi izole ediyor ve çıkan metrikler ne anlatıyor, hepsini tek yerde tutmak.

## 1. Çalışmanın Ana Sorusu

Bu çalışma tek cümleyle şunu test ediyor:

> Yukarıdan şehir görüntülerinde araçları maskelemek istersek, sadece metin prompt kullanan SAM tabanlı modeller ile önce detection yapıp sonra SAM'e kutu vermek arasında ne kadar fark var?

Bu soru üç parçaya ayrıldı:

- **Text-only / referring segmentation**: Model sadece "vehicle" gibi metinsel bilgiyle araç maskesi çıkarabiliyor mu?
- **YOLO + SAM**: Önce YOLO araç kutularını bulursa, SAM bu kutularla daha iyi maske üretir mi?
- **GT bbox + SAM**: Detection kusursuz olsaydı, yani kutular ground truth'tan gelseydi, SAM'in maske tarafındaki üst sınırı ne olurdu?

Buna ek olarak, sonuçların sadece genel ortalamada kalmaması için eval split özellikle 2x2 ayrıldı:

- bbox overlap yok / var
- toplam hedef maske alanı düşük / yüksek

Böylece hem **üst üste binen araçlar** hem de **az/çok maskelenecek alan** durumları ayrı ayrı görülebiliyor.

## 2. Neden iSAID?

İlk başta Semantic Drone tartışıldı. Semantic Drone güzel ve drone seviyesine yakın görüntüler içeriyor, ama ana problemimiz için zayıf kaldı: semantic mask var, fakat instance ayrımı ve yoğun araç örnekleri iSAID kadar güçlü değil.

iSAID burada daha iyi oturdu çünkü:

- overhead / aerial şehir görüntüleri var,
- araç sınıfları instance polygon olarak geliyor,
- `Small_Vehicle` ve `Large_Vehicle` birleştirilerek tek `vehicle` hedefi kurulabiliyor,
- GT mask ve GT bbox doğrudan instance anotasyondan türetilebiliyor,
- overlap ve küçük/yoğun araç senaryoları seçilebilir hale geliyor.

Final çalışmada **Semantic Drone connected-component bbox kullanılmadı**. Bu iSAID çalışmasında GT mask ve GT bbox, iSAID instance polygon anotasyonlarından geliyor.

## 3. Final Veri ve Split

Ana config:

```text
configs/isaid_vehicle_yolo26x_cpu_eval.yaml
```

Hazır dataset:

```text
data/isaid_vehicle/
```

Eval split:

```text
data/isaid_vehicle/eval/
```

Eval set 128 pozitif tile içeriyor:

| Stratum | Görüntü |
| --- | ---: |
| `no_overlap__low_mask_area` | 32 |
| `no_overlap__high_mask_area` | 32 |
| `overlap__low_mask_area` | 32 |
| `overlap__high_mask_area` | 32 |

Burada `low/high_mask_area`, eval split içindeki median vehicle mask alanına göre ayrıldı. Bu, sadece obje sayısına bakmaktan daha doğru oldu; çünkü bazen az sayıda ama büyük araç veya çok sayıda ama minicik araç görüntüsü olabilir.

## 4. Denenen Pipeline'lar

Final karşılaştırmada 9 pipeline var:

| Pipeline | Ne ölçüyor? |
| --- | --- |
| `SAM3 text-only` | SAM3 sadece metinle araç bulabiliyor mu? |
| `RemoteSAM text` | Earth observation için geliştirilmiş text/referring model kutusuz ne yapıyor? |
| `SegEarth-OV3 + SAM3` | SAM3 semantic head + instance head fusion, remote-sensing open-vocab yaklaşımı olarak ne katıyor? |
| `YOLO + SAM3` | Eğitilmiş YOLO kutuları SAM3'e verildiğinde fark ne? |
| `GT bbox + SAM3` | Detection kusursuz olsa SAM3 maskesi nereye kadar çıkar? |
| `YOLO + SAM2` | Aynı YOLO kutuları SAM2'de daha mı iyi çalışıyor? |
| `GroundingDINO + SAM2` | YOLO yerine zero-shot text-to-box kullanılırsa ne oluyor? |
| `YOLO + RingMo-SAM` | Remote-sensing fine-tuned RingMo-SAM, YOLO kutularıyla ne yapıyor? |
| `GT bbox + RingMo-SAM` | RingMo-SAM için detection kusursuz olursa ne değişiyor? |

## 5. YOLO Eğitim Durumu

YOLO26x eğitimi epoch 74'te durduruldu. En iyi checkpoint epoch 51'de geldi.

| Değer | Sonuç |
| --- | ---: |
| best epoch | 51 |
| total elapsed | 4h 55m 55s |
| precision | 0.7356 |
| recall | 0.5850 |
| mAP50 | 0.6097 |
| mAP50-95 | 0.3377 |

Checkpoint:

```text
runs/yolo26x_isaid_vehicle_s1024/train/weights/best.pt
```

Bu YOLO mükemmel değil. Özellikle küçük araçlarda recall sınırlı. Bu yüzden `YOLO + SAM` ile `GT bbox + SAM` arasındaki fark, detection kalitesinin mask kalitesine ne kadar etki ettiğini gösteriyor.

## 6. Genel Sonuçlar

| Pipeline | IoU | Dice | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| GT bbox + SAM3 | 0.5230 | 0.6510 | 0.5826 | 0.8072 |
| YOLO + SAM2 | 0.4336 | 0.5615 | 0.5494 | 0.6549 |
| YOLO + SAM3 | 0.4076 | 0.5328 | 0.5130 | 0.6485 |
| RemoteSAM text | 0.3850 | 0.5132 | 0.5244 | 0.5993 |
| SegEarth-OV3 + SAM3 | 0.2952 | 0.4027 | 0.3599 | 0.6449 |
| SAM3 text-only | 0.2739 | 0.3782 | 0.3589 | 0.5850 |
| GT bbox + RingMo-SAM | 0.2625 | 0.3592 | 0.7247 | 0.2815 |
| YOLO + RingMo-SAM | 0.2349 | 0.3266 | 0.6292 | 0.2630 |
| GroundingDINO + SAM2 | 0.0713 | 0.1196 | 0.1059 | 0.3734 |

Ana okuma:

- **GT bbox + SAM3** en iyi IoU/Dice ve recall sonucunu verdi. Yani kusursuz detection, SAM mask kalitesini belirgin artırıyor.
- **YOLO + SAM2**, bu çalışmada **YOLO + SAM3**'ten biraz daha iyi çıktı.
- **RemoteSAM text**, kutusuz/text-referring modeller içinde en iyi sonuç verdi; SAM3 text-only'i ciddi geçti.
- **SegEarth-OV3**, recall tarafını artırdı ama low-mask-area sahnelerde fazla geniş maske üretme eğilimi gösterdi.
- **RingMo-SAM**, precision yüksek ama recall düşük. Temiz ama eksik maske üretiyor.
- **GroundingDINO + SAM2**, bu task için iyi zero-shot detector olmadı; çok fazla yanlış/dağınık alan üretti.

## 7. Neden Böyle Çıktı?

### Text-only SAM3 neden zayıf kaldı?

Araçlar iSAID tile'larında çoğu zaman çok küçük. Sadece "vehicle" prompt'u verildiğinde model araç kavramını buluyor ama küçük objeleri ayrıştırmakta zorlanıyor. Bazı sahnelerde araç dışı parlak/sert kenarlı şehir dokularını da araca benzetiyor. Bu yüzden pred area ratio çok yüksek:

```text
SAM3 text-only mean_pred_area_ratio = 21.4385
```

Bu, modelin GT araç alanının çok üstünde alan maskediğini gösteriyor. Sonuç: recall fena değil, precision düşük.

### RemoteSAM neden text-only içinde daha iyi?

RemoteSAM Earth observation için tasarlanmış bir text/referring segmentasyon modeli. Prompt olarak `vehicles` ve `small vehicle` birlikte kullanıldı. Bu model küçük overhead objelere daha aşina olduğu için plain SAM3 text-only'e göre daha dengeli çıktı:

```text
RemoteSAM text IoU = 0.3850
SAM3 text-only IoU = 0.2739
```

Yine de YOLO-guided SAM2'yi geçemedi. Çünkü kutusuz yaklaşım hâlâ "nerede araç var?" sorusunu kendisi çözmek zorunda.

### SegEarth-OV3 neden beklediğimiz kadar yükselmedi?

SegEarth-OV3 SAM3'ün semantic head ve instance head çıktılarını birleştiren remote-sensing open-vocabulary yaklaşımıdır. Recall yüksek:

```text
SegEarth-OV3 recall = 0.6449
```

Ama low-mask-area sahnelerde fazla geniş alan maskeliyor. Özellikle `no_overlap__low_mask_area` için pred area ratio çok yüksek:

```text
SegEarth-OV3 no_overlap__low_mask_area mean_pred_area_ratio = 76.4738
```

Bu yüzden IoU/Dice sınırlı kaldı.

### YOLO + SAM neden işe yaradı?

YOLO, SAM'e "şu bölgeye bak" diyor. Bu, SAM'in bütün tile içinde metinle arama yaparken yaşadığı false-positive problemini azaltıyor. `YOLO + SAM3`, plain SAM3 text-only'i her 2x2 stratumda geçti.

### GT bbox + SAM3 neden üst sınır?

GT bbox detection hatasını ortadan kaldırıyor. Model sadece "bu kutunun içini doğru maskele" işine kalıyor. Bu yüzden:

```text
GT bbox + SAM3 IoU = 0.5230
YOLO + SAM3 IoU = 0.4076
```

Bu fark, YOLO tarafında hâlâ iyileştirme alanı olduğunu gösteriyor.

### RingMo-SAM neden precision yüksek ama IoU düşük?

RingMo-SAM remote-sensing fine-tuned olduğu için seçtiği bölgelerde daha temiz davranıyor. Bu yüzden precision yüksek:

```text
GT bbox + RingMo-SAM precision = 0.7247
```

Ama çok fazla aracı kaçırıyor:

```text
GT bbox + RingMo-SAM recall = 0.2815
```

Yani "az ama temiz" maske çıkarıyor. IoU/Dice için bu yeterli olmuyor.

### GroundingDINO + SAM2 neden kötü?

GroundingDINO burada task-specific eğitilmiş YOLO yerine zero-shot text-to-box detector olarak kullanıldı. Remote-sensing tiny vehicle sahnesinde çok fazla yanlış/çok geniş kutu üretti. SAM2 bu kutuları iyi maskelese bile input kutuları hatalı olduğu için final mask de kötü oluyor.

## 8. 2x2 Strata Nasıl Okunmalı?

### No overlap / Low area

Az maskelenecek araç alanı ve üst üste binmeyen kutular. En zor küçük-obje precision testlerinden biri. Text-only modeller burada fazla alan maskelemeye çok açık.

RemoteSAM burada iyi iş çıkardı:

```text
RemoteSAM text IoU = 0.3563
YOLO + SAM2 IoU = 0.3647
GT bbox + SAM3 IoU = 0.4980
```

### No overlap / High area

Araç alanı daha büyük, overlap yok. Detection-guided yöntemler rahatlıyor.

```text
GT bbox + SAM3 IoU = 0.6380
YOLO + SAM2 IoU = 0.5339
YOLO + SAM3 IoU = 0.5269
RemoteSAM text IoU = 0.4563
```

### Overlap / Low area

Hem overlap var hem toplam araç alanı düşük. Bu stratum küçük, sıkışık, ayırması zor araçları gösteriyor.

```text
GT bbox + SAM3 IoU = 0.3668
YOLO + SAM2 IoU = 0.2671
YOLO + SAM3 IoU = 0.2663
RemoteSAM text IoU = 0.2365
```

YOLO + SAM2 ve YOLO + SAM3 burada neredeyse aynı. Bu güzel bir gözlem: overlap-low-area sahnesinde asıl darboğaz SAM sürümünden çok detection ve küçük obje ayrımı.

### Overlap / High area

Çok araç var, overlap net, toplam maske alanı yüksek. En görsel ve sunumda en anlaşılır stratum bu.

```text
GT bbox + SAM3 IoU = 0.5892
YOLO + SAM2 IoU = 0.5690
YOLO + SAM3 IoU = 0.5293
RemoteSAM text IoU = 0.4911
```

Burada tüm güçlü yöntemler toparlanıyor çünkü maske alanı büyük ve sinyal daha fazla.

## 9. Sunumda Anlatılacak Kısa Hikaye

Sunum için önerilen anlatı:

1. "Sadece SAM3 text prompt yeterli mi?" diye başladık.
2. Araçlar küçük ve şehir dokusu karmaşık olduğu için SAM3 text-only fazla alan maskeledi.
3. RemoteSAM gibi remote-sensing text/referring model ekleyince kutusuz baseline ciddi iyileşti.
4. Ama YOLO kutusu hâlâ önemli: YOLO + SAM2/SAM3, text-only modelleri genel olarak geçti.
5. GT bbox + SAM3 en iyi IoU/Dice sonucu verdi; yani detection kalitesi hâlâ mask kalitesini sınırlıyor.
6. RingMo-SAM temiz ama eksik maskeliyor; remote-sensing fine-tune tek başına çözüm değil.
7. 2x2 strata bize şunu gösteriyor: low-area küçük araç sahneleri en zor; high-area overlap sahnelerinde yöntemler daha iyi ayrışıyor.

## 10. Saklanan Değerli Klasörler

Çalışmayla ilgili tutulması gereken yerler:

```text
/home/ssyzai/projects/yolo-sam/project-yolo-sam/
```

Repo ve kod.

```text
/home/ssyzai/projects/yolo-sam/project-yolo-sam/data/isaid_raw/
/home/ssyzai/projects/yolo-sam/project-yolo-sam/data/isaid_raw_downloads/
/home/ssyzai/projects/yolo-sam/project-yolo-sam/data/isaid_vehicle/
```

iSAID ham veri, indirme kaynakları ve hazırlanmış YOLO/COCO/eval split.

```text
/home/ssyzai/projects/yolo-sam/project-yolo-sam/models/sam3_hf/
/home/ssyzai/projects/yolo-sam/project-yolo-sam/models/remotesam_hf/
/home/ssyzai/projects/yolo-sam/project-yolo-sam/models/ringmo_sam_hf/
```

SAM3, RemoteSAM ve RingMo-SAM model dosyaları. Git'e girmiyorlar ama lokal tekrar üretim için değerli.

```text
/home/ssyzai/projects/yolo-sam/project-yolo-sam/runs/yolo26x_isaid_vehicle_s1024/train/
```

YOLO26x eğitim çıktısı ve `best.pt`.

```text
/home/ssyzai/projects/yolo-sam/project-yolo-sam/results/isaid_vehicle_*
```

Pipeline maskeleri, raw json çıktıları, stratified metrikler, curated qualitative kartlar ve final rapor.

```text
/home/ssyzai/projects/yolo-sam/presentation_isaid_vehicle_sam3_sam2_study/
```

Final PPTX/PDF sunum paketi.

## 11. Temizlenen Eski / Final Dışı Şeyler

Final çalışmayla ilgisiz olduğu için temizlenenler:

- DOTA raw download klasörleri
- Semantic Drone raw/prepared/results/run klasörleri
- eski iSAID probe split
- eski SAM3-only sunum klasörü
- eski Semantic Drone sunum klasörü
- YOLO smoke/aborted run klasörleri
- `tmp_logs`
- `__pycache__` ve benzeri geçici Python klasörleri
- `yolo26n.pt`

## 12. Fresh Clone İçin Dış Model Notu

`external_models/` Git'e alınmıyor. RemoteSAM ve SegEarth-OV3 tekrar çalıştırılacaksa:

```bash
python scripts/setup_external_models.py
```

Bu script:

- `earth-insights/SegEarth-OV-3` reposunu çeker,
- `1e12Leon/RemoteSAM` reposunu çeker,
- CPU-only çalıştırma için gerekli küçük patch'leri uygular,
- RemoteSAM'in eski `mmcv` beklentisi için minimal lokal stub ekler.

Ardından model checkpoint'leri zaten şu klasörlerde olmalı:

```text
models/sam3_hf/
models/remotesam_hf/
models/ringmo_sam_hf/
```

## 13. Reproduce Komutları

Final CPU eval akışı:

```bash
python scripts/run_sam3_text.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_remotesam_text.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_segearth_ov3.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_yolo_sam3.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_gt_box_sam3.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_yolo_sam2.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_grounded_sam2.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/run_ringmo_sam.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/evaluate_stratified_triplet.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/export_curated_qualitative_examples.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/write_isaid_experiment_report.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/export_isaid_presentation_pdf.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
python scripts/validate_isaid_experiment_outputs.py --config configs/isaid_vehicle_yolo26x_cpu_eval.yaml
```

GPU kullanmamak için:

```bash
CUDA_VISIBLE_DEVICES=''
```

## 14. En Kısa Sonuç

Bu çalışmanın net sonucu:

> Remote-sensing text/referring modelleri faydalı, özellikle RemoteSAM güçlü bir kutusuz baseline verdi. Ama küçük overhead araç segmentasyonunda detection guidance hâlâ belirgin değer katıyor. En iyi IoU/Dice, kusursuz kutu ile SAM3'te; pratikte ise YOLO + SAM2 bu koşuda en güçlü trained-detector pipeline oldu.
