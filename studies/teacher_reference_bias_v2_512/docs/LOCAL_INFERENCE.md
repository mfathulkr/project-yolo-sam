# 8 GB VRAM ile Yerel Inference

Bu belge canonical v2 çalışmasını RTX 4060 8 GB VRAM ve 32 GB RAM bulunan
yerel bilgisayarda çalıştırmak için gereken varlıkları ve komutları açıklar.

## GitHub'a Gönderilenler

- Kanonik raporda kullanılan iSAID plane seed 42 YOLO26x `best.pt`.
- Kanonik raporda kullanılan SAMRS SOTA plane seed 42 YOLO26x `best.pt`.
- Seed 123 ve 2026 ağırlıkları yalnız tarihsel yeniden üretilebilirlik için
  tutulur; güncel rapor hesaplarına girmez.
- Canonical tahmin, değerlendirme, analiz ve audit paketi: yaklaşık 102 MiB.
- Görüntü içermeyen prepared metadata/anotasyon paketi: yaklaşık 8 MiB.
- Bütün kod, config, test ve final raporlar.

Git LFS ile indirilen toplam ek içerik yaklaşık 787 MiB'dir.

## GitHub'a Gönderilmeyenler

| Varlık | Yaklaşık boyut | Neden |
|---|---:|---|
| Ham veri setleri | 42 GiB | Üçüncü taraf görüntüleri ve kullanım koşulları |
| V2 prepared train/validation/test görüntüleri | 7,1 GiB | Üçüncü taraf görüntü kopyaları |
| Yalnız iki test görüntü klasörü | 1,35 GiB | Git yerine private aktarım gerekir |
| SAM1 ViT-H | 2,4 GiB disk cache | Pinned Hugging Face revision'dan indirilir |
| SAM2.1 Hiera Large | 857 MiB disk cache | Pinned Hugging Face revision'dan indirilir |
| SAM3 Transformers checkpoint | 3,44 GB | Gated model; kullanıcının kendi erişimi gerekir |
| Tarihsel model ve sonuçlar | 10+ GiB | Canonical v2 inference için gerekli değildir |

iSAID görüntüleri Google Earth/DOTA kaynak koşullarına tabidir ve akademik
kullanımla sınırlandırılmıştır. SAMRS de araştırma amaçlı yayımlanmıştır.
Bu nedenle görüntüler public GitHub deposunda yeniden dağıtılmaz.

## 1. Clone ve LFS

```bash
git clone git@github.com:mfathulkr/project-yolo-sam.git
cd project-yolo-sam
git lfs install
git lfs pull
```

LFS dosyalarını doğrula:

```bash
python studies/teacher_reference_bias_v2_512/scripts/manage_local_assets.py \
  status --verify-hashes --strict
```

## 2. Python ve CUDA Ortamı

Önce NVIDIA sürücüsüne uygun CUDA destekli PyTorch kurulmalıdır. Ardından:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Kontrol:

```bash
.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Sonuç ve Metadata Paketlerini Geri Yükleme

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/manage_local_assets.py \
  restore prepared-metadata canonical-results
```

Bu işlem canonical sonuçları `results/`, anotasyon/metadata dosyalarını
`data/prepared/` altına açar. Dataset görüntüleri bu paketlerde yoktur.

## 4. Test Görüntülerini VM'den Private Aktarma

VM üzerinde 512 + 512 test görüntüsünü tek tar dosyasına koy:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/manage_local_assets.py \
  export-private-test-images --output /tmp/v2_private_test_images.tar
```

Dosya yaklaşık 1,35 GiB olur. Bunu `scp`, `rsync` veya harici disk ile yerel
bilgisayara aktar. Yerel repoda geri yükle:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/manage_local_assets.py \
  restore-private-test-images \
  --archive /path/to/v2_private_test_images.tar
```

## 5. SAM Modellerini İndirme

SAM1 ve SAM2:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/manage_local_assets.py \
  download-models sam1 sam2
```

SAM3 için önce Hugging Face hesabında `facebook/sam3` erişimini kabul et ve:

```bash
export HF_TOKEN=hf_...
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/manage_local_assets.py \
  download-models sam3
```

SAM3 indirmesi yalnız Transformers inference için gerekli dosyaları çeker;
ayrıca 3,45 GB büyüklüğündeki `sam3.pt` kopyasını indirmez.

## 6. 8 GB Profilini Smoke Test Etme

`protocol.local_8gb.yaml`, canonical protokoldeki veri ve model ayarlarını
korur; yalnız detector batch değerini `1`, SAM dtype değerini `float16`
yapar. Bu profil bellek içindir. Canonical raporlar `protocol.yaml` ile
üretilmiştir; float16 tekrarları bit düzeyinde aynı sonuç sayılmaz.

Önce bir gerçek GT-bbox örneği:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/smoke_test_matched_segmenter.py \
  --protocol \
    studies/teacher_reference_bias_v2_512/configs/protocol.local_8gb.yaml \
  --dataset \
    studies/teacher_reference_bias_v2_512/configs/datasets/isaid_plane.yaml \
  --model sam1 --device 0 \
  --output /tmp/isaid_sam1_local_smoke.json
```

SAM2 ve SAM3 için yalnız `--model` değeri değiştirilir. Modeller ayrı
process'lerde sırayla çalıştırılmalıdır; aynı anda GPU'ya yüklenmemelidir.

## 7. Tam Test Inference

GT bbox ile 512 iSAID görüntüsü:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/study.py infer \
  --protocol \
    studies/teacher_reference_bias_v2_512/configs/protocol.local_8gb.yaml \
  --dataset \
    studies/teacher_reference_bias_v2_512/configs/datasets/isaid_plane.yaml \
  --model sam1 --bbox-source gt_bbox --split test --device 0 --force
```

YOLO bbox ile çalıştırmak için LFS'den gelen detector ağırlığı ve restore
edilen canonical detector çıktıları kullanılır:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_v2_512/src \
.venv/bin/python \
  studies/teacher_reference_bias_v2_512/scripts/study.py infer \
  --protocol \
    studies/teacher_reference_bias_v2_512/configs/protocol.local_8gb.yaml \
  --dataset \
    studies/teacher_reference_bias_v2_512/configs/datasets/isaid_plane.yaml \
  --model sam1 --bbox-source yolo_bbox --seed 42 \
  --split test --device 0 --force
```

SAMRS için dataset config'i `samrs_sota_plane.yaml` yapılır. Sabit seed 42 ve
üç SAM modeli aynı komut şablonuyla sırayla çalıştırılabilir.

## Bellek Notları

- YOLO26x detector değerlendirmesi yerel profilde batch `1` kullanır.
- SAM modelleri `float16` ve tek process olarak çalışır.
- İlk deneme SAM2 ile yapılabilir; SAM1 ViT-H ve özellikle SAM3 daha ağırdır.
- SAM3 8 GB'da yine OOM verirse bu donanım sınırıdır. Model CPU'da
  çalıştırılabilir ancak çok yavaş olur; canonical VM sonucu bundle içinde
  zaten bulunmaktadır.
- Yerel `float16` çıktıları canonical `float32` raporun yerine geçirilmemeli,
  ayrı runtime doğrulaması olarak yorumlanmalıdır.
