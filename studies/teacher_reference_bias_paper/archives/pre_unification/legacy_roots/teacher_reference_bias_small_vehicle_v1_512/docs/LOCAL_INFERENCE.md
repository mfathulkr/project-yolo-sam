# Yerel Inference Notları

Bu çalışma VM'de `float32` ile üretilen kanonik sonuçları ve sabit seed 42
YOLO26x ağırlıklarını kullanır. Ham iSAID ve SAMRS görüntüleri lisans ve boyut
nedeniyle public GitHub deposuna eklenmez.

## Gerekli Varlıklar

- Repository kodu, config'leri, testleri ve full-metric raporları
- iSAID ve SAMRS için seed 42 `best.pt` detector ağırlıkları
- Görüntü içermeyen prepared metadata/anotasyon paketi
- Kanonik tahmin, değerlendirme ve analiz paketi
- Yerel kullanım için lisanslı ham veya prepared test görüntüleri
- SAM1, SAM2 ve SAM3 checkpointleri

VM'deki kanonik çalışma tamamlandıktan sonra görüntüsüz paketler şu komutla
üretilir:

```bash
.venv/bin/python \
  studies/teacher_reference_bias_small_vehicle_v1_512/scripts/build_portable_bundles.py
```

İki seed 42 `best.pt` dosyası ve iki `tar.gz` paketi Git LFS ile taşınır.
`bundles/manifest.json` dosyası boyut ve SHA-256 değerlerini kaydeder.

## Ortam

```bash
git clone git@github.com:mfathulkr/project-yolo-sam.git
cd project-yolo-sam
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

CUDA kontrolü:

```bash
.venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Komut Şablonu

iSAID small-vehicle üzerinde SAM1 GT-bbox çıkarımı:

```bash
PYTHONPATH=src:studies/teacher_reference_bias_small_vehicle_v1_512/src \
.venv/bin/python \
  studies/teacher_reference_bias_small_vehicle_v1_512/scripts/study.py infer \
  --dataset \
    studies/teacher_reference_bias_small_vehicle_v1_512/configs/datasets/isaid_small_vehicle.yaml \
  --model sam1 --bbox-source gt_bbox --split test --device 0
```

YOLO-bbox için `--bbox-source yolo_bbox --seed 42` kullanılır. SAMRS koşulunda
dataset config'i `samrs_sota_small_vehicle.yaml` yapılır. SAM2 ve SAM3 için
yalnız `--model` değeri değişir.

## 8 GB VRAM Sınırı

Kanonik protokol `float32` kullanır ve 8 GB VRAM'de özellikle SAM3 için yetersiz
kalabilir. Yerel smoke test amacıyla segmenter dtype'ını `float16`, detector
batch değerini `1` yapan ayrı bir config türetilebilir. Böyle bir koşu kanonik
raporun yerine geçirilmez; yalnız çalışma zamanı doğrulamasıdır. Modeller aynı
GPU'da ayrı process'lerde sırayla çalıştırılmalıdır.

## Yeniden Üretilebilirlik

Rapor üretiminde varsayılan veya elle yazılmış metrik yoktur. Gerekli detector,
tahmin veya evaluation manifesti eksikse analiz ve rapor üretimi hata vererek
durur. Public pakette görüntülerin bulunmaması nedeniyle tam inference için
veri setlerinin kullanım koşullarına uygun biçimde ayrıca edinilmesi gerekir.
