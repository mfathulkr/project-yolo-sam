# YOLO + SAM2 vs SAM3 on iSAID Swimming Pool

Bu repo, aerial goruntuler uzerinde tek sinifli bir deney kurar:

1. `Pipeline A`: YOLO detection -> bbox prompt -> SAM2 mask
2. `Pipeline B`: text prompt `"swimming pool"` -> SAM3 concept segmentation

Temel veri kaynagi `iSAID` veri setidir. Deneyde sadece `swimming pool` kategorisi tutulur.

## Neden iSAID?

- Resmi olarak aerial `instance segmentation` veri setidir.
- `swimming pool` sinifi vardir.
- Pixel-level anotasyon kullandigi icin gercek `mask IoU` hesaplanabilir.

Not: iSAID resmi sayfasina gore veri seti sadece akademik kullanim icindir.

## Deney Kurallari

- Tek kategori: `swimming pool`
- YOLO sadece bu sinifta egitilir
- Segmentasyon metrikleri ayni evaluation split uzerinde hesaplanir
- Varsayilan evaluation split `val`'dir
  - Cunku iSAID resmi `test` split'inde ground truth public degildir

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env.example` dosyasini `.env` olarak kopyalayip ihtiyaca gore su degiskenleri doldur:

- `HF_TOKEN`: local `facebook/sam3` modelini indirmek ve calistirmak icin gerekli
- `ROBOFLOW_API_KEY`: hosted SAM3 backend'ine donmek istersen gerekli

## Hizli Akis

### 1. Resmi iSAID train ve val splitlerini indir

```bash
python scripts/download_dataset.py
```

### 2. `swimming pool` tek-sinif subset'ini hazirla

```bash
python scripts/prepare_isaid_subset.py
```

Bu adim:

- train ve val anotasyonlarinda sadece `swimming pool` kategorisini birakir
- tum goruntuleri korur
- pool olmayan goruntuler icin bos label yazar
- YOLO detect egitimi icin `labels/`
- evaluation icin `_annotations.coco.json`
- YOLO icin `data.yaml`

### 3. Model dosyalarini indir

```bash
python scripts/download_models.py
```

Bu script:

- `SAM2` checkpoint'ini Ultralytics uzerinden indirir veya cache'den kullanir
- `SAM3` local model dosyalarini Hugging Face uzerinden indirir
  - `facebook/sam3` gated oldugu icin `HF_TOKEN` ve repo erisimi gerekir

Varsayilan `Pipeline B` local SAM3 backend kullanir.

### 4. YOLO modeli egit

```bash
python scripts/train_yolo.py
```

### 5. Pipeline A calistir

```bash
python scripts/run_pipeline_a.py
```

### 6. Pipeline B calistir

```bash
python scripts/run_pipeline_b.py
```

Not: Ilk calistirmada local `SAM3` model dosyalari yoksa indirilmeye calisilir. `HF_TOKEN` yoksa veya hesap `facebook/sam3` reposuna erisimli degilse bu adim gated-model hatasi verir.

### 7. Sonuclari degerlendir

```bash
python scripts/evaluate_experiment.py
```

CSV ciktilari `results/metrics/`, overlay'ler `results/visualizations/` altina yazilir.

## Dizin Yapisi

```text
project_yolo-sam/
├── configs/
│   └── experiment.yaml
├── data/
│   ├── isaid_raw/
│   ├── isaid_swimming_pool/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── results/
│   ├── metrics/
│   ├── pipeline_a/
│   ├── pipeline_b/
│   ├── visualizations/
│   └── .gitkeep
├── runs/
│   └── .gitkeep
├── scripts/
│   ├── download_dataset.py
│   ├── prepare_isaid_subset.py
│   ├── download_models.py
│   ├── run_pipeline_a.py
│   ├── run_pipeline_b.py
│   ├── train_yolo.py
│   └── evaluate_experiment.py
└── src/
    └── pool_segmentation_compare/
```

## Notlar

- `Pipeline A` local `YOLO + SAM2` kullanir.
- `Pipeline B` varsayilan olarak local `transformers + SAM3` kullanir.
- Istersen config'te `sam3.backend: hosted` yapip Roboflow serverless endpoint'ine donebilirsin.
- Local `facebook/sam3` gated oldugu icin Hugging Face erisimi gerekir.
- `mask IoU` varsayilan olarak sadece ground truth havuz bulunan goruntuler uzerinde raporlanir.
