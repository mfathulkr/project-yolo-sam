# Yerel Aktarım Paketleri

Bu dizindeki büyük dosyalar Git LFS ile tutulur. Normal `git clone` sonrasında
aşağıdaki komut gerçek dosyaları indirir:

```bash
git lfs install
git lfs pull
```

Paketler:

- `canonical_results_without_weights.tar.gz`: canonical tahmin, detector
  değerlendirmesi, segmentasyon değerlendirmesi, analiz ve audit çıktıları.
- `prepared_metadata_without_images.tar.gz`: hazırlanmış split yapısı,
  COCO anotasyonları, metadata ve YOLO label dosyaları. Üçüncü taraf görüntüler
  pakete dahil değildir.
- İki `best.pt`: iSAID plane ve SAMRS SOTA plane için sabit `seed 42` ile
  eğitilmiş YOLO26x detector ağırlıklarıdır. Ağırlıklar paket içinde
  kopyalanmaz; doğrudan `results/detectors/.../weights/best.pt` yollarında
  LFS ile tutulur.

Boyutlar ve SHA-256 değerleri [manifest.json](manifest.json) içindedir.
Geri yükleme ve doğrulama komutları
[../docs/LOCAL_INFERENCE.md](../docs/LOCAL_INFERENCE.md) belgesindedir.
