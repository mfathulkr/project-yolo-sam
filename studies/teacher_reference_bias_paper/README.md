# Teacher-Reference Bias Paper Study

Bu klasör, bildiride kullanılacak dört kontrollü deneyi tek ve yeniden üretilebilir bir çalışma altında toplar. Eski `teacher_reference_bias_*` klasörleri artık kanonik çalışma değildir.

## Bilimsel Soru

SAM tarafından üretilen maskeler test referansı yapıldığında, aynı dondurulmuş üretici checkpoint bağımsız insan referansındaki başarısına göre daha avantajlı görünebilir mi?

Çalışma bu soruyu aynı görüntüleri, instance'ları, bbox istemlerini ve dondurulmuş SAM1/2/3 tahminlerini sabit tutup yalnız değerlendirme referansını değiştirerek inceler.

## Dört Kanonik Deney

| Deney | Hedef | Görüntü | Instance | Temel referans | Rol |
| --- | --- | ---: | ---: | --- | --- |
| `isaid_plane` | Plane | 512 | 5.447 | İnsan maskesi | Ana bağımsız kontrol |
| `isaid_small_vehicle` | Small Vehicle | 512 | 12.051 | İnsan maskesi | Ana bağımsız kontrol |
| `samrs_plane` | Plane | 512 | 3.713 | Yayımlanmış SAMRS maskesi | Destekleyici SAM-türevi referans analizi |
| `samrs_small_vehicle` | Small Vehicle | 512 | 7.659 | Yayımlanmış SAMRS maskesi | Destekleyici SAM-türevi referans analizi |

Her deneyde dört referans vardır. iSAID için `human`, `pseudo_sam1`, `pseudo_sam2`, `pseudo_sam3`; SAMRS için `published_samrs_reference`, `reproduced_pseudo_sam1`, `pseudo_sam2`, `pseudo_sam3` kullanılır. Yayımlanmış SAMRS referansı insan ground truth değildir.

## Dizin Yapısı

- `configs/`: bütün deneylere ortak kanonik protokol ve düşük VRAM yerel profil.
- `experiments/`: dört deney; her birinde matched `config.yaml`, hamdan havuz üretimi için `master_config.yaml`, veri, sonuç, rapor ve deney belgeleri bulunur.
- `src/`: ortak değerlendirme, raporlama ve pseudo referans kodu.
- `scripts/`: yeniden üretim ve doğrulama komutları.
- `analysis/`: dört deney arasındaki ana çapraz analiz.
- `paper_writing/assets/`: bildiride kullanılacak kaynak-hash'li figür ve tablolar.
- `paper_writing/overleaf/`: kullanıcının verdiği `elektr` şablonunu koruyan yazım iskeleti.
- `literature_review/`: literatür sentezi ve arama denetimi.
- `docs/`: ortak protokol, reproducibility, handoff, worklog ve taşıma manifesti.

## Kanonik Çıktılar

Her deneyde dört adet legacy full-metric belge bulunur:

```text
experiments/<experiment>/reports/full_metrics/<reference>/
```

Her klasör MD, renkli DOCX, renkli PDF, beş maske tablosunun ve detector tablosunun CSV'leri, dört nitel görsel ve hash manifesti içerir. Her deneyin bütün referanslarını birlikte yorumlayan belge:

```text
experiments/<experiment>/reports/cross_analysis/
```

Dört deney arası ana belge:

```text
analysis/main_cross_analysis_colored.pdf
```

## Kritik Yorum Kuralları

1. iSAID insan referansı ana bağımsız kontrol kanıtıdır.
2. SAMRS yayımlanmış referansı SAM1-türevidir; insan başarısı olarak sunulmaz.
3. GT-bbox öz-referans diagonalindeki `1.000`, model performansı değil matematiksel identity/kapsama kontrolüdür.
4. Ana karşılaştırma, YOLO bbox koşulunda aynı dondurulmuş modelin kendi ürettiği etiketteki IoU'sundan diğer iki SAM etiketindeki ortalama IoU'yu çıkarır. Pozitif **ek IoU**, modelin kendi etiketinde daha yüksek puan aldığını gösterir. Ayrıntılı güven aralıkları ve ikincil istatistiksel kontrol analiz CSV'lerinde saklanır.
5. Pseudo etiketlerin eğitimde yararlı olması ile bağımsız test referansı olarak geçerli olması farklı sorulardır.
6. iSAID ve SAMRS farklı anotasyon ürünleridir ancak DOTA kökenli görüntüleri kısmen paylaşır; dört deney bağımsız replikasyon gibi sunulmaz ve tek ortalamada birleştirilmez.
7. Dört test kümesinin tamamı hedef-pozitiftir. Detector AP değerleri resmi benchmark değil, seçilmiş 512 görüntülük deney içi kontroldür.
8. Doğrudan kontrastlar ilk sonuçlar görüldükten sonra geliştirilmiştir; preregistered confirmatory test değildir, çoklu karşılaştırma düzeltmesi yoktur ve farklı checkpoint/seed ya da model ailesine genellenemez.

## Yeniden Üretim

Repo kökünden:

```bash
.venv/bin/python studies/teacher_reference_bias_paper/scripts/build_references.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/evaluate_reference_cubes.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/compile_experiment_analyses.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/deep_scientific_audit.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_experiment_figures.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_full_metric_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/write_cross_analysis_reports.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/generate_paper_assets.py
.venv/bin/python studies/teacher_reference_bias_paper/scripts/validate_paper_study.py
```

Inference ve detector eğitimi için ayrıntılı komutlar [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) ve deney klasörlerindeki belgelerde yer alır.

8 GB ekran kartında aynı CLI `--profile local_8gb` ile kullanılır. Ham veriden başlamak için önce `prepare-master`, sonra `prepare-matched` çalıştırılır; canonical raporlar varsayılan `canonical` profiline aittir.

## Ana Belgeler

- [SCIENTIFIC_PROTOCOL.md](docs/SCIENTIFIC_PROTOCOL.md)
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- [HANDOFF.md](docs/HANDOFF.md)
- [DEEP_SCIENTIFIC_AUDIT.md](docs/DEEP_SCIENTIFIC_AUDIT.md)
- [LITERATURE_REVIEW.md](literature_review/LITERATURE_REVIEW.md)
- [PAPER_STRUCTURE.md](paper_writing/PAPER_STRUCTURE.md)
