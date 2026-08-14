# Overleaf Kullanımı

`main.tex`, kullanıcının sağladığı Turkish Journal of Electrical Engineering & Computer Sciences `elektr` şablonunu değiştirmeden korur. `elektr.cls` ve `elksty.tex` dergi şablon paketinin dosyalarıdır; bu repoda üretilmemiştir.

Overleaf'te mevcut dergi projesine bu klasördeki `main.tex`, `ref.bib` ve seçilen `../assets/` dosyaları eklenmelidir. Yerelde derleme yapılacaksa derginin lisanslı/güncel şablon paketindeki `elektr.cls` ile `elksty.tex` aynı çalışma dizinine konmalıdır.

Figürlerde `assets/figures/*.pdf` vektör dosyaları kullanılmalıdır. Aynı
klasördeki PNG kopyaları yalnız yerel görsel kalite kontrolü içindir. Üretilen
dört figür 16 cm nihai genişliğe göre hazırlanmıştır; panel harfleri ve figür
içi metinler küçültme sonrasında okunacak boyuttadır. Ana metinde Figure 2 ile
Table 3 aynı veriyi tekrar ettiği için birlikte kullanılmamalıdır. Önerilen
dört figür + beş ana tablo seçimi, derginin en fazla 10 figür+tablo sınırı
içinde toplam dokuz görsel öğedir.

Üretilmiş tablo parçaları `booktabs` komutlarını kullanır; `main.tex` bu paketi
yükler. Geniş tablolar metne eklenirken `graphicx` paketinin sağladığı
`\resizebox{\textwidth}{!}{\input{...}}` kalıbı kullanılmalıdır. Repo dergiye
ait `elektr.cls` ve `elksty.tex` dosyalarını içermediği için bu VM'de tam dergi
derlemesi doğrulanamaz; dosyalar Overleaf'teki mevcut dergi projesine eklenmek
üzere hazırlanmıştır.
