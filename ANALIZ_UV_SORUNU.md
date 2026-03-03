# Facial Details Synthesis — UV / Isomap Sorunu Analizi

> **✅ ÇÖZÜLDÜ** — UV fix uygulandı ve doğrulandı. Convert betiği güncellendi (`--no-uv` flag desteği eklendi), .bin UV ile yeniden üretildi, pipeline 8/8 görüntüde başarılı çıktı verdi. No-UV deneyi ile UV'nin zorunluluğu doğrulandı.  
> **İlgili doküman:** [PIPELINE_ANALIZ.md](PIPELINE_ANALIZ.md) — Tam pipeline akışı, modül açıklamaları, kaynak kod incelemesi ve detay analizi.

**Proje:** [apchenstu/Facial_Details_Synthesis](https://github.com/apchenstu/Facial_Details_Synthesis)  
**EOS deposu:** [patrikhuber/eos](https://github.com/patrikhuber/eos)  
**EOS fork (bu projede):** [LansburyCH/eos-expression-aware-proxy](https://github.com/LansburyCH/eos-expression-aware-proxy) — `./src/proxyEstimator` alt modülü olarak bulunur, BFM2017 model fitting için kullanılır.  
**Referans issue:** [apchenstu/Facial_Details_Synthesis#35](https://github.com/apchenstu/Facial_Details_Synthesis/issues/35)

---

## Özet

`result.isomap.png` tamamen siyah çıkmasının temel nedeni, release paketinde **yanlış convert betiği** kullanılarak üretilen `bfm2017-1_bfm_nomouth.bin` dosyasında UV (`vt`) koordinatlarının bulunmamasıdır.

Orijinal çalışmada yazarlar, EOS fork'unda (`LansburyCH/eos-expression-aware-proxy`) kendi ürettikleri `texture_uv.mat` dosyasını kullanarak BFM2017 mesh'ine UV koordinatları eklemişlerdir. Bu dosya fork deposunun `bfm2017/texture_uv.mat` yolundadır (~760KB, per-vertex UV koordinatları). Fork'taki modifiye `convert-bfm2017-to-eos.py` betiği bu UV'leri `.bin` modele gömer. Ancak **release paketi bu dosyayı ve doğru convert betiğini içermez** — bu bir paketleme hatasıdır (oversight), kasıtlı bir sadeleştirme değildir.

GitHub Issue #35'te aynı sorun başka bir kullanıcı tarafından rapor edilmiş ve fork'un güncel sürümüne geçilerek çözülmüştür.

---

## Temel Bulgu: `texture_uv.mat` ve Fork'taki Convert Betiği

### Orijinal çalışma UV'yi nasıl üretmiş?

Yazarlar (Zhang Chen et al.) BFM2017 mesh topolojisi (53149 vertex) için kendi UV parametrizasyonlarını hesaplayıp `texture_uv.mat` dosyasına kaydetmişlerdir. Bu dosya:

- **Konum:** `LansburyCH/eos-expression-aware-proxy` deposunda `bfm2017/texture_uv.mat`
- **İçerik:** `N×2` MATLAB matrisi (per-vertex UV koordinatları), `texture_uv` anahtarıyla
- **Boyut:** ~760KB
- **Geçmişi:** İlk commit'te (`c1a0b5f`, 14/08/2019, `zhangchen8`) eklendi; `9019b32`'de güncellendi
- **Kaynak:** BFM2017'nin kendisinde UV yoktur; bu dosya tamamen yazarlar tarafından üretilmiştir

### Fork'taki convert betiği vs Release'deki convert betiği

| Özellik | Fork: `share/scripts/convert-bfm2017-to-eos.py` | Release: `convert-bfm2017-to-eos-v016.py` |
|---|---|---|
| `import scipy.io` | ✅ Var | ❌ Yok |
| `texture_uv.mat` yükleme | ✅ `scipy.io.loadmat('../../bfm2017/texture_uv.mat')['texture_uv']` | ❌ Yok |
| V-ekseni çevirme | ✅ `texture_uv[:, 1] = 1 - texture_uv[:, 1]` | ❌ Yok |
| UV parametre | ✅ `texture_coordinates=texture_uv` | ❌ `[]` (boş) |
| Sonuç `.bin` | UV'li model → OBJ'de `vt` satırları → geçerli isomap | UV'siz model → `vt=0` → siyah isomap |

### Fork'un convert betiği (tam):

```python
import numpy as np
import eos
import h5py
import scipy.io

bfm2017_file = r"../../bfm2017/model2017-1_bfm_nomouth.h5"

with h5py.File(bfm2017_file, 'r') as hf:
    shape_mean = np.array(hf['shape/model/mean'])
    shape_orthogonal_pca_basis = np.array(hf['shape/model/pcaBasis'])
    shape_pca_variance = np.array(hf['shape/model/pcaVariance'])
    triangle_list = np.array(hf['shape/representer/cells'])

    shape_model = eos.morphablemodel.PcaModel(shape_mean, shape_orthogonal_pca_basis, shape_pca_variance, triangle_list.transpose().tolist())

    color_mean = np.array(hf['color/model/mean'])
    color_orthogonal_pca_basis = np.array(hf['color/model/pcaBasis'])
    color_pca_variance = np.array(hf['color/model/pcaVariance'])
    color_model = eos.morphablemodel.PcaModel(color_mean, color_orthogonal_pca_basis, color_pca_variance, triangle_list.transpose().tolist())

    expression_mean = np.array(hf['expression/model/mean'])
    expression_pca_basis = np.array(hf['expression/model/pcaBasis'])
    expression_pca_variance = np.array(hf['expression/model/pcaVariance'])
    expression_model = eos.morphablemodel.PcaModel(expression_mean, expression_pca_basis, expression_pca_variance, triangle_list.transpose().tolist())

    # texture uv
    texture_uv = scipy.io.loadmat('../../bfm2017/texture_uv.mat')['texture_uv']
    texture_uv[:, 1] = 1 - texture_uv[:, 1]  # flip V coordinate
    texture_uv = texture_uv.tolist()

    model = eos.morphablemodel.MorphableModel(shape_model, expression_model, color_model, texture_coordinates=texture_uv)
    eos.morphablemodel.save_model(model, "../../bfm2017/bfm2017-1_bfm_nomouth.bin")
    print("Converted and saved model as bfm2017-1_bfm_nomouth.bin.")
```

### Issue #35: Aynı sorunun teyidi

[Issue #35](https://github.com/apchenstu/Facial_Details_Synthesis/issues/35) — "textureRender not output the result without error"

Kullanıcı **SuwoongHeo** (03/04/2020) kendi çözümünü paylaşmış:
> *"I found the problem, it was the problem of **old version of proxyEstimator which can be found in /src**. It **does not use the "texture_uv.mat"** in proxy estimation step. So the **result.obj has no texture coordinate information** resulting in erroneous behavior in "textureRender.exe". By **switching to the latest version** in the repo LansburyCH/eos-expression-aware-proxy, I could solve the problem."*

---

## EOS C++ Zinciri: UV Nasıl OBJ'ye Ulaşır

EOS fork'unda (`fit-model.exe`) UV'lerin OBJ'ye yazılma zinciri:

```
1. texture_uv.mat → convert betiği yükler
2. MorphableModel(..., texture_coordinates=texture_uv) → .bin'e gömülür
3. fit-model.exe → load_model() → model.get_texture_coordinates() (boş değilse)
4. fit_shape_and_pose() → sample_to_mesh(..., morphable_model.get_texture_coordinates())
5. sample_to_mesh():
     if (!texture_coordinates.empty()) {
         mesh.texcoords.resize(num_vertices);
         for (auto i = 0; i < num_vertices; ++i)
             mesh.texcoords[i] = Eigen::Vector2f(texture_coordinates[i][0], texture_coordinates[i][1]);
     }
6. write_textured_obj(mesh, outputfile) → vt satırları + f v/vt formatı yazılır
```

**`--save-texture` bayrağı OBJ'ye `vt` yazılmasını kontrol ETMEZ.** Bu bayrak sadece `render::extract_texture()` ile ayrı isomap PNG dosyası üretilip üretilmeyeceğini belirler. OBJ'ye `vt` yazılması tamamen `.bin` modelindeki UV varlığına bağlıdır.

---

## Detaylı Açıklama

### 1) OBJ'de neden `vt` yok?

- Release'deki `convert-bfm2017-to-eos-v016.py` betiği UV parametresi olarak boş `[]` geçirir.
- Bu nedenle `.bin` modelde `texture_coordinates` boştur (`len(model.get_texture_coordinates()) == 0`).
- `fit-model.exe` bu modeli yüklediğinde `mesh.texcoords` boş kalır.
- `write_textured_obj()` kodunda texcoords boşsa `vt` satırları yazılmaz.
- BFM2017 H5 dosyasında (`model2017-1_bfm_nomouth.h5`) UV/texcoord dataseti bulunmamaktadır (h5py ile doğrulandı).

**Doğrulama sonucu:** 5/5 OBJ dosyasında `vt=0`, `v=53149`, `f=105694`. Face format `f v/vt` ama `vt` verisi tanımsız.

### 2) `textureRender.exe` / `faceClip.exe` neden siyah üretir?

- `render_texture` komutu (orijinal): `textureRender.exe <obj> 3 <isomap.png> 0 <image> <camera> ./shaders`
- Mod 3 (unfold) vertex shader'ı (`OBJRender.vs`):
  ```glsl
  gl_Position = vec4(in_TexCoord.x*2-1, 1-in_TexCoord.y*2, 0, 1);
  ```
  Vertex pozisyonlarını UV koordinatlarından hesaplar. UV=0 olduğunda tüm vertex'ler tek bir noktaya çöker → hiçbir şey rasterize edilmez → siyah çıktı.
- Fragment shader (`OBJRender_unfold.frag`):
  ```glsl
  color.rgb = texture(rawImg, out_texcoord).rgb;
  ```
  Giriş görüntüsünden UV koordinatına göre örnek alır. UV boşsa (0,0) örnekler → siyah.

**Doğrulama sonucu:** 5/5 `isomap.png` dosyası 4096×4096, mean=0.00, nonzero=0/50331648 (%0.0) — tamamen siyah.

### 3) Displacement/Normal Map neden anlamsız?

- DFDN (pix2pix tabanlı ağ) `isomap.png` girdisini alarak `displacementmap.png` ve `normalmap.png` üretir.
- Girdi tamamen siyah olduğundan, DFDN her yüz için neredeyse aynı çıktıyı üretir.
- 5 displacement map'in MD5 hash'leri farklı ama istatistikleri neredeyse aynı (mean≈33170.7, min≈13958, max≈41766-41767).

### 4) README'deki teaser görseli neyi gösterir?

README'deki teaser, pipeline'ın **UV'li çalıştığında** ürettiği sonucu gösterir:
- Tek bir fotoğraftan üretilmiş 3D yüz modeli
- UV haritası üzerinden renklendirilmiş (isomap) proxy mesh — renkli, giriş görüntüsünden örneklenmiş yüz dokusu
- DFDN ile tahmin edilen detaylı displacement ve normal map'lerin uygulandığı, ince kırışıklıklar gibi yüz detaylarını içeren fotorealistik sonuç
- Pipeline çıktıları: `result.obj`, `result.isomap.png`, `result.displacementmap.png`, `result.normalmap.png`
- Bu kaliteye ulaşmak için OBJ'de UV olması **zorunludur**; UV yoksa isomap siyah çıkar ve tüm detay sentezi anlamsız hale gelir

### 5) Release'e özgü sorunlar

| Sorun | Detay |
|---|---|
| Convert betiği | Release'de upstream eos v0.16 betiği (`convert-bfm2017-to-eos-v016.py`) kullanılıyor; fork'un UV'li betiği dahil edilmemiş |
| `texture_uv.mat` | Release paketinde **yok**; fork deposunda `bfm2017/texture_uv.mat` yolunda mevcut |
| `render_texture()` argümanları | Release'deki `faceClip.exe` için DOĞRU (6 argüman, shader CWD'den, kamera baz isim olarak). `textureRender.exe` farklı binary (7 argüman). |
| Executable ismi | Release'de `faceClip.exe`, orijinal kaynak kodda `textureRender.exe` — farklı argüman formatları, `render_texture()` release versiyonu için doğru |

### 6) Mod 3 vs Mod 4 karşılaştırması

| Özellik | Mod 3 (Unfold) | Mod 4 (Project) |
|---|---|---|
| Vertex Shader | `OBJRender.vs` — UV'den pozisyon | `OBJRender_project.vs` — MVP matris projeksiyonu |
| Fragment Shader | `OBJRender_unfold.frag` — texture örnekleme | `OBJRender_project.frag` — Phong aydınlatma |
| UV gereksinimi | **Zorunlu** | Gereksiz (kamera matrisi kullanır) |
| Amacı | Giriş görüntüsünden UV haritasına açma | 3D render (aydınlatma ile) |

### 7) Kontrol tablosu

| Bileşen | Kontrol | Sonuç |
|---|---|---|
| `result.obj` | `vt` satırları var mı? | ✅ `vt=53149` (8/8 dosya) — FIX SONRASI |
| BFM2017 H5 | UV dataseti var mı? | ❌ Yok (BFM2017 UV sağlamaz) |
| `texture_uv.mat` | Release'de var mı? | ✅ Eklendi (fork'tan indirildi) |
| `.bin` model | `model.get_texture_coordinates()` | ✅ `len=53149` (UV'li üretildi, 309.17 MB) |
| Convert betiği | Fork'un UV'li betiği mi? | ✅ Güncellendi (fork ile birebir fonksiyonel, `--no-uv` flag eklendi) |
| `.bin` dosya sistemi | İki dosya mevcut mu? | ✅ `bfm2017-1_bfm_nomouth.bin` (aktif) + `_uv.bin` (yedek) |
| `isomap.png` | Renkli piksel var mı? | ✅ %77.8-77.9 nonzero, yüze özgü |
| Displacement map | Yüze özgü farklılık | ✅ 8 farklı MD5 hash, ~18MB, 16-bit PNG |
| No-UV deneyi | UV'siz mod test edildi mi? | ✅ 8/8 işlendi, UV zorunluluğu doğrulandı |
| `hmrenderer.exe` | Çalışıyor mu? | ❌ ACCESS_VIOLATION (OpenGL) |

---

## Çözüm

### Kesin Çözüm: `texture_uv.mat` ile `.bin` yeniden üretmek

Sorunun kök nedeni belirlenmiş, çözüm uygulanmış ve doğrulanmıştır:

1. **✅ `texture_uv.mat` dosyası indirildi:** `LansburyCH/eos-expression-aware-proxy` deposundan `bfm2017/texture_uv.mat` → `released/proxy/bfm2017/` dizinine kopyalandı.

2. **✅ Convert betiği güncellendi:** Orijinal `convert-bfm2017-to-eos-v016.py` betiğine minimum değişiklikle UV desteği eklendi. Ayrıca `--no-uv` flag ile UV'siz mod seçeneği de eklendi.

3. **✅ `.bin` modeli yeniden üretildi:** `py3_6_eos_convert` conda ortamında (Python 3.6.13, eos-py 0.16.1). İki dosya:
   - `bfm2017-1_bfm_nomouth.bin` — aktif model (pipeline bu ismi kullanır)
   - `bfm2017-1_bfm_nomouth_uv.bin` — UV'li kalıcı yedek (hızlı geri dönüş için)

4. **✅ Pipeline çalıştırıldı:** `facialDetails.py` ile 8 test görüntüsü işlendi (UV'li ve UV'siz modlarda).

5. **✅ Doğrulandı:** 
   - UV'li: OBJ `vt=53149`, isomap %77.8-77.9 nonzero, 8 farklı displacement MD5
   - UV'siz: OBJ `vt=0`, siyah isomap, 8 farklı ama düşük varyasyonlu displacement

### Convert Betiği Güncel Kullanım

```bash
# UV'li .bin üret (varsayılan) — pipeline tam detaylı çıktı üretir
python convert-bfm2017-to-eos-v016.py

# UV'siz .bin üret — orijinal release davranışı (siyah isomap, minimal detay)
python convert-bfm2017-to-eos-v016.py --no-uv
```

**Değişiklik boyutu:** Orijinal betiğe sadece `import argparse` + 3 satır argparse + `if/else` sarma + `_uv.bin` ek kaydı eklendi. Orijinal kod korundu.

### `render_texture()` argüman durumu

**İncelendi ve DOĞRU olduğu tespit edildi.** Release'deki `faceClip.exe` ile orijinal `textureRender.exe` FARKLI binary'lerdir:
- `faceClip.exe`: 6 argüman (shader yolu yok, CWD'den yükler; kamera = baz isim, `.affine_from_ortho.txt` uzantısını kendisi ekler)
- `textureRender.exe`: 7 argüman (shader yolu gerekir; kamera = tam yol)

Release'deki `render_texture()` fonksiyonu `faceClip.exe` için doğru çalışmaktadır — değişiklik gerekmiyor.

---

## No-UV Deney Sonuçları

`--no-uv` moduyla 8 görüntü işlenerek UV'nin zorunluluğu deneysel olarak kanıtlandı:

| Metrik | UV'li Mod | UV'siz Mod | Fark |
|---|---|---|---|
| OBJ `vt` count | 53149 | 0 | UV gerekli |
| Isomap nonzero | %77.8-77.9 | %0.0 (siyah) | Tamamen farklı |
| Displacement dinamik aralık | **%75** (49240/65535) | **%42** (27809/65535) | 1.8× azalma |
| Displacement std | 3122 (%4.8) | 593 (%0.9) | **5.3× azalma** |
| Displacement dosya boyutu | ~18 MB | ~5.8 MB | 3.1× azalma |
| Benzersiz MD5 hash | 8/8 | 8/8 | Her ikisinde farklı |

**Sonuç:** UV'siz modda DFDN farklı displacement map'üretse de (çünkü proxy mesh farklı), detay varyasyonu ihmal edilebilir seviyede. UV **zorunludur**.

---

## Paketleme Hatası Kanıtları

Bu sorun release paketinin bir **paketleme hatasıdır** (oversight), kasıtlı bir sadeleştirme değildir:

1. **Fork, UV desteği için özel olarak oluşturulmuştur.** `texture_uv.mat` dosyası ilk commit'ten (`c1a0b5f`, 14/08/2019) beri mevcuttur ve hiçbir commit'te kaldırılmamıştır.

2. **Pipeline UV olmadan çalışamaz.** Vertex shader (`OBJRender.vs`) UV'den pozisyon hesaplar — mod 3 "unfold" yapısal olarak UV'ye bağımlıdır. UV yoksa siyah çıktı üretir, alternatif yol yoktur.

3. **Issue #35 doğruluyor.** Başka bir kullanıcı (SuwoongHeo) aynı sorunu yaşamış, `texture_uv.mat` eksikliğini tespit etmiş ve fork'un güncel sürümüne geçerek çözmüştür.

4. **Release'deki convert betiği yanlış dosyadır.** `convert-bfm2017-to-eos-v016.py` upstream eos v0.16 betiğidir (UV'siz); fork'un `share/scripts/convert-bfm2017-to-eos.py` betiği ise UV'lidir. Yazar paketlerken yanlış betiği dahil etmiş görünmektedir.

5. **Aynı araştırma grubu.** Fork (`LansburyCH`/`zhangchen8`) ve ana repo (`apchenstu`) aynı araştırma grubunun üyeleridir. Fork, bu proje için UV desteğiyle özel olarak oluşturulmuştur.

6. **Submodule doğru fork'a bağlıdır.** `.gitmodules` dosyasında `src/proxyEstimator` → `LansburyCH/eos-expression-aware-proxy.git` commit `d8d4c7d` — bu commit'te UV'li convert betiği zaten mevcuttur.

---

## Notlar

- `result.displacementmap.png` ve `result.normalmap.png` varlığı geometrik fitting'in ve DFDN'in çalıştığını gösterir; ancak isomap siyah olduğu için displacement bilgisi anlamsızdır.
- `proxyPredictor.py` içinde `render_texture()` çağrısı yorum satırında (line ~175); yani proxy-only modda bu sorun görünmez.
- EOS deposu UV üretmez; modelin kendisinin UV sağlaması gerekir. BFM2017 de UV sağlamaz; yazarlar kendi UV parametrizasyonlarını hesaplayıp `texture_uv.mat` olarak kaydetmişlerdir.
- `--save-texture` bayrağı OBJ'ye `vt` yazılmasını kontrol etmez; sadece ayrı isomap PNG dosyası üretimini belirler. `vt` yazımı tamamen `.bin` modelindeki UV varlığına bağlıdır.
- OBJ'ye elle `vt` eklemek işe yaramaz; vertex-face indeksleme tutarlılığı gerekir ve `textureRender.exe` tinyobj üzerinden okur.
- README'deki teaser görselleri UV'li proxy mesh + normal map render efekti ile üretilmiştir. Detaylar gerçek mesh geometrisi DEĞİL, normal mapping ile aydınlatma pertürbasyonu sonucudur. Displacement map mesh'e uygulanmaz — sadece normal map türetmek için kullanılır. Detaylı analiz: [PIPELINE_ANALIZ.md](PIPELINE_ANALIZ.md) §5.
- **Kaynak kod incelemesi** (`src/` dizini) bu bulguları doğrulamıştır: `render_hm.vs` (standart MVP, vertex displacement yok), `render_hm.frag` (%100 normal mapping), `render_bssrdf.frag` (SSS, displacement yok). Detaylar: [PIPELINE_ANALIZ.md](PIPELINE_ANALIZ.md) §5A.
- **Displacement map 16-bit doğrulandı:** PNG header `bit_depth=16, color_type=0`. PIL `mode=I` (int32) olarak okur ama dosya gerçekte 16-bit. Merdivenlenme artefaktı yok.
- **DFDN patch detayları:** `areas.mat` — forehead=1547 patch (y=[128,2176] x=[128,3840]), mouse=1390 patch (y=[1856,3776] x=[128,3712]), weight mask=256×256 (5473 benzersiz değer). Toplam 2937 patch.
