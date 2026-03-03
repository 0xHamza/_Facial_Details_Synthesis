# Repo Analiz Raporu: apchenstu/Facial_Details_Synthesis

> **Analiz Tarihi:** 2 Mart 2026  
> **Analiz Perspektifi:** Fiziksel vertex deformasyonu, detaylı mesh üretimi (Detaylı mesh projesi bağlamı)  
> **Analiz Kaynakları:** Lokal klondaki kaynak kod incelemesi (released/ + src/), shader analizi, pipeline çalıştırma sonuçları, paper okuma, DeepWiki

---

## 1. Genel Bakış

| Özellik | Detay |
|---|---|
| **Repo** | [apchenstu/Facial_Details_Synthesis](https://github.com/apchenstu/Facial_Details_Synthesis) |
| **Paper** | "Photo-Realistic Facial Details Synthesis From Single Image" — ICCV 2019 |
| **Yazarlar** | Anpei Chen*, Zhang Chen*, Guli Zhang, Ziheng Zhang, Kenny Mitchell, Jingyi Yu |
| **İlk yayın** | 18 Ağustos 2019 |
| **Son güncelleme** | 7 Temmuz 2020 (~5.5 yıl önce, **aktif geliştirme yok**) |
| **Lisans** | MIT |
| **Platform** | Windows 10 (C++ exe'ler Windows'a bağımlı) |
| **Dil/Framework** | Python 3.7, PyTorch (DFDN), Keras/TF (opsiyonel emotion), C++ (EOS, OpenFace, OpenGL renderer) |
| **3DMM Modeli** | Basel Face Model 2017 (BFM2017) — 53.149 vertex, 105.694 face, sabit topoloji |
| **Eğitim Verisi** | 366 yüksek kaliteli 3D tarama (122 kişi × 3 ifade) + 163K in-the-wild yüz görüntüsü |

**Amaç:** Tek bir yüz fotoğrafından 3 aşamalı pipeline ile proxy mesh → isomap texture → displacement/normal map üretmek.

---

## 2. Pozitif Yönler

### 2.1 Öncü ve Pionier Çalışma
- 2019'da single-image'den mesoskopik yüz detayı (kırışıklık, gözenek) sentezleyen **ilk kapsamlı çalışmalardan biri**. DFDN (Deep Facial Detail Net) konsepti, sonraki çalışmalara (DECA, HRN vb.) ilham kaynağı olmuştur.

### 2.2 Gerçek 3D Tarama Verisi ile Eğitim
- 366 high-fidelity 3D scan (122 kişi × 3 ifade) ground truth olarak kullanılmış. Bu, tamamen sentetik veriyle eğitilen modellere kıyasla **gerçekçi detay dağılımı** öğrenme potansiyeli sağlar.

### 2.3 16-bit Displacement Map Çıktısı
- Displacement map **16-bit grayscale PNG** olarak kaydedilir (`facialDetails.py` satır 280-282):
  ```python
  displacementMap = (displacementMap+1)/2*65535
  Image.fromarray(displacementMap.astype('uint16')).save(save_path)
  ```
- 65.536 ton gri → **merdivenlenme/teraslanma artefaktı oluşmaz**. Bu, 8-bit kullanan birçok rakibinden üstündür.
- **Building Block değerlendirmesi:** Bu displacement map, kendi başına değerli bir **hammadde (A3 bileşeni)**. Geometriye uygulanmak üzere extract edilebilir.

### 2.4 Modüler Pipeline Mimarisi
5 bağımsız modül: Landmark Detection → Expression Prior → Proxy Fitting → Isomap Rendering → DFDN Detail Synthesis. Her modül ayrı ayrı test/değiştirilebilir.

### 2.5 PCA + Refine CNN Hibrit Mimarisi
DFDN sadece naive pix2pix değil:
- **UNet-8** → **FC (2048→1024→512→64)** → **PCA (64 dim) rekonstrüksiyon** → concat with input → **RefineCNN (UNet-6, Tanh)**
- PCA bottleneck, anatomik olarak olası displacement'ların alt-uzayında kalmasını sağlar. Bu, keyfi artefaktları azaltan akıllı bir tasarım.

### 2.6 Bölgesel İşleme
İki ayrı model: `forehead_1031` (alın, 1547 patch) ve `mouse_1031` (ağız/çene, 1390 patch). Toplam 2937 patch, ağırlıklı ortalama ile 4096×4096 canvas'a stitch edilir. Weight mask (256×256, 5473 benzersiz değer) yumuşak geçiş sağlar.

### 2.7 MIT Lisansı
Ticari kullanıma uygun, modifiye ve dağıtım özgürlüğü sağlar.

---

## 3. Eksiklikler ve Zayıf Noktalar

### 3.1 KRİTİK: Displacement → Geometri Uygulaması YOK

**Bu, detaylı mesh perspektifinden en büyük eksikliktir.**

Paper'ın Denklem 11'i açıkça belirtir:

$$P_{fine}(u,v) = P_{proxy}(u,v) + G(u,v) \cdot N_{proxy}(u,v)$$

Ancak **bu adım ne `released/` ne de `src/` kodlarında mevcut değildir**. Pipeline şu noktada durur:

```
facialDetails.py → displacement PNG kaydedilir → normal map PNG türetilir → BİTTİ
```

- `hmrenderer.exe` shader kaynak kodu incelendi:
  - `render_hm.vs`: Standart MVP transformasyonu. **Vertex displacement YOK.**
    ```glsl
    gl_Position = projection * view * model * vec4(pos, 1.0);
    ```
  - `render_hm.frag`: %100 normal mapping — tangent-space normal ile Phong aydınlatma pertürbasyonu. **Displacement örneklemesi YOK.**
  - `render_bssrdf.frag`: Subsurface scattering denemesi — yine displacement yok.

- `OBJRender.vs` (isomap renderer): UV'den pozisyon hesaplar — displacement ile ilgisi yok.

**Sonuç:** Çıktı mesh (`result.obj`) her zaman **53.149 vertex'li düz, pürüzsüz proxy mesh** olarak kalır. Kırışıklıklar, gözenekler **hiçbir zaman fiziksel geometri olarak mevcut değildir**. README'deki teaser görselleri **normal mapping render hilesi** ile üretilmiştir.

| Kriter | Durum |
|---|---|
| A1 (Detaylı ham mesh çıktısı) | ❌ Yok — proxy hep 53K vertex, düz |
| A2 (Displacement → vertex uygulama) | ❌ Yok — ne kod ne script |
| A3 (Displacement map hammadde) | ⚠️ Mevcut — 16-bit PNG, extract edilebilir |
| A4 (Normal map hammadde) | ⚠️ Mevcut — 8-bit RGB, displacement'tan türetilmiş |

### 3.2 Düşük Çözünürlüklü Proxy Mesh
- **53.149 vertex / 105.694 face** — Bu, BFM2017'nin sabit topolojisi.
- Displacement'ı mesh'e uygulamak istesek bile bu vertex sayısı **yetersiz**. 4096×4096 displacement map'teki detayı yakalamak için **subdivision (en az 3-4 seviye → ~500K-800K+ vertex)** gerekir.
- Pipeline'da subdivision adımı yok.

### 3.3 Displacement Map Kalitesi — Orta Seviye

Ölçümlerimize göre (8 test görüntüsü):

| Metrik | UV'li Mod | Değerlendirme |
|---|---|---|
| Bit Derinliği | 16-bit ✅ | Yeterli |
| Dinamik Aralık | %75 (49240/65535) | Orta — ideal %90+ |
| Standart Sapma | 3122 (%4.8) | Düşük varyasyon |
| Eğitim Verisi | 366 scan / 122 kişi | **Çok az** — modern standartlarda 1000+ kişi beklenir |
| DFDN Patch Boyutu | 256×256 | Sınırlayıcı — modern yöntemler 512+ kullanır |
| DFDN Mimarisi | UNet-8 + PCA(64) + UNet-6 | 2019 seviyesi — transformer/diffusion yok |
| Coverage | Sadece alın + ağız | Burun, yanak, göz çevresi **modellenmez** |

**Sınırlamalar:**
- DFDN sadece 2 bölge (forehead + mouse) işler → yüz yüzeyinin ~%60'ı coverage dışında
- 366 tarama yetersiz diversity → genel/topluluk detayları öğrenilir, kişiye özgü incelikler zayıf
- PCA 64 boyut → agresif bilgi sıkıştırması

### 3.4 Ciddi Paketleme Hatası — Release'de UV Eksik

Release paketi **yanlış convert betiği** (`convert-bfm2017-to-eos-v016.py` — upstream eos betiği) içerir. Bu betik UV koordinatı **eklemez**. Doğru betik fork'ta (`LansburyCH/eos-expression-aware-proxy`), ama release'e dahil edilmemiştir.

Sonuç:
- `.bin` modelde UV yok → OBJ'de `vt=0` → isomap tamamen siyah → DFDN siyah girdiden anlamsız çıktı üretir
- Kullanıcılar bunu fark etmeden "çalışıyor" sanır (pipeline hata vermez, sadece sonuçlar kötü)
- `texture_uv.mat` dosyası da release'de yok
- GitHub Issue #35'te aynı sorun rapor edilmiş

**Biz bu sorunu çözdük:** Fork'tan `texture_uv.mat` indirildi, convert betiğine UV desteği eklendi (`--no-uv` flag ile toggle), .bin yeniden üretildi. Detaylar: [ANALIZ_UV_SORUNU.md](ANALIZ_UV_SORUNU.md)

### 3.5 Kimlik Sadakati — Dolaylı ve Sınırlı

| Kriter | Durum |
|---|---|
| Kişiye özgü proxy | ✅ BFM2017 shape coefficients → genel yüz oranları korunur |
| Kişiye özgü displacement | ⚠️ Isomap'ten türetilir ama 366 scan'lik PCA alt-uzayı ile sınırlı |
| Kimlik benzerlik metrikleri | ❌ Paper'da LMD/RMSE var ama FaceNet/ArcFace benzerlik yok |
| İfade çeşitliliği | ⚠️ 3 ifade (nötr, gülümseme, ?) — sınırlı |
| Ben, çil, yara gibi özellikler | ❌ DFDN bunları öğrenmez (sadece geometrik kırışıklık deseni) |

BFM2017 kendi başına kimlik korumada iyi: 199 shape + expression PCA bileşeni → makul yüz oranları. Ancak DFDN'in displacement çıktısı kişiye özgü ince detayları yakalamakta **zayıf** (düşük training data diversity + agresif PCA sıkıştırma).

### 3.6 Kod Tabanı Sorunları

| Sorun | Detay |
|---|---|
| **Platform bağımlılığı** | Windows 10 only — tüm C++ exe'ler (fit-model.exe, FaceLandmarkImg.exe, faceClip.exe, hmrenderer.exe) |
| **Eski PyTorch** | Python 3.7 + PyTorch (CUDA 10.0 DLL'leri mevcut) — güncel GPU'larla uyumsuz |
| **Keras emotion (opsiyonel)** | TF1/Keras API — deprecated |
| **Hardcoded yollar** | `fit_model()` içinde .bin yolu hardcoded |
| **hmrenderer.exe** | ACCESS_VIOLATION (OpenGL crash) — test sistemimizde çalışmıyor |
| **Derleme kılavuzu** | README: "on the way ....." — 5+ yıldır gelmedi |
| **OneDrive linkleri** | Model ağırlıkları OneDrive'da — kalıcılığı şüpheli |
| **Kod içi yorum** | Minimal — DFDN models/ dosyalarında neredeyse yorum yok |
| **`calNormalMap()` scale** | `0.07` hardcoded — neden bu değer, dokümantasyon yok |
| **Eğitim kodu** | `src/DFDN/` eğitim scriptleri mevcut ama veri seti paylaşılmıyor |

### 3.7 Çıktı Formatı Sınırlamaları

| Çıktı | Format | Kalite | Fiziksel Geometri? |
|---|---|---|---|
| `result.obj` | OBJ (vt=53149 UV fix sonrası) | Pürüzsüz proxy | ❌ Displacement uygulanmamış |
| `result.isomap.png` | 8-bit RGB, 4096² | İyi doku | N/A (texture) |
| `result.displacementmap.png` | 16-bit grayscale, 4096² | Orta detay | ❌ Dosya olarak kalır |
| `result.normalmap.png` | 8-bit RGB, 4096² | displacement'tan türetilmiş | ❌ Render hilesi |
| STL/PLY export | ❌ Yok | — | — |

---

## 4. İyileştirme Önerileri

### 4.1 Kısa Vadeli (Hemen Uygulanabilir)

#### A. Displacement → Geometry Uygulama Script'i
**En kritik iyileştirme.** Pipeline'a eklenecek bir Python script'i:

```python
# Kavramsal pipeline:
# 1. result.obj yükle (trimesh/Open3D)
# 2. Loop subdivision (3-4 seviye → ~800K vertex)
# 3. Subdivide edilmiş mesh'in UV'lerinden displacement map'i örnekle
# 4. Her vertex'e: new_pos = pos + displacement * vertex_normal
# 5. Detaylı mesh'i OBJ/STL olarak kaydet
```

Alternatif: **Blender headless** ile `Subdivision Surface` + `Displace Modifier` → `Apply All` → export.

Bu tek ekleme, pipeline'ı **A3 hammadde → A1+A2 fiziksel geometri** seviyesine çıkarır.

#### B. Yüz Kaplama Alanı Genişletme
Mevcut `areas.mat` sadece forehead (1547 patch) + mouse (1390 patch) tanımlar. Yanak, burun, göz çevresi için yeni bölgeler eklenerek tüm yüz kaplanabilir. Bu, eğitim gerektirmeden `areas.mat` düzenleme + inference kodu değişikliği ile yapılabilir **ancak** bu bölgeler için eğitilmiş ağırlıklar yoktur.

#### C. Normal Map Format İyileştirmesi
Normal map şu anda 8-bit RGB. 16-bit'e çıkarılması (EXR/TIFF) aydınlatma kalitesini artırır:
```python
# Mevcut (8-bit):
normalMap = (normalMap+1)/2*255
Image.fromarray(normalMap.astype('uint8')).save(save_path)

# İyileştirilmiş (16-bit):
normalMap = (normalMap+1)/2*65535
Image.fromarray(normalMap.astype('uint16')).save(save_path)
```

#### D. UV Fix'in Upstream'e Bildirilmesi
Bizim uyguladığımız UV fix'in (`texture_uv.mat` + güncellenmiş convert betiği) GitHub issue veya PR olarak paylaşılması.

### 4.2 Orta Vadeli (Haftalık Çaba)

#### E. DFDN'i Güncel Backbone ile Değiştirme
UNet-8 + PCA(64) → **StyleGAN2/Diffusion tabanlı detail generator** ile değiştirme:
- **Daha yüksek patch çözünürlüğü:** 256→512 veya 1024
- **Daha geniş PCA/latent uzay:** 64→256+ veya tamamen latent diffusion
- **Eğitim verisi artırımı:** FaceScape (938 kişi, 20 ifade) veya REALY benchmark

#### F. Multi-View Fusion Desteği
Mevcut pipeline tek fotoğraf girişi alır. Birden fazla açıdan alınan görüntülerin fusion'ı:
- Displacement map'lerin farklı açılardan tahmin edilip güvenilirlik ağırlıklı birleştirilmesi
- Cross-view consistency check

#### G. Alternatif 3DMM/Shape Model
BFM2017 → **FLAME** (5023 vertex, ifade + poz ayrıştırması daha iyi) veya **FaceVerse** geçişi. FLAME'in UV layout'u displacement application için daha uygun.

### 4.3 Uzun Vadeli (Yeni Yaklaşım)

#### H. Neural Implicit / SDF Tabanlı Geçiş
Tüm parametrik 3DMM + 2D map pipeline'ını terk edip doğrudan **implicit surface** çıktısı üreten modern yaklaşımlara geçiş:
- **NeuS/VolSDF** tabanlı yüz rekonstrüksiyonu → doğrudan yüksek çözünürlüklü mesh
- **Difüzyon tabanlı 3D yüz** → vertex düzeyinde detay üretimi
- **3D Gaussian Splatting → mesh extraction** (SuGaR, 2DGS) → multi-view ile

#### I. End-to-End Detaylı Mesh Pipeline
Proxy + displacement iki aşamalı yerine, tek geçişte detaylı mesh üreten modern mimariler:
- Point-cloud diffusion + meshing
- Differentiable rendering tabanlı optimizasyon
- Mesh deformation network (vertex düzeyinde)

---

## 5. Sonuç ve Alternatifler

### Genel Değerlendirme

| Kriter | Puan (1-5) | Açıklama |
|---|---|---|
| Geometrik Fidelity | ⭐⭐ (2/5) | Displacement var ama geometriye uygulanmıyor. Proxy 53K vertex, düz. |
| Kimlik Sadakati | ⭐⭐⭐ (3/5) | BFM2017 ile makul shape, ama displacement kişiye özgü detayda zayıf |
| Displacement Hammadde | ⭐⭐⭐ (3/5) | 16-bit, %75 dinamik aralık — kullanılabilir ama mükemmel değil |
| Kod Kalitesi | ⭐⭐ (2/5) | Paketleme hatası, eski bağımlılıklar, platform kısıtlaması |
| Güncellik | ⭐ (1/5) | 2019 yayın, 2020 son güncelleme, aktif bakım yok |
| Detaylı Mesh Uygunluğu | ⭐⭐ (2/5) | Doğrudan kullanılamaz, displacement baking eklenirse hammadde sağlayıcı |

### Detaylı Mesh Projesi İçin Yeri

Bu repo, Detaylı mesh projesinin ihtiyacı olan **"tek fotoğraftan detaylı mesh"** hedefini **mevcut haliyle karşılayamaz**. Ancak:

1. **Hammadde sağlayıcı olarak değerlendirilebilir:** 16-bit displacement map extract edip, Blender/trimesh ile mesh'e uygulama (displacement application) pipeline'ı eklenirse fiziksel geometri elde edilebilir.
2. **Kavramsal referans:** "Proxy mesh + displacement detail" iki aşamalı yaklaşım, proje pipeline'ında da benzer şekilde kullanılabilir.
3. **Yetersiz displacement kalitesi:** 366 scan'lik eğitim verisi, 256×256 patch boyutu ve sadece 2 bölge coverage'ı → modern alternatiflere kıyasla zayıf.

### Önerilen Alternatif Projeler

| Proje | Yıl | Avantaj | Dezavantaj |
|---|---|---|---|
| **HRN** (younglbw/HRN) | 2023 | Zaten test ediyoruz; BFM tabanlı (Deep3DFaceRecon_pytorch base); deformation (mid-freq) + displacement (high-freq) iki aşamalı pipeline; single-view + multi-view desteği; REALY benchmark top-1; 04/2023'te mesh'e displacement export modu eklendi | Displacement→geometry export modu var ancak çok fazla artifact üretiyor; README: "exported mesh with high frequency details may not be as ideal as the rendered 2D image"; Linux-only |
| **DECA** (yfeng95/DECA) | 2021 | FLAME tabanlı, UV detail map, coarse+detail ayrıştırma | Displacement geometriye otomatik uygulanmıyor; FLAME 5K vertex |
| **3DDFA-V3** (wang-zidu/3DDFA-V3) | 2024 | Güncel, lightweight, hızlı | Detay seviyesi düşük (parametrik 3DMM odaklı) |
| **FaceScape** (zhuhao-nju/facescape) | 2020-2023 | 938 kişi × 20 ifade, yüksek kalite mesh + displacement | Multi-view girdi gerektirir; veri seti erişimi kısıtlı |
| **ToFu** (tiangexiang/ToFu) | 2022 | Transformer tabanlı, yüksek çözünürlüklü mesh | Eğitim verisi/model ağırlıkları paylaşılmamış |
| **REALY Benchmark** | 2023 | Değerlendirme standardı | Model değil, benchmark |
| **Implicit/SDF tabanlı** (NeuFace, HeadNeRF vb.) | 2023-2025+ | Doğrudan yüksek çözünürlüklü implicit geometry | Multi-view gerekebilir; mesh extraction ek adım |

### Detaylı Mesh İçin En Uygun Strateji

```
Kısa Vade: HRN'in mevcut displacement→geometry export modunu test et (artifact analizi)
           → DECA çıktılarına da displacement application script'i ekle
           → Proxy mesh + displacement map → subdivide → displace → STL
           
Orta Vade: FaceScape veya benzeri yüksek kalite veri ile fine-tune
           → Daha iyi displacement map kalitesi
           
Uzun Vade: Modern SDF/diffusion tabanlı end-to-end detaylı mesh pipeline
           → Doğrudan vertex-level detay
```

---

## 6. Referanslar

### Paper ve Repo
- Chen, A., Chen, Z., Zhang, G., et al. "Photo-Realistic Facial Details Synthesis From Single Image." ICCV 2019.
- GitHub: [apchenstu/Facial_Details_Synthesis](https://github.com/apchenstu/Facial_Details_Synthesis)
- EOS Fork: [LansburyCH/eos-expression-aware-proxy](https://github.com/LansburyCH/eos-expression-aware-proxy)

### Karşılaştırma Referansları
- Feng, Y., et al. "Learning an Animatable Detailed 3D Face Model from In-The-Wild Images." (DECA) SIGGRAPH 2021.
- Lei, B., et al. "A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images." (HRN) CVPR 2023.
- Wang, Z., et al. "3DDFA-V3: A 3D Morphable Model Fitting Library." 2024.
- Yang, H., et al. "FaceScape: A Large-scale High Quality 3D Face Dataset." CVPR 2020.
- Chai, Z., et al. "REALY: Rethinking the Evaluation of 3D Face Reconstruction." ECCV 2022.
- Xiang, T., et al. "ToFu: Topology-Free 3D Face Reconstruction." NeurIPS 2022 (Workshop).

### Bizim Analizlerimiz
- [PIPELINE_ANALIZ.md](PIPELINE_ANALIZ.md) — Tam pipeline akışı, DFDN mimarisi, shader analizi
- [ANALIZ_UV_SORUNU.md](ANALIZ_UV_SORUNU.md) — UV/isomap sorunu analizi ve çözüm

---

## Ek: Kritik Kod Referansları

| Dosya | Satır | İçerik |
|---|---|---|
| `released/facialDetails.py` | 280-282 | Displacement map 16-bit kayıt: `(displacementMap+1)/2*65535` → `uint16` |
| `released/facialDetails.py` | 150-163 | `calNormalMap()`: 11×11 kernel, FFT convolution, scale=0.07 |
| `released/facialDetails.py` | 289-292 | Normal map 8-bit kayıt: `(normalMap+1)/2*255` → `uint8` |
| `released/facialDetails.py` | 296 | `hmrenderer.exe` çağrısı — sadece normal map ile render |
| `released/DFDN/models/pix2pix_model.py` | 143-171 | Forward pass: netG → FC → PCA recon → concat → refineCNN |
| `released/DFDN/models/networks.py` | 397-430 | FC_layers: Linear(2048→1024→512→64) |
| `released/DFDN/options/base_options.py` | 12,22,42 | inSize=256, unet_256, dimention=64 |
| `released/DFDN/options/testsingle_options.py` | 50-51 | imageW=4096, pacthW=256 |
| `released/faceRender/shaders/render_hm.vs` | 20 | `gl_Position = projection * view * model * vec4(pos, 1.0)` — NO displacement |
| `released/faceRender/shaders/render_hm.frag` | 34-37 | Normal map sampling + TBN perturbation — pure normal mapping |
| `released/renderTexture/OBJRender.vs` | 20 | `gl_Position = vec4(in_TexCoord.x*2-1, ...)` — UV→position unfold |
