# Facial Details Synthesis — Pipeline Analizi & Teaser Karşılaştırması

**Paper:** "Photo-Realistic Facial Details Synthesis from Single Image"  
**Yazarlar:** Anpei Chen, Zhang Chen, Guli Zhang, Ziheng Zhang, Kenny Mitchell, Jingyi Yu  
**Venue:** ICCV 2019, pp. 9429–9439  
**arXiv:** [arxiv.org/abs/1903.10873](https://arxiv.org/abs/1903.10873)  
**Repo:** [apchenstu/Facial_Details_Synthesis](https://github.com/apchenstu/Facial_Details_Synthesis)  
**EOS Fork:** [LansburyCH/eos-expression-aware-proxy](https://github.com/LansburyCH/eos-expression-aware-proxy)  

---

## 0. Paper Özeti (ICCV 2019)

**Başlık:** "Photo-Realistic Facial Details Synthesis from Single Image"  
**Yazarlar:** Anpei Chen, Zhang Chen, Guli Zhang, Ziheng Zhang, Kenny Mitchell, Jingyi Yu  
**Kurum:** ShanghaiTech University + Edinburgh Napier University  
**Venue:** ICCV 2019, pp. 9429–9439 | [arXiv:1903.10873](https://arxiv.org/abs/1903.10873)

### Yöntemin Özü
Tek bir 2D yüz fotoğrafından, ifade bilgisini koruyarak ince geometrik detaylar (kırışıklıklar, gözenekler) içeren 3D yüz mesh'i üretme. İki ana yenilik:

1. **Emotion-driven proxy generation:** FACS + emotion prediction ile BFM2017 expression katsayılarını tahmin edip 3D proxy mesh fitting'de prior olarak kullanma → daha doğru ifade yakalama (özellikle yanak, nasolabial kıvrımlar)

2. **DFDN (Deep Facial Detail Net):** İki aşamalı CGAN — PDIM (PCA bazlı kaba displacement) + PDRM (refinement ile ince detay) — supervised (366+340 = 706 3D tarama) ve unsupervised (163K in-the-wild görüntü) eğitim kombinasyonu

### Paper'daki Kritik Formüller

**Proxy fitting (Denklem 4):**
$$E = \sum_k w_k \|L_k - P(l_k(\alpha, \beta))\|_2 + \lambda_s \|\alpha\|_2$$

**Displacement uygulama (Denklem 11):**
$$P_{fine}(u,v) = P_{proxy}(u,v) + G(u,v) \cdot N_{proxy}(u,v)$$

**Appearance loss (Denklem 14):**
$$I_{recon} = I_{albedo} \cdot S(N_{fine}), \quad L_{recon} = \|I_{input} - I_{recon}\|_1$$

### Çekirdek Bulgu: Paper vs Release Farkı
Paper'daki görseller (Figure 1, 7, 8) **displacement'ı geometriye uygulanmış** şekilde gösteriyor (Denklem 11). Ancak **release kodu bu adımı içermiyor** — displacement sadece PNG olarak kaydedilir, normal map türetimi yapılır ve `hmrenderer.exe` sadece normal mapping render yapar. OBJ hiçbir zaman modifiye edilmez.

**Kaynak koddan teyit edildi:** `src/` dizinindeki orijinal kaynak kodlar da (release'den farklı olsa bile) displacement→geometry uygulaması içermiyor. Detaylı analiz §5A'da.

### Capture Sistemi (Eğitim Verisi)
- 5 Cannon 760D DSLR + 9 polarize flaş
- 23 görüntü/tarama: 5 uniform aydınlatma (MVS) + 9 çift polarize (PS)
- Multi-view stereo + photometric stereo birleşimi
- Normal integration: $\min \int [(∇z - [p,q])^2 + \mu(z-z_0)^2] \, du \, dv$
- $\mu = 10^{-5}$ (detaylı) vs $\mu = 10^{-3}$ (detaysız) → ground truth displacement map

---

## 1. Tam Pipeline Akışı (6 Aşama)

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────────┐
│  1. LANDMARK │───▶│ 2. EXPRESSION│───▶│  3. PROXY MESH   │───▶│  4. ISOMAP   │───▶│ 5. DFDN  │───▶│ 6. RENDER    │
│  DETECTION   │    │    PRIOR     │    │    FITTING       │    │  EXTRACTION  │    │ DETAIL   │    │ VISUALIZATION│
│              │    │              │    │                  │    │              │    │ SYNTHESIS│    │              │
│ FaceLandmark │    │ find_nearest │    │ fit-model.exe    │    │ faceClip.exe │    │ PyTorch  │    │hmrenderer.exe│
│ Img.exe      │    │ (dict NPZ)   │    │ (EOS fork)       │    │ (OpenGL)     │    │ CGAN     │    │ (OpenGL)     │
└─────────────┘    └──────────────┘    └─────────────────┘    └──────────────┘    └──────────┘    └──────────────┘
     ↓                   ↓                    ↓                     ↓                  ↓                ↓
  .pts file          expression.txt      result.obj +          result.isomap     displacement    hmrenderer.exe
  .txt (FACS)        (BFM2017 PCA       result.affine_from      .png (4096²)     map + normal    ekranında 
  .box               coefficients)       _ortho.txt                               map (4096²)     görselleştirme
```

### Aşama 1: Landmark Detection
- **Modül:** `landmarkDetector/FaceLandmarkImg.exe` (basitleştirilmiş OpenFace)
- **Girdi:** 2D yüz fotoğrafı (JPG/PNG)
- **Çıktı:**
  - `.pts` — 68 noktalı iBUG landmark koordinatları (x,y)
  - `.txt` — FACS (Facial Action Coding System) Action Unit yoğunlukları
  - `.box` — yüz bounding box
- **Algoritma:** CLM/CE-CLM landmark dedektörü + FACS AU regresyonu

### Aşama 2: Expression Prior (İfade Öncülü) — Opsiyonel
- **Modül:** `find_nearest()` in `facialDetails.py`
- **Girdi:** FACS features (`.txt`) VEYA EmotionNet tahminleri
- **Çıktı:** `expression.txt` — BFM2017 ifade PCA katsayıları
- **Algoritma:** Önceden hesaplanmış sözlükte (`dictionary.npz`) L1 en yakın komşu arama. FACS -> BFM2017 ifade parametreleri dönüşümü.
- **İki mod:**
  - `--FAC 1`: FACS AU yoğunlukları → sözlük → BFM ifade parametreleri
  - `--emotion 1`: Keras EmotionNet → sözlük → BFM ifade parametreleri

### Aşama 3: Proxy Mesh Fitting (3D Mesh Üretimi)
- **Modül:** `proxyEstimator/fit-model.exe` (LansburyCH EOS fork)
- **Girdi:** Orijinal görüntü + .pts landmark + expression.txt + BFM2017 .bin model
- **Çıktı:**
  - `result.obj` — 53.149 vertex, 105.694 face, UV koordinatları (.bin'de varsa)
  - `result.affine_from_ortho.txt` — ortografik projeksiyon kamera matrisi
- **Algoritma:** İteratif 3DMM (3D Morphable Model) fitting:
  $$\mathbf{S} = \bar{\mathbf{S}} + \sum_{i=1}^{n_s} \alpha_i \cdot \mathbf{s}_i + \sum_{j=1}^{n_e} \beta_j \cdot \mathbf{e}_j$$
  - $\bar{\mathbf{S}}$: Ortalama yüz (mean shape)
  - $\mathbf{s}_i$: Kimlik PCA baz vektörleri (identity)
  - $\mathbf{e}_j$: İfade PCA baz vektörleri (expression)
  - $\alpha_i, \beta_j$: Optimize edilen katsayılar
  - 2D landmark'lara ortografik projeksiyon hatası minimize edilir
  - Her zaman BFM2017 topolojisinde (sabit 53K vertex, 105K face)

### Aşama 4: Isomap (UV Texture) Çıkarma
- **Modül:** `textureRender/faceClip.exe` (OpenGL)
- **Girdi:** result.obj + orijinal görüntü + kamera matrisi
- **Çıktı:** `result.isomap.png` — 4096×4096 RGB UV-space texture haritası
- **Algoritma:** OpenGL mod 3 "unfold" render:
  ```glsl
  // Vertex Shader (OBJRender.vs)
  gl_Position = vec4(in_TexCoord.x*2-1, 1-in_TexCoord.y*2, 0, 1);
  
  // Fragment Shader (OBJRender_unfold.frag)
  color.rgb = texture(rawImg, out_texcoord).rgb;
  ```
  Her vertex UV koordinatına göre konumlandırılır → orijinal görüntüden renk örneklenir → yüz dokusu UV uzayına açılır.
- **KRİTİK:** UV koordinatları (`vt`) OBJ'de **zorunlu**. UV yoksa tüm vertex'ler (0,0)'a çöker → siyah çıktı.

### Aşama 5: DFDN — Facial Detail Synthesis (Detay Sentezi)
- **Modül:** `DFDN/` — Deep Facial Detail Network (PyTorch, pix2pix tabanlı CGAN)
- **Girdi:** `result.isomap.png` (4096×4096'ya resize edilir → 256×256 patch'lere bölünür)
- **Çıktı:**
  - `result.displacementmap.png` — 16-bit grayscale displacement map (4096×4096)
  - `result.normalmap.png` — 8-bit RGB normal map (4096×4096, displacement'tan türetilir)

**İki aşamalı mimari (paper Section 4.1'den doğrulanmış):**

PDIM (Partial Detail Inference Module) — orta frekans detaylar (kırışıklıklar):
PDRM (Partial Detail Refinement Module) — yüksek frekans detaylar (gözenekler):

```
Input Patch (256×256, HSV V-channel)
        │
        ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────────┐
│   PDIM       │────▶│   FC Layers   │────▶│ PCA Recon.       │
│ (UNet-8)     │     │ 2048→1024→    │     │ 64 PCA baz       │
│ Feature Ext. │     │ 512→64 dim    │     │ vektörü ile      │
│ + ReLU       │     │ + ReLU (son   │     │ kaba displacement│
└──────────────┘     │ hariç)        │     └────────┬─────────┘
                     └───────────────┘              │ concat
                                                     ▼
                                            ┌──────────────────┐
                                            │ PDRM             │
                                            │ (UNet-6)         │
                                            │ 4×4 kernel,      │
                                            │ stride=2, pad=1  │
                                            │ LeakyReLU + tanh │
                                            │ [PCA_recon +     │
                                            │  input] → final  │
                                            └──────────────────┘
```

**Geometry Loss (paper Denklem 10):**
$$L_{scans} = \|PCA(PDIM(x,z)) - y\|_1 + \|PDRM(PCA(PDIM(x,z))) - y\|_1$$

**GAN Loss (paper Denklem 6-7):**
$$\min_G \max_D (L_{cGAN}(G,D) + \lambda L_{L1}(G)), \quad \lambda = 100$$

**Semi-supervised Loss (paper Denklem 8):**
$$L_{L1}(G) = L_{scans}(x,z,y) + \eta \cdot L_{recon}(x), \quad \eta = 0.5$$
3. **Normal Map Hesaplama:** `calNormalMap()` — displacement map'in gradyanları FFT konvolüsyon ile hesaplanır (paper Denklem 12-13 ile ilişkili):
   ```python
   # 11×11 Sobel-benzeri kernel ile FFT konvolüsyon
   GX = fftconvolve(displacement, kernel_x)  # yatay gradyan
   GY = fftconvolve(displacement, kernel_y)  # dikey gradyan
   normal = [GX*0.07, GY*0.07, 1.0]          # scale=0.07 sabit
   normal = normalize(normal)                  # birim uzunluk → RGB: (normal+1)/2*255
   ```

**Bölgesel İşleme (areas.mat analizi ile doğrulanmış):**
- **`forehead_1031`** modeli: Alın bölgesi — **1547 patch**, y=[128,2176] x=[128,3840]
- **`mouse_1031`** modeli: Ağız/alt yüz bölgesi — **1390 patch**, y=[1856,3776] x=[128,3712]
- **Toplam:** 2937 patch (4096×4096 canvas'a stitch edilir)
- **Weight mask:** 256×256, 5473 benzersiz ağırlık değeri (yumuşak patch birleştirme için)
- Patch boyutu: 256×256 (`args.pacthW=256`)
- Çıktı boyutu: 4096×4096 (`args.imageW=4096`)

**Displacement Map Bit Derinliği (doğrulanmış):**
- **16-bit grayscale PNG** — PNG header: `bit_depth=16, color_type=0`
- Değer dönüşümü: DFDN çıktısı `[-1,1]` → `(x+1)/2 × 65535` → `uint16`
- **65536 ton gri** → merdivenlenme/teraslanma artefaktı **YOK** (8-bit'te 256 ton → stair-stepping olurdu)
- PIL `mode=I` (int32) olarak okur ama dosya içinde 16-bit
- Normal map ise **8-bit RGB** (`uint8`, 256 ton per kanal)

**Eğitim verisi (paper'dan doğrulanmış):**
- **Geometri (supervised):** 706 yüksek hassasiyetli 3D tarama → 366 kendi çekim (122 kişi × 3 ifade) + 340 ICT-3DRFE [47]
- **Görünüm (unsupervised):** 163K vahşi doğada yüz görüntüsü (AffectNet dataset)
- **Eğitim şeması:** İlk 15 epoch sadece supervised geometry loss, sonra supervised/unsupervised alternating, toplam 250 epoch
- **Öğrenme hızı:** 0.0001 → 0 (100. epoch'tan itibaren lineer azalma)
- **GAN parametreleri:** λ=100 (CGAN ağırlığı), η=0.5 (reconstruction loss ağırlığı)
- **Patch örnekleme:** 10K supervised + 12K unsupervised, 50% overlap ile uniform sampling (inference'da)

### Aşama 6: Görselleştirme (Opsiyonel)
- **Modül:** `faceRender/hmrenderer.exe` (OpenGL + ImGui)
- **Girdi:** result.obj + result.normalmap.png + shader dizini
- **Çıktı:** İnteraktif OpenGL penceresi — Phong aydınlatma + normal map pertürbasyonu
- **NOT:** README: "The visualizer currently only supports mesh + normalMap"
  - Displacement map KULLANILMAZ — sadece normal map uygulanır
  - Tessellation/subdivision YOK — düşük çokgenli proxy mesh doğrudan render edilir

---

## 2. Her Modülün Görevi

| Modül | Görev | Girdi | Çıktı |
|---|---|---|---|
| `landmarkDetector/FaceLandmarkImg.exe` | 68 nokta yüz landmark'ı + FACS AU tespiti | 2D fotoğraf | .pts, .txt, .box |
| `find_nearest()` | FACS/emotion → BFM2017 ifade katsayıları | FACS features | expression.txt |
| `fit-model.exe` (EOS fork) | 3DMM fitting → 3D proxy mesh | görüntü + landmark + BFM2017 | result.obj + kamera matrisi |
| `faceClip.exe` | UV-space texture çıkarma (isomap) | OBJ + görüntü + kamera | result.isomap.png (4096²) |
| DFDN (PyTorch) | İnce detay displacement/normal tahmin | isomap → 256² patch'ler | displacement + normal map |
| `hmrenderer.exe` | Normal map ile Phong render | OBJ + normalmap | İnteraktif GL penceresi |

---

## 3. EOS ve BFM2017'nin Rolü

### EOS Nedir? (`patrikhuber/eos`)
Açık kaynak C++ kütüphanesi — **3D Morphable Model (3DMM) fitting**. Yetenekleri:
- `.bin` formatında morphable model yükleme
- Shape/expression/color PCA katsayılarını 2D landmark'lara fit etme
- OBJ mesh çıktısı + kamera matrisi üretme
- BFM2009, BFM2017, Surrey Face Model desteği

### LansburyCH Fork Neler Ekler?
Paper yazarının (Zhang Chen) fork'u şu ekleri içerir:
1. **Expression prior girdi:** `--init-expression-coeffs-fp` bayrağı — dosyadan expression katsayı başlangıç değeri
2. **Expression sabitleme:** `--fix-expression-coeffs` — OBJ fitting'de expression katsayılarını dondurma
3. **UV koordinatları:** `texture_uv.mat` — BFM2017 için özel UV parametrizasyonu (BFM2017'de native UV yoktur)
4. **Modifiye convert betiği:** UV'leri `.bin`'e gömme desteği

### BFM2017 (Basel Face Model 2017)
Basel Üniversitesi'nin istatistiksel 3D yüz modeli:
- **Topoloji:** 53.149 vertex, 105.694 üçgen — sabit topoloji, ağız içi yok
- **Shape PCA:** Kimlik varyasyonu (yüz şekli)
- **Expression PCA:** İfade varyasyonu (gülme, kaş çatma vb.)
- **Color PCA:** Cilt rengi/albedo varyasyonu
- **UV:** Orijinalde YOK → yazarlar `texture_uv.mat` olarak üretip fork'a eklemiş

### Proxy Mesh ↔ BFM2017 İlişkisi
Proxy mesh, BFM2017'nin **belirli bir örneği** (instance). Her zaman aynı topolojide (53K vertex, 105K face) — sadece vertex pozisyonları değişir. PCA katsayıları fit edilerek girdi yüzüne uygun şekil + ifade üretilir.

---

## 4. Isomap, Displacement Map, Normal Map

### Isomap (`result.isomap.png`)
- **Ne:** UV-space texture haritası — yüz dokusunun BFM2017 UV koordinat sistemine "açılmış" hali
- **Boyut:** 4096×4096 RGB
- **İçerik:** Giriş fotoğrafından örneklenmiş yüz renkleri, UV uzayında düzenlenmiş
- **Üretim:** OpenGL mod 3 — UV koordinatlarından vertex pozisyonu, kamera matrisi ile fotoğraftan renk örnekleme
- **Amaç:** DFDN'ye girdi sağlamak — baş pozisyonundan bağımsız kanonik UV uzayında görünüm bilgisi

### Displacement Map (`result.displacementmap.png`)
- **Ne:** Yükseklik haritası — her piksel yüzeyin normal yönünde ne kadar yükseltileceği/alçaltılacağı
- **Boyut:** 4096×4096, 16-bit grayscale
- **Değer aralığı:** 0–65535 (DFDN çıktısı [-1,1] → `(value+1)/2*65535` dönüşümü)
- **İçerik:** İnce geometrik detay — kırışıklıklar, gözenekler, alın çizgileri, burun kıvrımları
- **Koordinat:** UV space (isomap ile aynı düzen)
- **Kullanım imkanları:**
  1. **Render (sadece görsel):** Shader'da normal pertürbasyonu veya tessellation
  2. **Geometri modifikasyonu (gerçek mesh):** Her vertex'i normali yönünde öteleme → gerçek 3D detay

### Normal Map (`result.normalmap.png`)
- **Ne:** Her pikselin RGB'si yüzey normal yönünü kodlar (XYZ → RGB)
- **Boyut:** 4096×4096, 8-bit RGB
- **Üretim:** Displacement map'ten gradyan hesabı ile türetilir (`calNormalMap()`)
- **Amaç:** `hmrenderer.exe`'de normal mapping — yüzey normallerini fragment shader'da bozarak aydınlatma etkisi

### Üçlü İlişki:
```
isomap (görünüm) ──▶ DFDN ──▶ displacement map (yükseklik) ──▶ calNormalMap() ──▶ normal map
                                      │                                              │
                              [geometri modifikasyonu]                      [render trick]
                              (pipeline YAPMIYOR)                          (hmrenderer yapar)
```

---

## 5. KRİTİK SORU: Teaser Detayları MESH mi RENDER mı?

### Cevap: Paper'da GERÇEK GEOMETRİ gösteriliyor — ama RELEASE kodu bunu YAPMIYOR.

### Paper'ın Kendisi Ne Diyor?

Figure 1 açıklaması (paper sayfa 1, kelimesi kelimesine):
> *"From left to right: input face image; proxy 3D face, texture and displacement map produced by our framework; **detailed face geometry with estimated displacement map applied on the proxy 3D face**; and re-rendered facial image."*

Paper Denklem 11 — Displacement uygulaması:
$$P_{fine}(u,v) = P_{proxy}(u,v) + G(u,v) \cdot N_{proxy}(u,v)$$

Burada:
- $P_{proxy}$: Proxy mesh'in position map'i (UV space'de)
- $G(u,v)$: DFDN'nin ürettiği displacement değeri
- $N_{proxy}$: Proxy mesh'in normal'i
- $P_{fine}$: **Displacement uygulanmış detaylı pozisyon haritası**

Yani paper'ın formülasyonunda displacement **gerçek geometriye uygulanıyor** — her UV noktasında proxy pozisyonu + displacement × normal = fine pozisyon.

Paper Denklem 12-14 — Normal ve görünüm hesabı:
- $N_{fine} = F(P_{fine})$ — detaylı geometrinin normalleri (cross product ile)
- $I_{recon} = I_{albedo} \cdot S(N_{fine})$ — detaylı normallerle re-render
- $L_{recon} = ||I_{input} - I_{recon}||_1$ — unsupervised appearance loss

**Paper'daki teaser görselleri gerçek displaced geometry gösteriyor** — Figure 7'deki close-up'lar "synthesized meshes" olarak etiketleniyor ve mesh wireframe detayları görünüyor.

### Ama Release Kodu Bunu YAPMIYOR

#### A. Kod analizi — OBJ hiçbir zaman modifiye edilmiyor
`facialDetails.py` satır satır incelendiğinde:
1. `fit_model()` → `result.obj` üretir (düz BFM2017 proxy mesh, 53K vertex)
2. `render_texture()` → `result.isomap.png` üretir
3. `predict_details()` → NumPy array olarak displacement + normal döner
4. Bu array'ler PNG olarak kaydedilir
5. **OBJ dosyası bir daha açılmaz, düzenlenmez, kaydedilmez.** Displacement mesh'e uygulanmaz.

#### B. README açıkça belirtiyor
> "The visualizer currently only supports mesh + normalMap, but will also support displacementMap in the near future."  

Displacement desteği **gelecek çalışma** olarak planlanmış ancak **hiçbir zaman uygulanmamıştır**.

#### C. hmrenderer.exe kodunda tessellation yok
- Tessellation shader'lar kullanılmaz (`GL_TESS_CONTROL_SHADER`, `GL_TESS_EVALUATION_SHADER` yok)
- Subdivision yapılmaz
- Displacement map GPU'ya yüklenmez
- Sadece normal map fragment shader'da texture olarak örneklenir → aydınlatma pertürbasyonu

### 5A. Kaynak Kod İncelemesi (src/ vs released/) — Displacement Hiçbir Yerde Yok

`src/` dizinindeki orijinal kaynak kodlar detaylıca incelendi. Sonuç: **Ne `src/` ne de `released/` displacement→geometry uygulaması içeriyor.**

#### src/faceRender — hmrenderer kaynak kodu
- Git submodule → `gg-z/face_rendering` deposuna işaret eder
- **`render_hm.vs` (vertex shader):** Standart MVP transformasyonu. **Vertex displacement YOK.**
  ```glsl
  gl_Position = projection * view * model * vec4(aPos, 1.0);
  ```
- **`render_hm.frag` (fragment shader):** %100 normal mapping — tangent-space normal map'ten pertürbe edilmiş normal ile Phong aydınlatma. **Displacement örneklemesi YOK.**
- **`render_bssrdf.frag`:** Subsurface scattering denemesi — yine displacement yok.

#### src/renderTexture/faceClip — isomap renderer kaynak kodu
- C++ kaynak + GLSL shader'ları
- Mod 3 unfold: UV→pozisyon, görüntüden renk örnekleme (isomap üretimi)
- Displacement ile ilgisi yok

#### src/DFDN — eğitim kodu
- Eğitim script'leri ve veri yükleme kodu
- Displacement map'i **Python array olarak hesaplar** ama **mesh'e uygulamaz**

#### src/facialDetails.py vs released/facialDetails.py
- Küçük farklar: PIL vs cv2, textureRender vs faceClip, patch normalizasyonu
- Her ikisinde de **displacement→mesh uygulaması YOK**

**Sonuç:** Paper'ın Denklem 11 ($P_{fine} = P_{proxy} + G \cdot N_{proxy}$) hiçbir zaman kod olarak yayınlanmamıştır.

#### D. `hmrenderer.exe` çağrısı (facialDetails.py satır 293):
```python
cmd = 'hmrenderer.exe "%s" "%s" "%s"' % (
    save_obj_path + '.obj',       # düz proxy mesh (53K vertex)
    save_path,                     # result.normalmap.png
    args.face_render_path + '/shaders'
)
```
**Sadece normalmap geçiriliyor — displacement map geçirilMİYOR.**

### Release Kodu vs Paper Arasındaki Fark:

| Özellik | Paper (ICCV 2019) | Release Kodu |
|---|---|---|
| Displacement → geometri | ✅ Denklem 11 ile uygulanıyor | ❌ Sadece PNG olarak kaydedilir |
| Görselleştirme | Displaced mesh render | Normal map Phong render |
| Figure 1 görseli | Gerçek displaced geometry | hmrenderer = sadece normalmap |
| Figure 7 close-up | "Synthesized meshes" | Düz proxy + normalmap |
| Texture çözünürlüğü | 2048×2048 (paper) | 4096×4096 (release default) |
| Displacement uygulama | Position map tabanlı, UV space'de | Hiç uygulanmıyor |

### Düz 3D viewer'da result.obj nasıl görünür?
**Tamamen düz, detaysız, manken yüzü.** Burnun çıkıntısı, çene hattı gibi büyük özellikler var ama kırışıklık, gözenek gibi hiçbir ince detay YOK.

### Teaser'daki detay nasıl üretilmiş?
Paper'ın kendi render pipeline'ında (eğitim sırasında kullanılan, Denklem 11-14):
1. Proxy mesh'in UV-space position map'i üretilir ($P_{proxy}$)
2. DFDN displacement map'i ($G(u,v)$) ile çarpılarak pozisyon güncellenmesi yapılır
3. Yeni pozisyonlardan normal map hesaplanır ($N_{fine}$)
4. Albedo × SH lighting × fine normal ile re-render

**Bu position-map tabanlı yaklaşım, displacement'ı piksel seviyesinde uyguladığı için mesh'in vertex sayısına bağlı değil** — render çözünürlüğünde detay üretir.

Ancak release'deki `hmrenderer.exe` bunun yerine basit normal mapping yapıyor:
- Her pikselde yerel normal yönü RGB'den okunur
- Bu bozulmuş normal ile Phong aydınlatma hesaplanır
- Sonuç: kırışıklıklar, gözenekler gibi görünen gölge/ışık varyasyonları
- Mesh geometrisi değişmez → siluette detay görünmez → fiziksel mesh'te detay YOK

---

## 6. Displacement'ı Gerçek Geometriye Dönüştürme (Bu Pipeline Yapmıyor)

Eğer displacement map'i gerçek 3D geometriye uygulamak isterseniz (detaylı mesh üretimi için):

### Yöntem A: Per-vertex displacement (düşük çözünürlük)
```python
# Her vertex'i normalı boyunca öteleme
for v in mesh.vertices:
    uv = vertex_uv[v]
    disp = sample_displacement_map(uv)
    v.position += v.normal * disp * scale
```
**Sorun:** Sadece 53K vertex → büyük UV alanı → gözenek seviyesi detay MÜMKÜN DEĞİL.

### Yöntem B: Subdivision + displacement (yüksek çözünürlük, doğru yaklaşım)
```python
# Mesh'i subdivide et, sonra her yeni vertex'i ötele
mesh = subdivide(mesh, iterations=3)  # 53K → ~3.4M vertex
for v in subdivided_mesh.vertices:
    uv = interpolated_uv[v]
    disp = sample_displacement_map(uv)
    v.position += v.normal * disp * scale
```
**Bu Blender'da yapılabilir:** Subdivision Surface modifier + Displace modifier (UV tabanlı).

### Yöntem C: GPU tessellation (gerçek zamanlı)
OpenGL/Vulkan tessellation shader'ları ile runtime'da subdivide + displace. `hmrenderer.exe`'de planlanmış ama uygulanmamış.

---

## 7. Kalite Beklentileri — Gerçekçi Değerlendirme

### Displacement Map Kalite Analizi (Ölçümlerle Doğrulanmış)

| Metrik | UV'li Mod (isomap var) | UV'siz Mod (siyah isomap) |
|---|---|---|
| Bit derinliği | **16-bit** (65536 ton) | **16-bit** (65536 ton) |
| Dinamik aralık kullanımı | **%75** (49240/65535) | **%42** (27809/65535) |
| Standart sapma | 3122 (%4.8) | 593 (%0.9) |
| Aktif alan | %80.7 | %99.9 |
| Değer aralığı | min=7241, max=56481 | min=13958, max=41767 |
| Ortalama | ~32810 (median=32767) | ~33170 (median=33093) |
| Yüze özgülük | ✅ 8 farklı MD5 hash | ✅ 8 farklı MD5 hash |
| Dosya boyutu | ~18 MB | ~5.8 MB |

**Önemli:** UV'siz modda bile (siyah isomap girdisiyle) DFDN **farklı** displacement map'ler üretiyor (8/8 benzersiz hash). Ancak dinamik aralık ve standart sapma çok düşük → detay neredeyse düz.

### Pipeline Çıktılarının Kalitesi

| Bileşen | Kalite | Not |
|---|---|---|
| Proxy mesh (result.obj) | ✓ İyi | Doğru yüz şekli, 53K vertex, düz yüzey |
| Isomap (result.isomap.png) | ✓ İyi (UV fix sonrası) | Renkli yüz dokusu, 4096² çözünürlük |
| Displacement map (16-bit) | ⚠ Orta | Patch-bazlı (256²), sadece alın+ağız, 366 taramadan öğrenilmiş, %75 dinamik aralık |
| Normal map (8-bit) | ⚠ Orta | Displacement'tan sayısal türev, scale=0.07 sabit, 11×11 kernel |
| Detaylı mesh | ❌ Yok | Displacement mesh'e uygulanmıyor |

### Teaser Kalitesine Neden Ulaşılamayabilir?

| Kısıtlama | Doğası | Açıklama |
|---|---|---|
| Patch sınır artefaktları | **Yapısal** | 256×256 patch'lerin overlap bölgesinde ağırlıklı ortalama → detay kaybı |
| Bölge kapsama sınırlılığı | **Yapısal** | Sadece alın (forehead) ve ağız (mouse) bölgeleri → yanak, burun köprüsü, çene detaysız |
| Eğitim verisi | **Yapısal** | 366 tarama (122 kişi × 3 ifade) — sınırlı varyasyon |
| Çıktı çözünürlüğü | 4096² | Bu yeterli ancak patch'ler 256² ile sınırlı |
| Normal map scale | **Sabit 0.07** | Her görüntü için aynı — uyarlanabilir değil |
| Release vs Paper | **Farklı** | README Q&A: "Proxy result is different with showing in the paper" — farklı parametreler kullanılmış |

### "Displacement as Texture" vs "Displacement as Geometry"

| Özellik | Texture (mevcut pipeline) | Geometry (gerekli olan) |
|---|---|---|
| **Depolama** | PNG dosyası (OBJ'den ayrı) | Vertex pozisyonları OBJ/STL'de |
| **Vertex sayısı** | Değişmez (53K) | Subdivision gerekir (milyonlarca) |
| **Görsel kalite** | Özel renderer'da mükemmel | Her yerde mükemmel |
| **Siluet** | Düz (konturda detay yok) | Detaylı (profilde kırışıklıklar görünür) |
| **Detaylı mesh** | İşe yaramaz (mesh düz) | Uygun |
| **Dosya boyutu** | Küçük OBJ + büyük PNG | Çok büyük OBJ/STL |
| **Yazılım desteği** | Normal map okuyan renderer gerekir | Evrensel — her viewer gösterir |

---

## 8. Convert Betiği: Güncel Durum ve Karşılaştırma

### Mevcut `convert-bfm2017-to-eos-v016.py` — `--no-uv` Flag Desteği

Orijinal fork betiğine minimum değişiklikle `--no-uv` argüman desteği eklendi:

```python
# Eklenen: import argparse + 3 satır argparse + if/else sarma
parser = argparse.ArgumentParser(description='BFM2017 -> eos .bin converter')
parser.add_argument('--no-uv', action='store_true', help='UV koordinatsiz .bin olustur')
args = parser.parse_args()
```

**Kullanım:**
```bash
# UV'li .bin üret (varsayılan) — pipeline isomap + detaylı displacement üretir
python convert-bfm2017-to-eos-v016.py

# UV'siz .bin üret — pipeline siyah isomap + düz displacement üretir
python convert-bfm2017-to-eos-v016.py --no-uv
```

**İki .bin dosya sistemi:**

| Dosya | Boyut | İçerik | Açıklama |
|---|---|---|---|
| `bfm2017-1_bfm_nomouth.bin` | 308.36 / 309.17 MB | Aktif model | Pipeline bu ismi kullanır. İçeriği son çalıştırmaya bağlı. |
| `bfm2017-1_bfm_nomouth_uv.bin` | 309.17 MB | UV'li kalıcı kopya | UV'li modda otomatik kaydedilir. Hızlı geri dönüş için. |

### Bizim Versiyon vs Fork Karşılaştırması

| Özellik | Fork (LansburyCH) | Bizim Versiyon | Fark |
|---|---|---|---|
| `scipy.io` import | ✅ | ✅ | Aynı |
| H5 dosya yolu | `../../bfm2017/...` | `./...` | Sadece yol (doğru) |
| `texture_uv.mat` yolu | `../../bfm2017/...` | `./...` | Sadece yol (doğru) |
| V-flip | `texture_uv[:, 1] = 1 - texture_uv[:, 1]` | Aynı | Aynı |
| `.tolist()` | ✅ | ✅ | Aynı |
| `MorphableModel(...)` | `texture_coordinates = texture_uv` | `texture_coordinates=texture_uv` | Aynı (whitespace) |
| `--no-uv` desteği | ❌ | ✅ (argparse) | UV'siz mod eklendi |
| `_uv.bin` yedek | ❌ | ✅ (otomatik kayıt) | Hızlı mod değişimi |
| Assertion'lar | ❌ | ✅ (3 adet) | Ekstra güvenlik |
| Doğrulama adımı | ❌ | ✅ (load + check) | Ekstra güvenlik |

**Sonuç: UV dalı fonksiyonel olarak fork ile aynı.** `--no-uv` dalı orijinal release davranışını (UV'siz) korur.

---

## 9. Sonraki Adım — Displacement'ı Mesh'e Uygulama (Gelecek Görev)

UV fix sonrasında pipeline düzgün çalışıyor. Ancak çıktılar:
- `result.obj`: Düz proxy mesh (kırışıklıksız)
- `result.displacementmap.png`: Detay bilgisi (sadece PNG olarak)
- `result.normalmap.png`: Render efekti (sadece PNG olarak)

**Detaylı mesh üretmek için:**
1. Blender'da `result.obj` yükle
2. Subdivision Surface modifier (2-4 seviye)
3. Displace modifier → `result.displacementmap.png` texture olarak
4. UV koordinatlarını seç
5. Displacement gücünü kalibre et (16-bit değerler → uygun ölçek)
6. Modifier'ları uygula → yüksek-poly STL/OBJ olarak dışa aktar

**Alternatif:** Python ile trimesh/PyVista kullanarak programatik subdivision + displacement.

---

## 10. Çıktı Karşılaştırması: UV Fix Öncesi vs Sonrası

| Metrik | UV Fix Öncesi (NO_UV) | UV Fix Sonrası | İyileşme |
|---|---|---|---|
| OBJ vt count | 0 | 53.149 | ✅ UV mevcut |
| Isomap nonzero | %0.0 (tamamen siyah) | %77.8-77.9 | ✅ Renkli doku |
| Isomap boyut | ~56 KB | 7.4-14.9 MB | ✅ Gerçek veri |
| Isomap ortalama piksel | 0.0 | 61.9-133.4 | ✅ Her yüz farklı |
| Isomap MD5 hash | Hepsi aynı | 8 farklı hash | ✅ Yüze özgü |
| Displacement ortalama | ~33170.7 (hep aynı) | ~32700-32825 (farklı) | ✅ Yüze özgü |
| Displacement boyut | ~5.8 MB | ~18 MB | ✅ Daha zengin veri |
| Displacement MD5 | Neredeyse aynı | 8 farklı hash | ✅ Yüze özgü |
| Normal map MD5 | — | 8 farklı hash | ✅ Yüze özgü |

---

## 11. No-UV Deney Sonuçları (8 Görüntü, `samples/details`)

`--no-uv` modunda 8 görüntü (`samples/details`) üzerinde pipeline çalıştırıldı. Amaç: UV'siz (orijinal release davranışı) vs UV'li çıktıları karşılaştırmak.

### Sonuçlar:

| Görüntü | OBJ (vt) | Isomap | Displacement (mean, size) | Normal Map |
|---|---|---|---|---|
| 013578 | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |
| 017133 | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |
| 019615 | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |
| 025652 | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |
| 0_0362 | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |
| D3CMF1 | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |
| DCWTDP | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |
| E2P3NP | vt=0 | %0.0 (siyah) | mean=33170.7, 5792KB | mean=169.4, 2323KB |

### Bulgular:
- **8/8 displacement map benzersiz MD5 hash'e sahip** — siyah isomap'e rağmen DFDN farklı çıktılar üretiyor
- **Ancak istatistikler neredeyse aynı** (mean≈33170.7, std≈593) → detay minimal, dinamik aralık %42
- **UV'li modda aynı görüntüler** → mean≈32810, std≈3122, dinamik aralık %75 → **5.3× daha fazla detay varyasyonu**
- **Sonuç:** UV'siz mod pipelinee'ı çalıştırır ama anlamlı displacement üretmez. UV **zorunludur**.

---

## Referanslar

- **Paper:** Chen, A., Chen, Z., Zhang, G., Zhang, Z., Mitchell, K., & Yu, J. (2019). Photo-Realistic Facial Details Synthesis from Single Image. ICCV 2019.
- **EOS:** Huber, P. et al. (2016). A Multiresolution 3D Morphable Face Model and Fitting Framework. VISAPP.
- **BFM2017:** Gerig, T. et al. (2018). Morphable Face Models - An Open Framework. IEEE FG 2018.
- **pix2pix:** Isola, P. et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017.
