# Released Paketi Kurulum & İndirme Rehberi

> **Tarih:** 3 Mart 2026  
> **Repo:** [0xHamza/_Facial_Details_Synthesis](https://github.com/0xHamza/_Facial_Details_Synthesis) (fork of [apchenstu/Facial_Details_Synthesis](https://github.com/apchenstu/Facial_Details_Synthesis))  
> **Not:** Bu fork'a sadece küçük kod dosyaları (~8.6 MB) push edilmiştir. Büyük model/binary dosyalar (~5.5 GB) git'te yoktur ve aşağıdaki kaynaklardan indirilmelidir.

---

## Genel Durum

| Klasör | Boyut | Git'te? | İndirme Kaynağı |
|--------|------:|:-------:|-----------------|
| `DFDN/` (kod dosyaları) | ~0.5 MB | ✅ | — (repo'da mevcut) |
| `DFDN/checkpoints/` | 3,488 MB | ❌ | Google Drive / OneDrive |
| `proxy/` | 1,013 MB | ❌ | Release v0.1.0 + BFM2017 |
| `landmarks/` | 953 MB | ❌ | Release v0.1.0 + Google Drive |
| `renderTexture/` | 86 MB | ❌ | Release v0.1.0 |
| `emotion/` | 28 MB | ❌ | Google Drive / OneDrive |
| `faceRender/` (shaderlar) | 0.7 MB | ✅ | — (repo'da mevcut) |
| `samples/` | 8 MB | ✅ | — (repo'da mevcut) |
| `facialDetails.py` | 11 KB | ✅ | — (repo'da mevcut) |
| `proxyPredictor.py` | 8 KB | ✅ | — (repo'da mevcut) |

---

## 1. Released Paket (Ana Paket)

Released v0.1.0 paketi, aşağıdaki linklerden tek seferde indirilebilir. Bu paket `landmarks/`, `proxy/`, `renderTexture/`, `faceRender/`, `DFDN/` (kod), `samples/` ve script dosyalarını içerir.

**İndirme:**
- [Google Drive](https://drive.google.com/file/d/1n1gB4bb9TOiFgp8IqfscqFOS3LHzKUIN/view?usp=sharing)
- [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMc5auQCu-G4uKQDo?e=8C9378)
- [GitHub Release](https://github.com/apchenstu/Facial_Details_Synthesis/releases/tag/V0.1.0) (sadece kaynak kod, released paket yok)

> **ÖNEMLİ:** GitHub Release sayfasında sadece kaynak kodun zip/tar.gz dosyaları var. Asıl `released/` paketi (exe, dll, model dosyalarıyla birlikte) **Google Drive veya OneDrive**'dan indirilmelidir.

**İçerik:**
```
released/
├── DFDN/           # Kod + boş checkpoints klasörü
├── emotion/        # EmotionNet (opsiyonel)
├── faceRender/     # OpenGL renderer
├── landmarks/      # OpenFace landmark detector
├── proxy/          # EOS proxy estimator
├── renderTexture/  # Isomap texture renderer
├── samples/        # Test resimleri
├── facialDetails.py
└── proxyPredictor.py
```

---

## 2. DFDN Checkpoints (3.5 GB) — AYRI İNDİRME GEREKLİ

DFDN model ağırlıkları released pakete dahil **DEĞİLDİR**. Ayrıca indirilmelidir.

**İndirme:**
- [Google Drive](https://drive.google.com/file/d/1taK985IJr3m15HG1S7k70bvI-SuusYom/view?usp=sharing)
- [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMeVDOvgXWwZJG57I?e=K1tMIV)

**Hedef:** `released/DFDN/checkpoints/` altına çıkartılacak

**Dosya yapısı (indirdikten sonra):**
```
released/DFDN/checkpoints/
├── checkpoints/
│   ├── forehead_1031/
│   │   ├── latest_net_FC.pth      (522 MB)
│   │   ├── latest_net_G.pth       (208 MB)
│   │   └── latest_net_REFINE.pth  (112 MB)
│   └── mouse_1031/
│       ├── latest_net_FC.pth      (522 MB)
│       ├── latest_net_G.pth       (208 MB)
│       └── latest_net_REFINE.pth  (112 MB)
├── forehead_1031/                  (aynı dosyalar — duplikat)
├── mouse_1031/                     (aynı dosyalar — duplikat)
└── PCA/
    ├── forehead/project.mat        (31 MB)
    └── mouse/project.mat           (31 MB)
```

> **Not:** İçeride duplikat yapı var (checkpoints/checkpoints/). Orijinal paket böyle gelmiştir.  
> **3 model tipi:**  
> - `FC` (Fully Connected): PCA bottleneck — displacement alt-uzay projeksiyonu  
> - `G` (Generator): UNet-8 ana ağ  
> - `REFINE`: UNet-6 ince detay iyileştirme  
> **2 bölge:** `forehead_1031` (alın) ve `mouse_1031` (ağız/çene)

---

## 3. Landmark Modelleri (953 MB) — AYRI İNDİRME GEREKLİ

Landmarks detector modeli ve kütüphaneleri ayrıca indirilmelidir.

**İndirme:**
- [Google Drive](https://drive.google.com/file/d/1rNNkXf372XvtBNiMu4kJe27p9v7nRKgX/view?usp=sharing)
- [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMeJG_0W5UOwTPIM4?e=MSA8BM)

**Hedef:** `released/landmarks/` (released paket ile geleni üzerine yazacak/tamamlayacak)

**Büyük dosyalar:**
| Dosya | Boyut | Açıklama |
|-------|------:|----------|
| `model/patch_experts/cen_patches_0.50_of.dat` | 147 MB | OpenFace CE-CLM patch uzmanı |
| `model/patch_experts/cen_patches_1.00_of.dat` | 147 MB | OpenFace CE-CLM patch uzmanı |
| `dictionary.npz` | 78 MB | FACS→BFM ifade sözlüğü |
| `opencv_world410.dll` | 70 MB | OpenCV runtime |
| `model/patch_experts/cen_patches_0.25_of.dat` | 58 MB | OpenFace CE-CLM patch uzmanı |
| `model/patch_experts/cen_patches_0.35_of.dat` | 58 MB | OpenFace CE-CLM patch uzmanı |
| `lib/Debug/dlib.lib` | 54+40 MB | dlib kütüphanesi (Debug) |
| `exe/x64/openblas.dll` | 26 MB | OpenBLAS matematik kütüphanesi |
| `exe/LandmarkDetector.lib` | 38 MB | Landmark detector kütüphanesi |

---

## 4. BFM2017 Modeli (proxy/ içinde) — MANUEL İNDİRME

Proxy estimator BFM2017 (Basel Face Model 2017) kullanır. Model dosyası akademik lisans gerektirir.

**İndirme:**
1. [BFM2017 resmi sayfası](https://faces.dmi.unibas.ch/bfm/bfm2017.html) — kayıt gerekli
2. `model2017-1_bfm_nomouth.h5` dosyasını indir (310 MB)
3. `released/proxy/bfm2017/` altına koy
4. `python convert-bfm2017-to-eos.py` ile `.bin` dosyaları üret

**Proxy klasörü büyük dosyaları:**
| Dosya | Boyut | Açıklama |
|-------|------:|----------|
| `model2017-1_bfm_nomouth.h5` | 310 MB | BFM2017 orijinal HDF5 model |
| `bfm2017-1_bfm_nomouth_uv.bin` | 309 MB | EOS formatına dönüştürülmüş (UV'li) |
| `bfm2017-1_bfm_nomouth.bin` | 308 MB | EOS formatına dönüştürülmüş (UV'siz) |
| `bfm2017-1_bfm_nomouth_edge_topology.json` | 30 MB | Kenar topolojisi |
| OpenCV DLL'leri | 55 MB | opencv_core343.dll + opencv_imgproc343.dll |
| `fit-model.exe` | 0.6 MB | Proxy fitting çalıştırılabilir dosyası |

> **Not:** `.bin` ve `.json` dosyaları `model2017-1_bfm_nomouth.h5`'ten `convert-bfm2017-to-eos.py` scripti ile üretilmiştir. Orijinal yayından bu yana bu dosyalar released pakete dahil geliyordu.

---

## 5. EmotionNet Checkpoints (28 MB) — OPSİYONEL

EmotionNet, expression prior için opsiyonel alternatiftir (`--emotion 1` flag'i ile).

**İndirme:**
- [Google Drive](https://drive.google.com/file/d/1pTz0wqJLwa_QQxxownu5KsifocDLBF-6/view?usp=sharing)
- [OneDrive](https://1drv.ms/u/s!Ard0t_p4QWIMdwW7nEx0pV0w56I?e=rQta0m)

**Hedef:** `released/emotion/` (veya `released/emotionNet/checkpoints/`)

**Dosyalar:**
| Dosya | Boyut | Açıklama |
|-------|------:|----------|
| `large.h5` | 28 MB | Keras emotion model ağırlıkları |
| `large.json` | ~1 KB | Model mimarisi JSON |
| `fertrainImgLarge.py` | ~1 KB | Eğitim scripti |

---

## 6. renderTexture (86 MB) — Released Pakette Gelir

İsomap UV texture çıkarma modülü.

**Kaynak:** Released v0.1.0 paketine dahil

**Büyük dosyalar:**
| Dosya | Boyut | Açıklama |
|-------|------:|----------|
| `opencv_world330.dll` | 86 MB | OpenCV 3.30 runtime |
| `faceClip.exe` | 0.1 MB | Isomap çıkarma çalıştırılabilir |
| Shader dosyaları | ~5 KB | OBJRender*.frag/*.vs |

---

## Hızlı Kurulum Özeti

```bash
# 1. Released paketi indir ve çıkart
#    Google Drive veya OneDrive'dan -> released/ klasörüne

# 2. DFDN checkpoints indir
#    -> released/DFDN/checkpoints/ altına çıkart

# 3. Landmark modelleri indir  
#    -> released/landmarks/ altına çıkart (mevcut dosyaları tamamlar)

# 4. [Opsiyonel] EmotionNet indir
#    -> released/emotion/ altına çıkart

# 5. BFM2017 model dosyası (proxy için zaten mevcut; yoksa)
#    faces.dmi.unibas.ch/bfm/bfm2017.html'den indir
#    -> released/proxy/bfm2017/ altına koy

# 6. Conda environment
conda create -n facial_details python=3.7 pytorch torchvision -c pytorch
conda activate facial_details
pip install opencv-python pillow scipy
pip install eos-py==0.16.1  # proxy estimator için

# 7. Test çalıştırma
cd released/
python facialDetails.py -i ./samples/details/019615.jpg -o ./results
python proxyPredictor.py -i ./samples/proxy -o ./results
```

---

## İndirme Linkleri Özet Tablosu

| Bileşen | Google Drive | OneDrive | Boyut |
|---------|:---:|:---:|------:|
| **Released v0.1.0 paketi** | [Link](https://drive.google.com/file/d/1n1gB4bb9TOiFgp8IqfscqFOS3LHzKUIN/view?usp=sharing) | [Link](https://1drv.ms/u/s!Ard0t_p4QWIMc5auQCu-G4uKQDo?e=8C9378) | ~100 MB (sıkıştırılmış) |
| **DFDN checkpoints** | [Link](https://drive.google.com/file/d/1taK985IJr3m15HG1S7k70bvI-SuusYom/view?usp=sharing) | [Link](https://1drv.ms/u/s!Ard0t_p4QWIMeVDOvgXWwZJG57I?e=K1tMIV) | ~1.7 GB (sıkıştırılmış) |
| **Landmark modelleri** | [Link](https://drive.google.com/file/d/1rNNkXf372XvtBNiMu4kJe27p9v7nRKgX/view?usp=sharing) | [Link](https://1drv.ms/u/s!Ard0t_p4QWIMeJG_0W5UOwTPIM4?e=MSA8BM) | ~500 MB (sıkıştırılmış) |
| **EmotionNet** (opsiyonel) | [Link](https://drive.google.com/file/d/1pTz0wqJLwa_QQxxownu5KsifocDLBF-6/view?usp=sharing) | [Link](https://1drv.ms/u/s!Ard0t_p4QWIMdwW7nEx0pV0w56I?e=rQta0m) | ~28 MB |
| **BFM2017** (akademik) | — | — | 310 MB |
| **Paper PDF** | [arXiv](https://arxiv.org/abs/1903.10873) | — | — |

> **⚠️ Link Durumu (Mart 2026):** Bu linkler 2019'dan beri aktif. Linkler zaman içinde erişilemez hale gelebilir. Bu durumda orijinal repo'nun [README](https://github.com/apchenstu/Facial_Details_Synthesis#released-version) ve [Issues](https://github.com/apchenstu/Facial_Details_Synthesis/issues) sayfalarını kontrol edin.
