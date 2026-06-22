# RESNET_USADA
## Medical Plant Detection — Perbandingan ResNet50, MobileNetV3-Small, dan ViT-Base/16 di GPU CUDA

Penelitian identifikasi tanaman obat menggunakan dan membandingkan performa
3 arsitektur deep learning yang berbeda filosofi: CNN klasik (ResNet50),
CNN ringan (MobileNetV3-Small), dan Vision Transformer (ViT-Base/16) —
dengan analisis mendalam terhadap performa komputasi GPU CUDA.

---

## Spesies Tanaman (Fase 1 — 5 spesies data terbanyak)
| Label | Spesies                  | Jumlah Asli |
|-------|--------------------------|-------------|
| 0     | Symphytum_officinale     | 106         |
| 1     | Euchresta_horsfieldii    |  96         |
| 2     | Tabernaemontana_sp       |  89         |
| 3     | Zingiber_purpureum       |  82         |
| 4     | Erythrina_hypaphorus     |  81         |

---

## Model yang Dibandingkan

| Model | Tipe | Karakteristik |
|---|---|---|
| **ResNet50** | CNN (heavy) | Residual/skip connection, baseline kuat |
| **MobileNetV3-Small** | CNN (lightweight) | Depthwise separable conv, untuk efisiensi |
| **ViT-Base/16** | Transformer | Self-attention antar patch 16×16, non-konvolusi |

---

## Urutan Eksekusi

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Split data asli → train/val/test (dipakai BERSAMA oleh 3 model)
python split_dataset.py

# 3. Augmentasi data training (dipakai BERSAMA oleh 3 model)
python augmentation.py

# 4. Training SEMUA model secara otomatis berurutan
#    (ResNet50 → MobileNetV3-Small → ViT-Base/16)
python train.py

# 5. Evaluasi SEMUA model + tabel perbandingan akhir
python evaluate.py

# 6. Prediksi satu gambar
python inference.py path/ke/gambar.jpg                   # default: resnet50
python inference.py path/ke/gambar.jpg --model mobilenet
python inference.py path/ke/gambar.jpg --model vit
python inference.py path/ke/gambar.jpg --model all        # bandingkan ketiganya
```

---

## Mengubah Daftar Model

Edit `MODEL_LIST` di `config.py`:
```python
MODEL_LIST = ["resnet50", "mobilenet", "vit"]   # urutan training otomatis
```
Untuk uji coba cepat satu model saja:
```python
MODEL_LIST = ["mobilenet"]
```

---

## Metrik yang Diukur (untuk setiap model)

| Metrik | Keterangan |
|--------|------------|
| Accuracy & F1-Score | Kualitas prediksi |
| Training Time (s/epoch) | Kecepatan belajar di GPU |
| Inference Time (ms/gambar) | Kecepatan prediksi real-time (batch=1, adil antar model) |
| GPU Memory Usage (MB) | Efisiensi VRAM |
| FLOPs | Jumlah operasi komputasi |
| Parameters (juta) | Ukuran model |

---

## Hyperparameter per Model

Setiap model punya `batch_size` dan `learning_rate` sendiri (lihat `MODEL_HYPERPARAMS` di `config.py`),
karena ViT lebih sensitif terhadap learning rate besar dan butuh batch lebih kecil (VRAM lebih besar per sampel),
sementara MobileNet lebih toleran dan ringan.

| Model | Batch Size | Learning Rate |
|---|---|---|
| ResNet50 | 16 | 0.0001 |
| MobileNetV3-Small | 32 | 0.0005 |
| ViT-Base/16 | 8 | 0.00003 |

---

## Gradual Unfreeze (semua model)

Untuk dataset terbatas, backbone setiap model di-freeze total di awal,
lalu dibuka bertahap:
- Epoch 11 → buka blok terdalam ke-1
- Epoch 21 → buka blok ke-2
- Epoch 31 → buka blok ke-3

Jadwal layer spesifik per arsitektur ada di `UNFREEZE_SCHEDULE` (model.py).

---

## Struktur Output

```
outputs/
├── resnet50/
│   ├── checkpoints/best_model.pth
│   ├── metrics/train_result.json, gpu_metrics.json, classification_report.txt
│   └── plots/accuracy_curve.png, loss_curve.png, gpu_usage.png, confusion_matrix.png
│
├── mobilenet/
│   └── (struktur sama)
│
├── vit/
│   └── (struktur sama)
│
└── comparison/
    ├── training_comparison.json       ← ringkasan training 3 model
    ├── evaluation_comparison.json     ← ringkasan evaluasi 3 model
    └── model_comparison_chart.png     ← bar chart 4 metrik utama
```
