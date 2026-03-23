# MSTT: Multivariate Spatio-Temporal Transformer for High-Resolution Drought Forecasting

## 📌 Overview

This repository provides the implementation of a **Multivariate Spatio-Temporal Transformer (MSTT)** model for high-resolution drought forecasting using Earth Observation (EO) data.

The model is designed to predict the **Standardized Precipitation Index (SPI)** at 1 km spatial resolution by jointly modeling spatial and temporal dependencies from multivariate EO signals.

---

## 🌍 Study Area

United Arab Emirates (UAE) — a hyper-arid region characterized by:

* low precipitation
* high evapotranspiration
* strong spatial heterogeneity

Time period: **2000–2023**

---

## 📊 Input Data

The model integrates the following EO variables:

* **NDVI** (MODIS MOD13A3) → vegetation health
* **Land Surface Temperature (LST)** (MODIS MOD11A1)
* **Soil Moisture** (GLDAS NOAH)
* **SPI (target)** derived from CHIRPS precipitation

All data are:

* harmonized to **1 km resolution**
* aggregated at **monthly scale**

---

## 🧠 Model Architecture

The MSTT model is based on a **dual-attention mechanism**:

### 🔹 Spatial Attention

* captures dependencies between grid cells
* models geographic interactions

### 🔹 Temporal Attention

* captures monthly temporal dynamics
* models short-term and seasonal dependencies

### 🔹 Key Features

* multivariate input fusion
* transformer-based architecture
* scalable to large EO datasets
* interpretable via attention mechanisms

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/MSTT-Drought-Forecasting-UAE.git
cd MSTT-Drought-Forecasting-UAE
pip install -r requirements.txt
```

---

## ▶️ Quick Test (MANDATORY FOR CAGEO)

```bash
python src/train.py
```

This script:

* generates synthetic EO data
* trains the MSTT model
* outputs performance metrics

---

## 📁 Repository Structure

```text
.
├── README.md
├── requirements.txt
├── LICENSE
│
├── src/
│   ├── train.py          # training script
│   ├── model.py          # MSTT architecture
│   ├── utils.py
│
├── data/
│   ├── sample_data.csv
│   └── README_data.md
│
├── notebooks/
│   └── example_run.ipynb
│
├── results/
│   └── example_output.png
│
└── config/
    └── config.yaml
```

---

## 🔁 Reproducibility

To reproduce experiments:

1. Prepare EO datasets (NDVI, LST, SM)
2. Align to 1 km / monthly resolution
3. Update configuration file
4. Run:

```bash
python src/train.py
```

---

## 📈 Results

The MSTT model achieves:

* **R² ≈ 0.93**
* Lower RMSE and MAE compared to:

  * LSTM
  * GRU
  * Informer
  * EarthFormer
  * DLinear

---

## 📜 License

MIT License

---

## 📩 Contact

Ali Ben Abbes
Manouba School of Engineering
University of Manouba, Tunisia

Email: [ali.benabbes@mse.uma.tn](mailto:ali.benabbes@mse.uma.tn)
