# Western Sahel Convective Core Nowcasting Model  
### LSTM Encoder → CNN Decoder (Object-to-Map Architecture)

This model predicts future convective core probabilities over the **Western Sahel** using an **LSTM encoder** and **CNN decoder**, trained separately for **1-hour**, **3-hour**, and **6-hour** lead times.

---

## Input Specification

Each convective core is described using **14 features**:

**Core-level (10):**  
`lat`, `lon`, `lat_min`, `lat_max`, `lon_min`, `lon_max`, `size`, `intensity`, `lag`, `mask`

**Time features (4):**  
`month_sin`, `month_cos`, `tod_sin`, `tod_cos`

At each of the **5 time steps** (`t₀−2h`, `−1.5h`, `−1h`, `−0.5h`, `t₀`):

- **50 cores** (padded if fewer available)  
- **14 features per core**

Final input tensor: **(5, 50, 14)** → flattened per timestep → **(5, 700)**

---

## Model Architecture

### 1. LSTM Encoder
- Input size: 700  
- Sequence length: 5  
- Hidden size: 512–1024  
- Layers: 1–2  
- Dropout enabled  
- Final hidden state used as latent representation

### 2. Latent Projection
- Fully connected layer  
- Reshaped to a **32 × 32** latent map  

### 3. CNN Decoder
- Bilinear upsampling + convolutional blocks  
- Outputs a **512 × 512** probability map over the Western Sahel  

Separate model per lead time:
- **Model_1h** → `(1, 512, 512)`  
- **Model_3h** → `(1, 512, 512)`  
- **Model_6h** → `(1, 512, 512)`

---

## Loss Function

A hybrid loss is used:

1. **Binary Cross-Entropy (BCE)**  
   - Uses `pos_weight` to address class imbalance  
2. **Multi-scale Fraction Skill Score (FSS)**  
   - Encourages spatial coherence  
   - Improves medium-range performance

---

## Regularisation

- Dropout (encoder and projection layers)  
- Early stopping on validation loss or validation FSS  

---

## Data Selection

Only **JJAS** months (June to September) are used to align with the West African monsoon.

---

## Temporal Split

- **Training:** 2004–2019  
- **Validation:** 2020  
- **Test:** 2021–2024  

This avoids temporal leakage and ensures robust monsoon-season evaluation.

---

## Summary

- **Input:** `(5 timesteps × 50 cores × 14 features)`  
- **Model:** `LSTM encoder → 32×32 latent map → CNN decoder → 512×512 output`  
- **Output:** separate probability models for +1h, +3h, +6h  
- **Loss:** BCE (pos_weight) + multi-scale FSS  
- **Season:** JJAS  
- **Data Range:** 2004–2024  

This provides an efficient and physically meaningful Western Sahel convective core nowcasting framework tailored to regional storm structure.
