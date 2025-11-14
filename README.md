# Western Sahel Convective Core Nowcasting Model  
### LSTM Encoder → CNN Decoder (Object-to-Map Architecture)

This model predicts future convective core probabilities over the **Western Sahel** using an **LSTM encoder** and **CNN decoder**, trained separately for **1-hour**, **3-hour**, and **6-hour** lead times.

---

## Input Specification

Each convective core is represented using **14 features**:

**Core-level (10):**  
`lat`, `lon`, `lat_min`, `lat_max`, `lon_min`, `lon_max`, `size`, `intensity`, `lag`, `mask`

**Time features (4):**  
`month_sin`, `month_cos`, `tod_sin`, `tod_cos`

At each of the **5 time steps** (`t₀−2h`, `−1.5h`, `−1h`, `−0.5h`, `t₀`):

- **100 cores** (padded if fewer available)
- **14 features per core**

Final input tensor: (5, 100, 14) → flattened per timestep → (5, 1400)



---

## Model Architecture

**1. LSTM Encoder**  
- Input size: 1400  
- Sequence length: 5  
- Hidden size: 512–1024  
- Layers: 1–2  
- Dropout enabled  
- Final hidden state used as latent representation  

**2. Latent Projection**  
- Fully connected layer  
- Reshape to a **32 × 32** latent map  

**3. CNN Decoder**  
- Bilinear upsampling + convolutional blocks  
- Outputs a **512 × 512** probability map over Western Sahel  

**Separate model per lead time:**  
- Model_1h → `(1, 512, 512)`  
- Model_3h → `(1, 512, 512)`  
- Model_6h → `(1, 512, 512)`

---

## Loss Function

A hybrid loss is used:

1. **Binary Cross-Entropy (BCE)**  
   - `pos_weight` applied to handle class imbalance  
2. **Multi-scale Fraction Skill Score (FSS)**  
   - Encourages spatial coherence  
   - Improves performance at long lead times  

---

## Regularisation

- Dropout (encoder + projection layers)  
- Early stopping on validation loss or validation FSS  

---

## Data Selection

Only **JJAS** months are used (June, July, August, September) to focus on the West African monsoon.

---

## Temporal Split

- **Training:** 2004–2019  
- **Validation:** 2020  
- **Test:** 2021–2024  

This avoids temporal leakage and ensures robust monsoon-season evaluation.

---

## Summary

**Input:** `(5 timesteps × 100 cores × 14 features)`  
**Model:** `LSTM encoder → latent 32×32 → CNN decoder → 512×512 map`  
**Output:** separate probability models for +1h, +3h, +6h  
**Loss:** BCE (pos_weight) + multi-scale FSS  
**Season:** JJAS  
**Data Range:** 2004–2024  

This forms a coherent, physically meaningful Western Sahel convective core nowcasting framework.

