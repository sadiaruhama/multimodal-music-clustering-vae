<h1 align="center">🎧 Multi-Modal Music Clustering using VAEs</h1>
<p align="center">
Bridging Sound and Semantics through Hybrid Audio–Text Representation Learning
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-blue">
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red">
  <img src="https://img.shields.io/badge/Status-Research-green">
  <img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

---

## 📌 Overview

This project explores **unsupervised music clustering** using **Variational Autoencoders (VAEs)** across **audio and text modalities**.

The goal is to understand:

- How well models can learn **latent representations**
- Whether clusters reflect **true musical semantics (genre)**
- How **multimodal fusion (audio + lyrics)** affects performance

---

## 🎯 Key Idea

> 💡 **Better clustering structure does NOT guarantee correct semantic grouping**

---

## 🚀 Contributions

- 🎵 Hybrid **audio + lyrics clustering framework**
- 🧠 Multi-task evaluation:
  - Easy → Basic VAE
  - Medium → Conv-VAE + Multimodal Fusion
  - Hard → Beta-VAE, CVAE, Autoencoder
- 📊 Multi-scale dataset design
- 🔬 Extensive comparison across models and features

---

## 📂 Dataset Strategy

| Dataset | Samples | Purpose |
|--------|--------|--------|
| Hybrid (Audio + Projected Text) | 9,202 | Scalability testing |
| Aligned Multimodal | 837 | Main evaluation |
| Paired Dataset | 337 | Controlled multimodal learning |

---

## 🧠 Models Used

### 🔹 Easy Task
- Fully Connected VAE  
- PCA baseline  

### 🔹 Medium Task
- Convolutional VAE (Conv-VAE)
- Audio + Text Fusion  

### 🔹 Hard Task
- Beta-VAE  
- Conditional VAE (CVAE)  
- Autoencoder  
- Lyrics-only VAE  
- PCA baselines  

---

## 📊 Evaluation Metrics

### Structure-Based Metrics
- Silhouette Score ↑  
- Davies–Bouldin Index ↓  
- Calinski–Harabasz Index ↑  

### Label-Based Metrics
- Adjusted Rand Index (ARI)  
- Normalized Mutual Information (NMI)  
- Cluster Purity  

---

## 📈 Key Results

| Model | Insight |
|------|--------|
| VAE | Better clustering than PCA |
| Conv-VAE | Strong separation but poor semantic alignment |
| Beta-VAE | Best geometric clusters |
| Autoencoder | Best label alignment |
| CVAE | Poor performance |
| Multimodal Fusion | Degrades performance due to noise |

---

## ⚠️ Important Findings

- ❌ Multimodal is NOT always better  
- ❌ High Silhouette ≠ correct clusters  
- ✅ Data quality > Model complexity  
- ✅ Modality alignment is critical  

---

## 🛠️ Tech Stack

- Python  
- PyTorch  
- Librosa  
- Scikit-learn  
- Sentence Transformers  
- Whisper (for pseudo-lyrics)  

---

## 🧪 Example Workflow

```python
# Encode data into latent space
z = encoder(x)

# Perform clustering
from sklearn.cluster import KMeans
clusters = KMeans(n_clusters=10).fit_predict(z)
