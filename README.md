# 🧠 EfficientNet-GAT-Fuzzy Cervical Cytology Classification

A research-based deep learning project for **five-class cervical cell classification** using an **EfficientNet-B0 Region-Graph Attention Network with a Fuzzy Confidence Head**.

The project explores whether combining convolutional feature extraction, region-based graph modeling, Graph Attention Networks (GAT), and fuzzy confidence learning can improve cervical cytology image classification while using a single compact CNN backbone.

---

## 📌 Project Overview

Automated cervical cytology classification is a challenging computer vision problem because different cervical cell categories can share similar visual characteristics such as:

- Nuclear shape
- Cytoplasmic appearance
- Texture
- Staining patterns
- Cell boundaries
- Background characteristics

This research proposes a model based on:

- **EfficientNet-B0** for deep visual feature extraction
- A **five-node region graph**
- **Graph Attention Networks (GAT)**
- A **Fuzzy Confidence and Separation Head**
- **Five-fold stratified cross-validation**
- **Grad-CAM based interpretability**
- Multiple controlled **ablation experiments**

---

## 🏗️ Proposed Architecture

The complete model follows the pipeline:

```text
Input Image
    ↓
Image Preprocessing
    ↓
EfficientNet-B0
    ↓
1280 × 7 × 7 Feature Map
    ↓
1 × 1 Convolution
    ↓
160 × 7 × 7 Feature Map
    ↓
Region Graph Construction
    ↓
4 Local Region Nodes + 1 Global Node
    ↓
Graph Attention Layer 1
    ↓
Graph Attention Layer 2
    ↓
Graph Mean Pooling
    ↓
MLP Classifier
    ↓
Softmax Probabilities
    ↓
Fuzzy Confidence / Separation Scores
    ↓
Five-Class Prediction
```

The final graph contains **five nodes**:

- Top-Left region
- Top-Right region
- Bottom-Left region
- Bottom-Right region
- Global image-context node

The graph allows local image regions to exchange information with one another and with the overall image representation.

---

## 🧩 Model Components

### 🔹 EfficientNet-B0

An ImageNet-pretrained EfficientNet-B0 is used as the feature extraction backbone.

The original classification layer is removed, producing:

```text
1280 × 7 × 7
```

feature maps.

A learnable `1 × 1` convolution reduces these features to:

```text
160 × 7 × 7
```

before constructing the graph.

---

### 🔹 Region Graph

The feature map is divided into four spatial regions:

```text
Top Left       Top Right

Bottom Left    Bottom Right
```

A fifth node represents the complete global feature map.

Adaptive average pooling converts every region into a **160-dimensional feature vector**.

The resulting node matrix has the shape:

```text
5 × 160
```

---

### 🔹 Graph Attention Network

Two Graph Attention layers are applied to the region graph.

Each GAT layer learns how much attention should be assigned to neighboring nodes, allowing the model to combine:

- Local visual information
- Global image context
- Relationships between different image regions

After the GAT layers, graph mean pooling produces a single graph representation.

---

### 🔹 MLP Classifier

The pooled graph representation is passed through an MLP:

```text
160 → 128 → 5
```

to generate logits for the five cervical cell classes.

---

### 🔹 Fuzzy Confidence Head

The model also includes a fuzzy confidence mechanism.

The fuzzy score considers:

- Predicted class probability
- Difference between the two highest class probabilities
- Trainable global confidence and separation parameters

A fuzzy reward term is also included in the training objective.

---

# 📊 Dataset

The experiments use the **SIPaKMeD cervical cytology dataset**.

The dataset contains **4,049 isolated cervical cell images** belonging to five classes.

| Class | Images |
|---|---:|
| Dyskeratotic | 813 |
| Koilocytotic | 825 |
| Metaplastic | 793 |
| Parabasal | 787 |
| Superficial-Intermediate | 831 |
| **Total** | **4,049** |

### ⚠️ Dataset Availability

The dataset is **not included in this repository**.

Users must obtain the SIPaKMeD dataset separately and update the dataset path inside the notebooks before execution.

---

# 🧪 Experimental Setup

The experiments were conducted using **stratified five-fold cross-validation**.

Main training configuration:

| Setting | Value |
|---|---|
| Input Size | 224 × 224 |
| Cross Validation | 5-Fold Stratified |
| Random Seed | 42 |
| Batch Size | 16 |
| Maximum Epochs | 14 |
| Backbone Learning Rate | 8 × 10⁻⁵ |
| New-Layer Learning Rate | 4 × 10⁻⁴ |
| Optimizer | AdamW |
| Weight Decay | 1 × 10⁻⁴ |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | 4 Epochs |
| Label Smoothing | 0.05 |
| GAT Dropout | 0.20 |
| MLP Dropout | 0.35 |
| Fuzzy Reward Coefficient | 0.10 |
| Mixed Precision | Enabled with CUDA |

---

# 📈 Full Model Results

The proposed model achieved the following average five-fold performance:

| Metric | Result |
|---|---:|
| Accuracy | **97.46%** |
| Macro Precision | **97.48%** |
| Macro Recall | **97.47%** |
| Macro F1 Score | **97.47%** |

Across all out-of-fold predictions:

```text
Total Images        : 4,049
Correct Predictions : 3,946
Incorrect Predictions: 103
```

---

## 📊 Fold-Level Performance

| Fold | Best Epoch | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Fold 1 | 5 | 95.93% | 95.97% | 95.94% | 95.94% |
| Fold 2 | 8 | 98.52% | 98.52% | 98.53% | 98.53% |
| Fold 3 | 11 | 97.41% | 97.44% | 97.41% | 97.42% |
| Fold 4 | 8 | 98.15% | 98.17% | 98.16% | 98.16% |
| Fold 5 | 6 | 97.28% | 97.31% | 97.30% | 97.29% |
| **Mean** | **7.60** | **97.46%** | **97.48%** | **97.47%** | **97.47%** |

---

# 🔬 Ablation Study

Several controlled experiments were performed to study the contribution of individual model components.

| Model Variant | Parameters | Mean Accuracy | Macro F1 |
|---|---:|---:|---:|
| Full Model | 4,286,243 | 97.4561% | 97.4676% |
| One GAT Layer | 4,260,003 | 97.4315% | 97.4405% |
| No GAT Layers | 4,233,763 | 97.4561% | 97.4623% |
| No Global Node | 4,286,243 | 97.4314% | 97.4403% |
| No Fuzzy Reward | 4,286,243 | 97.3573% | 97.3699% |
| **EfficientNet-B0 Only** | **4,013,953** | **97.6291%** | **97.6305%** |

### 🔎 Ablation Findings

The experiments showed that:

- Removing one GAT layer caused only a very small performance change.
- Removing both GAT layers produced almost the same mean accuracy as the full model.
- Removing the global node also produced only a small change.
- Removing the fuzzy reward slightly reduced performance.
- The **EfficientNet-B0-only model achieved the highest mean accuracy** among the tested variants.
- EfficientNet-B0 therefore contributed the majority of the predictive performance.
- The additional graph and fuzzy components did not produce a clear mean-accuracy improvement under the current evaluation protocol.

These findings are useful because the ablation study evaluates not only whether the complete architecture performs well, but also whether its additional complexity is justified.

---

# 🧮 Softmax vs Fuzzy Decision Analysis

An analytical experiment was also performed to compare:

```text
Softmax Argmax
```

with:

```text
Fuzzy Score Argmax
```

The implemented fuzzy transformation preserves the class ordering when the shared fuzzy scaling parameters are positive.

A numerical verification using **1,000,000 randomly generated five-class probability vectors** produced:

```text
Agreement     : 100%
Disagreements : 0
```

Therefore, the post-training fuzzy transformation itself does not change the predicted class ordering under these conditions.

Its observable contribution primarily comes from the **fuzzy reward used during training**.

---

# 🔍 Class-Wise Performance

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dyskeratotic | 97.05% | 97.05% | 97.05% |
| Koilocytotic | 95.22% | 94.18% | 94.70% |
| Metaplastic | 95.91% | 97.60% | 96.75% |
| Parabasal | 100.00% | 99.11% | 99.55% |
| Superficial-Intermediate | 99.16% | 99.40% | 99.28% |

The **Koilocytotic** class was the most difficult category for the model.

The most frequent misclassifications occurred between:

```text
Koilocytotic → Metaplastic
Dyskeratotic → Koilocytotic
Koilocytotic → Dyskeratotic
Metaplastic → Koilocytotic
```

---

# 🔥 Grad-CAM Interpretability

Grad-CAM was used to examine which image regions influenced model predictions.

For many correctly classified images, the model focused on:

- Central cellular structures
- Nuclear regions
- Important cell morphology

However, some incorrect predictions showed attention toward:

- Background regions
- Image borders
- Broad cell outlines

These observations indicate that Grad-CAM is useful for inspecting model behavior, although it should not be interpreted as proof of clinical reasoning.

---

# 📁 Repository Structure

```text
EfficientNet-GAT-Fuzzy-Cervical-Cytology/
│
├── .gitignore
├── requirements.txt
│
├── notebooks/
│   ├── 00_main_full_model.ipynb
│   │
│   └── ablations/
│       ├── 01_one_gat_layer.ipynb
│       ├── 02_no_gat_layers.ipynb
│       ├── 03_no_global_node.ipynb
│       ├── 04_no_fuzzy_reward.ipynb
│       ├── 05_softmax_vs_fuzzy_argmax.ipynb
│       └── 06_efficientnetb0_only.ipynb
│
├── figures/
│   └── Research figures and visualizations
│
└── paper/
    ├── EfficientNet_GAT_Fuzzy_Cervical_Cytology_Paper.pdf
    └── main.tex
```

---

# 📓 Notebooks

### 00 — Full Proposed Model

```text
notebooks/00_main_full_model.ipynb
```

Implements the complete:

```text
EfficientNet-B0
      +
Region Graph
      +
2 GAT Layers
      +
Global Node
      +
Fuzzy Reward
```

architecture.

---

## Ablation Experiments

### 01 — One GAT Layer

```text
notebooks/ablations/01_one_gat_layer.ipynb
```

Tests the architecture using only one graph attention layer.

---

### 02 — No GAT Layers

```text
notebooks/ablations/02_no_gat_layers.ipynb
```

Removes the graph attention layers to study their contribution.

---

### 03 — No Global Node

```text
notebooks/ablations/03_no_global_node.ipynb
```

Tests the region graph without the global-context node.

---

### 04 — No Fuzzy Reward

```text
notebooks/ablations/04_no_fuzzy_reward.ipynb
```

Removes the fuzzy reward from the training objective.

---

### 05 — Softmax vs Fuzzy Argmax

```text
notebooks/ablations/05_softmax_vs_fuzzy_argmax.ipynb
```

Analytically and numerically compares the class selected using softmax probabilities with the class selected using fuzzy scores.

---

### 06 — EfficientNet-B0 Only

```text
notebooks/ablations/06_efficientnetb0_only.ipynb
```

Uses the EfficientNet-B0 backbone without the region graph, GAT layers, fuzzy head, or fuzzy reward.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/FatimaSohailCheema19/EfficientNet-GAT-Fuzzy-Cervical-Cytology
```

Move into the project directory:

```bash
cd EfficientNet-GAT-Fuzzy-Cervical-Cytology
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

The main Python libraries used in this project include:

```text
numpy
pandas
matplotlib
Pillow
scikit-learn
torch
torchvision
```

GPU acceleration using CUDA is recommended for model training.

---

# ▶️ Running the Project

The experiments were originally executed using a GPU-enabled notebook environment.

To reproduce the experiments:

1. Obtain the SIPaKMeD dataset.
2. Update the dataset path inside the notebook.
3. Install all dependencies from `requirements.txt`.
4. Open the required Jupyter Notebook.
5. Run the cells sequentially.
6. Use a CUDA-enabled GPU where possible.
7. Run the main model before comparing the ablation experiments.

Start with:

```text
notebooks/00_main_full_model.ipynb
```

---

# 📄 Research Paper

The complete research paper is available inside:

```text
paper/
```

The paper contains:

- Introduction
- Literature Review
- Proposed Methodology
- Mathematical Formulation
- Experimental Setup
- Five-Fold Evaluation
- Out-of-Fold Analysis
- Ablation Study
- Grad-CAM Analysis
- Limitations
- Future Work

---

# ⚠️ Limitations

The current study has several limitations:

- The evaluation is performed on a single isolated-cell dataset.
- Data splitting is performed at image level because patient and slide identifiers were unavailable.
- The held-out fold is used for both checkpoint selection and final fold evaluation.
- External clinical validation was not performed.
- The graph structure is fixed for every image.
- The current GAT implementation uses a single attention head.
- Grad-CAM has limited spatial resolution.
- Confidence scores are not fully calibrated.
- Several incorrect predictions remain highly confident.
- The proposed system should not be considered a clinically validated diagnostic system.

---

# 🚀 Future Work

Possible future improvements include:

- External dataset validation
- Patient-level or slide-level splitting
- Nested cross-validation
- Learned graph connectivity
- GATv2
- Multi-head graph attention
- Residual GAT architectures
- Class-specific fuzzy membership functions
- Confidence calibration
- Supervised contrastive learning
- Model latency analysis
- Memory and energy benchmarking
- Whole-slide image evaluation
- Clinical validation using data from external laboratories

---

# 👩‍💻 Authors

**Nayab Nasir**  
**Namra Noor**  
**Fatima Sohail**

GIFT University, Gujranwala, Pakistan

---

# 🎓 Academic Context

This repository contains a research-oriented deep learning project developed for academic purposes.

The work investigates the application of convolutional neural networks, graph attention mechanisms, fuzzy confidence modeling, model interpretability, and controlled ablation experiments to cervical cytology classification.

---

# ⚕️ Disclaimer

This project is intended for **research and educational purposes only**.

The model has not undergone clinical validation and must not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## ⭐ Repository

If you find this project useful for learning or research, consider giving the repository a ⭐.
