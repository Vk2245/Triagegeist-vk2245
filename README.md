<div align="center">

# 🏥 Triagegeist — Hierarchical Multimodal Triage Intelligence

### *A Three-Tier Safety-First Clinical AI System for Emergency Severity Prediction*

[![Competition](https://img.shields.io/badge/Competition-Triagegeist-blue?style=for-the-badge&logo=kaggle)](https://kaggle.com/competitions/triagegeist)
[![QWK Score](https://img.shields.io/badge/OOF%20QWK-0.9987-brightgreen?style=for-the-badge)](https://kaggle.com/competitions/triagegeist)
[![Baseline](https://img.shields.io/badge/Baseline%20QWK-0.7120-red?style=for-the-badge)](https://kaggle.com/competitions/triagegeist)
[![Improvement](https://img.shields.io/badge/Improvement-%2B0.2867-orange?style=for-the-badge)](https://kaggle.com/competitions/triagegeist)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20T4%2015.6GB-76B900?style=for-the-badge&logo=nvidia)](https://kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

<br/>

> *Can AI meaningfully support triage decisions in the emergency department?*
> **This system answers yes — with clinical rigour, safety guardrails, and near-perfect accuracy.**

</div>

---

## 📋 Table of Contents

- [Clinical Problem Statement](#-clinical-problem-statement)
- [System Architecture](#-system-architecture)
- [Performance Results](#-performance-results)
- [Feature Engineering](#-feature-engineering)
- [Model Stack](#-model-stack)
- [Dataset](#-dataset)
- [Quickstart](#-quickstart)
- [Project Structure](#-project-structure)
- [Key Findings](#-key-findings)
- [Limitations](#-limitations)
- [Citations](#-citations)

---

## 🩺 Clinical Problem Statement

Emergency Severity Index (ESI) triage is the primary gatekeeping decision in modern emergency departments — a five-level algorithm determining who receives immediate resuscitation and who waits. It operates under a **dual-axis logic** that most AI systems fail to recognise:

| ESI Level | Determining Factor | Clinical Meaning |
|-----------|-------------------|-----------------|
| **ESI-1** | Patient acuity | Immediate life-saving intervention required |
| **ESI-2** | Patient acuity | High risk of rapid deterioration |
| **ESI-3** | Resource prediction | Stable, but needs ≥2 ED resources |
| **ESI-4** | Resource prediction | Stable, needs exactly 1 resource |
| **ESI-5** | Resource prediction | No resources needed |

**Failure modes addressed:**
1. **Undertriage** — misclassifying high-acuity patients to lower urgency → delayed intervention → preventable death
2. **ESI-2/3 boundary confusion** — the most dangerous classification error, caused by treating ESI as a simple linear severity scale

This system addresses both with a **three-tier hierarchical architecture** that mirrors human triage decision logic.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATIENT INTAKE DATA                          │
│         (Vitals · Demographics · History · Chief Complaint)     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1 — DETERMINISTIC SAFETY GUARDRAIL                        │
│                                                                 │
│  ESI-1 Triggers:  GCS ≤ 8  │  HR > 150  │  SBP < 80 mmHg       │
│                   "cardiac arrest" │ "agonal breathing" │ ...    │
│                                                                 │
│  ESI-2 Triggers:  SpO₂ < 90%  │  SBP < 90  │  AMS               │
│                   "chest pain" │ "stroke" │ "seizure" │ ...     │
│                                                                 │
│  Rule: ONLY upgrades — never downgrades. Safety-first.          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 2 — CLINICAL FEATURE ENGINEERING  (200 features)          │
│                                                                 │
│  ├── NEWS2         (RCP 2017, 6-parameter exact calculator)     │
│  ├── Shock Index + Modified Shock Index (HR/MAP)                │
│  ├── Charlson Comorbidity Index (21 hx_ flag mappings)          │
│  ├── Missingness Flags  (9 vital signs — clinically informative)│
│  ├── 16 Interaction Terms  (elderly×AMS, ambulance×shock, ...)  │
│  └── NLP  (TF-IDF 400 features + SVD 60 dims, float32)          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 3 — MULTIMODAL META-ENSEMBLE                              │
│                                                                 │
│   LightGBM 4.6  ──┐                                             │
│   CatBoost 1.2  ──┼──► Grid-searched OOF blend                  │
│   XGBoost 3.2   ──┘    (LGB:0.50 · CBT:0.35 · XGB:0.15)        │
│                                                                 │
│   5-fold Stratified CV  │  GPU-accelerated  │  Early stopping    │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING                                                │
│  ├── Nelder-Mead Threshold Optimisation  (5 restarts, QWK-direct)│
│  └── Predictive Entropy Audit  (ESI-2/3 safety boundary)        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
                   FINAL ESI PREDICTION
```

---

## 📊 Performance Results

### Model Progression

| System Component | OOF QWK | Δ vs Baseline |
|-----------------|---------|--------------|
| Published Baseline (LGB + TF-IDF-200) | 0.7120 | — |
| **LightGBM (v2 features)** | **0.9986** | **+0.2866** |
| CatBoost | 0.9986 | +0.2866 |
| XGBoost | 0.9984 | +0.2864 |
| Blend (LGB:0.50 · CBT:0.35 · XGB:0.15) | 0.9987 | +0.2867 |
| + Nelder-Mead Thresholds | 0.9987 | +0.2867 |
| **+ Entropy Audit (FINAL)** | **0.9987** | **+0.2867** |

### Per-Class Performance (OOF)

| ESI Level | Precision | Recall | F1 | Support |
|-----------|-----------|--------|----|---------|
| ESI-1 (Immediate) | 0.966 | **0.986** | 0.976 | 3,222 |
| ESI-2 (Emergent) | 0.995 | 0.991 | 0.993 | 13,439 |
| ESI-3 (Urgent) | 1.000 | 0.999 | **0.999** | 28,921 |
| ESI-4 (Less Urgent) | 1.000 | 1.000 | 1.000 | 23,020 |
| ESI-5 (Non-Urgent) | 0.999 | 1.000 | 1.000 | 11,398 |
| **Overall Accuracy** | — | — | **0.997** | **80,000** |

### Cross-Validation Stability

| Model | CV Mean | CV Std | Interpretation |
|-------|---------|--------|---------------|
| LightGBM | 0.9986 | ±0.0004 | Very stable |
| CatBoost | 0.9986 | ±0.0002 | Extremely stable |
| XGBoost | 0.9984 | ±0.0002 | Extremely stable |

### Triage Bias Audit

Among 45,582 high-acuity patients (true ESI 1-3):
- **Overall undertriage rate: 0.1%** — near-zero systematic bias
- No statistically significant undertriage differences across sex, insurance type, language, or arrival mode (all p > 0.05)
- The model does not amplify real-world demographic undertriage patterns

---

## 🔬 Feature Engineering

### Clinical Physiological Indices

#### NEWS2 (National Early Warning Score 2)
Explicitly implemented per **Royal College of Physicians (2017)** specification:

| Parameter | Normal Range | Score 3 (Critical) |
|-----------|-------------|-------------------|
| Respiratory Rate | 12–20 bpm | ≤8 or ≥25 |
| SpO₂ | ≥96% | ≤91% |
| Systolic BP | 111–219 mmHg | ≤90 or ≥220 |
| Heart Rate | 51–90 bpm | ≤40 or ≥131 |
| Temperature | 36.1–38.0°C | ≤35.0°C |
| Consciousness (ACVPU) | Alert | Any new confusion |

**NEWS2 ≥ 5 = urgent clinical review trigger** (retained as binary feature)

#### Haemodynamic Indices
```
Shock Index (SI)          = HR / SBP           (normal: 0.5–0.7; shock risk: >0.9)
Modified Shock Index (MSI) = HR / MAP           (sensitive for early sepsis)
Mean Arterial Pressure     = DBP + (SBP–DBP)/3
Pulse Pressure             = SBP – DBP
```

#### Charlson Comorbidity Index (CCI)
Mapped from 21 binary `hx_` comorbidity flags:

| Condition | CCI Weight | Clinical Relevance |
|-----------|-----------|-------------------|
| Dementia, COPD, CHF, Stroke | +1 each | ESI-2 risk via AMS/respiratory |
| Diabetes w/ complications, Renal Disease, Malignancy | +2 each | Resource intensity |
| Cirrhosis | +3 | High resource/acuity |
| Metastatic Cancer, AIDS | +6 each | Highest complexity |

### Missingness as Clinical Signal

> *"A triage nurse who doesn't record blood pressure is making an implicit clinical judgment."*

9 binary absence flags for vital signs — missingness is **non-random** and **inversely correlated with acuity** (lower-acuity patients have more missing vitals).

### Interaction Features

| Feature | Clinical Rationale |
|---------|-------------------|
| `elderly × altered_ms` | AMS in elderly = high undertriage risk |
| `ambulance × shock_risk` | Prehospital activation + shock = ESI-1 |
| `ambulance × news2_critical` | Double confirmation of severity |
| `cci_score × news2_total` | Compound: complex patient + abnormal vitals |
| `cci_score × hypotension` | Chronic disease + haemodynamic instability |
| `pediatric × fever` | Infant fever = ESI-2 automatic trigger |
| `geri_msi_risk` | Beta-blocker masked tachycardia in elderly |

### NLP (Chief Complaint)
- **TF-IDF**: 400 features, 1–3 ngrams, sublinear TF, float32
- **Truncated SVD**: 60 components (memory-efficient, ~5% QWK contribution)

---

## 🤖 Model Stack

### LightGBM (Weight: 0.50)
```python
n_estimators=2000, learning_rate=0.03, num_leaves=127,
subsample=0.80, colsample_bytree=0.70,
class_weight='balanced', device='gpu'
```
- Leaf-wise growth captures non-linear physiological interactions
- Fastest convergence: ~500 best iterations on T4 GPU
- Training time: 5.6 min (5-fold)

### CatBoost (Weight: 0.35)
```python
iterations=2000, learning_rate=0.03, depth=8,
auto_class_weights='Balanced', task_type='GPU'
```
- Native categorical feature handling via ordered boosting
- Symmetric trees minimise overfitting on clinical categoricals
- Training time: 2.8 min (5-fold)

### XGBoost (Weight: 0.15)
```python
n_estimators=2000, learning_rate=0.03, max_depth=7,
tree_method='hist', device='cuda',
early_stopping_rounds=100  # constructor arg in XGB ≥2.0
```
- Histogram method for GPU memory efficiency
- Diverse regularisation path from LGB
- Training time: 3.5 min (5-fold)

### Post-Processing
- **Nelder-Mead optimisation**: 5 random restarts on cumulative probability thresholds
- **Entropy audit**: Upgrades uncertain ESI-3 (high Shannon entropy) → ESI-2

---

## 📁 Dataset

| File | Rows | Cols | Description |
|------|------|------|-------------|
| `train.csv` | 80,000 | 40 | Features + target (`triage_acuity`) |
| `test.csv` | 20,000 | 37 | Features, no target |
| `chief_complaints.csv` | 100,000 | 3 | Free-text chief complaint narratives |
| `patient_history.csv` | 100,000 | 26 | 25 binary comorbidity flags |
| `sample_submission.csv` | 20,000 | 2 | Submission format |

**Source:** Triagegeist Dataset, Laitinen-Fredriksson Foundation (2026). Synthetic data calibrated to MIMIC-IV-ED and NHAMCS distributions.

**Target distribution (train):**
```
ESI-1 (Immediate):    4.0%  ██
ESI-2 (Emergent):    16.8%  ██████████
ESI-3 (Urgent):      36.2%  █████████████████████
ESI-4 (Less Urgent): 28.8%  █████████████████
ESI-5 (Non-Urgent):  14.2%  ████████
```

---

## 🚀 Quickstart

### Requirements

```bash
pip install lightgbm>=4.0 catboost>=1.2 xgboost>=2.0 scikit-learn>=1.4 shap pandas numpy scipy matplotlib seaborn
```

### Run on Kaggle

1. Fork the notebook: `triage_Vk2245.ipynb`
2. Add the **Triagegeist** competition dataset as input
3. Enable **GPU accelerator** (T4 recommended)
4. Run all cells — end-to-end in ~15 minutes

### Local Setup

```bash
git clone https://github.com/Vk2245/Triagegeist-vk2245
cd triagegeist-vk224

# Install dependencies
pip install -r requirements.txt

# Update DATA path in notebook Cell 1:
DATA = '/path/to/triagegeist/data/'

# Run notebook
jupyter notebook triage_Vk2245.ipynb
```

### requirements.txt

```
lightgbm>=4.6.0
catboost>=1.2.10
xgboost>=3.2.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.44.0
torch>=2.0.0
```

---

## 📂 Project Structure

```
triagegeist-vk224/
│
├── 📓 triage_Vk2245.ipynb    # Main Kaggle notebook (run end-to-end)
├── 📄 README.md                       # This file
├── 📋 requirements.txt                # Python dependencies
├── 📝 writeup.pdf                      # Competition writeup
│
├── outputs/                           # Generated during notebook run
│   ├── submission.csv                 # Final predictions
│   ├── eda.png                        # EDA visualisations
│   ├── performance.png                # Confusion matrix + F1
│   ├── feature_importance.png         # Top 40 features
│   ├── bias_audit.png                 # Undertriage bias analysis
│   └── dashboard.png                  # Final results dashboard
│
└── docs/
    └── cover.png                      # Competition cover image (560×280px)
```

---

## 🔍 Key Findings

### 1. Near-Perfect Classification of Synthetic Data
The OOF QWK of **0.9987** reflects that the synthetic Triagegeist dataset has highly learnable, deterministic structure calibrated to real ED distributions. All three GBDT models converge to similar performance (0.9984–0.9986), indicating this is the dataset's information ceiling — not model-specific overfitting.

### 2. ESI-1 Recall = 98.6%
The most clinically critical metric. The model misses only 1.4% of immediate-resuscitation patients — a direct function of the NEWS2, GCS, and shock index features dominating ESI-1 classification (confirmed by SHAP).

### 3. Model Knows What Matters Clinically
SHAP analysis confirms the top predictors align precisely with established triage literature: `news2_calc`, `gcs_total`, `shock_index`, `mod_shock_index`, `spo2`, and `systolic_bp_missing` — not demographic or textual features.

### 4. No Demographic Undertriage Bias
The bias audit (chi-squared tests across sex, insurance, language, arrival mode) found no statistically significant undertriage differences in any subgroup (all p > 0.05). The model does not replicate real-world bias patterns into the synthetic data.

### 5. Model Confidence is Extremely High
Entropy audit: **0 patients** had high predictive entropy at the ESI-2/3 boundary. The model's probability distributions are sharp and confident — consistent with synthetic data's deterministic generation rules.

---

## ⚠️ Limitations

| Limitation | Description |
|-----------|-------------|
| **Synthetic data** | All findings are on simulated data. Real-world validation is mandatory before clinical deployment |
| **Tier 1 aggressiveness** | The keyword guardrail causes ~26% test prediction overrides — appropriate for safety but creates distribution shift |
| **No trajectory modeling** | Patient deterioration during ED wait time is not captured |
| **ESI-3 irreducible ambiguity** | The ESI-3/4 boundary has fundamental clinical ambiguity for any model |
| **CatBoost non-convergence** | best_iter ~1998/2000 — may benefit from 3000+ iterations |

---

## 📚 Citations

```bibtex
@dataset{triagegeist2026,
  title  = {Triagegeist Dataset},
  author = {{Laitinen-Fredriksson Foundation}},
  year   = {2026},
  note   = {Synthetic ED data, competition license}
}

@article{rcp_news2_2017,
  title  = {National Early Warning Score (NEWS) 2},
  author = {{Royal College of Physicians}},
  year   = {2017},
  publisher = {RCP London}
}

@article{charlson1987,
  title   = {A new method of classifying prognostic comorbidity in longitudinal studies},
  author  = {Charlson, M. E. and others},
  journal = {Journal of Chronic Diseases},
  volume  = {40},
  number  = {5},
  pages   = {373--383},
  year    = {1987}
}

@dataset{mimic_iv_ed,
  title  = {MIMIC-IV-ED (version 2.2)},
  author = {Johnson, A. E. W. and others},
  year   = {2023},
  doi    = {10.13026/5ntk-km72}
}

@misc{nhamcs2023,
  title  = {National Hospital Ambulatory Medical Care Survey (NHAMCS)},
  author = {{National Center for Health Statistics}},
  year   = {2023},
  publisher = {CDC}
}
```

---

<div align="center">

**Built for the Triagegeist Competition — Laitinen-Fredriksson Foundation**

*"Every minute counts in the emergency department."*

[![Kaggle](https://img.shields.io/badge/View%20on-Kaggle-20BEFF?style=for-the-badge&logo=kaggle)](https://kaggle.com/competitions/triagegeist)

</div>
