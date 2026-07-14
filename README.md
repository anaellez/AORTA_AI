# Aorta AI — Predicting Post-Operative Complication Risk After Aortic Surgery

> Machine Learning models to predict the risk of post-operative complications in patients with aortic (aorta) disease, helping the clinical team assess patient risk and make informed decisions before and after surgery.

**Final project report — June 2026**
Afeka College of Engineering

| | |
|---|---|
| **Authors** | Anael Maimon, Daniel Berih |
| **Client** | Dr. Maysam Shehab — Senior Vascular & Endovascular Surgeon, Aortic Fellow, Uppsala University Hospital |
| **Academic supervisor** | Dr. Sharon Yalov-Handzel |

---

## ⚠️ Confidentiality

This repository is a **public summary of the project only**. Both the **patient data** and the **source code** used in this project are **confidential and are not shared publicly**.

- The dataset consists of **real, de-identified medical records** provided by a general hospital in collaboration with Dr. Maysam Shehab. It contains sensitive protected health information and **cannot be distributed, published, or reproduced**.
- The training/modeling **code, notebooks, and trained model artifacts are private** and are not included in this repository.

Please do not request access to the raw data or private code — they remain restricted for medical-ethics and privacy reasons.

---

## 1. Overview

Aortic disease and the surgeries that treat it (both open and endovascular) carry a significant risk of post-operative complications. The goal of **Aorta AI** is to build machine-learning models that **predict the risk of these complications**, giving the medical team a data-driven decision-support tool for risk assessment before and after surgery.

The project defines **eight target variables**, each representing a different type of complication, grouped into two families:

- **Early complications** — events occurring within **30 days** after surgery. Treated as **binary classification** problems (XGBoost, ANN, Ensemble).
- **Long-term complications** — events occurring within **36 months** after surgery. Treated as **time-to-event / survival** problems (Cox Proportional Hazards).

## 2. Data & Preprocessing

The starting point was a real hospital dataset of **2,423 patient records** and **203 features (columns)**. A thorough cleaning and reduction process was applied to keep only the most relevant, high-quality information.

**Column reduction (203 → 48 features):**
- Removed columns with more than 20% missing values, and columns containing a single unique value (**−101 columns**).
- Manually removed non-predictive columns such as IDs and near-duplicates (**−41 columns**).
- Removed columns with strong multicollinearity (**−13 columns**).

**Cleaning & enrichment:**
- Unified all the different representations of missing values (`NaN`, blanks, dots, etc.) into a single consistent null value.
- Imputed remaining missing values (for columns with <20% missing) using the mean or median.
- Ordinal encoding for ranked categorical variables (e.g. socio-economic status `1–5`).
- **One-Hot Encoding** for nominal categorical variables (e.g. gender), growing the feature set from **48 → 65 columns**.

To prevent **data leakage**, features that are only known *after* surgery were removed from the early-complication models (leaving 54 features for the Early Death model).

## 3. Methodology

| Method | Role |
|---|---|
| **XGBoost** | Gradient-boosted decision trees; strong, efficient baseline for tabular data. |
| **ANN** | Feed-forward neural network with two hidden layers, to capture non-linear relationships. |
| **Ensemble** | Combination of XGBoost + ANN to leverage the strengths of both. |
| **Cox Proportional Hazards** | Survival model estimating risk as a function of time and features — used for the long-term targets. |
| **SMOTE** | Synthetic over-sampling of the minority class to handle class imbalance. |
| **Hyperparameter Tuning** | Systematic search for the best model configuration. |
| **5-Fold Cross-Validation** | Stable, reliable performance estimation and reduced overfitting risk. |
| **SHAP** | Explainability — identifies and selects the most influential features. |

**Evaluation metrics.** Because the data is heavily imbalanced (far fewer patients experienced complications than didn't), the primary metrics are:

- **PR-AUC** (Precision–Recall AUC) — the main metric for the classification tasks; best suited to imbalanced data and to correctly identifying the minority (at-risk) group.
- **ROC-AUC** — overall separability between the two groups.
- **C-index** (Concordance Index) — the main metric for the Cox survival models; measures how well the model ranks patient risk over time.

Supporting metrics: Accuracy, Precision, Recall (Sensitivity), F1-Score.

## 4. Results by Target

The best model was selected per target based on the central metric (bolded below).

| Target | Type | Central metric | Selected model | Reliable? |
|---|---|---|---|---|
| **Early Death** (≤30 days) | Classification | PR-AUC **0.271** | ANN + Tuning | ✅ |
| **Early MACCE** | Classification | ROC-AUC **0.71** | XGBoost + SHAP | ✅ |
| **Early CAD** | Classification | PR-AUC **0.059** | XGBoost + SHAP (10 feat.) | ⚠️ weak |
| **Early CHF** | Classification | PR-AUC **0.199** | XGBoost + SHAP (10 feat.) | ✅ |
| **Early Paraplegia** | Classification | ROC-AUC **0.553** | XGBoost + SHAP | ❌ not reliable |
| **Early Reintervention** | Classification | ROC-AUC **0.625** | Ensemble (XGBoost + ANN) | ⚠️ modest |
| **Death up to 36 months** | Survival | C-index **0.857** | Cox Proportional Hazards | ✅ |
| **Reintervention up to 36 months** | Survival | C-index **0.630** | Cox Proportional Hazards | ⚠️ modest |

*Early Stroke was also explored but had too few positive cases to train a reliable model, and was excluded from the final comparison.*

**Highlights:**
- **Death up to 36 months** was the strongest result (C-index 0.857, ROC-AUC 0.862) — the Cox survival model excellently separates risk levels and ranks patients by 3-year mortality risk.
- **Early Death** reached PR-AUC 0.271 / ROC-AUC 0.814; hyperparameter tuning of the ANN meaningfully improved over the XGBoost baseline. The model flags ~58% of high-risk patients *before* surgery.
- **Early MACCE** selected features are clinically meaningful (CHF, CAD, albumin, creatinine, urea, BMI), reinforcing that the model learns real clinical relationships.

## 5. Key Findings

- **Prediction quality depends heavily on how frequent the complication is.** Common complications (e.g. death up to 36 months) were predicted well; very rare complications (Paraplegia, Stroke) could not be modeled reliably.
- **Different methods excelled at different tasks:** a tuned ANN for Early Death, the XGBoost+ANN Ensemble for Early Reintervention, and Cox survival models for the long-term targets.
- **Most influential features across targets** were consistent and clinically recognized risk factors: baseline **CHF** (`BL_CHF`), baseline **CAD** (`BL_CAD`), **albumin**, **creatinine**, **urea**, **BMI**, and **age** — evidence that the models learn genuine medical patterns rather than noise.

## 6. Limitations & Future Work

- **Class imbalance** was the central limitation: the number of patients with complications was far smaller than those without, making high precision difficult. For the rarest complications there were simply too few positive cases to train a clinically useful model.
- Some targets (e.g. early CAD and early Reintervention) remain hard to predict from the currently available data.
- **Future directions:** expand the dataset (especially for rare complications), add more clinical variables, and explore additional advanced techniques for handling imbalance.
- These models are intended as a **decision-support** tool alongside — **not as a replacement for** — clinical judgment.

## 7. Repository Structure

```
.
├── README.md                 # This file
└── Aorta_AI_Report_Final.pdf # Full project report (Hebrew)
```

> Data and modeling code are intentionally **not** included — see the Confidentiality section above.

---

*Data provided in collaboration with Dr. Maysam Shehab — Senior Vascular & Endovascular Surgeon, Aortic Fellow, Uppsala University Hospital.*
