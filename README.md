# Biological Age & DXA Body Composition Study
**Predicting HbA1c from NHANES 2017–2018 Body Composition Data**

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon_DB-336791) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![NHANES](https://img.shields.io/badge/Data-NHANES_2017--2018-green)

---

## Overview

This project investigates whether body composition measurements from Dual-Energy X-ray Absorptiometry (DXA) scans can predict HbA1c (glycated hemoglobin), a key biomarker for diabetes risk. Using publicly available NHANES 2017–2018 data, we built a full data pipeline from raw XPT files through a normalized PostgreSQL database to seven regression models, unsupervised clustering, and SHAP-based explainability.

**Team:** Caitlyn Maung · Bonnie Hines · Larry Pastor · Yeyi Su · Maxine Ma  
**Course:** BAX 452 — UC Davis Master of Science in Business Analytics

---

## Table of Contents
1. [Business Problem](#business-problem)
2. [Dataset](#dataset)
3. [Database Architecture](#database-architecture)
4. [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
5. [Models & Results](#models--results)
6. [Unsupervised Analysis](#unsupervised-analysis)
7. [Model Explainability (SHAP)](#model-explainability-shap)
8. [Deployment](#deployment)
9. [Setup & Usage](#setup--usage)
10. [Team Contributions](#team-contributions)

---

## Business Problem

Early diabetes detection typically requires fasting blood draws, creating a barrier to access for many patients. This project explores whether routinely collected body composition measurements — particularly from DXA scans — can serve as a cost-free triage layer, flagging high-risk individuals for follow-up lab testing without requiring blood work upfront.

This project follows the **CRISP-DM** framework across six phases: Business Understanding → Data Understanding → Data Preparation → Modeling → Evaluation → Deployment.

---

## Dataset

**Source:** [CDC NHANES 2017–2018](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017) (publicly available)

| File | Contents |
|------|----------|
| `DEMO_J.XPT` | Demographics (age, sex, race/ethnicity) |
| `BMX_J.XPT` | Body measurements (BMI, waist circumference, height, weight) |
| `GHB_J.XPT` | Laboratory — HbA1c (target variable) |
| `DXX_J.XPT` | DXA body composition (regional lean mass, fat mass) |

- **Final sample:** 3,199 adult participants after joining on `SEQN` and dropping rows with missing HbA1c
- **Target variable:** `LBXGH` — HbA1c (%) from glycohemoglobin file

---

## Database Architecture

Raw XPT files were ingested into a **Neon DB PostgreSQL** instance using a normalized relational schema:

```
person       (SEQN, age, sex, race, ...)
body_comp    (SEQN, bmi, waist_cm, weight_kg, height_cm, ...)  ← FK → person
labs         (SEQN, hba1c, ...)                                ← FK → person
```

Tables are joined on `SEQN` (NHANES sequence number) using SQLAlchemy. Database credentials are loaded from environment variables — **never hardcoded**.

```python
import os
NEON_URL = os.environ.get("NEON_URL")  # Store in .env, never in code
engine = create_engine(NEON_URL)
```

---

## Feature Engineering & Preprocessing

### DXA Feature Aggregation
Limb-level DXA columns (arms, legs, trunk) were aggregated into three composite features:

| Feature | Description |
|---------|-------------|
| `lean_mass_total` | Sum of lean mass across all body regions |
| `fat_mass_total` | Sum of fat mass across all body regions |
| `lean_fat_ratio` | `lean_mass_total / fat_mass_total` |

### Multicollinearity — VIF Analysis
Variance Inflation Factor (VIF) analysis identified severe multicollinearity:

- `weight_kg` (VIF > 10) → **removed**
- `height_cm` (VIF > 10) → **removed**

**Final feature set:** `age`, `bmi`, `waist_cm`, `fat_mass_total`, `lean_mass_total`

### Missing Data
- Rows with missing **HbA1c** (target): dropped entirely
- Missing values in `age`, `bmi`, `waist_cm`: imputed with **column median**

---

## Models & Results

Seven models were trained and evaluated on an 80/20 train-test split. All continuous features were standardized (zero mean, unit variance) before modeling.

| Model | Test R² | RMSE | MAE | AIC |
|-------|---------|------|-----|-----|
| **Neural Network (MLP)** ⭐ | **0.2763** | **0.7054** | **0.4070** | **-172.62** |
| K-Nearest Neighbors | 0.1920 | 0.7454 | 0.4259 | -142.21 |
| Ridge Regression ✅ | 0.1762 | 0.7526 | 0.4301 | -146.88 |
| OLS Regression | 0.1762 | 0.7526 | 0.4301 | 2783.83 |
| Lasso Regression | 0.1632 | 0.7586 | 0.4091 | -142.54 |
| Random Forest | 0.1049 | 0.7845 | 0.4377 | — |
| Gradient Boosting | 0.0949 | 0.7889 | 0.4202 | 469.10 |

⭐ Best performance · ✅ Selected for deployment

### Model Selection Rationale
The **Neural Network** (MLP: 64→32 units, ReLU activation) achieved the highest predictive accuracy (R² = 0.276, AIC = -172.62). However, **Ridge Regression** was selected for deployment due to:
- **Interpretability**: Linear coefficients are explainable to clinical stakeholders
- **Stability**: Consistent, regularized performance across validation folds
- **Regulatory alignment**: Transparent models are preferred in healthcare risk screening contexts

> **AIC (Akaike Information Criterion):** Balances model fit against complexity. Formula: `n × log(MSE) + 2k`. Lower AIC = better trade-off between accuracy and parsimony. OLS's large positive AIC (2783.83) reflects its penalization for parameter count without L2 regularization.

---

## Unsupervised Analysis

### K-Means Clustering — Metabolic Phenotypes
K-Means (k=4) was applied to body composition features to identify distinct metabolic profiles:

| Cluster | Label | Profile |
|---------|-------|---------|
| 0 | Lean Healthy | Lowest HbA1c, high lean-to-fat ratio |
| 1 | Low Lean Mass | Below-average lean mass |
| 2 | Intermediate Risk | Moderate fat and lean metrics |
| 3 | High Body Fat | Highest fat mass, elevated HbA1c |

Optimal k was selected using the elbow method on within-cluster sum of squares (WCSS).

### PCA — Dimensionality Reduction
Principal Component Analysis was applied to the 6 body measurement variables:
- **3 components explain 93% of total variance**
- Used to visualize cluster separation and confirm feature structure

---

## Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) values were computed for the Gradient Boosting model to interpret feature contributions at the individual level.

**Key finding:** For the highest-risk patient in the test set:
- `waist_cm` contributed **+2.55** to the predicted HbA1c — the single most influential feature
- High waist circumference is strongly associated with visceral adiposity and insulin resistance

SHAP waterfall plots confirm that abdominal fat, not overall fat mass, drives HbA1c predictions for high-risk individuals.

---

## Deployment

The Ridge Regression model is serialized via `joblib` and loaded for inference. The screening pipeline accepts new patient body composition inputs and returns:

1. **Predicted HbA1c** (continuous value)
2. **Diabetes Risk Category** based on clinical thresholds:
   - Normal: HbA1c < 5.7%
   - Prediabetes: 5.7% – 6.4%
   - Diabetes: ≥ 6.5%

---

## Setup & Usage

### Requirements

```bash
pip install pandas numpy sqlalchemy psycopg2-binary statsmodels matplotlib seaborn scikit-learn shap joblib python-dotenv
```

### Clone the Repo

```bash
git clone https://github.com/<your-username>/bax452-biological-age-dxa.git
cd bax452-biological-age-dxa
```

### Environment Variables

Create a `.env` file in the project root — **this file must never be committed to GitHub**:

```
NEON_URL=postgresql://<user>:<password>@<host>/neondb?sslmode=require
```

Load it in the notebook:

```python
from dotenv import load_dotenv
import os
load_dotenv()
NEON_URL = os.environ.get("NEON_URL")
```

A `.gitignore` should include:
```
.env
*.env
```

### Run

```bash
jupyter notebook BAX_452_Biological_Age_and_DXA_Study_v2.ipynb
```

---

## Team Contributions

| Team Member | Contributions |
|-------------|--------------|
| **Larry Pastor** | NHANES data ingestion, PostgreSQL schema design, SQLAlchemy pipeline; Report sections 1 & 2 |
| **Caitlyn Maung** | EDA visualizations, final candidate models, diabetes risk prediction pipeline; Report sections 2, 3 & 4 |
| **Bonnie Hines** | EDA visualizations, PCA implementation, SHAP analysis, debugging; Report sections 1–6 revision |
| **Yeyi Su** | Report sections 5 & 6 |
| **Maxine Ma** | Report section 3 |

---

## Key Takeaways

- DXA body composition features explain ~18–28% of HbA1c variance — a meaningful signal given that dietary, genetic, and medication data were excluded
- **Waist circumference** is the strongest single predictor of elevated HbA1c
- Removing multicollinear features (weight, height) via VIF analysis improved model stability
- K-Means clustering reveals 4 clinically distinct metabolic risk phenotypes that complement regression predictions
- Ridge regularization outperforms OLS on generalization while maintaining interpretability for clinical deployment

---

## Data Citation

National Center for Health Statistics. National Health and Nutrition Examination Survey Data. U.S. Department of Health and Human Services, Centers for Disease Control and Prevention, 2017–2018. https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017

*UC Davis Master of Science in Business Analytics — BAX 452 Final Project*
