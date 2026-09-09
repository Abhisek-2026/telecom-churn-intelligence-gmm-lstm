# Telecom Churn Intelligence

### GMM Behavioral Segmentation + Cluster-wise LSTM Forecasting

An AI/ML framework for analyzing telecom churn behavior by combining **Gaussian Mixture Model (GMM) behavioral segmentation** with **cluster-specific LSTM forecasting**.

The project is designed around a practical telecom analytics problem:

> Different network/customer groups exhibit different churn patterns, so a single forecasting model may not adequately represent all behavioral segments.

The proposed approach first identifies behavioral segments using GMM and then trains a separate LSTM forecasting model for each segment.

---

## 🚀 Key Highlights

* Gaussian Mixture Model (GMM) for behavioral segmentation
* Four behavioral clusters
* Soft cluster probabilities for risk estimation
* Cluster-wise LSTM forecasting
* Early stopping for model training
* Hybrid churn risk scoring
* PCA-based cluster visualization
* Regression evaluation using:

  * MSE
  * RMSE
  * MAE
  * R²
* Classification-oriented evaluation framework
* SHAP-based explainability planned/integrated for model interpretation
* Streamlit dashboard architecture
* Designed for telecom/network analytics use cases

---

## 🏗️ Solution Architecture

```text
             Telecom Churn Data
                     │
                     ▼
          Data Cleaning & Scaling
                     │
                     ▼
        Gaussian Mixture Model (GMM)
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Cluster 0  Cluster 1  Cluster 2  Cluster 3
          │          │          │          │
          ▼          ▼          ▼          ▼
       LSTM-0     LSTM-1     LSTM-2     LSTM-3
          │          │          │          │
          └──────────┼──────────┘
                     ▼
            Future Churn Prediction
                     │
                     ▼
            Hybrid Risk Score
                     │
                     ▼
        LOW / MEDIUM / HIGH RISK
                     │
                     ▼
            Analytics Dashboard
```

---

# 🎯 Business Problem

Telecom churn is not always driven by a single pattern.

Different sites, grids, customer groups, or behavioral segments can demonstrate:

* Stable churn
* Gradually increasing churn
* Highly variable churn
* Persistently high churn
* Sudden churn spikes

A single forecasting model assumes that all observations follow broadly similar temporal behavior.

This project explores an alternative:

**First understand behavioral segments, then forecast each segment separately.**

---

# 🧠 Methodology

## 1. Data preprocessing

Monthly churn observations are cleaned and transformed before modeling.

The current implementation uses monthly observations from:

```text
Mar'25
Apr'25
May'25
Jun'25
Jul'25
Aug'25
Sep'25
Oct'25
Nov'25
Dec'25
```

Missing values are handled and the monthly features are standardized using `StandardScaler`.

---

# 2. GMM Behavioral Segmentation

A Gaussian Mixture Model is used to identify behavioral groups.

Current implementation:

```text
Number of clusters = 4
Covariance type   = full
Random state      = 42
```

Unlike hard clustering approaches, GMM provides a probability distribution over clusters.

For each observation:

```text
P(Cluster 0)
P(Cluster 1)
P(Cluster 2)
P(Cluster 3)
```

This provides a useful foundation for risk modeling because an observation can have partial membership across multiple behavioral groups.

---

# 3. Behavioral Cluster Interpretation

The clusters are interpreted based on their churn behavior.

A typical interpretation framework is:

| Cluster   | Possible behavior                 |
| --------- | --------------------------------- |
| Cluster 0 | Relatively stable / lower churn   |
| Cluster 1 | Higher-risk churn behavior        |
| Cluster 2 | Moderate / fluctuating churn      |
| Cluster 3 | Transitional or variable behavior |

**Important:** GMM cluster labels are model-generated labels. Their business interpretation should be assigned after examining the actual cluster statistics rather than assuming that a particular cluster number always represents a particular condition.

The current implementation identifies the high-risk cluster based on its observed churn characteristics.

---

# 4. Cluster-wise LSTM Forecasting

Instead of training one LSTM for the complete population, the implementation trains separate LSTM models for the four GMM clusters.

Conceptually:

```text
GMM
 │
 ├── Cluster 0 → LSTM 0
 ├── Cluster 1 → LSTM 1
 ├── Cluster 2 → LSTM 2
 └── Cluster 3 → LSTM 3
```

The current forecasting formulation uses:

```text
Mar'25 → Nov'25
       ↓
   LSTM sequence
       ↓
Dec'25 prediction
```

The LSTM architecture contains:

```text
Input
  ↓
LSTM(64)
  ↓
Dropout(0.30)
  ↓
LSTM(32)
  ↓
Dense(1)
```

The model is trained using the Adam optimizer and MSE loss.

---

# 🛑 Early Stopping

The implementation uses early stopping instead of blindly training for a fixed number of epochs.

Conceptually:

```python
EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)
```

This allows training to stop when validation performance stops improving and restores the best-performing model weights.

This helps reduce unnecessary training and provides some protection against overfitting.

---

# 📊 Current Model Performance

The current baseline implementation produced:

| Metric |     Result |
| ------ | ---------: |
| MSE    | **31.222** |
| RMSE   |  **5.588** |
| MAE    |  **3.706** |
| R²     |  **0.543** |

### Interpretation

The current model explains approximately **54.3% of the variance** in the held-out target under the current experimental setup.

This is a useful baseline, but it should **not** yet be presented as production-ready performance.

The next objective is to investigate whether performance can be improved through:

* richer telecom/network features
* improved temporal windowing
* feature engineering
* hyperparameter optimization
* alternative sequence models
* ensemble methods
* better validation methodology

---

# 🔬 Hybrid Risk Score

The project combines:

1. GMM-based behavioral risk
2. Forecasted future churn

The current implementation uses:

```python
hybrid_risk = (
    churn_probability *
    np.maximum(predicted_next_churn, 0)
)
```

The resulting score is mapped into risk categories:

```text
LOW
MEDIUM
HIGH
```

This provides a mechanism for moving from:

> "What will churn look like?"

toward:

> "Which observations should be prioritized?"

---

# 📈 Explainability

SHAP is used/planned as an explainability layer to understand which input variables contribute most strongly to the model prediction.

Example questions:

* Which historical churn periods influence the prediction?
* Which features push an observation toward higher risk?
* Which behavioral segments have stronger churn signals?

SHAP explanations should be interpreted as **model contributions**, not causal proof of why churn occurred.

---

# 📊 Visualization

The project includes visual analysis such as:

### GMM cluster visualization

PCA is used to project the behavioral feature space into two dimensions for visualization.

### Forecasting analysis

Actual versus predicted churn can be visualized to understand model performance.

### Classification analysis

Where a valid business-defined classification threshold is available:

* Confusion matrix
* ROC curve
* ROC-AUC

can be generated.

---

# 💻 Dashboard

A Streamlit dashboard is planned/implemented as the visualization layer.

Potential dashboard modules include:

```text
Overview
   │
   ├── Cluster Distribution
   ├── Churn Forecast
   ├── Risk Distribution
   ├── High-Risk Sites/Groups
   ├── Model Performance
   └── Explainability
```

---

# 🛠️ Technology Stack

### Programming

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Gaussian Mixture Model
* PCA
* StandardScaler

### Deep Learning

* TensorFlow
* Keras
* LSTM

### Explainable AI

* SHAP

### Visualization

* Matplotlib

### Dashboard

* Streamlit

### Development

* Jupyter Notebook
* Anaconda

---

# 📁 Repository Structure

```text
telecom-churn-intelligence-gmm-lstm/
│
├── data/
├── src/
├── notebooks/
├── dashboard/
├── models/
├── outputs/
├── docs/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

# 🔐 Data Privacy

The original telecom/network dataset used during development may contain proprietary or confidential information.

Therefore:

**No confidential Airtel/customer/network data should be uploaded to this repository.**

The public repository should contain only:

* synthetic data
* anonymized data
* generated examples
* model outputs that do not expose confidential information

---

# 🔭 Future Improvements

The current implementation is a baseline and can be extended significantly.

### 1. Multivariate telecom features

Integrate network KPIs such as:

```text
DL Throughput
UL Throughput
PRB Utilization
Packet Loss
Latency
RSRP
RSRQ
SINR
Drop Rate
Handover Success Rate
Availability
Traffic Volume
```

This can potentially provide stronger predictive signals than historical churn alone.

---

### 2. Sliding-window forecasting

Instead of using only one fixed historical window:

```text
Mar → Nov → Dec
```

future versions can use:

```text
Mar-Apr-May → Jun
Apr-May-Jun → Jul
May-Jun-Jul → Aug
...
```

This creates more temporal training samples.

---

### 3. Model comparison

Compare:

```text
LSTM
GRU
XGBoost
Random Forest
Temporal CNN
Transformer
```

against simple statistical baselines.

---

### 4. Automated cluster selection

The current implementation uses four GMM clusters.

Future experiments should evaluate different values of K using:

* BIC
* AIC
* silhouette score
* cluster stability
* business interpretability

---

### 5. Advanced ensemble

A future architecture could combine:

```text
GMM
 +
LSTM / GRU
 +
XGBoost
 +
Network KPI features
```

to investigate whether an ensemble provides more robust forecasting.

---

### 6. Production Architecture

A possible future architecture:

```text
Network / Customer Data
          ↓
       Data Lake
          ↓
   Feature Engineering
          ↓
       ML Pipeline
          ↓
   GMM Segmentation
          ↓
 Cluster-specific Models
          ↓
    Risk Prediction
          ↓
 Dashboard / API / Alerts
          ↓
 Network Operations
```

---

# 📌 Current Project Status

| Component                      | Status |
| ------------------------------ | ------ |
| Data preprocessing             | ✅      |
| GMM clustering                 | ✅      |
| 4 behavioral clusters          | ✅      |
| PCA visualization              | ✅      |
| Cluster-wise LSTM              | ✅      |
| Early stopping                 | ✅      |
| Hybrid risk score              | ✅      |
| Regression metrics             | ✅      |
| Confusion matrix               | 🔄     |
| ROC-AUC                        | 🔄     |
| SHAP analysis                  | 🔄     |
| Streamlit dashboard            | 🔄     |
| Multivariate network KPI model | 🔄     |

---

# 👨‍💻 Author

**Abhisek De**

Senior Manager – 5G RAN / Network Operations
Telecom | 5G | AI/ML | Network Intelligence

Background:

* 18+ years in telecom/RAN
* 4G/5G network operations and optimization
* Multi-vendor RAN
* IIT Kanpur eMasters – Next Generation Wireless Technologies
* Python / Machine Learning / Deep Learning

---

# ⚠️ Disclaimer

This repository is an independent technical/portfolio project.

It does not contain or disclose confidential, proprietary, customer, or operational data from any telecom operator.

Results presented in this repository are based on the documented experimental dataset and methodology and should not be interpreted as production performance guarantees.
