# Machine Learning: Precision Agriculture & Li-Ion Battery Thermal Forecasting

This repository contains two end-to-end machine learning pipelines developed in Python (`scikit-learn`, `pandas`, `seaborn`) addressing distinct domain challenges: **agronomic zone segmentation** and **EV battery thermal anomaly detection**.

## Project Overview

### Part A: Precision Agriculture & Crop Recommendation
* **Data Sanitization & Leakage Prevention:** Stripped target-leaking features and applied One-Hot Encoding and `StandardScaler` normalization across soil and climatic variables.
* **Unsupervised Spatial Segmentation:** Utilized **K-Means Clustering** ($k=5$, justified via Silhouette Score) and **Principal Component Analysis (PCA)** to compress multidimensional environmental metrics into distinct, actionable cropping zones.
* **Supervised Classification & Critical Evaluation:** Trained and cross-validated **Logistic Regression**, **Random Forest**, and **Support Vector Machine (SVM)** classifiers, benchmarked against a 9.09% random-chance baseline to audit synthetic data limitations.

<img width="926" height="790" alt="Feature corelation" src="https://github.com/user-attachments/assets/e6be1760-3a19-4ea8-94b2-9bb34c50c941" />

### Part B: Lithium-Ion Battery Thermal Behaviour (BMS)
* **Time-Series Feature Engineering:** Flattened longitudinal charging cycles into tabular thermodynamic features, extracting total temperature rise (Delta T), peak temperature, maximum heating rate (dT/dt), charge duration, average current, and average voltage.
* **Thermal Regression Modelling:** Evaluated **Linear Regression**, **Decision Tree**, and **Random Forest Regressors** (80/20 train-test split) using $R^2$ and RMSE metrics, extracting feature importance weights to eliminate black-box ambiguity.
* **Automated High-Risk Anomaly Detection:** Built a statistical rule-based BMS (Battery Management System) logic gate that flags hazardous charging cycles breaching both the **90th percentile of Delta T** and the **historical mean heating rate (dT/dt)**.

<img width="555" height="446" alt="Charging profile" src="https://github.com/user-attachments/assets/a1548faf-4f00-499c-be7e-2b6e4066a0a2" />

## Tech Stack
* **Language:** Python 3.12
* **Libraries:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`
* **Environment:** Jupyter Notebook
