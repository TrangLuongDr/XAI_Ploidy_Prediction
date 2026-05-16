---
title: XAI Ploidy Prediction
emoji: 🧬
colorFrom: teal
colorTo: indigo
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
license: mit
---

# XAI Ploidy Prediction

Gradio web app for embryo ploidy prediction using a Random Forest classifier.

The app predicts the probability of Euploidy from clinical and embryo time-lapse features, then shows explainability outputs:

- Global Random Forest feature importance
- Local LIME explanation for the submitted sample

## Model

The deployment artifact was generated from `XAI_model_deploy.ipynb` using:

- `IterativeImputer` for missing numeric values
- `RandomForestClassifier`
- Decision threshold: `0.402962` for Euploidy

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

## Retrain

The original clinical dataset is not included in this repository. To rebuild the model artifact from the local dataset path used in the notebook:

```bash
python train_model.py --data "/path/to/Train_Internal_test.csv"
```
