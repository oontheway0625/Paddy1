# Paddy Yield Prediction — Streamlit Dashboard v2

This version is ready to upload to GitHub and deploy on Streamlit Community Cloud.

## Main sections

- `Prediction` — interactive four-class paddy-yield prediction.
- `Data Understanding` — original dataset overview, missing values, duplicate checks, summary statistics, full 33-input correlation heatmap, numerical EDA, categorical EDA and target distribution.
- `Data Preparation` — cleaned class distribution, correlation evidence, feature distributions, categorical analysis, StandardScaler evidence and feature selection.
- `Model Performance` — scorecard, 5-fold cross-validation, confusion matrices, ROC/PR curves, feature importance, learning curve and tuning evidence.
- `About` — project objective, target definition and preprocessing flow.

## Files

- `app.py` — dashboard UI
- `paddydataset.csv` — source dataset
- `paddy_dashboard_bundle.joblib` — fitted models + saved evaluation evidence
- `requirements.txt` — Python dependencies
- `train_model.py` — optional model regeneration script

## Deploy

1. Create or open a GitHub repository.
2. Upload all files in this folder to the same repository folder.
3. In Streamlit Community Cloud, create a new app from the repository.
4. Set **Main file path** to `app.py`.
5. Deploy.

You do not need to run `train_model.py` before deploying because the fitted bundle is already included.

## Prediction target

- Low: `0–9,999 kg`
- Moderate: `10,000–19,999 kg`
- High: `20,000–29,999 kg`
- Very High: `30,000–39,999 kg`

The app follows the 33 selected predictors, duplicate removal, stratified 75/25 split and `random_state=42` used in the supplied notebook.
