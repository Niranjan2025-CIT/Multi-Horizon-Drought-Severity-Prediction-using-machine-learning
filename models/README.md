# DROUGHT-X Trained Model Weights

This directory hosts the serialized machine learning models for multi-horizon drought severity forecasting.

## Multi-Horizon Models
- `model_1w.pkl`: 1-Week Ahead Drought Severity Classifier (Random Forest)
- `model_2w.pkl`: 2-Week Ahead Drought Severity Classifier (Random Forest)
- `model_3w.pkl`: 3-Week Ahead Drought Severity Classifier (Random Forest)
- `model_4w.pkl`: 4-Week Ahead Drought Severity Classifier (Random Forest)

> **Note**: Model `.pkl` binaries (~800 MB to 1.5 GB each) exceed GitHub's 100 MB file limit and are excluded via `.gitignore`. 
> Models can be trained using `notebooks/` pipelines or downloaded from external model storage.
