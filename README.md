# 💧 Smart Water Quality Prediction Using Advanced ML and XAI

## Overview

Smart Water Quality Prediction is a machine learning-based system that predicts whether water is Safe or Unsafe for consumption using physicochemical parameters. The project integrates Explainable Artificial Intelligence (XAI) to provide transparent and interpretable predictions.

## Features

* Water quality prediction using Machine Learning
* Manual parameter input
* Bulk prediction through CSV/XLSX upload
* Explainable AI using SHAP
* Interactive dashboards and visualizations
* Prediction history management
* User-friendly Streamlit interface

## Input Parameters

* pH
* Hardness
* Solids
* Chloramines
* Sulfate
* Conductivity
* Organic Carbon
* Trihalomethanes
* Turbidity

## Technology Stack

### Frontend

* Streamlit
* HTML
* CSS

### Backend

* Python

### Machine Learning & XAI

* Scikit-learn
* SHAP
* Joblib

### Data Processing & Visualization

* Pandas
* NumPy
* Matplotlib
* Seaborn

### Database

* Oracle Database

## System Workflow

1. User enters water quality parameters or uploads a dataset.
2. Data is preprocessed and passed to the trained ML model.
3. The model predicts whether water is Safe or Unsafe.
4. SHAP generates explanations for the prediction.
5. Results are displayed through dashboards and tables.

## Project Structure

```text
Water-Quality-Prediction/
│
├── app_frontend.py
├── train_model.py
├── model.pkl
├── water_potability.csv
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Hari-06-Prakash/Water-Quality-Prediction.git
cd Water-Quality-Prediction
pip install -r requirements.txt
streamlit run app_frontend.py
```

## Future Enhancements

* Real-time IoT sensor integration
* Cloud deployment
* Mobile application support
* Advanced analytics dashboard
* Multi-model comparison

## Author

Hari Prakash

## License

This project is developed for educational and research purposes.
