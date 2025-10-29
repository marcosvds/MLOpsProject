# 🏙️ Milan Rent Prices MLOps Project

This project operationalizes a regression model that predicts **apartment rent prices in Milan** using advanced feature engineering and an XGBoost model.  
It was developed as part of the **Machine Learning Ops & Interviews** course.

---

## 📘 Project Overview

The base model was built through a **structured and iterative process** including:
- **Exploratory Data Analysis (EDA)**  
- **Data cleaning and imputation** using median and prediction-based strategies  
- **Feature engineering** from descriptive text and categorical variables  
- **Geospatial enhancement** — distance to Duomo di Milano computed with *geopy*  
- **Model selection and tuning**, where **XGBoost** outperformed LightGBM, Ridge, and ensemble models  
- **Target transformation** using square root to stabilize variance and reduce skewness  

The final model integrates interaction features (e.g., spatial × structural) and achieved the best validation performance among tested configurations.

---

## ⚙️ Requirements

- Python 3.12  
- pandas, numpy, scikit-learn  
- xgboost  
- mlflow (for future experiment tracking)  
- joblib (for model persistence)  

---

## 🧱 Repo Structure
- `src/`: training and preprocessing scripts  
- `data/`: raw and processed datasets  
- `models/`: trained model artifacts  
- `notebooks/`: exploratory analysis

---

## 🚀 How to Run
pip install -r requirements.txt
python src/train.py

