# Credit Card Fraud Detection System

Machine Learning system for detecting fraudulent credit card transactions using Logistic Regression and Random Forest algorithms with SMOTE oversampling for handling imbalanced data.

## 🚀 Features

- **Data Preprocessing**: Standard scaling of Amount and Time features
- **SMOTE Oversampling**: Handles class imbalance effectively
- **Model Training**: Logistic Regression & Random Forest with evaluation
- **Real-time API**: Flask API for instant fraud predictions
- **Power BI Integration**: Export predictions for visualization
- **Model Persistence**: Save trained models and scalers

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Libraries**: pandas, numpy, scikit-learn, imbalanced-learn, Flask, joblib
- **Models**: Logistic Regression, Random Forest Classifier

## 📋 Prerequisites

```bash
pip install -r requirements.txt

## 📊 Dataset

### Download Credit Card Fraud Dataset

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Click **Download** button
3. Save `creditcard.csv` to the project root folder

**OR** use Python to download:

```python
import kaggle
kaggle.api.dataset_download_files('mlg-ulb/creditcardfraud', path='.', unzip=True)
