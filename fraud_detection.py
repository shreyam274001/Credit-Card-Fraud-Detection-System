# fraud_detection.py
# Credit Card Fraud Detection System
# Author: Shreyam
# Description: ML-based fraud detection using Logistic Regression & Random Forest with SMOTE

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# --------------------------------------------------
# 0. CHECK FOR DATASET
# --------------------------------------------------
def check_dataset():
    """Check if dataset exists, if not provide instructions"""
    if not os.path.exists("creditcard.csv"):
        print("="*60)
        print("❌ ERROR: creditcard.csv not found!")
        print("="*60)
        print("\n📥 Please download the dataset from:")
        print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("\n🔄 Or run the download script:")
        print("   python download_data.py")
        print("="*60)
        return False
    return True

# --------------------------------------------------
# 1. LOAD AND EXPLORE DATA
# --------------------------------------------------
def load_data():
    """Load and perform initial data exploration"""
    print("\n" + "="*60)
    print("📊 CREDIT CARD FRAUD DETECTION SYSTEM")
    print("="*60)
    
    print("\n📥 Loading dataset...")
    df = pd.read_csv("creditcard.csv")
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Total transactions: {len(df):,}")
    print(f"   - Features: {len(df.columns)-1}")
    print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Basic statistics
    print("\n📊 Dataset Statistics:")
    print(f"   - Legitimate transactions: {len(df[df['Class']==0]):,} ({len(df[df['Class']==0])/len(df)*100:.2f}%)")
    print(f"   - Fraudulent transactions: {len(df[df['Class']==1]):,} ({len(df[df['Class']==1])/len(df)*100:.4f}%)")
    print(f"   - Fraud ratio: {len(df[df['Class']==1])/len(df[df['Class']==0])*100:.4f}%")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"   ⚠️  Missing values: {missing}")
    else:
        print(f"   ✅ No missing values")
    
    return df

# --------------------------------------------------
# 2. PREPROCESSING
# --------------------------------------------------
def preprocess_data(df):
    """Preprocess the data: scale features and prepare for modeling"""
    print("\n" + "="*60)
    print("🔧 PREPROCESSING DATA")
    print("="*60)
    
    # Features & Target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    print(f"\n📋 Feature matrix shape: {X.shape}")
    print(f"📋 Target vector shape: {y.shape}")
    
    # Scale the "Amount" and "Time" columns
    print("\n📏 Scaling 'Amount' and 'Time' features...")
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])
    
    print("✅ Scaling complete!")
    print(f"   - Mean of Amount: {X_scaled['Amount'].mean():.2f}")
    print(f"   - Std of Amount: {X_scaled['Amount'].std():.2f}")
    
    return X_scaled, y, scaler

# --------------------------------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------------------------------
def split_data(X, y):
    """Split data into training and testing sets"""
    print("\n" + "="*60)
    print("✂️  SPLITTING DATA")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✅ Data split complete!")
    print(f"   - Training set: {len(X_train):,} samples")
    print(f"   - Test set: {len(X_test):,} samples")
    print(f"\n📊 Training set distribution:")
    print(f"   - Legitimate: {sum(y_train==0):,} ({sum(y_train==0)/len(y_train)*100:.2f}%)")
    print(f"   - Fraud: {sum(y_train==1):,} ({sum(y_train==1)/len(y_train)*100:.4f}%)")
    
    return X_train, X_test, y_train, y_test

# --------------------------------------------------
# 4. APPLY SMOTE
# --------------------------------------------------
def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("\n" + "="*60)
    print("🔄 APPLYING SMOTE OVERSAMPLING")
    print("="*60)
    
    print("\n⚠️  Class imbalance detected! Applying SMOTE...")
    print(f"   Before SMOTE:")
    print(f"   - Legitimate: {sum(y_train==0):,}")
    print(f"   - Fraud: {sum(y_train==1):,}")
    
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    print(f"\n✅ SMOTE applied successfully!")
    print(f"   After SMOTE:")
    print(f"   - Legitimate: {sum(y_train_res==0):,}")
    print(f"   - Fraud: {sum(y_train_res==1):,}")
    print(f"   - Total samples: {len(X_train_res):,}")
    
    return X_train_res, y_train_res, sm

# --------------------------------------------------
# 5. TRAIN MODELS
# --------------------------------------------------
def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and compare performance"""
    print("\n" + "="*60)
    print("🤖 TRAINING MODELS")
    print("="*60)
    
    models = {}
    results = []
    
    # 1. Logistic Regression
    print("\n📊 Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        C=1.0
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Random Forest
    print("📊 Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # 3. XGBoost (if available)
    try:
        from xgboost import XGBClassifier
        print("📊 Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb
    except ImportError:
        print("⚠️  XGBoost not installed, skipping...")
    
    # Evaluate all models
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION RESULTS")
    print("="*60)
    
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"🔍 {name}")
        print('='*50)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"📈 Performance Metrics:")
        print(f"   - Accuracy:  {accuracy:.4f}")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall:    {recall:.4f}")
        print(f"   - F1-Score:  {f1:.4f}")
        print(f"   - ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        print(f"\n📊 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"   TN: {cm[0][0]:,}  FP: {cm[0][1]:,}")
        print(f"   FN: {cm[1][0]:,}  TP: {cm[1][1]:,}")
        
        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = (name, model)
    
    return models, best_model, results

# --------------------------------------------------
# 6. VISUALIZE RESULTS
# --------------------------------------------------
def visualize_results(models, X_test, y_test):
    """Create visualizations for model performance"""
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        # ROC Curves
        plt.figure(figsize=(10, 8))
        
        for name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Credit Card Fraud Detection')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✅ ROC curves saved as 'roc_curves.png'")
        plt.close()
        
        # Feature Importance (Random Forest)
        if 'Random Forest' in models:
            rf = models['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': X_test.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            top_features = feature_importance.head(15)
            plt.barh(top_features['feature'], top_features['importance'])
            plt.xlabel('Feature Importance')
            plt.title('Top 15 Features - Random Forest')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
            print("✅ Feature importance saved as 'feature_importance.png'")
            plt.close()
            
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

# --------------------------------------------------
# 7. SAVE MODELS AND RESULTS
# --------------------------------------------------
def save_models(best_model, scaler, results):
    """Save the best model, scaler, and results"""
    print("\n" + "="*60)
    print("💾 SAVING MODELS AND RESULTS")
    print("="*60)
    
    # Save best model
    name, model = best_model
    model_filename = f"fraud_model_{name.replace(' ', '_').lower()}.pkl"
    joblib.dump(model, model_filename)
    print(f"✅ Best model saved as '{model_filename}'")
    
    # Save scaler
    joblib.dump(scaler, "scaler.pkl")
    print(f"✅ Scaler saved as 'scaler.pkl'")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    results_df.to_csv("model_performance.csv", index=False)
    print(f"✅ Model performance saved as 'model_performance.csv'")
    
    # Print summary
    print(f"\n🏆 BEST MODEL: {name}")
    print(f"   - F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")
    print(f"   - Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# --------------------------------------------------
# 8. EXPORT PREDICTIONS FOR POWER BI
# --------------------------------------------------
def export_predictions(model, X_test, y_test, scaler):
    """Export predictions for Power BI visualization"""
    print("\n" + "="*60)
    print("📤 EXPORTING PREDICTIONS FOR POWER BI")
    print("="*60)
    
    # Create predictions dataframe
    df_predictions = X_test.copy()
    df_predictions["Actual"] = y_test
    df_predictions["Predicted"] = model.predict(X_test)
    df_predictions["Probability"] = model.predict_proba(X_test)[:, 1]
    df_predictions["Prediction_Status"] = df_predictions["Predicted"].map({
        0: "Legitimate", 
        1: "Fraud"
    })
    df_predictions["Actual_Status"] = df_predictions["Actual"].map({
        0: "Legitimate", 
        1: "Fraud"
    })
    
    # Add transaction ID for tracking
    df_predictions["Transaction_ID"] = range(1, len(df_predictions) + 1)
    
    # Reorder columns for better readability
    cols = ["Transaction_ID"] + [col for col in df_predictions.columns if col not in ["Transaction_ID"]]
    df_predictions = df_predictions[cols]
    
    # Save to CSV
    filename = f"fraud_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_predictions.to_csv(filename, index=False)
    print(f"✅ Predictions exported to '{filename}'")
    
    # Also save as fixed name for Power BI
    df_predictions.to_csv("fraud_predictions_for_powerBI.csv", index=False)
    print(f"✅ Predictions saved as 'fraud_predictions_for_powerBI.csv' for Power BI")
    
    # Summary statistics
    print(f"\n📊 Prediction Summary:")
    print(f"   - Total predictions: {len(df_predictions):,}")
    print(f"   - Predicted Fraud: {sum(df_predictions['Predicted']==1):,}")
    print(f"   - Actual Fraud: {sum(df_predictions['Actual']==1):,}")
    print(f"   - Correctly identified fraud: {sum((df_predictions['Actual']==1) & (df_predictions['Predicted']==1)):,}")
    
    return df_predictions

# --------------------------------------------------
# 9. MAIN EXECUTION
# --------------------------------------------------
def main():
    """Main execution function"""
    start_time = datetime.now()
    
    # Check dataset
    if not check_dataset():
        sys.exit(1)
    
    try:
        # Load and explore data
        df = load_data()
        
        # Preprocess data
        X, y, scaler = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Apply SMOTE
        X_train_res, y_train_res, sm = apply_smote(X_train, y_train)
        
        # Train models
        models, best_model, results = train_models(X_train_res, y_train_res, X_test, y_test)
        
        # Visualize results
        visualize_results(models, X_test, y_test)
        
        # Save models
        save_models(best_model, scaler, results)
        
        # Export predictions
        _, best_model_obj = best_model
        export_predictions(best_model_obj, X_test, y_test, scaler)
        
        # Final summary
        elapsed_time = datetime.now() - start_time
        print("\n" + "="*60)
        print("✅ PROCESS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"⏱️  Total execution time: {elapsed_time.total_seconds():.2f} seconds")
        print("\n📁 Generated Files:")
        print("   - model_performance.csv (model comparison)")
        print("   - fraud_predictions_for_powerBI.csv (Power BI ready)")
        print("   - roc_curves.png (visualization)")
        print("   - feature_importance.png (visualization)")
        print("   - *.pkl (model and scaler files)")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# --------------------------------------------------
# 10. SCRIPT EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    main()
