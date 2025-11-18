# -*- coding: utf-8 -*-
"""
Task 4: Classification with Logistic Regression
Objective: Build a binary classifier to predict Breast Cancer diagnosis (Malignant/Benign).

Dataset: Breast Cancer Wisconsin (Diagnostic) Data Set - Loaded from 'data.csv'
Author: FIBIN MN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, roc_auc_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ==========================================
# 1. Load and Explore Data
# ==========================================
def load_data(filepath):
    """
    Loads the dataset and performs basic cleaning and target encoding.
    """
    print("------------------------------------------------------------------")
    print("1. Data Loading and Cleaning")
    print("------------------------------------------------------------------")
    
    # Load data
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    # 'id' is just an identifier. 'Unnamed: 32' is often an empty artifact in this dataset.
    cols_to_drop = ['id', 'Unnamed: 32']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], axis=1)

    # Encode Target Variable: 'M' (Malignant) -> 1, 'B' (Benign) -> 0
    # This is essential for binary classification algorithms.
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    print(f"Data loaded successfully. Final Shape: {df.shape}")
    print("Target Variable ('diagnosis') Distribution:")
    print(df['diagnosis'].value_counts())
    print("\n")
    return df

# ==========================================
# 2. Preprocessing
# ==========================================
def preprocess_data(df):
    """
    Splits data into features/target, train/test sets, and scales features.
    """
    print("------------------------------------------------------------------")
    print("2. Data Preprocessing (Splitting and Scaling)")
    print("------------------------------------------------------------------")
    
    # Split Features (X) and Target (y)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    # Train/Test Split (80% Train, 20% Test)
    # Using stratify=y ensures both splits have the same proportion of target classes.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    
    # Standardization (Feature Scaling)
    # Logistic Regression relies on gradient descent and regularization,
    # making it sensitive to feature scaling. Standardizing ensures fair weighting.
    scaler = StandardScaler()
    
    # Fit scaler ONLY on the training data to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Apply the same transformation to the test data
    X_test_scaled = scaler.transform(X_test)
    
    print("Features standardized successfully.\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# ==========================================
# 3. Model Training
# ==========================================
def train_model(X_train, y_train):
    """
    Fits the Logistic Regression model.
    """
    print("------------------------------------------------------------------")
    print("3. Model Training")
    print("------------------------------------------------------------------")
    
    # Initialize and train the Logistic Regression model
    model = LogisticRegression(random_state=42, solver='liblinear') 
    # 'liblinear' is good for small datasets and L1/L2 regularization
    model.fit(X_train, y_train)
    print("Logistic Regression Model training complete.")
    
    return model

# ==========================================
# 4. Evaluation
# ==========================================
def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluates the model and prints key metrics.
    """
    print("------------------------------------------------------------------")
    print("4. Model Evaluation (Default Threshold = 0.5)")
    print("------------------------------------------------------------------")
    
    # Predictions (Class labels: 0 or 1)
    y_pred = model.predict(X_test)
    
    # Probability predictions (floats between 0 and 1)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 4.1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Benign (0)', 'Malignant (1)'])
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.cividis, values_format='d')
    plt.title("Confusion Matrix")
    plt.show()
    
    # 4.2. Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
    
    # 4.3. ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc_score:.4f})', 
             color='darkred', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal random classifier
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # 4.4. Feature Importance (Coefficients)
    print("\nModel Coefficients (Impact on Malignant outcome):")
    
    # Get the coefficients and corresponding feature names
    coefficients = model.coef_[0]
    feature_impact = pd.Series(coefficients, index=feature_names).sort_values(ascending=False)
    
    print("Top 5 Positive Predictors (Likely to be Malignant):")
    print(feature_impact.head(5))
    print("\nTop 5 Negative Predictors (Likely to be Benign):")
    print(feature_impact.tail(5))
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    feature_impact.plot(kind='barh', color=(feature_impact > 0).map({True: 'salmon', False: 'skyblue'}))
    plt.title('Feature Coefficients (Impact on Log-Odds of Malignancy)')
    plt.xlabel('Coefficient Value (Higher positive value = stronger Malignant predictor)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    return y_pred_proba

# ==========================================
# 5. Theoretical Concepts & Threshold Tuning
# ==========================================
def plot_sigmoid_function():
    """
    Visualizes the Sigmoid function (The core of Logistic Regression).
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    z = np.linspace(-10, 10, 100)
    phi_z = sigmoid(z)

    plt.figure(figsize=(8, 5))
    plt.plot(z, phi_z, color='darkgreen', lw=3)
    plt.axvline(0.0, color='gray', linestyle='--')
    plt.axhline(y=0.5, color='red', linestyle=':', label='Decision Boundary (Threshold)')
    plt.yticks([0.0, 0.5, 1.0])
    plt.ylim(-0.05, 1.05)
    plt.title('Sigmoid Function $\phi (z)$')
    plt.xlabel('z (Linear combination of features: $w_1x_1 + w_2x_2 + ... + b$)')
    plt.ylabel('Probability $\phi (z)$')
    plt.legend()
    plt.grid(True, linestyle='dotted', alpha=0.6)
    plt.show()

def threshold_tuning_demo(y_test, y_pred_proba):
    """
    Demonstrates how changing the decision threshold affects Precision and Recall.
    """
    print("------------------------------------------------------------------")
    print("5. Threshold Tuning Demonstration")
    print("------------------------------------------------------------------")
    
    plot_sigmoid_function()
    
    # New, conservative threshold: maximizes precision (fewer false alarms)
    high_threshold = 0.85 
    y_pred_high = (y_pred_proba >= high_threshold).astype(int)
    
    print(f"\n--- Analysis with Conservative Threshold = {high_threshold} (High Precision Focus) ---")
    print("We only classify as Malignant if probability is > 85%.")
    print(classification_report(y_test, y_pred_high, target_names=['Benign', 'Malignant']))
    
    # New, sensitive threshold: maximizes recall (fewer missed cases)
    low_threshold = 0.25 
    y_pred_low = (y_pred_proba >= low_threshold).astype(int)
    
    print(f"\n--- Analysis with Sensitive Threshold = {low_threshold} (High Recall Focus) ---")
    print("We classify as Malignant if probability is > 25%.")
    print(classification_report(y_test, y_pred_low, target_names=['Benign', 'Malignant']))
    print("------------------------------------------------------------------")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # --- Configuration ---
    FILE_PATH = 'data.csv' 
    
    try:
        # 1. Load and Clean Data
        df = load_data(FILE_PATH)
        
        # 2. Preprocessing: Split and Scale
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        
        # 3. Train Model
        model = train_model(X_train, y_train)
        
        # 4. Evaluate Model (Default 0.5 Threshold)
        y_pred_proba = evaluate_model(model, X_test, y_test, feature_names)
        
        # 5. Explore Theory and Tuning
        threshold_tuning_demo(y_test, y_pred_proba)
        
    except FileNotFoundError:
        print("\n\nError: 'data.csv' not found. Please ensure the file is uploaded to the Colab environment or the correct path is specified.")
    except Exception as e:
        print(f"\n\nAn unexpected error occurred during execution: {e}")