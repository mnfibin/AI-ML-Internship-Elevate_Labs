# Linear Regression for House Price Prediction
# Author: FIBIN MN
# Date: 17 November 2025

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 70)
print("LINEAR REGRESSION - HOUSE PRICE PREDICTION")
print("=" * 70)

# ============================================================================
# STEP 2: LOAD AND EXPLORE THE DATASET
# ============================================================================

# Load the dataset
df = pd.read_csv('Housing.csv')

print("\n1. DATASET OVERVIEW")
print("-" * 70)
print(f"Dataset Shape: {df.shape}")
print(f"Number of Records: {df.shape[0]}")
print(f"Number of Features: {df.shape[1]}")

print("\n2. FIRST FEW ROWS")
print("-" * 70)
print(df.head())

print("\n3. DATASET INFORMATION")
print("-" * 70)
print(df.info())

print("\n4. STATISTICAL SUMMARY")
print("-" * 70)
print(df.describe())

print("\n5. MISSING VALUES CHECK")
print("-" * 70)
print(df.isnull().sum())

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Create a copy for preprocessing
data = df.copy()

# Encode categorical variables
print("\n6. ENCODING CATEGORICAL VARIABLES")
print("-" * 70)

# List of categorical columns
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                   'airconditioning', 'prefarea', 'furnishingstatus']

# Encode binary variables (yes/no) to 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
               'airconditioning', 'prefarea']

for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})
    print(f"{col}: yes→1, no→0")

# Encode furnishing status (ordinal encoding)
furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
data['furnishingstatus'] = data['furnishingstatus'].map(furnishing_map)
print(f"furnishingstatus: unfurnished→0, semi-furnished→1, furnished→2")

print("\n7. PREPROCESSED DATA")
print("-" * 70)
print(data.head())

# ============================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Correlation matrix
print("\n8. CORRELATION WITH PRICE")
print("-" * 70)
correlations = data.corr()['price'].sort_values(ascending=False)
print(correlations)

# Visualize correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of All Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Distribution of target variable
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(data['price'], bins=50, edgecolor='black', color='skyblue')
plt.xlabel('Price', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of House Prices', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(data['price'], vert=True)
plt.ylabel('Price', fontsize=12)
plt.title('Boxplot of House Prices', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('price_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 5: PREPARE FEATURES AND TARGET
# ============================================================================

print("\n" + "=" * 70)
print("FEATURE AND TARGET PREPARATION")
print("=" * 70)

# Separate features (X) and target (y)
X = data.drop('price', axis=1)
y = data['price']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"\nFeature columns: {list(X.columns)}")

# ============================================================================
# STEP 6: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================================

print("\n9. TRAIN-TEST SPLIT")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# ============================================================================
# STEP 7: BUILD AND TRAIN LINEAR REGRESSION MODEL
# ============================================================================

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
print("\n10. Training Linear Regression model...")
model.fit(X_train, y_train)
print("✓ Model training complete!")

# Display model parameters
print("\n11. MODEL COEFFICIENTS")
print("-" * 70)
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print(coefficients.to_string(index=False))
print(f"\nIntercept: {model.intercept_:,.2f}")

# Visualize coefficients
plt.figure(figsize=(12, 6))
colors = ['green' if x > 0 else 'red' for x in coefficients['Coefficient']]
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors, alpha=0.7, edgecolor='black')
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Coefficients in Linear Regression Model', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 8: MAKE PREDICTIONS
# ============================================================================

print("\n" + "=" * 70)
print("PREDICTIONS")
print("=" * 70)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n12. SAMPLE PREDICTIONS vs ACTUAL VALUES")
print("-" * 70)
comparison = pd.DataFrame({
    'Actual Price': y_test.values[:10],
    'Predicted Price': y_test_pred[:10],
    'Difference': y_test.values[:10] - y_test_pred[:10]
})
print(comparison.to_string(index=False))

# ============================================================================
# STEP 9: MODEL EVALUATION
# ============================================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

# Calculate metrics for training set
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(y_train, y_train_pred)

# Calculate metrics for test set
test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(y_test, y_test_pred)

print("\n13. TRAINING SET METRICS")
print("-" * 70)
print(f"Mean Absolute Error (MAE):  ₹{train_mae:,.2f}")
print(f"Mean Squared Error (MSE):   {train_mse:,.2f}")
print(f"Root Mean Squared Error:    ₹{train_rmse:,.2f}")
print(f"R² Score:                   {train_r2:.4f} ({train_r2*100:.2f}%)")

print("\n14. TEST SET METRICS")
print("-" * 70)
print(f"Mean Absolute Error (MAE):  ₹{test_mae:,.2f}")
print(f"Mean Squared Error (MSE):   {test_mse:,.2f}")
print(f"Root Mean Squared Error:    ₹{test_rmse:,.2f}")
print(f"R² Score:                   {test_r2:.4f} ({test_r2*100:.2f}%)")

# Create metrics comparison table
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R² Score'],
    'Training Set': [train_mae, train_mse, train_rmse, train_r2],
    'Test Set': [test_mae, test_mse, test_rmse, test_r2]
})

print("\n15. METRICS COMPARISON")
print("-" * 70)
print(metrics_df.to_string(index=False))

# ============================================================================
# STEP 10: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# 1. Actual vs Predicted (Training Set)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title(f'Training Set: Actual vs Predicted\nR² = {train_r2:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, color='orange', edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title(f'Test Set: Actual vs Predicted\nR² = {test_r2:.4f}', 
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Residual Plot
plt.figure(figsize=(14, 6))

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

plt.subplot(1, 2, 1)
plt.scatter(y_train_pred, train_residuals, alpha=0.5, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot - Training Set', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, test_residuals, alpha=0.5, color='orange', edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Price', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residual Plot - Test Set', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Distribution of Residuals
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.hist(train_residuals, bins=50, edgecolor='black', color='skyblue', alpha=0.7)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Residuals - Training Set', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(test_residuals, bins=50, edgecolor='black', color='orange', alpha=0.7)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Residuals - Test Set', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Feature Importance (based on absolute coefficients)
plt.figure(figsize=(12, 6))
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_)
}).sort_values('Importance', ascending=True)

plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
         color='teal', alpha=0.7, edgecolor='black')
plt.xlabel('Absolute Coefficient Value', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance (Absolute Coefficients)', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ All visualizations saved successfully!")

# ============================================================================
# STEP 11: MODEL INTERPRETATION
# ============================================================================

print("\n" + "=" * 70)
print("MODEL INTERPRETATION")
print("=" * 70)

print("\n16. KEY INSIGHTS")
print("-" * 70)
print(f"""
1. The model explains {test_r2*100:.2f}% of the variance in house prices (R² score)
2. On average, predictions are off by ₹{test_mae:,.0f} (MAE)
3. The root mean squared error is ₹{test_rmse:,.0f}

Top 3 Most Important Features:
""")

top_features = coefficients.head(3)
for idx, row in top_features.iterrows():
    impact = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"   • {row['Feature']}: Each unit increase {impact} price by ₹{abs(row['Coefficient']):,.2f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)