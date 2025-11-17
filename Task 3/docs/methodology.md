# Detailed Methodology - Linear Regression for House Price Prediction

## Table of Contents
1. [Introduction to Linear Regression](#introduction)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Steps](#implementation-steps)
4. [Feature Engineering](#feature-engineering)
5. [Model Assumptions](#model-assumptions)
6. [Evaluation Metrics Explained](#evaluation-metrics)
7. [Interpretation Guide](#interpretation-guide)

---

## 1. Introduction to Linear Regression

### What is Linear Regression?

Linear regression is a supervised machine learning algorithm used to predict a continuous target variable based on one or more input features. It assumes a linear relationship between the input variables (X) and the output variable (y).

### Types of Linear Regression

**Simple Linear Regression**: One independent variable
```
y = β₀ + β₁x + ε
```

**Multiple Linear Regression**: Multiple independent variables
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```

Where:
- y = target variable (price)
- β₀ = intercept (baseline price)
- β₁, β₂, ..., βₙ = coefficients (feature weights)
- x₁, x₂, ..., xₙ = independent variables (features)
- ε = error term

---

## 2. Mathematical Foundation

### Objective Function

Linear regression minimizes the **Residual Sum of Squares (RSS)**:

```
RSS = Σ(yᵢ - ŷᵢ)²
```

Where:
- yᵢ = actual value
- ŷᵢ = predicted value
- n = number of samples

### Ordinary Least Squares (OLS)

The optimal coefficients are found using OLS method:

```
β = (XᵀX)⁻¹Xᵀy
```

Where:
- X = feature matrix
- y = target vector
- β = coefficient vector

### Gradient Descent (Alternative Method)

Iteratively update coefficients:

```
β := β - α ∂(Cost Function)/∂β
```

Where α is the learning rate.

---

## 3. Implementation Steps

### Step 1: Data Collection
- Load housing dataset with 545 samples
- 13 features (6 numerical, 7 categorical)
- Target variable: house price

### Step 2: Data Exploration
```python
# Check basic statistics
df.describe()

# Check data types
df.info()

# Check missing values
df.isnull().sum()
```

### Step 3: Data Preprocessing

#### 3.1 Handling Missing Values
- Check for null values
- Impute or remove if necessary

#### 3.2 Encoding Categorical Variables

**Binary Encoding** (Yes/No → 1/0):
```python
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
               'airconditioning', 'prefarea']
for col in binary_cols:
    data[col] = data[col].map({'yes': 1, 'no': 0})
```

**Ordinal Encoding** (Ordered categories):
```python
furnishing_map = {
    'unfurnished': 0, 
    'semi-furnished': 1, 
    'furnished': 2
}
data['furnishingstatus'] = data['furnishingstatus'].map(furnishing_map)
```

**Why not One-Hot Encoding?**
- Binary features don't need one-hot encoding
- Ordinal encoding preserves the natural order in furnishing status
- Reduces dimensionality

### Step 4: Feature Selection

Analyze correlation with target variable:
```python
correlations = data.corr()['price'].sort_values(ascending=False)
```

**High Correlation Features** (in our dataset):
- Area: 0.53
- Bathrooms: 0.51
- Bedrooms: 0.37

### Step 5: Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**Why 80-20 split?**
- Standard practice
- Sufficient training data (436 samples)
- Adequate test data for evaluation (109 samples)

### Step 6: Model Training

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns:
- **Coefficients (β)**: Impact of each feature on price
- **Intercept (β₀)**: Baseline price when all features are zero

### Step 7: Prediction

```python
y_pred = model.predict(X_test)
```

For each house:
```
predicted_price = intercept + (coef₁ × area) + (coef₂ × bedrooms) + ...
```

---

## 4. Feature Engineering

### Feature Scaling

**Not Applied** in this project because:
- Linear regression coefficients adjust naturally
- All features are on reasonable scales
- Results are more interpretable without scaling

**When to scale:**
- Using regularization (Lasso, Ridge)
- Using gradient descent optimization
- Features have vastly different scales (e.g., 0-1 vs 0-100000)

### Feature Creation (Potential Improvements)

Could create new features:
```python
# Price per square foot
data['price_per_sqft'] = data['price'] / data['area']

# Total rooms
data['total_rooms'] = data['bedrooms'] + data['bathrooms']

# Luxury score
data['luxury_score'] = (data['basement'] + data['airconditioning'] + 
                        data['furnishingstatus'])
```

---

## 5. Model Assumptions

Linear regression assumes:

### 1. Linearity
**Assumption**: Linear relationship between features and target

**Check**: Scatter plots of features vs. price
```python
plt.scatter(X['area'], y)
```

### 2. Independence
**Assumption**: Observations are independent

**Check**: No autocorrelation in residuals

### 3. Homoscedasticity
**Assumption**: Constant variance of residuals

**Check**: Residual plot (should show random scatter)
```python
plt.scatter(y_pred, residuals)
```

### 4. Normality of Residuals
**Assumption**: Residuals are normally distributed

**Check**: Histogram of residuals
```python
plt.hist(residuals)
```

### 5. No Multicollinearity
**Assumption**: Features are not highly correlated

**Check**: Correlation matrix, VIF (Variance Inflation Factor)
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

---

## 6. Evaluation Metrics Explained

### Mean Absolute Error (MAE)

```python
MAE = (1/n) Σ|yᵢ - ŷᵢ|
```

**Interpretation**: Average absolute prediction error
- **Units**: Same as target variable (₹)
- **Range**: [0, ∞), lower is better
- **Pros**: Easy to interpret, not sensitive to outliers
- **Cons**: Doesn't penalize large errors heavily

**Example**: MAE = ₹873,221
- On average, predictions are off by ₹8.73 lakhs

### Mean Squared Error (MSE)

```python
MSE = (1/n) Σ(yᵢ - ŷᵢ)²
```

**Interpretation**: Average squared prediction error
- **Units**: Squared units of target (₹²)
- **Range**: [0, ∞), lower is better
- **Pros**: Differentiable, penalizes large errors
- **Cons**: Sensitive to outliers, not intuitive

### Root Mean Squared Error (RMSE)

```python
RMSE = √MSE
```

**Interpretation**: Square root of MSE
- **Units**: Same as target variable (₹)
- **Range**: [0, ∞), lower is better
- **Pros**: Same units as target, penalizes large errors
- **Cons**: Sensitive to outliers

**Example**: RMSE = ₹1,239,701
- Standard deviation of prediction errors

### R² Score (Coefficient of Determination)

```python
R² = 1 - (SS_res / SS_tot)

Where:
SS_res = Σ(yᵢ - ŷᵢ)²  (Residual Sum of Squares)
SS_tot = Σ(yᵢ - ȳ)²   (Total Sum of Squares)
```

**Interpretation**: Proportion of variance explained
- **Range**: (-∞, 1], higher is better
- **Perfect model**: R² = 1 (100% variance explained)
- **Baseline model**: R² = 0 (predicting mean)
- **Worse than baseline**: R² < 0

**Example**: R² = 0.6506
- Model explains 65.06% of price variance
- Remaining 34.94% is unexplained variance

**Rule of Thumb**:
- R² > 0.7: Good model
- 0.5 < R² < 0.7: Moderate model
- R² < 0.5: Weak model

---

## 7. Interpretation Guide

### Interpreting Coefficients

Each coefficient represents the change in target for a unit change in feature, **holding all other features constant**.

**Example from our model**:

```
Price = 1,092,263.67 + 393.93(area) + 698,285.36(bathrooms) - 235,668.07(bedrooms) + ...
```

**Interpretation**:
1. **Intercept (₹10.9 lakhs)**: Base price when all features are zero (theoretical)

2. **Area coefficient (+₹393.93)**: 
   - Each additional square foot increases price by ₹393.93
   - A 1000 sq ft increase → ₹3.94 lakhs increase

3. **Bathrooms coefficient (+₹698,285)**: 
   - Each additional bathroom increases price by ₹6.98 lakhs
   - Holding area constant

4. **Bedrooms coefficient (-₹235,668)**: 
   - Surprising negative coefficient!
   - Possible reasons:
     - Multicollinearity with area
     - Smaller bedrooms mean same area → lower price/sqft
     - More bedrooms = older/cheaper houses in dataset

### Feature Importance

**Absolute coefficient values** indicate importance:
- Larger absolute value = stronger impact
- Sign (+ or -) indicates direction

### Multicollinearity Effects

When features are correlated:
- Coefficients become unreliable
- Signs may be counterintuitive
- Standard errors increase

**Detection**:
```python
# Correlation matrix
data.corr()

# Variance Inflation Factor (VIF)
VIF > 10 indicates multicollinearity
```

**Solution**:
- Remove one of the correlated features
- Use regularization (Ridge/Lasso)
- Principal Component Analysis (PCA)

### Residual Analysis

**Good Model**:
- Residuals randomly scattered around zero
- No patterns in residual plot
- Normally distributed residuals

**Problems**:
- **Non-random pattern**: Non-linear relationship (use polynomial regression)
- **Funnel shape**: Heteroscedasticity (use weighted least squares)
- **Outliers**: Investigate and possibly remove

---

## Best Practices

### 1. Data Quality
- Handle missing values appropriately
- Remove or investigate outliers
- Check for data entry errors

### 2. Feature Engineering
- Create meaningful derived features
- Encode categorical variables properly
- Consider interaction terms

### 3. Model Validation
- Use train-test split or cross-validation
- Check assumptions (linearity, homoscedasticity)
- Analyze residuals

### 4. Model Improvement
- Try polynomial features for non-linear relationships
- Use regularization to prevent overfitting
- Consider ensemble methods if linear regression underperforms

### 5. Documentation
- Record all preprocessing steps
- Document feature engineering decisions
- Explain model limitations

---

## Limitations of Linear Regression

1. **Assumes Linearity**: Real-world relationships often non-linear
2. **Sensitive to Outliers**: Extreme values heavily influence model
3. **Multicollinearity Issues**: Correlated features cause problems
4. **No Automatic Feature Selection**: Manual feature engineering needed
5. **Cannot Capture Complex Patterns**: Limited modeling capacity

---

## When to Use Linear Regression

**Good For**:
- Linear relationships
- Interpretability is important
- Simple baseline model
- Small to medium datasets

**Not Recommended For**:
- Highly non-linear relationships
- Complex interactions
- High-dimensional data (without regularization)
- When only prediction accuracy matters (use ensemble methods)

---

## Further Improvements

1. **Polynomial Regression**: Capture non-linear relationships
2. **Regularization**: Ridge/Lasso for feature selection
3. **Feature Scaling**: StandardScaler or MinMaxScaler
4. **Cross-Validation**: K-fold for robust evaluation
5. **Ensemble Methods**: Random Forest, Gradient Boosting for better accuracy

---

*This methodology document provides a comprehensive understanding of the linear regression implementation. For questions or suggestions, please open an issue on GitHub.*
