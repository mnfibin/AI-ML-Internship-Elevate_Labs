# 🚢 Titanic Dataset - Exploratory Data Analysis (EDA)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-green.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-yellow.svg)](https://seaborn.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0%2B-purple.svg)](https://plotly.com/)


A complete, production-ready exploratory data analysis of the Titanic dataset, demonstrating statistical analysis, data visualization, and insight extraction using industry-standard Python libraries.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Analysis Workflow](#-analysis-workflow)
- [Key Insights](#-key-insights)
- [Visualizations](#-visualizations)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project performs a **comprehensive Exploratory Data Analysis (EDA)** on the famous Titanic dataset to uncover patterns, relationships, and insights that influenced passenger survival rates. The analysis follows industry best practices and demonstrates proficiency in data science fundamentals.

### **Objective**
To understand the Titanic dataset through:
- Statistical analysis
- Data visualization
- Pattern recognition
- Feature relationship exploration
- Anomaly detection

### **Dataset Description**
The Titanic dataset contains information about passengers aboard the RMS Titanic, including:
- **891 passengers** (rows)
- **12 features** (columns)
- Mix of numerical and categorical variables
- Target variable: `Survived` (0 = No, 1 = Yes)

**Key Features:**
| Feature | Description | Type |
|---------|-------------|------|
| PassengerId | Unique identifier | Integer |
| Survived | Survival status (0/1) | Integer |
| Pclass | Passenger class (1/2/3) | Integer |
| Name | Passenger name | String |
| Sex | Gender (male/female) | String |
| Age | Age in years | Float |
| SibSp | Number of siblings/spouses aboard | Integer |
| Parch | Number of parents/children aboard | Integer |
| Ticket | Ticket number | String |
| Fare | Passenger fare | Float |
| Cabin | Cabin number | String |
| Embarked | Port of embarkation (C/Q/S) | String |

---

## ✨ Features

### 🔍 **Comprehensive Analysis**
- ✅ Summary statistics for all features
- ✅ Missing value analysis and visualization
- ✅ Distribution analysis (histograms, KDE plots)
- ✅ Outlier detection using boxplots
- ✅ Skewness and kurtosis calculations
- ✅ Correlation matrix and heatmaps
- ✅ Bivariate and multivariate analysis
- ✅ Interactive visualizations with Plotly

### 📊 **20+ Visualizations**
- Histograms with mean/median lines
- Boxplots for outlier detection
- Count plots for categorical features
- Survival rate analysis by demographics
- Correlation heatmaps
- Pairplots for feature relationships
- 3D scatter plots
- Interactive dashboards

### 📈 **Statistical Insights**
- Central tendency measures (mean, median, mode)
- Dispersion measures (std, variance, range)
- Distribution analysis (skewness, kurtosis)
- Correlation coefficients
- Cross-tabulation analysis

### 🎨 **Professional Visualization**
- Clean, publication-ready plots
- Consistent color schemes
- Annotated charts with insights
- Interactive Plotly dashboards
- Comprehensive legends and labels

---

## 🛠 Technologies Used

### **Core Libraries**
```python
pandas==1.3.0+          # Data manipulation and analysis
numpy==1.21.0+          # Numerical computing
matplotlib==3.4.0+      # Static visualizations
seaborn==0.11.0+        # Statistical visualizations
plotly==5.0.0+          # Interactive visualizations
```

### **Environment**
- **Python**: 3.8+
- **Platform**: Google Colab / Jupyter Notebook
- **IDE**: Any Python IDE (VS Code, PyCharm, etc.)

---

## 📥 Installation

### **Option 1: Google Colab (Recommended)**
1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. All libraries are pre-installed! Just run the code.

### **Option 2: Local Installation**

#### **Step 1: Clone the Repository**
```bash
git clone https://github.com/yourusername/titanic-eda.git
cd titanic-eda
```

#### **Step 2: Create Virtual Environment**
```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

#### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

#### **Step 4: Launch Jupyter Notebook**
```bash
jupyter notebook
```

---

## 🚀 Usage

### **Quick Start (Google Colab)**

1. **Upload the Code**
   ```python
   # Copy the entire Python script into a Colab cell
   ```

2. **Upload Dataset**
   ```python
   # The script will prompt you to upload Titanic-Dataset.csv
   # Click "Choose Files" and select your CSV file
   ```

3. **Run Analysis**
   ```python
   # Execute the cell - the complete analysis will run automatically
   ```

### **Step-by-Step Execution**

```python
# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# 2. Load Data
df = pd.read_csv('Titanic-Dataset.csv')

# 3. Explore Data
print(df.head())
print(df.info())
print(df.describe())

# 4. Run Complete Analysis
# Execute the provided script for comprehensive EDA
```

### **Expected Output**
- Console output with statistical summaries
- 20+ visualization plots
- Key insights and patterns
- Data quality report
- Correlation analysis
- Actionable recommendations

---

## 📁 Project Structure

```
Task 2/
│
├── data/
│   └── Titanic-Dataset.csv          # Dataset file
│
├── notebooks/
│   └── Titanic_EDA.ipynb            # Jupyter notebook with analysis
│
├── src/
│   └── titanic_eda.py               # Main Python script
│
├── visualizations/
│   ├── distributions/               # Distribution plots
│   ├── correlations/                # Correlation heatmaps
│   ├── survival_analysis/           # Survival rate plots
│   └── interactive/                 # Plotly visualizations
│
├── reports/
│   └── EDA_Report.pdf               # Detailed analysis report
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file

```

---

## 🔄 Analysis Workflow

### **Phase 1: Data Loading & Initial Exploration**
```
1. Import libraries
2. Load dataset
3. Display first/last rows
4. Check data types and shape
5. Generate dataset info
```

### **Phase 2: Data Quality Assessment**
```
1. Identify missing values
2. Calculate missing percentages
3. Visualize missing data
4. Check for duplicates
5. Assess data integrity
```

### **Phase 3: Summary Statistics**
```
1. Generate descriptive statistics
2. Calculate central tendency (mean, median, mode)
3. Measure dispersion (std, variance, IQR)
4. Analyze distributions
5. Identify outliers
```

### **Phase 4: Univariate Analysis**
```
Numerical Features:
- Histograms with KDE
- Boxplots for outliers
- Skewness analysis
- Distribution patterns

Categorical Features:
- Count plots
- Frequency tables
- Percentage distributions
- Category balance
```

### **Phase 5: Bivariate Analysis**
```
1. Survival rate by class
2. Survival rate by gender
3. Survival rate by age groups
4. Survival rate by embarkation port
5. Cross-tabulation analysis
6. Grouped bar charts
```

### **Phase 6: Multivariate Analysis**
```
1. Correlation matrix
2. Correlation heatmap
3. Pairplots
4. 3D scatter plots
5. Feature interactions
6. Multicollinearity detection
```

### **Phase 7: Advanced Visualization**
```
1. Interactive Plotly charts
2. Animated visualizations
3. Dashboard creation
4. Comprehensive reporting
```

### **Phase 8: Insight Extraction**
```
1. Pattern identification
2. Trend analysis
3. Anomaly detection
4. Key findings summary
5. Actionable recommendations
```

---

## 💡 Key Insights

### **1. Survival Statistics**
- **Overall Survival Rate**: 38.38%
- **Female Survival Rate**: 74.20%
- **Male Survival Rate**: 18.89%
- **Key Finding**: Women had ~4x higher survival rate than men

### **2. Class-Based Analysis**
| Class | Survival Rate | Average Fare |
|-------|---------------|--------------|
| 1st Class | 62.96% | $84.15 |
| 2nd Class | 47.28% | $20.66 |
| 3rd Class | 24.24% | $13.68 |

**Insight**: First-class passengers had 2.6x better survival odds than third-class

### **3. Age Demographics**
- **Average Age (Survivors)**: 28.34 years
- **Average Age (Non-survivors)**: 30.63 years
- **Children (<18) Survival Rate**: 53.98%
- **Insight**: Children had better survival rates

### **4. Family Size Impact**
- **Optimal Family Size**: 2-4 members (best survival rates)
- **Solo Travelers**: 30.35% survival rate
- **Large Families (5+)**: Decreased survival rates
- **Insight**: Small families had survival advantage

### **5. Embarkation Port**
| Port | Code | Survival Rate |
|------|------|---------------|
| Cherbourg | C | 55.36% |
| Queenstown | Q | 38.96% |
| Southampton | S | 33.70% |

**Insight**: Cherbourg passengers (wealthier, 1st class) had highest survival

### **6. Fare Analysis**
- **Correlation with Survival**: +0.257 (positive)
- **Average Fare (Survivors)**: $48.40
- **Average Fare (Non-survivors)**: $22.12
- **Insight**: Higher fare = higher survival probability

### **7. Correlation Findings**
**Strong Correlations:**
- Pclass & Fare: -0.549 (negative - expected)
- SibSp & Parch: +0.415 (family size related)
- Pclass & Survived: -0.338 (lower class = lower survival)

**Weak Correlations:**
- Age & Survived: -0.077 (minimal impact)
- Fare & Age: +0.096 (not strongly related)

### **8. Data Quality Issues**
- **Age**: 177 missing values (19.87%)
- **Cabin**: 687 missing values (77.10%)
- **Embarked**: 2 missing values (0.22%)
- **Action Needed**: Imputation or feature engineering

### **9. Outlier Detection**
- **Fare**: Extreme outliers detected (max: $512.33)
- **Age**: Few outliers in 65-80 range
- **SibSp/Parch**: Outliers in large families
- **Recommendation**: Consider capping or log transformation

### **10. Distribution Patterns**
- **Age**: Slightly right-skewed (skewness: ~0.39)
- **Fare**: Heavily right-skewed (skewness: ~4.79)
- **Survival**: Imbalanced (38% survived, 62% died)
- **Action**: Consider resampling techniques for modeling

---

## 📊 Visualizations

### **Distribution Analysis**
![Distributions](https://via.placeholder.com/800x400?text=Distribution+Plots)
- Histograms for all numerical features
- KDE plots overlaid
- Mean and median lines
- Outlier identification

### **Survival Analysis**
![Survival](https://via.placeholder.com/800x400?text=Survival+Analysis)
- Survival rates by demographics
- Gender-based analysis
- Class-based comparison
- Age group distributions

### **Correlation Matrix**
![Correlation](https://via.placeholder.com/800x400?text=Correlation+Heatmap)
- Feature relationships
- Multicollinearity detection
- Survival correlations
- Color-coded intensity

### **Interactive Dashboards**
![Dashboard](https://via.placeholder.com/800x400?text=Interactive+Dashboard)
- Plotly 3D visualizations
- Dynamic filtering
- Hover information
- Real-time updates

---

## 🚀 Future Enhancements

### **Phase 1: Advanced Analysis**
- [ ] Time-series analysis (if temporal data available)
- [ ] Geospatial analysis of embarkation ports
- [ ] Text mining on passenger names/titles
- [ ] Network analysis of family relationships
- [ ] Survival prediction modeling

### **Phase 2: Feature Engineering**
- [ ] Title extraction from names (Mr., Mrs., Master, etc.)
- [ ] Deck extraction from cabin numbers
- [ ] Family size categories
- [ ] Fare per person calculation
- [ ] Age group binning strategies

### **Phase 3: Advanced Visualization**
- [ ] Animated timeline visualizations
- [ ] Interactive Dash/Streamlit dashboard
- [ ] 3D visualizations with Three.js
- [ ] Sankey diagrams for passenger flow
- [ ] Word clouds from text features

### **Phase 4: Automation**
- [ ] Automated EDA report generation
- [ ] Scheduled analysis pipeline
- [ ] API for real-time insights
- [ ] CI/CD integration
- [ ] Docker containerization

### **Phase 5: Machine Learning**
- [ ] Build predictive models
- [ ] Ensemble methods comparison
- [ ] Deep learning approaches
- [ ] Model interpretability (SHAP, LIME)
- [ ] Deployment pipeline

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### **How to Contribute**

1. **Fork the repository**
```bash
git clone https://github.com/yourusername/titanic-eda.git
```

2. **Create a feature branch**
```bash
git checkout -b feature/AmazingFeature
```

3. **Make your changes**
```bash
# Add your improvements
git add .
```

4. **Commit your changes**
```bash
git commit -m "Add: Amazing new feature"
```

5. **Push to the branch**
```bash
git push origin feature/AmazingFeature
```

6. **Open a Pull Request**

### **Contribution Guidelines**
- Follow PEP 8 style guide
- Add comments and documentation
- Include unit tests for new features
- Update README with new features
- Ensure all visualizations render correctly

### **Areas for Contribution**
- 🐛 Bug fixes
- 📊 New visualizations
- 📝 Documentation improvements
- ✨ Feature enhancements
- 🧪 Test coverage
- 🌐 Internationalization

---
