#!/usr/bin/env python3
"""
================================================================================
TITANIC DATASET - EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
File: titanic_eda.py
Author: FIBIN MN
Date: November 2025
Version: 1.0.0

Description:
    Comprehensive exploratory data analysis of the Titanic dataset including
    statistical analysis, visualization, pattern detection, and insight
    extraction using Pandas, Matplotlib, Seaborn, and Plotly.

Usage:
    python titanic_eda.py --input path/to/Titanic-Dataset.csv --output reports/

Dependencies:
    - pandas >= 1.3.0
    - numpy >= 1.21.0
    - matplotlib >= 3.4.0
    - seaborn >= 0.11.0
    - plotly >= 5.0.0

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import argparse
import warnings
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ==============================================================================
# CONSTANTS
# ==============================================================================
NUMERICAL_FEATURES = ['Age', 'SibSp', 'Parch', 'Fare']
CATEGORICAL_FEATURES = ['Survived', 'Pclass', 'Sex', 'Embarked']
TARGET = 'Survived'

# Color palettes
COLOR_SURVIVED = ['#e74c3c', '#2ecc71']  # Red for died, Green for survived
COLOR_PALETTE = 'viridis'

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def print_header(text: str, char: str = "=") -> None:
    """Print formatted section header."""
    print(f"\n{char * 80}")
    print(f"{text}")
    print(f"{char * 80}\n")


def print_subheader(text: str) -> None:
    """Print formatted subsection header."""
    print(f"\n{'─' * 80}")
    print(f"📊 {text}")
    print(f"{'─' * 80}")


def save_plot(filename: str, output_dir: str = "visualizations/") -> None:
    """Save current plot to file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filepath}")


def calculate_skewness_category(skew: float) -> str:
    """Categorize skewness value."""
    if abs(skew) < 0.5:
        return "Approximately Symmetric"
    elif skew > 0:
        return "Right Skewed (Positive Skew)"
    else:
        return "Left Skewed (Negative Skew)"


# ==============================================================================
# DATA LOADING AND VALIDATION
# ==============================================================================

class TitanicDataLoader:
    """Handle data loading and initial validation."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
    
    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        print_header("LOADING DATASET")
        
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        try:
            self.df = pd.read_csv(self.filepath)
            print(f"✓ Dataset loaded successfully from: {self.filepath}")
            print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def validate_data(self) -> Dict[str, bool]:
        """Validate dataset structure."""
        print_subheader("Data Validation")
        
        validation = {
            'has_target': TARGET in self.df.columns,
            'has_numerical': all(col in self.df.columns for col in NUMERICAL_FEATURES),
            'has_categorical': all(col in self.df.columns for col in CATEGORICAL_FEATURES),
            'no_duplicates': self.df.duplicated().sum() == 0
        }
        
        for check, passed in validation.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check.replace('_', ' ').title()}: {passed}")
        
        return validation


# ==============================================================================
# EXPLORATORY DATA ANALYSIS
# ==============================================================================

class TitanicEDA:
    """Comprehensive EDA for Titanic dataset."""
    
    def __init__(self, df: pd.DataFrame, output_dir: str = "visualizations/"):
        self.df = df
        self.output_dir = output_dir
        self.insights = []
        os.makedirs(output_dir, exist_ok=True)
    
    # ==========================================================================
    # 1. INITIAL EXPLORATION
    # ==========================================================================
    
    def initial_exploration(self) -> None:
        """Display initial dataset information."""
        print_header("INITIAL DATA EXPLORATION")
        
        print_subheader("First 5 Rows")
        print(self.df.head())
        
        print_subheader("Dataset Information")
        self.df.info()
        
        print_subheader("Dataset Shape")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        
        print_subheader("Column Names")
        print(self.df.columns.tolist())
    
    # ==========================================================================
    # 2. SUMMARY STATISTICS
    # ==========================================================================
    
    def summary_statistics(self) -> Dict:
        """Generate and display summary statistics."""
        print_header("SUMMARY STATISTICS")
        
        print_subheader("Numerical Features")
        print(self.df.describe())
        
        print_subheader("Categorical Features")
        print(self.df.describe(include='object'))
        
        # Survival statistics
        print_subheader("Survival Statistics")
        survival_rate = self.df[TARGET].mean() * 100
        survived_count = self.df[TARGET].sum()
        died_count = len(self.df) - survived_count
        
        stats = {
            'survival_rate': survival_rate,
            'survived_count': survived_count,
            'died_count': died_count
        }
        
        print(f"Overall Survival Rate: {survival_rate:.2f}%")
        print(f"Total Survived: {survived_count}")
        print(f"Total Died: {died_count}")
        
        self.insights.append(f"Overall survival rate: {survival_rate:.2f}%")
        
        return stats
    
    # ==========================================================================
    # 3. MISSING VALUES ANALYSIS
    # ==========================================================================
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """Analyze and visualize missing values."""
        print_header("MISSING VALUES ANALYSIS")
        
        # Calculate missing values
        missing = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df) * 100).round(2)
        })
        
        missing = missing[missing['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        if len(missing) > 0:
            print(missing)
            
            # Visualize
            plt.figure(figsize=(12, 6))
            sns.barplot(data=missing, x='Missing_Count', y='Column', palette='viridis')
            plt.title('Missing Values Count by Feature', fontsize=16, fontweight='bold')
            plt.xlabel('Number of Missing Values')
            plt.ylabel('Features')
            save_plot('missing_values.png', self.output_dir)
            plt.close()
            
            # Add to insights
            for _, row in missing.iterrows():
                self.insights.append(
                    f"{row['Column']}: {row['Missing_Count']} missing "
                    f"({row['Missing_Percentage']:.2f}%)"
                )
        else:
            print("✓ No missing values detected!")
        
        return missing
    
    # ==========================================================================
    # 4. UNIVARIATE ANALYSIS - NUMERICAL
    # ==========================================================================
    
    def analyze_numerical_features(self) -> None:
        """Analyze numerical features with distributions and boxplots."""
        print_header("UNIVARIATE ANALYSIS - NUMERICAL FEATURES")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = len(numerical_cols)
        
        # Distribution plots
        print_subheader("Distribution Analysis")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                data = self.df[col].dropna()
                axes[idx].hist(data, bins=30, color='steelblue', 
                              edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                
                # Add statistics
                mean_val = data.mean()
                median_val = data.median()
                axes[idx].axvline(mean_val, color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {mean_val:.2f}')
                axes[idx].axvline(median_val, color='green', linestyle='--', 
                                 linewidth=2, label=f'Median: {median_val:.2f}')
                axes[idx].legend()
        
        # Remove empty subplots
        for idx in range(n_cols, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        save_plot('numerical_distributions.png', self.output_dir)
        plt.close()
        
        # Boxplots for outlier detection
        print_subheader("Outlier Detection (Boxplots)")
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                sns.boxplot(data=self.df, y=col, ax=axes[idx], color='lightblue')
                axes[idx].set_title(f'Boxplot of {col}', 
                                   fontsize=12, fontweight='bold')
        
        for idx in range(n_cols, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        save_plot('numerical_boxplots.png', self.output_dir)
        plt.close()
        
        # Skewness analysis
        print_subheader("Skewness Analysis")
        for col in ['Age', 'Fare']:
            if col in self.df.columns:
                skew = self.df[col].skew()
                category = calculate_skewness_category(skew)
                print(f"{col}: {skew:.3f} - {category}")
                self.insights.append(f"{col} skewness: {skew:.3f} ({category})")
    
    # ==========================================================================
    # 5. UNIVARIATE ANALYSIS - CATEGORICAL
    # ==========================================================================
    
    def analyze_categorical_features(self) -> None:
        """Analyze categorical features with count plots."""
        print_header("UNIVARIATE ANALYSIS - CATEGORICAL FEATURES")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for idx, col in enumerate(CATEGORICAL_FEATURES):
            if col in self.df.columns and idx < len(axes):
                sns.countplot(data=self.df, x=col, ax=axes[idx], palette='Set2')
                axes[idx].set_title(f'Distribution of {col}', 
                                   fontsize=12, fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Count')
                
                # Add value labels
                for container in axes[idx].containers:
                    axes[idx].bar_label(container)
        
        plt.tight_layout()
        save_plot('categorical_distributions.png', self.output_dir)
        plt.close()
        
        # Print value counts
        print_subheader("Categorical Features Value Counts")
        for col in CATEGORICAL_FEATURES:
            if col in self.df.columns:
                print(f"\n{col}:")
                print(self.df[col].value_counts())
                print(f"\nPercentage:")
                print((self.df[col].value_counts(normalize=True) * 100).round(2))
    
    # ==========================================================================
    # 6. BIVARIATE ANALYSIS
    # ==========================================================================
    
    def analyze_survival_rates(self) -> Dict[str, pd.Series]:
        """Analyze survival rates across different features."""
        print_header("BIVARIATE ANALYSIS - SURVIVAL RATES")
        
        survival_rates = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Survival by Pclass
        survival_pclass = self.df.groupby('Pclass')[TARGET].mean()
        survival_rates['pclass'] = survival_pclass
        sns.barplot(x=survival_pclass.index, y=survival_pclass.values, 
                   ax=axes[0, 0], palette='coolwarm')
        axes[0, 0].set_title('Survival Rate by Passenger Class', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Survival Rate')
        for i, v in enumerate(survival_pclass.values):
            axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
        
        # Survival by Sex
        survival_sex = self.df.groupby('Sex')[TARGET].mean()
        survival_rates['sex'] = survival_sex
        sns.barplot(x=survival_sex.index, y=survival_sex.values, 
                   ax=axes[0, 1], palette='Set1')
        axes[0, 1].set_title('Survival Rate by Gender', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Survival Rate')
        for i, v in enumerate(survival_sex.values):
            axes[0, 1].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
        
        # Survival by Embarked
        survival_embarked = self.df.groupby('Embarked')[TARGET].mean()
        survival_rates['embarked'] = survival_embarked
        sns.barplot(x=survival_embarked.index, y=survival_embarked.values, 
                   ax=axes[1, 0], palette='viridis')
        axes[1, 0].set_title('Survival Rate by Port of Embarkation', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Survival Rate')
        for i, v in enumerate(survival_embarked.values):
            axes[1, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
        
        # Age distribution by Survival
        self.df[self.df[TARGET]==0]['Age'].hist(bins=30, ax=axes[1, 1], 
                                                alpha=0.6, label='Died', color='red')
        self.df[self.df[TARGET]==1]['Age'].hist(bins=30, ax=axes[1, 1], 
                                                alpha=0.6, label='Survived', color='green')
        axes[1, 1].set_title('Age Distribution by Survival', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Age')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        save_plot('survival_analysis.png', self.output_dir)
        plt.close()
        
        # Print insights
        print_subheader("Survival Rate Insights")
        print(f"Female survival rate: {survival_sex['female']:.2%}")
        print(f"Male survival rate: {survival_sex['male']:.2%}")
        print(f"1st Class survival: {survival_pclass[1]:.2%}")
        print(f"3rd Class survival: {survival_pclass[3]:.2%}")
        
        self.insights.append(f"Female survival: {survival_sex['female']:.2%} vs Male: {survival_sex['male']:.2%}")
        self.insights.append(f"1st Class survival: {survival_pclass[1]:.2%} vs 3rd Class: {survival_pclass[3]:.2%}")
        
        return survival_rates
    
    # ==========================================================================
    # 7. CORRELATION ANALYSIS
    # ==========================================================================
    
    def analyze_correlations(self) -> pd.DataFrame:
        """Analyze and visualize feature correlations."""
        print_header("CORRELATION ANALYSIS")
        
        # Select numerical columns
        corr_cols = [TARGET, 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        corr_df = self.df[corr_cols].dropna()
        
        # Calculate correlation matrix
        correlation_matrix = corr_df.corr()
        
        print_subheader("Correlation Matrix")
        print(correlation_matrix)
        
        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.3f', 
                   cmap='coolwarm', square=True, linewidths=1)
        plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
        save_plot('correlation_heatmap.png', self.output_dir)
        plt.close()
        
        # Correlation with survival
        print_subheader("Features Correlation with Survival")
        survival_corr = correlation_matrix[TARGET].sort_values(ascending=False)
        print(survival_corr)
        
        plt.figure(figsize=(10, 6))
        survival_corr.drop(TARGET).plot(kind='barh', color='teal')
        plt.title('Feature Correlation with Survival', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Correlation Coefficient')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        save_plot('survival_correlation.png', self.output_dir)
        plt.close()
        
        # Add insights
        top_corr = survival_corr.drop(TARGET).abs().sort_values(ascending=False).head(3)
        for feature, corr_val in top_corr.items():
            self.insights.append(f"{feature} correlation with survival: {corr_val:.3f}")
        
        return correlation_matrix
    
    # ==========================================================================
    # 8. MULTIVARIATE ANALYSIS
    # ==========================================================================
    
    def multivariate_analysis(self) -> None:
        """Perform multivariate analysis with pairplots."""
        print_header("MULTIVARIATE ANALYSIS")
        
        print_subheader("Generating Pairplot")
        pairplot_cols = [TARGET, 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        pairplot_df = self.df[pairplot_cols].dropna()
        
        pairplot = sns.pairplot(pairplot_df, hue=TARGET, palette='Set1', 
                               diag_kind='kde', plot_kws={'alpha': 0.6}, height=2.5)
        pairplot.fig.suptitle('Pairplot - Feature Relationships by Survival', 
                             y=1.01, fontsize=16, fontweight='bold')
        save_plot('pairplot.png', self.output_dir)
        plt.close()
        
        print("✓ Pairplot generated successfully")
    
    # ==========================================================================
    # 9. GENERATE INSIGHTS REPORT
    # ==========================================================================
    
    def generate_insights_report(self) -> str:
        """Generate comprehensive insights report."""
        print_header("KEY INSIGHTS AND PATTERNS")
        
        report = []
        report.append("="*80)
        report.append("TITANIC DATASET - KEY INSIGHTS")
        report.append("="*80)
        report.append("")
        
        # Add all collected insights
        for i, insight in enumerate(self.insights, 1):
            report.append(f"{i}. {insight}")
            print(f"  • {insight}")
        
        report.append("")
        report.append("="*80)
        report.append(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = os.path.join(self.output_dir, "insights_report.txt")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\n✓ Insights report saved to: {report_path}")
        
        return report_text
    
    # ==========================================================================
    # 10. RUN COMPLETE ANALYSIS
    # ==========================================================================
    
    def run_complete_analysis(self) -> None:
        """Execute complete EDA pipeline."""
        print_header("STARTING COMPREHENSIVE EDA")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            self.initial_exploration()
            self.summary_statistics()
            self.analyze_missing_values()
            self.analyze_numerical_features()
            self.analyze_categorical_features()
            self.analyze_survival_rates()
            self.analyze_correlations()
            self.multivariate_analysis()
            self.generate_insights_report()
            
            print_header("EDA COMPLETED SUCCESSFULLY!")
            print(f"✓ All visualizations saved to: {self.output_dir}")
            print(f"✓ Total insights generated: {len(self.insights)}")
            
        except Exception as e:
            print(f"\n✗ Error during analysis: {str(e)}")
            raise


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Titanic Dataset Exploratory Data Analysis'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='Titanic-Dataset.csv',
        help='Path to input CSV file (default: Titanic-Dataset.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations/',
        help='Output directory for visualizations (default: visualizations/)'
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # ASCII Art Header
    print("""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║                                                                          ║
    ║                    TITANIC DATASET - EDA ANALYSIS                        ║
    ║                                                                          ║
    ║                    🚢 RMS Titanic Data Explorer 🚢                       ║
    ║                                                                          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Load data
        loader = TitanicDataLoader(args.input)
        df = loader.load_data()
        loader.validate_data()
        
        # Run EDA
        eda = TitanicEDA(df, output_dir=args.output)
        eda.run_complete_analysis()
        
        print("\n" + "="*80)
        print("SUCCESS! Analysis complete. Check the output directory for results.")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: {str(e)}")
        print(f"{'='*80}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()