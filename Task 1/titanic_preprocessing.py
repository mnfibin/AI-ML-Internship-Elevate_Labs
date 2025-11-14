"""
Titanic Dataset: Data Cleaning & Preprocessing Pipeline
========================================================
Author: AIML Internship Task
Version: 1.0.0
Description: Modular preprocessing script for Titanic dataset

This script can be used as:
1. Standalone script: python titanic_preprocessing.py
2. Importable module: from titanic_preprocessing import preprocess_titanic
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os
from typing import Tuple, Dict, List

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'dataset_path': 'Titanic-Dataset.csv',
    'output_dir': 'outputs',
    'random_state': 42,
    'test_size': 0.2,
    'remove_outliers': True,
    'scaling_method': 'standard'  # 'standard' or 'minmax'
}


class TitanicPreprocessor:
    """
    A comprehensive preprocessing pipeline for the Titanic dataset.
    
    Attributes:
        df (pd.DataFrame): Original dataframe
        df_cleaned (pd.DataFrame): Cleaned dataframe
        scaler (StandardScaler): Fitted scaler object
        label_encoders (Dict): Dictionary of fitted label encoders
    """
    
    def __init__(self, filepath: str, verbose: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            filepath (str): Path to the Titanic CSV file
            verbose (bool): Whether to print progress messages
        """
        self.filepath = filepath
        self.verbose = verbose
        self.df = None
        self.df_cleaned = None
        self.scaler = None
        self.label_encoders = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(CONFIG['output_dir'], exist_ok=True)
        
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        self._print("\n" + "=" * 70)
        self._print("TITANIC DATASET: DATA CLEANING & PREPROCESSING")
        self._print("=" * 70)
        self._print(f"\n[STEP 1] Loading Dataset from {self.filepath}...")
        self._print("-" * 70)
        
        try:
            self.df = pd.read_csv(self.filepath)
            self._print(f"✓ Dataset loaded successfully!")
            self._print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            return self.df
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset not found at {self.filepath}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def explore_data(self):
        """Explore and display basic information about the dataset."""
        self._print("\n[STEP 2] Exploring Dataset...")
        self._print("-" * 70)
        
        if self.verbose:
            print("\nFirst 5 Rows:")
            print(self.df.head())
            print("\n\nDataset Info:")
            print(self.df.info())
            print("\n\nBasic Statistics:")
            print(self.df.describe())
    
    def analyze_missing_values(self) -> pd.DataFrame:
        """
        Analyze and visualize missing values in the dataset.
        
        Returns:
            pd.DataFrame: Summary of missing values
        """
        self._print("\n[STEP 3] Analyzing Missing Values...")
        self._print("-" * 70)
        
        missing_values = self.df.isnull().sum()
        missing_percent = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing_Count': missing_values.values,
            'Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Count', ascending=False
        )
        
        if not missing_df.empty:
            self._print("\nColumns with Missing Values:")
            self._print(missing_df.to_string(index=False))
            
            # Visualize missing values
            plt.figure(figsize=(10, 6))
            plt.barh(missing_df['Column'], missing_df['Percentage'], color='coral')
            plt.xlabel('Percentage of Missing Values', fontsize=12)
            plt.ylabel('Columns', fontsize=12)
            plt.title('Missing Values Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{CONFIG['output_dir']}/missing_values.png", dpi=300, bbox_inches='tight')
            self._print(f"\n✓ Visualization saved: {CONFIG['output_dir']}/missing_values.png")
            plt.show()
        else:
            self._print("\n✓ No missing values found!")
        
        return missing_df
    
    def handle_missing_values(self):
        """Handle missing values using appropriate imputation strategies."""
        self._print("\n[STEP 4] Handling Missing Values...")
        self._print("-" * 70)
        
        self.df_cleaned = self.df.copy()
        
        # Age: Fill with median
        if self.df_cleaned['Age'].isnull().sum() > 0:
            age_median = self.df_cleaned['Age'].median()
            self.df_cleaned['Age'].fillna(age_median, inplace=True)
            self._print(f"✓ Age: Filled {self.df['Age'].isnull().sum()} missing values with median = {age_median:.2f}")
        
        # Embarked: Fill with mode
        if self.df_cleaned['Embarked'].isnull().sum() > 0:
            embarked_mode = self.df_cleaned['Embarked'].mode()[0]
            self.df_cleaned['Embarked'].fillna(embarked_mode, inplace=True)
            self._print(f"✓ Embarked: Filled {self.df['Embarked'].isnull().sum()} missing values with mode = '{embarked_mode}'")
        
        # Cabin: Create binary feature
        self.df_cleaned['Has_Cabin'] = self.df_cleaned['Cabin'].notna().astype(int)
        self._print(f"✓ Cabin: Created binary feature 'Has_Cabin' (1 if cabin known, 0 otherwise)")
        
        # Fare: Fill with median
        if self.df_cleaned['Fare'].isnull().sum() > 0:
            fare_median = self.df_cleaned['Fare'].median()
            self.df_cleaned['Fare'].fillna(fare_median, inplace=True)
            self._print(f"✓ Fare: Filled missing values with median = {fare_median:.2f}")
        
        total_missing = self.df_cleaned.isnull().sum().sum()
        self._print(f"\n✓ Missing values after cleaning: {total_missing}")
    
    def feature_engineering(self):
        """Create new features from existing data."""
        self._print("\n[STEP 5] Feature Engineering...")
        self._print("-" * 70)
        
        # Family Size
        self.df_cleaned['FamilySize'] = (
            self.df_cleaned['SibSp'] + self.df_cleaned['Parch'] + 1
        )
        self._print(f"✓ Created 'FamilySize' = SibSp + Parch + 1")
        
        # Is Alone
        self.df_cleaned['IsAlone'] = (self.df_cleaned['FamilySize'] == 1).astype(int)
        self._print(f"✓ Created 'IsAlone' binary feature")
        
        # Extract Title
        self.df_cleaned['Title'] = self.df_cleaned['Name'].str.extract(
            ' ([A-Za-z]+)\.', expand=False
        )
        
        # Simplify titles
        title_mapping = {
            'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
            'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
            'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
            'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
            'Capt': 'Rare', 'Sir': 'Rare'
        }
        self.df_cleaned['Title'] = self.df_cleaned['Title'].map(title_mapping)
        self.df_cleaned['Title'].fillna('Rare', inplace=True)
        self._print(f"✓ Extracted and simplified 'Title' from Name column")
        self._print(f"  Unique titles: {self.df_cleaned['Title'].unique()}")
    
    def encode_categorical_features(self):
        """Encode categorical variables to numerical format."""
        self._print("\n[STEP 6] Encoding Categorical Variables...")
        self._print("-" * 70)
        
        # Drop unnecessary columns
        features_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        self.df_cleaned = self.df_cleaned.drop(columns=features_to_drop)
        
        # Label encode categorical features
        categorical_features = ['Sex', 'Embarked', 'Title']
        
        for feature in categorical_features:
            le = LabelEncoder()
            self.df_cleaned[feature] = le.fit_transform(self.df_cleaned[feature])
            self.label_encoders[feature] = le
            
            encoding_map = dict(zip(le.classes_, le.transform(le.classes_)))
            self._print(f"✓ Encoded '{feature}': {encoding_map}")
        
        self._print("\n✓ Categorical encoding completed!")
    
    def detect_outliers(self) -> Dict[str, int]:
        """
        Detect outliers using IQR method.
        
        Returns:
            Dict[str, int]: Dictionary with outlier counts per feature
        """
        self._print("\n[STEP 7] Detecting Outliers...")
        self._print("-" * 70)
        
        numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        outlier_counts = {}
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(numerical_features):
            # Calculate IQR
            Q1 = self.df_cleaned[feature].quantile(0.25)
            Q3 = self.df_cleaned[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = self.df_cleaned[
                (self.df_cleaned[feature] < lower_bound) | 
                (self.df_cleaned[feature] > upper_bound)
            ]
            outlier_counts[feature] = len(outliers)
            
            # Visualize
            sns.boxplot(data=self.df_cleaned, y=feature, ax=axes[idx], color='skyblue')
            axes[idx].set_title(f'Boxplot: {feature}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(feature, fontsize=10)
            axes[idx].text(
                0.5, 0.95, f'Outliers: {len(outliers)}',
                transform=axes[idx].transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10
            )
        
        # Remove empty subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/outlier_boxplots.png", dpi=300, bbox_inches='tight')
        self._print(f"✓ Visualization saved: {CONFIG['output_dir']}/outlier_boxplots.png")
        plt.show()
        
        self._print("\nOutlier Summary:")
        for feature, count in outlier_counts.items():
            self._print(f"  {feature:12s}: {count:3d} outliers")
        
        return outlier_counts
    
    def remove_outliers(self):
        """Remove outliers from the dataset."""
        if not CONFIG['remove_outliers']:
            self._print("\n[STEP 8] Outlier Removal: SKIPPED (disabled in config)")
            return
        
        self._print("\n[STEP 8] Removing Outliers...")
        self._print("-" * 70)
        
        initial_rows = len(self.df_cleaned)
        
        # Remove outliers from Fare (most significant)
        Q1 = self.df_cleaned['Fare'].quantile(0.25)
        Q3 = self.df_cleaned['Fare'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.df_cleaned = self.df_cleaned[
            (self.df_cleaned['Fare'] >= lower_bound) & 
            (self.df_cleaned['Fare'] <= upper_bound)
        ]
        
        removed = initial_rows - len(self.df_cleaned)
        self._print(f"✓ Removed {removed} outliers from 'Fare'")
        self._print(f"  Dataset shape: {initial_rows} → {len(self.df_cleaned)} rows")
    
    def scale_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Scale numerical features using StandardScaler.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Scaled features (X) and target (y)
        """
        self._print("\n[STEP 9] Feature Scaling...")
        self._print("-" * 70)
        
        # Separate features and target
        X = self.df_cleaned.drop('Survived', axis=1)
        y = self.df_cleaned['Survived']
        
        # Scale numerical features
        self.scaler = StandardScaler()
        numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
        
        X_scaled = X.copy()
        X_scaled[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        self._print(f"✓ Applied StandardScaler to numerical features:")
        self._print(f"  Features scaled: {numerical_cols}")
        self._print("\n  Scaling Statistics:")
        for col in numerical_cols:
            self._print(f"    {col:12s} - Mean: {X_scaled[col].mean():7.4f}, Std: {X_scaled[col].std():7.4f}")
        
        return X_scaled, y
    
    def correlation_analysis(self, X: pd.DataFrame, y: pd.Series):
        """
        Perform correlation analysis and visualization.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
        """
        self._print("\n[STEP 10] Correlation Analysis...")
        self._print("-" * 70)
        
        # Create correlation matrix
        correlation_data = X.copy()
        correlation_data['Survived'] = y
        
        plt.figure(figsize=(14, 12))
        correlation_matrix = correlation_data.corr()
        
        sns.heatmap(
            correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}
        )
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{CONFIG['output_dir']}/correlation_heatmap.png", dpi=300, bbox_inches='tight')
        self._print(f"✓ Visualization saved: {CONFIG['output_dir']}/correlation_heatmap.png")
        plt.show()
        
        # Display top correlations with survival
        self._print("\nTop 5 Features Correlated with Survival:")
        survival_corr = correlation_matrix['Survived'].sort_values(ascending=False)[1:6]
        for feature, corr in survival_corr.items():
            self._print(f"  {feature:15s}: {corr:6.3f}")
    
    def display_summary(self, X: pd.DataFrame, y: pd.Series):
        """
        Display final summary of preprocessed data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
        """
        self._print("\n" + "=" * 70)
        self._print("FINAL PREPROCESSED DATA SUMMARY")
        self._print("=" * 70)
        
        self._print(f"\nFinal Dataset Shape: {X.shape[0]} rows × {X.shape[1]} features")
        self._print(f"Target Variable: Survived")
        self._print(f"  - Survived: {y.sum()} ({y.mean()*100:.1f}%)")
        self._print(f"  - Did not survive: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
        
        self._print("\n\nFeature List:")
        for i, col in enumerate(X.columns, 1):
            self._print(f"  {i:2d}. {col}")
        
        if self.verbose:
            self._print("\n\nFirst 5 Rows of Preprocessed Data:")
            print(X.head())
            
            self._print("\n\nPreprocessed Data Statistics:")
            print(X.describe())
        
        self._print("\n" + "=" * 70)
        self._print("✓ DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
        self._print("=" * 70)
        self._print("\nYou can now use X (features) and y (target) for ML modeling!")
    
    def preprocess(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Execute the complete preprocessing pipeline.
        
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features (X) and target (y)
        """
        # Execute all steps
        self.load_data()
        self.explore_data()
        self.analyze_missing_values()
        self.handle_missing_values()
        self.feature_engineering()
        self.encode_categorical_features()
        self.detect_outliers()
        self.remove_outliers()
        X, y = self.scale_features()
        self.correlation_analysis(X, y)
        self.display_summary(X, y)
        
        return X, y
    
    def save_processed_data(self, X: pd.DataFrame, y: pd.Series, output_path: str = None):
        """
        Save preprocessed data to CSV.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target variable
            output_path (str): Output file path
        """
        if output_path is None:
            output_path = f"{CONFIG['output_dir']}/titanic_preprocessed.csv"
        
        processed_df = X.copy()
        processed_df['Survived'] = y
        processed_df.to_csv(output_path, index=False)
        
        self._print(f"\n✓ Preprocessed data saved to: {output_path}")


def preprocess_titanic(filepath: str = CONFIG['dataset_path'], 
                       verbose: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Main function to preprocess the Titanic dataset.
    
    Args:
        filepath (str): Path to the Titanic CSV file
        verbose (bool): Whether to print progress messages
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Preprocessed features (X) and target (y)
    
    Example:
        >>> X, y = preprocess_titanic('Titanic-Dataset.csv')
        >>> print(X.shape, y.shape)
    """
    preprocessor = TitanicPreprocessor(filepath, verbose)
    X, y = preprocessor.preprocess()
    
    # Optionally save processed data
    preprocessor.save_processed_data(X, y)
    
    return X, y


def main():
    """Main execution function."""
    try:
        X, y = preprocess_titanic(CONFIG['dataset_path'], verbose=True)
        
        print("\n" + "=" * 70)
        print("PREPROCESSING COMPLETE!")
        print("=" * 70)
        print(f"\nProcessed Data:")
        print(f"  - Features shape: {X.shape}")
        print(f"  - Target shape: {y.shape}")
        print(f"  - Output directory: {CONFIG['output_dir']}/")
        print("\nNext Steps:")
        print("  1. Use X and y for model training")
        print("  2. Try different ML algorithms (Logistic Regression, Random Forest, etc.)")
        print("  3. Perform cross-validation")
        print("  4. Evaluate model performance")
        print("=" * 70)
        
        return X, y
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    # Set visualization style
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Run preprocessing
    X, y = main()