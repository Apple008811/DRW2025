#!/usr/bin/env python3
"""
Support Vector Regression (SVR) Training Script
===============================================

Trains and evaluates SVR model for cryptocurrency market prediction.

Author: Yixuan
Date: 2025-01-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Memory optimization
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Machine Learning Models
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SVRTrainer:
    def __init__(self):
        """Initialize SVR training."""
        self.train = None
        self.test = None
        self.train_features = None
        self.test_features = None
        
        # Model results storage
        self.models = {}
        self.results = {}
        
        # File paths
        self.train_file = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        self.test_file = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
        self.features_file = '/kaggle/working/engineered_features.parquet'
        
        # Create results directory
        os.makedirs('/kaggle/working/results', exist_ok=True)
        
    def load_data(self):
        """Load data and engineered features."""
        print("Loading data and engineered features...")
        
        # Load original data
        self.train = pd.read_parquet(self.train_file)
        self.test = pd.read_parquet(self.test_file)
        
        # Load engineered features (from Phase 3)
        try:
            features_data = pd.read_parquet(self.features_file)
            self.train_features = features_data[features_data.index < len(self.train)]
            self.test_features = features_data[features_data.index >= len(self.train)]
            print(f"SUCCESS: Engineered features loaded: {self.train_features.shape}")
        except:
            print("WARNING: Engineered features not found, using original features")
            self.train_features = self.train.drop(['label'], axis=1, errors='ignore')
            self.test_features = self.test.copy()
        
        print(f"SUCCESS: Train data: {self.train.shape}")
        print(f"SUCCESS: Test data: {self.test.shape}")
        
    def prepare_data(self):
        """Prepare data for training."""
        print("\n" + "="*80)
        print("DATA PREPARATION")
        print("="*80)
        
        # Get target variable
        self.y_train = self.train['label']
        
        # Prepare feature columns
        feature_cols = [col for col in self.train_features.columns if col != 'label']
        self.X_train = self.train_features[feature_cols]
        self.X_test = self.test_features[feature_cols]
        
        # Handle missing values
        self.X_train = self.X_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        # Remove infinite values
        self.X_train = self.X_train.replace([np.inf, -np.inf], 0)
        self.X_test = self.X_test.replace([np.inf, -np.inf], 0)
        
        print(f"SUCCESS: Training features: {self.X_train.shape}")
        print(f"SUCCESS: Test features: {self.X_test.shape}")
        print(f"SUCCESS: Target variable: {self.y_train.shape}")
        
        # Memory optimization
        gc.collect()
        
    def time_series_cv(self, model, X, y, n_splits=5):
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Fit model
            model.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred = model.predict(X_val_fold)
            
            # Calculate correlation
            score = pearsonr(y_val_fold, y_pred)[0]
            scores.append(score)
            
        return np.mean(scores), np.std(scores), scores
        
    def train_svr(self, X, y, feature_cols):
        """Train SVR model."""
        print("\nTraining Support Vector Regression...")
        
        # Use a subset of data for SVR (computationally expensive)
        sample_size = min(10000, len(X))
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        
        X_sample = X.iloc[sample_idx]
        y_sample = y.iloc[sample_idx]
        
        # Standardize features for SVR
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Time series cross-validation
        model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
        avg_score, std_score, scores = self.time_series_cv(model, X_sample, y_sample)
        
        # Train final model
        model.fit(X_scaled, y_sample)
        predictions = model.predict(X_test_scaled)
        
        # Store results
        self.models['SVR'] = {'model': model, 'scaler': scaler}
        self.results['SVR'] = {
            'avg_score': avg_score,
            'std_score': std_score,
            'scores': scores,
            'predictions': predictions
        }
        
        print(f"SUCCESS: SVR: {avg_score:.4f} ± {std_score:.4f}")
        return predictions
        
    def create_submission_file(self, predictions, model_name):
        """Create submission file for Kaggle."""
        print(f"\nCreating submission file for {model_name}...")
        
        # Ensure correct number of rows (538,150)
        expected_rows = 538150
        if len(predictions) != expected_rows:
            print(f"WARNING: Expected {expected_rows} rows, got {len(predictions)}")
            if len(predictions) < expected_rows:
                # Pad with last prediction
                predictions = list(predictions) + [predictions[-1]] * (expected_rows - len(predictions))
            else:
                # Truncate
                predictions = predictions[:expected_rows]
        
        # Create submission dataframe with correct format (like LightGBM)
        submission_ids = list(range(1, expected_rows + 1))
        submission = pd.DataFrame({
            'id': submission_ids,
            'prediction': predictions
        })
        
        # Save submission file (same path as LightGBM)
        filename = f'/kaggle/working/{model_name.lower()}_submission.csv'
        submission.to_csv(filename, index=False)
        print(f"SUCCESS: Submission file saved: {filename}")
        print(f"   Predictions: {len(submission)} rows")
        print(f"   Prediction range: {min(predictions):.4f} to {max(predictions):.4f}")
        print(f"   Prediction mean: {np.mean(predictions):.4f}")
        print(f"   Prediction std: {np.std(predictions):.4f}")
        
        return filename
        
    def save_results(self):
        """Save model results and predictions."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save results summary
        results_summary = []
        for model_name, result in self.results.items():
            results_summary.append({
                'Model': model_name,
                'Avg_Score': result['avg_score'],
                'Std_Score': result['std_score'],
                'Min_Score': min(result['scores']),
                'Max_Score': max(result['scores'])
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Avg_Score', ascending=False)
        
        # Save results
        results_df.to_csv('/kaggle/working/results/svr_results.csv', index=False)
        print("SUCCESS: Results summary saved")
        
        # Print results
        print("\n" + "="*80)
        print("SVR RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Create submission files
        for model_name, result in self.results.items():
            self.create_submission_file(result['predictions'], model_name)
        
        # Memory optimization
        gc.collect()
        
    def run_training(self):
        """Run complete SVR training pipeline."""
        print("="*80)
        print("SVR TRAINING PIPELINE")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Get feature columns
        feature_cols = [col for col in self.X_train.columns if col != 'label']
        
        # Train model
        print("\n" + "="*80)
        print("TRAINING SVR MODEL")
        print("="*80)
        
        # SVR Model
        self.train_svr(self.X_train, self.y_train, feature_cols)
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("SVR TRAINING COMPLETED")
        print("="*80)

def main():
    """Main function to run the training pipeline."""
    trainer = SVRTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main() 