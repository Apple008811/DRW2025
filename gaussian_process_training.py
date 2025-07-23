#!/usr/bin/env python3
"""
Gaussian Process Regression Training Script
==========================================

Trains and evaluates Gaussian Process Regression model for cryptocurrency market prediction.

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

# Gaussian Process
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
    GP_AVAILABLE = True
except ImportError:
    print("WARNING: sklearn.gaussian_process not available, skipping GP model")
    GP_AVAILABLE = False

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GaussianProcessTrainer:
    def __init__(self):
        """Initialize Gaussian Process training."""
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
        
    def time_series_cv(self, model, X, y, n_splits=3):
        """Perform time series cross-validation."""
        from sklearn.model_selection import TimeSeriesSplit
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
        
    def train_gaussian_process(self, X, y, feature_cols):
        """Train Gaussian Process Regression model."""
        if not GP_AVAILABLE:
            print("SKIPPED: Gaussian Process model (sklearn.gaussian_process not available)")
            return None
            
        print("\nTraining Gaussian Process Regression...")
        
        try:
            # Use a subset for computational efficiency (GP is expensive)
            sample_size = min(5000, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Standardize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            X_test_scaled = scaler.transform(self.X_test)
            
            # Try different kernels
            kernels = [
                RBF(length_scale=1.0),
                Matern(length_scale=1.0, nu=1.5),
                RationalQuadratic(length_scale=1.0, alpha=1.0)
            ]
            
            best_score = -1
            best_model = None
            best_kernel = None
            
            for i, kernel in enumerate(kernels):
                print(f"  Testing kernel {i+1}/{len(kernels)}: {type(kernel).__name__}")
                
                model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    random_state=42,
                    n_restarts_optimizer=5
                )
                
                # Simple validation (GP is too slow for full CV)
                split_idx = int(len(X_scaled) * 0.8)
                X_train_gp = X_scaled[:split_idx]
                y_train_gp = y_sample.iloc[:split_idx]
                X_val_gp = X_scaled[split_idx:]
                y_val_gp = y_sample.iloc[split_idx:]
                
                # Fit model
                model.fit(X_train_gp, y_train_gp)
                
                # Predict validation set
                y_pred = model.predict(X_val_gp)
                score = pearsonr(y_val_gp, y_pred)[0]
                
                print(f"    Score: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_kernel = kernel
            
            print(f"  Best kernel: {type(best_kernel).__name__} with score: {best_score:.4f}")
            
            # Train final model with best kernel on full sample
            final_model = GaussianProcessRegressor(
                kernel=best_kernel,
                alpha=1e-6,
                random_state=42,
                n_restarts_optimizer=5
            )
            final_model.fit(X_scaled, y_sample)
            
            # Predict test set
            predictions = final_model.predict(X_test_scaled)
            
            # Store results
            self.models['Gaussian Process'] = {'model': final_model, 'scaler': scaler}
            self.results['Gaussian Process'] = {
                'avg_score': best_score,
                'std_score': 0.0,
                'scores': [best_score],
                'predictions': predictions
            }
            
            print(f"SUCCESS: Gaussian Process: {best_score:.4f}")
            return predictions
            
        except Exception as e:
            print(f"ERROR: Gaussian Process training failed: {e}")
            return None
        
    def create_submission_file(self, predictions, model_name):
        """Create submission file for Kaggle."""
        print(f"\nCreating submission file for {model_name}...")
        
        # Create submission dataframe
        submission = pd.DataFrame({
            'prediction': predictions
        })
        
        # Ensure correct number of rows (538,150)
        expected_rows = 538150
        if len(submission) != expected_rows:
            print(f"WARNING: Expected {expected_rows} rows, got {len(submission)}")
            if len(submission) < expected_rows:
                # Pad with last prediction
                padding = pd.DataFrame({
                    'prediction': [predictions[-1]] * (expected_rows - len(submission))
                })
                submission = pd.concat([submission, padding], ignore_index=True)
            else:
                # Truncate
                submission = submission.head(expected_rows)
        
        # Save submission file
        filename = f'/kaggle/working/results/{model_name.lower().replace(" ", "_")}_submission.csv'
        submission.to_csv(filename, index=False)
        print(f"SUCCESS: Submission file saved: {filename}")
        
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
        results_df.to_csv('/kaggle/working/results/gaussian_process_results.csv', index=False)
        print("SUCCESS: Results summary saved")
        
        # Print results
        print("\n" + "="*80)
        print("GAUSSIAN PROCESS RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Create submission files
        for model_name, result in self.results.items():
            self.create_submission_file(result['predictions'], model_name)
        
        # Memory optimization
        gc.collect()
        
    def run_training(self):
        """Run complete Gaussian Process training pipeline."""
        print("="*80)
        print("GAUSSIAN PROCESS TRAINING PIPELINE")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Get feature columns
        feature_cols = [col for col in self.X_train.columns if col != 'label']
        
        # Train model
        print("\n" + "="*80)
        print("TRAINING GAUSSIAN PROCESS MODEL")
        print("="*80)
        
        # Gaussian Process Model
        self.train_gaussian_process(self.X_train, self.y_train, feature_cols)
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("GAUSSIAN PROCESS TRAINING COMPLETED")
        print("="*80)

def main():
    """Main function to run the training pipeline."""
    trainer = GaussianProcessTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main() 