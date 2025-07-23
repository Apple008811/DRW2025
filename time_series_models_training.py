#!/usr/bin/env python3
"""
Time Series Models Training Script
==================================

Trains and evaluates time series models (ARIMA, Prophet) for cryptocurrency market prediction.

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

# Time Series Models
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    print("WARNING: statsmodels not available, skipping ARIMA model")
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("WARNING: Prophet not available, skipping Prophet model")
    PROPHET_AVAILABLE = False

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TimeSeriesModelsTrainer:
    def __init__(self):
        """Initialize time series models training."""
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
        
        # Prepare time series data
        self.y_series = self.y_train.reset_index(drop=True)
        
        print(f"SUCCESS: Time series data: {self.y_series.shape}")
        
        # Memory optimization
        gc.collect()
        
    def train_arima_model(self):
        """Train ARIMA model."""
        if not ARIMA_AVAILABLE:
            print("SKIPPED: ARIMA model (statsmodels not available)")
            return None
            
        print("\nTraining ARIMA model...")
        
        try:
            # Use a subset for computational efficiency
            sample_size = min(5000, len(self.y_series))
            y_sample = self.y_series.head(sample_size)
            
            # Fit ARIMA model
            arima_model = ARIMA(y_sample, order=(1, 1, 1))
            arima_fitted = arima_model.fit()
            
            # Simple validation
            train_size = int(len(y_sample) * 0.8)
            y_train_arima = y_sample[:train_size]
            y_val_arima = y_sample[train_size:]
            
            # Fit on training data
            arima_train = ARIMA(y_train_arima, order=(1, 1, 1))
            arima_fitted_train = arima_train.fit()
            
            # Predict validation set
            arima_forecast = arima_fitted_train.forecast(steps=len(y_val_arima))
            score = pearsonr(y_val_arima, arima_forecast)[0]
            
            # Predict test set
            arima_test_forecast = arima_fitted.forecast(steps=len(self.test))
            
            # Store results
            self.models['ARIMA'] = arima_fitted
            self.results['ARIMA'] = {
                'avg_score': score,
                'std_score': 0.0,
                'scores': [score],
                'predictions': arima_test_forecast
            }
            
            print(f"SUCCESS: ARIMA: {score:.4f}")
            return arima_test_forecast
            
        except Exception as e:
            print(f"ERROR: ARIMA training failed: {e}")
            return None
        
    def train_prophet_model(self):
        """Train Prophet model."""
        if not PROPHET_AVAILABLE:
            print("SKIPPED: Prophet model (prophet not available)")
            return None
            
        print("\nTraining Prophet model...")
        
        try:
            # Prepare data for Prophet
            sample_size = min(2000, len(self.y_series))
            y_sample = self.y_series.head(sample_size)
            
            # Create Prophet dataframe
            prophet_df = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(y_sample), freq='H'),
                'y': y_sample.values
            })
            
            # Fit Prophet model
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative'
            )
            prophet_model.fit(prophet_df)
            
            # Simple validation
            train_size = int(len(prophet_df) * 0.8)
            prophet_train = prophet_df.head(train_size)
            prophet_val = prophet_df.tail(len(prophet_df) - train_size)
            
            # Fit on training data
            prophet_model_train = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative'
            )
            prophet_model_train.fit(prophet_train)
            
            # Predict validation set
            prophet_forecast = prophet_model_train.predict(prophet_val)
            score = pearsonr(prophet_val['y'], prophet_forecast['yhat'])[0]
            
            # Predict test set
            test_dates = pd.date_range(
                start=prophet_df['ds'].iloc[-1] + pd.Timedelta(hours=1),
                periods=len(self.test),
                freq='H'
            )
            test_df = pd.DataFrame({'ds': test_dates})
            prophet_test_forecast = prophet_model.predict(test_df)
            
            # Store results
            self.models['Prophet'] = prophet_model
            self.results['Prophet'] = {
                'avg_score': score,
                'std_score': 0.0,
                'scores': [score],
                'predictions': prophet_test_forecast['yhat'].values
            }
            
            print(f"SUCCESS: Prophet: {score:.4f}")
            return prophet_test_forecast['yhat'].values
            
        except Exception as e:
            print(f"ERROR: Prophet training failed: {e}")
            return None
        
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
        results_df.to_csv('/kaggle/working/results/time_series_models_results.csv', index=False)
        print("SUCCESS: Results summary saved")
        
        # Print results
        print("\n" + "="*80)
        print("TIME SERIES MODELS RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Create submission files
        for model_name, result in self.results.items():
            self.create_submission_file(result['predictions'], model_name)
        
        # Memory optimization
        gc.collect()
        
    def run_training(self):
        """Run complete time series models training pipeline."""
        print("="*80)
        print("TIME SERIES MODELS TRAINING PIPELINE")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train models
        print("\n" + "="*80)
        print("TRAINING TIME SERIES MODELS")
        print("="*80)
        
        # ARIMA Model
        self.train_arima_model()
        
        # Prophet Model
        self.train_prophet_model()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("TIME SERIES MODELS TRAINING COMPLETED")
        print("="*80)

def main():
    """Main function to run the training pipeline."""
    trainer = TimeSeriesModelsTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main() 