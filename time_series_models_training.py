#!/usr/bin/env python3
"""
Ultra-Lightweight Time Series Models Training for Kaggle
Optimized for memory constraints and kernel stability
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Disable GPU and suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

class UltraLightTimeSeriesModels:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_columns = None
        
    def load_data(self):
        """Load data with memory optimization"""
        print("Loading data...")
        
        # Load train data
        train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
        print(f"Train data shape: {train_data.shape}")
        
        # Load test data
        test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
        print(f"Test data shape: {test_data.shape}")
        
        # Create ID and timestamp columns if they don't exist
        if 'id' not in train_data.columns:
            train_data['id'] = range(len(train_data))
        if 'timestamp' not in train_data.columns:
            train_data['timestamp'] = range(len(train_data))
            
        if 'id' not in test_data.columns:
            test_data['id'] = range(len(test_data))
        if 'timestamp' not in test_data.columns:
            test_data['timestamp'] = range(len(test_data))
        
        gc.collect()
        return train_data, test_data
    
    def create_time_series_features(self, df, is_train=True):
        """Create time series features"""
        print("Creating time series features...")
        
        # Basic time features
        df['hour'] = df['timestamp'] % 24
        df['day_of_week'] = (df['timestamp'] // 24) % 7
        
        # Select only a few important features to save memory
        feature_cols = [col for col in df.columns if col.startswith('X')]
        if len(feature_cols) > 20:  # Limit to top 20 features for time series
            feature_cols = feature_cols[:20]
        
        # Add selected features to the dataframe
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        if is_train:
            # For training data, create target variable
            if 'label' in df.columns:
                target_col = 'label'
            else:
                # Use first feature as target
                target_col = feature_cols[0] if feature_cols else 'X1'
                print(f"Using {target_col} as target variable")
            
            # Create lag features
            df['target_lag1'] = df[target_col].shift(1)
            df['target_lag1'].fillna(0, inplace=True)
            
            # Simple rolling mean
            df['target_rolling_mean'] = df[target_col].rolling(window=5, min_periods=1).mean()
        else:
            # For test data, use zeros for lag features
            df['target_lag1'] = 0
            df['target_rolling_mean'] = 0
        
        # Fill NaN values
        df.fillna(0, inplace=True)
        
        return df, feature_cols
    
    def train_arima(self, train_data, target_col):
        """Train ARIMA model"""
        print("Training ARIMA model...")
        
        try:
            # Use small sample for ARIMA training
            sample_size = min(3000, len(train_data))
            sample_data = train_data.sample(n=sample_size, random_state=42)
            
            # Get target series
            target_series = sample_data[target_col]
            
            # Fit ARIMA model with simple parameters
            model = ARIMA(target_series, order=(1, 0, 1))
            fitted_model = model.fit()
            
            print(f"ARIMA model fitted successfully")
            print(f"AIC: {fitted_model.aic:.2f}")
            
            return fitted_model
            
        except Exception as e:
            print(f"ERROR: ARIMA training failed: {e}")
            return None
    
    def train_sarima(self, train_data, target_col):
        """Train SARIMA model"""
        print("Training SARIMA model...")
        
        try:
            # Use small sample for SARIMA training
            sample_size = min(2000, len(train_data))
            sample_data = train_data.sample(n=sample_size, random_state=42)
            
            # Get target series
            target_series = sample_data[target_col]
            
            # Fit SARIMA model with simple parameters
            model = SARIMAX(target_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 24))
            fitted_model = model.fit(disp=False)
            
            print(f"SARIMA model fitted successfully")
            print(f"AIC: {fitted_model.aic:.2f}")
            
            return fitted_model
            
        except Exception as e:
            print(f"ERROR: SARIMA training failed: {e}")
            return None
    
    def generate_predictions(self, model, n_periods):
        """Generate predictions from time series model"""
        try:
            if model is None:
                return np.zeros(n_periods)
            
            # Generate forecasts
            forecast = model.forecast(steps=n_periods)
            
            # Ensure we have the right number of predictions
            if len(forecast) < n_periods:
                # Pad with last prediction
                padding = [forecast.iloc[-1]] * (n_periods - len(forecast))
                forecast = np.concatenate([forecast.values, padding])
            elif len(forecast) > n_periods:
                # Truncate
                forecast = forecast[:n_periods].values
            else:
                forecast = forecast.values
            
            return forecast
            
        except Exception as e:
            print(f"ERROR: Prediction generation failed: {e}")
            return np.zeros(n_periods)
    
    def train(self):
        """Train ultra-lightweight time series models"""
        print("Starting time series models training...")
        
        try:
            # Load data
            train_data, test_data = self.load_data()
            
            # Create features
            train_data, feature_cols = self.create_time_series_features(train_data, is_train=True)
            test_data, _ = self.create_time_series_features(test_data, is_train=False)
            
            # Get target variable
            if 'label' in train_data.columns:
                target_col = 'label'
            else:
                target_col = feature_cols[0] if feature_cols else 'X1'
            
            print(f"Training on {len(train_data)} samples with target: {target_col}")
            
            # Train models
            results = {}
            
            # Train ARIMA
            arima_model = self.train_arima(train_data, target_col)
            if arima_model is not None:
                arima_predictions = self.generate_predictions(arima_model, len(test_data))
                results['arima'] = {
                    'model': arima_model,
                    'predictions': arima_predictions
                }
                print(f"ARIMA predictions: {len(arima_predictions)}")
            
            # Train SARIMA
            sarima_model = self.train_sarima(train_data, target_col)
            if sarima_model is not None:
                sarima_predictions = self.generate_predictions(sarima_model, len(test_data))
                results['sarima'] = {
                    'model': sarima_model,
                    'predictions': sarima_predictions
                }
                print(f"SARIMA predictions: {len(sarima_predictions)}")
            
            # Clean up memory
            del train_data, test_data
            gc.collect()
            
            print(f"SUCCESS: Time series models trained")
            
            return results
            
        except Exception as e:
            print(f"ERROR: Training failed: {e}")
            return None
    
    def create_submission(self, results):
        """Create submission files for each model"""
        if results is None:
            print("ERROR: No results to create submission")
            return
        
        print("Creating submission files...")
        
        for model_name, result in results.items():
            predictions = result['predictions']
            
            # Create submission dataframe with correct format
            expected_rows = 538150
            
            # Ensure we have the correct number of predictions
            if len(predictions) != expected_rows:
                print(f"WARNING: Expected {expected_rows} predictions, got {len(predictions)}")
                if len(predictions) < expected_rows:
                    # Pad with last prediction value
                    padding = [predictions[-1]] * (expected_rows - len(predictions))
                    predictions = np.concatenate([predictions, padding])
                else:
                    # Truncate to expected length
                    predictions = predictions[:expected_rows]
            
            submission = pd.DataFrame({
                'id': range(1, expected_rows + 1),  # IDs from 1 to 538150
                'prediction': predictions
            })
            
            # Save to Kaggle working directory
            output_path = f'/kaggle/working/{model_name}_submission.csv'
            submission.to_csv(output_path, index=False)
            
            print(f"✅ {model_name} submission saved: {output_path}")
            print(f"   Rows: {len(submission)} (expected: {expected_rows})")
            print(f"   Mean: {submission['prediction'].mean():.6f}")
            print(f"   Std: {submission['prediction'].std():.6f}")
            print(f"   Min: {submission['prediction'].min():.6f}")
            print(f"   Max: {submission['prediction'].max():.6f}")
            
            # Verify submission format
            if len(submission) == expected_rows and submission['id'].min() == 1 and submission['id'].max() == expected_rows:
                print(f"   ✅ Submission format is correct!")
            else:
                print(f"   ❌ Submission format error!")

def main():
    """Main execution function"""
    print("="*80)
    print("ULTRA-LIGHTWEIGHT TIME SERIES MODELS TRAINING")
    print("="*80)
    import pytz
    pst = pytz.timezone('US/Pacific')
    current_time = datetime.now(pst)
    print(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')} PST")
    print(f"Memory optimization: ENABLED")
    print(f"GPU: DISABLED")
    print(f"Models: ARIMA(1,0,1), SARIMA(1,0,1)(1,0,1,24)")
    print("="*80)
    
    # Create trainer
    trainer = UltraLightTimeSeriesModels()
    
    # Train models
    results = trainer.train()
    
    # Create submissions
    if results is not None:
        trainer.create_submission(results)
        print(f"\n🎯 Ready for submission!")
    else:
        print("\n❌ Training failed - no submission created")
    
    print("="*80)
    print("COMPLETED")

if __name__ == "__main__":
    main() 