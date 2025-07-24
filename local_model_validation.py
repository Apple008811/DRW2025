#!/usr/bin/env python3
"""
Local Model Validation Script
Validate model performance locally without uploading to Kaggle
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

# Machine Learning Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Time Series Models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    print("WARNING: statsmodels not available, skipping ARIMA models")
    ARIMA_AVAILABLE = False

class LocalModelValidator:
    def __init__(self):
        self.results = {}
        
    def load_sample_data(self):
        """Load sample data for local validation"""
        print("Loading sample data for local validation...")
        
        # Check if we have the sample data
        if os.path.exists('train_sample.csv'):
            train_data = pd.read_csv('train_sample.csv')
            print(f"Loaded train_sample.csv: {train_data.shape}")
        else:
            # Create synthetic data for testing
            print("Creating synthetic data for testing...")
            np.random.seed(42)
            n_samples = 10000
            
            # Create synthetic features
            feature_data = np.random.randn(n_samples, 50)
            feature_cols = [f'X{i+1}' for i in range(50)]
            
            # Create synthetic target with some correlation
            target = (feature_data[:, 0] * 0.3 + 
                     feature_data[:, 1] * 0.2 + 
                     feature_data[:, 2] * 0.1 + 
                     np.random.randn(n_samples) * 0.5)
            
            train_data = pd.DataFrame(feature_data, columns=feature_cols)
            train_data['label'] = target
            train_data['id'] = range(len(train_data))
            train_data['timestamp'] = range(len(train_data))
            
            print(f"Created synthetic data: {train_data.shape}")
        
        # Split into train and validation
        split_idx = int(len(train_data) * 0.8)
        train = train_data[:split_idx].copy()
        val = train_data[split_idx:].copy()
        
        print(f"Train: {train.shape}, Validation: {val.shape}")
        return train, val
    
    def create_features(self, df, is_train=True):
        """Create features for training"""
        print("Creating features...")
        
        # Basic time features
        df['hour'] = df['timestamp'] % 24
        df['day_of_week'] = (df['timestamp'] // 24) % 7
        
        # Select features
        feature_cols = [col for col in df.columns if col.startswith('X')]
        if len(feature_cols) > 20:
            feature_cols = feature_cols[:20]
        
        if is_train:
            # Create lag features for training data
            if 'label' in df.columns:
                target_col = 'label'
            else:
                target_col = feature_cols[0]
            
            df['target_lag1'] = df[target_col].shift(1)
            df['target_lag1'].fillna(0, inplace=True)
            df['target_rolling_mean'] = df[target_col].rolling(window=5, min_periods=1).mean()
        else:
            # For validation data
            df['target_lag1'] = 0
            df['target_rolling_mean'] = 0
        
        df.fillna(0, inplace=True)
        return df, feature_cols
    
    def validate_linear_models(self, train, val):
        """Validate linear models"""
        print("\n" + "="*60)
        print("VALIDATING LINEAR MODELS")
        print("="*60)
        
        # Prepare data
        train, feature_cols = self.create_features(train, is_train=True)
        val, _ = self.create_features(val, is_train=False)
        
        if 'label' in train.columns:
            target_col = 'label'
        else:
            target_col = feature_cols[0]
        
        all_feature_cols = ['hour', 'day_of_week', 'target_lag1', 'target_rolling_mean'] + feature_cols
        
        X_train = train[all_feature_cols]
        y_train = train[target_col]
        X_val = val[all_feature_cols]
        y_val = val[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train and validate models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1)
        }
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            train_corr = pearsonr(y_train, train_pred)[0]
            val_corr = pearsonr(y_val, val_pred)[0]
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            print(f"  Train Correlation: {train_corr:.6f}")
            print(f"  Val Correlation: {val_corr:.6f}")
            print(f"  Train MSE: {train_mse:.6f}")
            print(f"  Val MSE: {val_mse:.6f}")
            
            self.results[model_name] = {
                'train_corr': train_corr,
                'val_corr': val_corr,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'predictions': val_pred
            }
    
    def validate_svr(self, train, val):
        """Validate SVR model"""
        print("\n" + "="*60)
        print("VALIDATING SVR MODEL")
        print("="*60)
        
        # Prepare data
        train, feature_cols = self.create_features(train, is_train=True)
        val, _ = self.create_features(val, is_train=False)
        
        if 'label' in train.columns:
            target_col = 'label'
        else:
            target_col = feature_cols[0]
        
        all_feature_cols = ['hour', 'day_of_week', 'target_lag1', 'target_rolling_mean'] + feature_cols[:10]  # Limit features for SVR
        
        X_train = train[all_feature_cols]
        y_train = train[target_col]
        X_val = val[all_feature_cols]
        y_val = val[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train SVR
        print("Training SVR...")
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        svr.fit(X_train_scaled, y_train)
        
        # Predict
        train_pred = svr.predict(X_train_scaled)
        val_pred = svr.predict(X_val_scaled)
        
        # Calculate metrics
        train_corr = pearsonr(y_train, train_pred)[0]
        val_corr = pearsonr(y_val, val_pred)[0]
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        
        print(f"  Train Correlation: {train_corr:.6f}")
        print(f"  Val Correlation: {val_corr:.6f}")
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Val MSE: {val_mse:.6f}")
        
        self.results['SVR'] = {
            'train_corr': train_corr,
            'val_corr': val_corr,
            'train_mse': train_mse,
            'val_mse': val_mse,
            'predictions': val_pred
        }
    
    def validate_time_series_models(self, train, val):
        """Validate time series models"""
        if not ARIMA_AVAILABLE:
            print("\n" + "="*60)
            print("SKIPPING TIME SERIES MODELS (statsmodels not available)")
            print("="*60)
            return
        
        print("\n" + "="*60)
        print("VALIDATING TIME SERIES MODELS")
        print("="*60)
        
        # Prepare data
        train, feature_cols = self.create_features(train, is_train=True)
        val, _ = self.create_features(val, is_train=False)
        
        if 'label' in train.columns:
            target_col = 'label'
        else:
            target_col = feature_cols[0]
        
        # Use smaller sample for time series
        sample_size = min(2000, len(train))
        train_sample = train.sample(n=sample_size, random_state=42)
        
        # Train ARIMA
        print("Training ARIMA...")
        try:
            target_series = train_sample[target_col]
            arima_model = ARIMA(target_series, order=(1, 0, 1))
            arima_fitted = arima_model.fit()
            
            # Generate predictions
            forecast = arima_fitted.forecast(steps=len(val))
            
            # Calculate metrics
            val_corr = pearsonr(val[target_col], forecast)[0]
            val_mse = mean_squared_error(val[target_col], forecast)
            
            print(f"  Val Correlation: {val_corr:.6f}")
            print(f"  Val MSE: {val_mse:.6f}")
            
            self.results['ARIMA'] = {
                'val_corr': val_corr,
                'val_mse': val_mse,
                'predictions': forecast
            }
        except Exception as e:
            print(f"  ERROR: ARIMA failed - {e}")
    
    def run_validation(self):
        """Run complete validation"""
        print("="*80)
        print("LOCAL MODEL VALIDATION")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Load data
        train, val = self.load_sample_data()
        
        # Validate models
        self.validate_linear_models(train, val)
        self.validate_svr(train, val)
        self.validate_time_series_models(train, val)
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results to display")
            return
        
        # Create summary table
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Val Correlation': f"{result.get('val_corr', 0):.6f}",
                'Val MSE': f"{result.get('val_mse', 0):.6f}",
                'Train Correlation': f"{result.get('train_corr', 0):.6f}",
                'Train MSE': f"{result.get('train_mse', 0):.6f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Val Correlation', ascending=False)
        
        print(summary_df.to_string(index=False))
        
        # Best model
        best_model = summary_df.iloc[0]
        print(f"\n🏆 Best Model: {best_model['Model']}")
        print(f"   Validation Correlation: {best_model['Val Correlation']}")
        print(f"   Validation MSE: {best_model['Val MSE']}")
        
        # Save results
        summary_df.to_csv('local_validation_results.csv', index=False)
        print(f"\n💾 Results saved to: local_validation_results.csv")

def main():
    """Main execution function"""
    validator = LocalModelValidator()
    results = validator.run_validation()
    
    print("\n" + "="*80)
    print("LOCAL VALIDATION COMPLETED")
    print("="*80)
    print("💡 This gives you an estimate of model performance before uploading to Kaggle")
    print("📊 Higher correlation values indicate better performance")
    print("🎯 Use this to choose which models to run on Kaggle")

if __name__ == "__main__":
    main() 