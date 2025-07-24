#!/usr/bin/env python3
"""
Ultra-Lightweight Linear Models Training for Kaggle
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

# Machine Learning Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

class UltraLightLinearModels:
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
    
    def create_simple_features(self, df, is_train=True):
        """Create simple features to minimize memory usage"""
        print("Creating simple features...")
        
        # Basic time features
        df['hour'] = df['timestamp'] % 24
        df['day_of_week'] = (df['timestamp'] // 24) % 7
        
        # Select only a few important features to save memory
        feature_cols = [col for col in df.columns if col.startswith('X')]
        if len(feature_cols) > 50:  # Limit to top 50 features
            feature_cols = feature_cols[:50]
        
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
    
    def train_models(self):
        """Train ultra-lightweight linear models"""
        print("Starting linear models training...")
        
        try:
            # Load data
            train_data, test_data = self.load_data()
            
            # Create features
            train_data, feature_cols = self.create_simple_features(train_data, is_train=True)
            test_data, _ = self.create_simple_features(test_data, is_train=False)
            
            # Get target variable
            if 'label' in train_data.columns:
                target_col = 'label'
            else:
                target_col = feature_cols[0] if feature_cols else 'X1'
            
            # Select features for training
            all_feature_cols = ['hour', 'day_of_week', 'target_lag1', 'target_rolling_mean'] + feature_cols
            
            # Use small sample for training
            sample_size = min(10000, len(train_data))  # Small sample
            sample_idx = np.random.choice(len(train_data), sample_size, replace=False)
            
            X_train = train_data.iloc[sample_idx][all_feature_cols]
            y_train = train_data.iloc[sample_idx][target_col]
            
            X_test = test_data[all_feature_cols]
            
            print(f"Training on {len(X_train)} samples with {len(all_feature_cols)} features")
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            models = {
                'linear_regression': LinearRegression(),
                'ridge_regression': Ridge(alpha=1.0),
                'lasso_regression': Lasso(alpha=0.1)
            }
            
            results = {}
            
            for model_name, model in models.items():
                print(f"\nTraining {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                train_corr = pearsonr(y_train, train_pred)[0]
                train_mse = mean_squared_error(y_train, train_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                
                print(f"  Train Correlation: {train_corr:.6f}")
                print(f"  Train MSE: {train_mse:.6f}")
                print(f"  Train MAE: {train_mae:.6f}")
                
                results[model_name] = {
                    'model': model,
                    'train_corr': train_corr,
                    'train_mse': train_mse,
                    'train_mae': train_mae,
                    'predictions': test_pred
                }
                
                # Clean up memory
                del train_pred
                gc.collect()
            
            # Clean up memory
            del X_train, y_train, X_train_scaled, X_test_scaled
            gc.collect()
            
            print(f"\nSUCCESS: All linear models trained on {sample_size} samples")
            
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
    print("ULTRA-LIGHTWEIGHT LINEAR MODELS TRAINING")
    print("="*80)
    import pytz
    pst = pytz.timezone('US/Pacific')
    current_time = datetime.now(pst)
    print(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')} PST")
    print(f"Memory optimization: ENABLED")
    print(f"GPU: DISABLED")
    print("="*80)
    
    # Create trainer
    trainer = UltraLightLinearModels()
    
    # Train models
    results = trainer.train_models()
    
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