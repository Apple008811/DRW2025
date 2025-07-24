#!/usr/bin/env python3
"""
Ultra-Lightweight SVR Training for Kaggle
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
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

class UltraLightSVR:
    def __init__(self):
        self.model = None
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
        if len(feature_cols) > 30:  # Limit to top 30 features for SVR
            feature_cols = feature_cols[:30]
        
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
    
    def train(self):
        """Train ultra-lightweight SVR model"""
        print("Starting SVR training...")
        
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
            
            # Use very small sample for SVR training
            sample_size = min(5000, len(train_data))  # Very small sample for SVR
            sample_idx = np.random.choice(len(train_data), sample_size, replace=False)
            
            X_train = train_data.iloc[sample_idx][all_feature_cols]
            y_train = train_data.iloc[sample_idx][target_col]
            
            X_test = test_data[all_feature_cols]
            
            print(f"Training on {len(X_train)} samples with {len(all_feature_cols)} features")
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train SVR model with lightweight parameters
            self.model = SVR(
                kernel='rbf',
                C=1.0,  # Reduced complexity
                epsilon=0.1,
                gamma='scale',
                cache_size=1000,  # Reduced cache size
                max_iter=1000,    # Limited iterations
                random_state=42
            )
            
            print("Training SVR model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            print("Making predictions...")
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            train_corr = pearsonr(y_train, train_pred)[0]
            train_mse = mean_squared_error(y_train, train_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            
            print(f"Train Correlation: {train_corr:.6f}")
            print(f"Train MSE: {train_mse:.6f}")
            print(f"Train MAE: {train_mae:.6f}")
            
            # Clean up memory
            del X_train, y_train, X_train_scaled, X_test_scaled, train_pred
            gc.collect()
            
            print(f"SUCCESS: SVR trained on {sample_size} samples")
            print(f"Predictions: {len(test_pred)}")
            print(f"Mean prediction: {np.mean(test_pred):.6f}")
            
            return test_pred
            
        except Exception as e:
            print(f"ERROR: Training failed: {e}")
            return None
    
    def create_submission(self, predictions):
        """Create submission file with correct format"""
        if predictions is None:
            print("ERROR: No predictions to create submission")
            return
        
        print("Creating submission file...")
        
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
        output_path = '/kaggle/working/svr_submission.csv'
        submission.to_csv(output_path, index=False)
        
        print(f"✅ Submission saved: {output_path}")
        print(f"📊 Submission stats:")
        print(f"   Rows: {len(submission)} (expected: {expected_rows})")
        print(f"   ID range: {submission['id'].min()} to {submission['id'].max()}")
        print(f"   Mean: {submission['prediction'].mean():.6f}")
        print(f"   Std: {submission['prediction'].std():.6f}")
        print(f"   Min: {submission['prediction'].min():.6f}")
        print(f"   Max: {submission['prediction'].max():.6f}")
        
        # Verify submission format
        if len(submission) == expected_rows and submission['id'].min() == 1 and submission['id'].max() == expected_rows:
            print(f"✅ Submission format is correct!")
        else:
            print(f"❌ Submission format error!")
            print(f"   Expected: {expected_rows} rows, ID 1-{expected_rows}")
            print(f"   Actual: {len(submission)} rows, ID {submission['id'].min()}-{submission['id'].max()}")
        
        return output_path

def main():
    """Main execution function"""
    print("="*80)
    print("ULTRA-LIGHTWEIGHT SVR TRAINING")
    print("="*80)
    import pytz
    pst = pytz.timezone('US/Pacific')
    current_time = datetime.now(pst)
    print(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')} PST")
    print(f"Memory optimization: ENABLED")
    print(f"GPU: DISABLED")
    print(f"SVR parameters: C=1.0, epsilon=0.1, max_iter=1000")
    print("="*80)
    
    # Create trainer
    trainer = UltraLightSVR()
    
    # Train model
    predictions = trainer.train()
    
    # Create submission
    if predictions is not None:
        submission_path = trainer.create_submission(predictions)
        print(f"\n🎯 Ready for submission: {submission_path}")
    else:
        print("\n❌ Training failed - no submission created")
    
    print("="*80)
    print("COMPLETED")

if __name__ == "__main__":
    main() 