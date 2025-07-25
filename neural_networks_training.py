#!/usr/bin/env python3
"""
Lightweight Neural Network Training for Kaggle
Optimized for memory constraints and kernel stability
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Disable GPU to avoid memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import TensorFlow with memory optimization
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set TensorFlow to use CPU only
tf.config.set_visible_devices([], 'GPU')

class UltraLightNeuralNetwork:
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
        print(f"Train columns: {len(train_data.columns)} columns")
        print(f"Sample columns: {list(train_data.columns[:5])}...")
        
        # Load test data
        test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')
        print(f"Test data shape: {test_data.shape}")
        print(f"Test columns: {len(test_data.columns)} columns")
        print(f"Sample columns: {list(test_data.columns[:5])}...")
        
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
        if len(feature_cols) > 20:  # Limit to top 20 features
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
    
    def create_model(self, input_dim):
        """Create ultra-lightweight neural network"""
        print("Creating neural network model...")
        
        model = Sequential([
            Dense(32, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(8, activation='relu'),
            BatchNormalization(),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"Model parameters: {model.count_params():,}")
        return model
    
    def train(self):
        """Train ultra-lightweight neural network"""
        print("Starting neural network training...")
        
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
            sample_size = min(5000, len(train_data))  # Very small sample
            sample_idx = np.random.choice(len(train_data), sample_size, replace=False)
            
            X_train = train_data.iloc[sample_idx][all_feature_cols]
            y_train = train_data.iloc[sample_idx][target_col]
            
            X_test = test_data[all_feature_cols]
            
            print(f"Training on {len(X_train)} samples with {len(all_feature_cols)} features")
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            self.model = self.create_model(X_train_scaled.shape[1])
            
            # Callbacks for early stopping
            callbacks = [
                EarlyStopping(
                    monitor='loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
            
            # Train with very small batch size
            history = self.model.fit(
                X_train_scaled, y_train,
                epochs=20,  # Fewer epochs
                batch_size=256,  # Smaller batch size
                callbacks=callbacks,
                verbose=1,
                validation_split=0.2
            )
            
            # Make predictions
            print("Making predictions...")
            predictions = self.model.predict(X_test_scaled, batch_size=256)
            predictions = predictions.flatten()
            
            # Clean up memory
            del X_train, y_train, X_train_scaled, X_test_scaled
            gc.collect()
            
            print(f"SUCCESS: Neural Network trained on {sample_size} samples")
            print(f"Predictions: {len(predictions)}")
            print(f"Mean prediction: {np.mean(predictions):.6f}")
            
            return predictions
            
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
        output_path = '/kaggle/working/neural_network_submission.csv'
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
    print("ULTRA-LIGHTWEIGHT NEURAL NETWORK TRAINING")
    print("="*80)
    import pytz
    pst = pytz.timezone('US/Pacific')
    current_time = datetime.now(pst)
    print(f"Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')} PST")
    print(f"Memory optimization: ENABLED")
    print(f"GPU: DISABLED")
    print("="*80)
    
    # Create trainer
    trainer = UltraLightNeuralNetwork()
    
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