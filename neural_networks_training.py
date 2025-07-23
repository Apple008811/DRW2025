#!/usr/bin/env python3
"""
Neural Networks Training Script
===============================

Trains and evaluates neural network models (LSTM, GRU, Transformer) for cryptocurrency market prediction.

Author: Yixuan
Date: 2025-01-22
"""

import pandas as pd
import numpy as np
import os
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc

# Suppress warnings
warnings.filterwarnings('ignore')

class LightweightNeuralNetwork:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_column = 'target'
        
    def load_and_prepare_data(self, data_path='/kaggle/input/drw-2/data/train.csv'):
        """Load and prepare data with memory optimization"""
        print("=== Loading and Preparing Data ===")
        
        # Load data in chunks to save memory
        chunk_size = 100000
        chunks = []
        
        for chunk in pd.read_csv(data_path, chunksize=chunk_size):
            # Keep only essential columns to save memory
            essential_cols = ['id', 'target'] + [col for col in chunk.columns 
                                               if col.startswith(('feature_', 'lag_', 'rolling_'))]
            chunk = chunk[essential_cols]
            chunks.append(chunk)
            
        df = pd.concat(chunks, ignore_index=True)
        del chunks
        gc.collect()
        
        print(f"📊 Loaded data: {df.shape}")
        print(f"📊 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(0)
        
        # Separate features and target
        self.feature_columns = [col for col in df.columns if col not in ['id', 'target']]
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        print(f"📊 Features: {X.shape[1]}")
        print(f"📊 Target range: {y.min():.4f} to {y.max():.4f}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        del X, y, X_train, X_val
        gc.collect()
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def create_model(self, input_dim):
        """Create a lightweight neural network"""
        print("=== Creating Neural Network Model ===")
        
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.1),
            
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"📊 Model parameters: {model.count_params():,}")
        model.summary()
        
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val):
        """Train the neural network"""
        print("=== Training Neural Network ===")
        
        # Create model
        self.model = self.create_model(X_train.shape[1])
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train with smaller batch size to save memory
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=512,  # Smaller batch size
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        y_pred = self.model.predict(X_val, batch_size=512)
        
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"\n📊 Validation Results:")
        print(f"   MSE: {mse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R²: {r2:.6f}")
        
        return history, mse, mae, r2
    
    def predict_test_data(self, test_path='/kaggle/input/drw-2/data/test.csv'):
        """Make predictions on test data"""
        print("=== Making Test Predictions ===")
        
        # Load test data in chunks
        chunk_size = 100000
        test_chunks = []
        
        for chunk in pd.read_csv(test_path, chunksize=chunk_size):
            # Keep only feature columns
            chunk = chunk[['id'] + self.feature_columns]
            test_chunks.append(chunk)
            
        test_df = pd.concat(test_chunks, ignore_index=True)
        del test_chunks
        gc.collect()
        
        print(f"📊 Test data: {test_df.shape}")
        
        # Handle missing values
        test_df = test_df.fillna(method='ffill').fillna(0)
        
        # Scale features
        X_test = test_df[self.feature_columns].values
        X_test_scaled = self.scaler.transform(X_test)
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled, batch_size=512)
        predictions = predictions.flatten()
        
        print(f"📊 Predictions: {len(predictions)}")
        print(f"📊 Range: {predictions.min():.6f} to {predictions.max():.6f}")
        print(f"📊 Mean: {predictions.mean():.6f}")
        print(f"📊 Std: {predictions.std():.6f}")
        
        return test_df['id'].values, predictions
    
    def save_results(self, test_ids, predictions, mse, mae, r2):
        """Save results and create submission file"""
        print("=== Saving Results ===")
        
        # Create results directory
        os.makedirs('/kaggle/working/results', exist_ok=True)
        
        # Save detailed results
        results_df = pd.DataFrame({
            'metric': ['MSE', 'MAE', 'R²'],
            'value': [mse, mae, r2]
        })
        
        results_path = '/kaggle/working/results/neural_network_results.csv'
        results_df.to_csv(results_path, index=False)
        print(f"📁 Detailed results saved: {results_path}")
        
        # Create submission file
        submission_df = pd.DataFrame({
            'id': test_ids,
            'prediction': predictions
        })
        
        submission_path = '/kaggle/working/results/neural_network_submission.csv'
        submission_df.to_csv(submission_path, index=False)
        
        print(f"📁 Submission file saved: {submission_path}")
        print(f"📊 Submission rows: {len(submission_df)}")
        print(f"📊 ID range: {submission_df['id'].min()} to {submission_df['id'].max()}")
        
        # Show first few rows
        print(f"\n📄 First 5 rows of submission:")
        print(submission_df.head())
        
        return submission_path

def main():
    """Main function"""
    print("=" * 60)
    print("LIGHTWEIGHT NEURAL NETWORK TRAINING")
    print("=" * 60)
    
    # Set memory growth for GPU
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
    except:
        print("⚠️ GPU configuration failed, using CPU")
    
    # Create model
    nn_model = LightweightNeuralNetwork()
    
    # Load and prepare data
    X_train, X_val, y_train, y_val = nn_model.load_and_prepare_data()
    
    # Train model
    history, mse, mae, r2 = nn_model.train_model(X_train, X_val, y_train, y_val)
    
    # Make predictions
    test_ids, predictions = nn_model.predict_test_data()
    
    # Save results
    submission_path = nn_model.save_results(test_ids, predictions, mse, mae, r2)
    
    print(f"\n✅ Neural Network training completed!")
    print(f"📤 Ready to submit: {submission_path}")
    
    # Clean up memory
    del X_train, X_val, y_train, y_val
    gc.collect()

if __name__ == "__main__":
    main() 