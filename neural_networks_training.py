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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Memory optimization
import gc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available, skipping neural network models")
    TENSORFLOW_AVAILABLE = False

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NeuralNetworksTrainer:
    def __init__(self):
        """Initialize neural networks training."""
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
        
    def create_sequences(self, data, target, sequence_length=10):
        """Create sequences for time series prediction."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(target[i + sequence_length])
        return np.array(X), np.array(y)
        
    def build_lstm_model(self, input_shape):
        """Build LSTM model."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def build_gru_model(self, input_shape):
        """Build GRU model."""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def build_transformer_model(self, input_shape, num_heads=4):
        """Build Transformer model."""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=32)(inputs, inputs)
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
        
        # Feed forward network
        ffn_output = Dense(128, activation='relu')(attention_output)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
        
        # Global average pooling and output
        pooled_output = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        outputs = Dense(1)(pooled_output)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def train_lstm(self, X, y, feature_cols):
        """Train LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            print("SKIPPED: LSTM model (TensorFlow not available)")
            return None
            
        print("\nTraining LSTM model...")
        
        try:
            # Use a subset for computational efficiency
            sample_size = min(20000, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            X_test_scaled = scaler.transform(self.X_test)
            
            # Create sequences
            sequence_length = 10
            X_seq, y_seq = self.create_sequences(X_scaled, y_sample.values, sequence_length)
            
            # Split data
            split_idx = int(len(X_seq) * 0.8)
            X_train_seq = X_seq[:split_idx]
            y_train_seq = y_seq[:split_idx]
            X_val_seq = X_seq[split_idx:]
            y_val_seq = y_seq[split_idx:]
            
            # Build and train model
            model = self.build_lstm_model((sequence_length, X_scaled.shape[1]))
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            val_pred = model.predict(X_val_seq)
            score = pearsonr(y_val_seq, val_pred.flatten())[0]
            
            # Predict test set
            # Create sequences for test data
            test_sequences = []
            for i in range(len(X_test_scaled) - sequence_length + 1):
                test_sequences.append(X_test_scaled[i:(i + sequence_length)])
            test_sequences = np.array(test_sequences)
            
            test_pred = model.predict(test_sequences)
            
            # Pad predictions to match test size
            predictions = np.zeros(len(self.X_test))
            predictions[:len(test_pred)] = test_pred.flatten()
            if len(test_pred) < len(predictions):
                predictions[len(test_pred):] = test_pred[-1]
            
            # Store results
            self.models['LSTM'] = {'model': model, 'scaler': scaler}
            self.results['LSTM'] = {
                'avg_score': score,
                'std_score': 0.0,
                'scores': [score],
                'predictions': predictions
            }
            
            print(f"SUCCESS: LSTM: {score:.4f}")
            return predictions
            
        except Exception as e:
            print(f"ERROR: LSTM training failed: {e}")
            return None
        
    def train_gru(self, X, y, feature_cols):
        """Train GRU model."""
        if not TENSORFLOW_AVAILABLE:
            print("SKIPPED: GRU model (TensorFlow not available)")
            return None
            
        print("\nTraining GRU model...")
        
        try:
            # Use a subset for computational efficiency
            sample_size = min(20000, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            X_test_scaled = scaler.transform(self.X_test)
            
            # Create sequences
            sequence_length = 10
            X_seq, y_seq = self.create_sequences(X_scaled, y_sample.values, sequence_length)
            
            # Split data
            split_idx = int(len(X_seq) * 0.8)
            X_train_seq = X_seq[:split_idx]
            y_train_seq = y_seq[:split_idx]
            X_val_seq = X_seq[split_idx:]
            y_val_seq = y_seq[split_idx:]
            
            # Build and train model
            model = self.build_gru_model((sequence_length, X_scaled.shape[1]))
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            val_pred = model.predict(X_val_seq)
            score = pearsonr(y_val_seq, val_pred.flatten())[0]
            
            # Predict test set
            test_sequences = []
            for i in range(len(X_test_scaled) - sequence_length + 1):
                test_sequences.append(X_test_scaled[i:(i + sequence_length)])
            test_sequences = np.array(test_sequences)
            
            test_pred = model.predict(test_sequences)
            
            # Pad predictions to match test size
            predictions = np.zeros(len(self.X_test))
            predictions[:len(test_pred)] = test_pred.flatten()
            if len(test_pred) < len(predictions):
                predictions[len(test_pred):] = test_pred[-1]
            
            # Store results
            self.models['GRU'] = {'model': model, 'scaler': scaler}
            self.results['GRU'] = {
                'avg_score': score,
                'std_score': 0.0,
                'scores': [score],
                'predictions': predictions
            }
            
            print(f"SUCCESS: GRU: {score:.4f}")
            return predictions
            
        except Exception as e:
            print(f"ERROR: GRU training failed: {e}")
            return None
        
    def train_transformer(self, X, y, feature_cols):
        """Train Transformer model."""
        if not TENSORFLOW_AVAILABLE:
            print("SKIPPED: Transformer model (TensorFlow not available)")
            return None
            
        print("\nTraining Transformer model...")
        
        try:
            # Use a subset for computational efficiency
            sample_size = min(15000, len(X))
            sample_idx = np.random.choice(len(X), sample_size, replace=False)
            
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            X_test_scaled = scaler.transform(self.X_test)
            
            # Create sequences
            sequence_length = 10
            X_seq, y_seq = self.create_sequences(X_scaled, y_sample.values, sequence_length)
            
            # Split data
            split_idx = int(len(X_seq) * 0.8)
            X_train_seq = X_seq[:split_idx]
            y_train_seq = y_seq[:split_idx]
            X_val_seq = X_seq[split_idx:]
            y_val_seq = y_seq[split_idx:]
            
            # Build and train model
            model = self.build_transformer_model((sequence_length, X_scaled.shape[1]))
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=30,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            val_pred = model.predict(X_val_seq)
            score = pearsonr(y_val_seq, val_pred.flatten())[0]
            
            # Predict test set
            test_sequences = []
            for i in range(len(X_test_scaled) - sequence_length + 1):
                test_sequences.append(X_test_scaled[i:(i + sequence_length)])
            test_sequences = np.array(test_sequences)
            
            test_pred = model.predict(test_sequences)
            
            # Pad predictions to match test size
            predictions = np.zeros(len(self.X_test))
            predictions[:len(test_pred)] = test_pred.flatten()
            if len(test_pred) < len(predictions):
                predictions[len(test_pred):] = test_pred[-1]
            
            # Store results
            self.models['Transformer'] = {'model': model, 'scaler': scaler}
            self.results['Transformer'] = {
                'avg_score': score,
                'std_score': 0.0,
                'scores': [score],
                'predictions': predictions
            }
            
            print(f"SUCCESS: Transformer: {score:.4f}")
            return predictions
            
        except Exception as e:
            print(f"ERROR: Transformer training failed: {e}")
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
        filename = f'/kaggle/working/results/{model_name.lower()}_submission.csv'
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
        results_df.to_csv('/kaggle/working/results/neural_networks_results.csv', index=False)
        print("SUCCESS: Results summary saved")
        
        # Print results
        print("\n" + "="*80)
        print("NEURAL NETWORKS RESULTS")
        print("="*80)
        print(results_df.to_string(index=False))
        
        # Create submission files
        for model_name, result in self.results.items():
            self.create_submission_file(result['predictions'], model_name)
        
        # Memory optimization
        gc.collect()
        
    def run_training(self):
        """Run complete neural networks training pipeline."""
        print("="*80)
        print("NEURAL NETWORKS TRAINING PIPELINE")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Get feature columns
        feature_cols = [col for col in self.X_train.columns if col != 'label']
        
        # Train models
        print("\n" + "="*80)
        print("TRAINING NEURAL NETWORK MODELS")
        print("="*80)
        
        # LSTM Model
        self.train_lstm(self.X_train, self.y_train, feature_cols)
        
        # GRU Model
        self.train_gru(self.X_train, self.y_train, feature_cols)
        
        # Transformer Model
        self.train_transformer(self.X_train, self.y_train, feature_cols)
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("NEURAL NETWORKS TRAINING COMPLETED")
        print("="*80)

def main():
    """Main function to run the training pipeline."""
    trainer = NeuralNetworksTrainer()
    trainer.run_training()

if __name__ == "__main__":
    main() 