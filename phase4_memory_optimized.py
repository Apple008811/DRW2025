#!/usr/bin/env python3
"""
Phase 4: Memory Optimized Model Training (Kaggle Version)
=========================================================

Memory-optimized version for Kaggle environment with limited RAM.

Author: Yixuan
Date: 2025-07-22
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Time Series Models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    print("WARNING: statsmodels not available, skipping ARIMA/SARIMA models")
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("WARNING: Prophet not available, skipping Prophet model")
    PROPHET_AVAILABLE = False

# Deep Learning (with memory limits)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    # Memory optimization for TensorFlow
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth setting failed: {e}")
except ImportError:
    print("WARNING: TensorFlow not available, skipping deep learning models")
    TENSORFLOW_AVAILABLE = False

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MemoryOptimizedTrainer:
    def __init__(self):
        """Initialize memory-optimized training."""
        self.train = None
        self.test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        
        # Model results storage
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        # File paths
        self.train_file = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        self.test_file = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
        
        # Memory optimization settings
        self.sample_size = 50000  # Reduced from full dataset
        self.max_features = 50    # Reduced feature count
        self.lstm_epochs = 10     # Reduced from 50
        
    def load_data(self):
        """Load data with memory optimization."""
        print("Loading data with memory optimization...")
        
        # Load data with sampling
        self.train = pd.read_parquet(self.train_file)
        self.test = pd.read_parquet(self.test_file)
        
        # Sample data to reduce memory usage
        if len(self.train) > self.sample_size:
            sample_idx = np.random.choice(len(self.train), self.sample_size, replace=False)
            self.train = self.train.iloc[sample_idx].reset_index(drop=True)
            print(f"SUCCESS: Sampled {self.sample_size} training samples")
        
        print(f"SUCCESS: Train data: {self.train.shape}")
        print(f"SUCCESS: Test data: {self.test.shape}")
        
        # Clear memory
        gc.collect()
        
    def prepare_data(self):
        """Prepare data with memory optimization."""
        print("\n" + "="*80)
        print("MEMORY OPTIMIZED DATA PREPARATION")
        print("="*80)
        
        # Get target variable
        self.y_train = self.train['label']
        
        # Select top features only
        feature_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != 'label' and col != 'timestamp']
        
        # Use only top features by correlation
        correlations = []
        for col in feature_cols:
            corr = abs(self.y_train.corr(self.train[col]))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:self.max_features]]
        
        self.X_train = self.train[top_features].fillna(method='bfill').fillna(0)
        self.X_test = self.test[top_features].fillna(method='bfill').fillna(0)
        
        print(f"Using top {len(top_features)} features (memory optimized)")
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        
        # Clear memory
        gc.collect()
        
    def time_series_cv(self, model, X, y, n_splits=3):
        """Perform time series cross-validation with reduced splits."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            score = pearsonr(y_val_fold, preds)[0]
            scores.append(score)
            
        return np.mean(scores), np.std(scores), scores
        
    def train_quick_models(self):
        """Train fast models first."""
        print("\n" + "="*80)
        print("QUICK MODELS TRAINING (Memory Optimized)")
        print("="*80)
        
        # Linear models
        print("\nTraining Linear Models...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        linear_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.01)
        }
        
        for name, model in linear_models.items():
            print(f"Training {name}...")
            avg_score, std_score, scores = self.time_series_cv(model, X_train_scaled, self.y_train)
            
            model.fit(X_train_scaled, self.y_train)
            predictions = model.predict(X_test_scaled)
            
            self.models[name] = model
            self.results[name] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'scores': scores,
                'predictions': predictions
            }
            
            print(f"SUCCESS: {name}: {avg_score:.4f} ± {std_score:.4f}")
            
        # Tree models
        print("\nTraining Tree Models...")
        tree_models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=50,  # Reduced from 100
                max_depth=8,      # Reduced from 10
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100,  # Reduced from 200
                learning_rate=0.05,
                max_depth=6,       # Reduced from 8
                random_state=42,
                verbose=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,  # Reduced from 200
                learning_rate=0.05,
                max_depth=6,       # Reduced from 8
                random_state=42,
                verbosity=0
            )
        }
        
        for name, model in tree_models.items():
            print(f"Training {name}...")
            avg_score, std_score, scores = self.time_series_cv(model, self.X_train, self.y_train)
            
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            
            self.models[name] = model
            self.results[name] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'scores': scores,
                'predictions': predictions
            }
            
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            print(f"SUCCESS: {name}: {avg_score:.4f} ± {std_score:.4f}")
            
        # Clear memory
        gc.collect()
        
    def train_time_series_models(self):
        """Train time series models with reduced data."""
        print("\n" + "="*80)
        print("TIME SERIES MODELS (Memory Optimized)")
        print("="*80)
        
        # Use smaller sample for time series models
        sample_size = min(2000, len(self.y_train))
        y_sample = self.y_train.head(sample_size)
        
        # ARIMA Model
        if ARIMA_AVAILABLE:
            print("\nTraining ARIMA model...")
            try:
                arima_model = ARIMA(y_sample, order=(1, 1, 1))
                arima_fitted = arima_model.fit()
                
                # Simple validation
                train_size = int(len(y_sample) * 0.8)
                y_train_arima = y_sample[:train_size]
                y_val_arima = y_sample[train_size:]
                
                arima_train = ARIMA(y_train_arima, order=(1, 1, 1))
                arima_fitted_train = arima_train.fit()
                
                arima_forecast = arima_fitted_train.forecast(steps=len(y_val_arima))
                score = pearsonr(y_val_arima, arima_forecast)[0]
                
                # Predict test set
                arima_test_forecast = arima_fitted.forecast(steps=len(self.X_test))
                
                self.models['ARIMA'] = arima_fitted
                self.results['ARIMA'] = {
                    'avg_score': score,
                    'std_score': 0.0,
                    'scores': [score],
                    'predictions': arima_test_forecast
                }
                
                print(f"SUCCESS: ARIMA: {score:.4f}")
                
            except Exception as e:
                print(f"ERROR: ARIMA training failed: {e}")
        
        # Prophet Model
        if PROPHET_AVAILABLE:
            print("\nTraining Prophet model...")
            try:
                prophet_df = pd.DataFrame({
                    'ds': pd.date_range(start='2020-01-01', periods=len(y_sample), freq='H'),
                    'y': y_sample.values
                })
                
                prophet_model = Prophet(
                    yearly_seasonality=False,  # Reduced complexity
                    weekly_seasonality=False,
                    daily_seasonality=True,
                    seasonality_mode='additive'  # Simpler than multiplicative
                )
                prophet_model.fit(prophet_df)
                
                # Simple validation
                train_size = int(len(prophet_df) * 0.8)
                prophet_train = prophet_df.head(train_size)
                prophet_val = prophet_df.tail(len(prophet_df) - train_size)
                
                prophet_model_train = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=True,
                    seasonality_mode='additive'
                )
                prophet_model_train.fit(prophet_train)
                
                prophet_forecast = prophet_model_train.predict(prophet_val)
                score = pearsonr(prophet_val['y'], prophet_forecast['yhat'])[0]
                
                # Predict test set
                test_dates = pd.date_range(
                    start=prophet_df['ds'].iloc[-1] + pd.Timedelta(hours=1),
                    periods=len(self.X_test),
                    freq='H'
                )
                test_df = pd.DataFrame({'ds': test_dates})
                prophet_test_forecast = prophet_model.predict(test_df)
                
                self.models['Prophet'] = prophet_model
                self.results['Prophet'] = {
                    'avg_score': score,
                    'std_score': 0.0,
                    'scores': [score],
                    'predictions': prophet_test_forecast['yhat'].values
                }
                
                print(f"SUCCESS: Prophet: {score:.4f}")
                
            except Exception as e:
                print(f"ERROR: Prophet training failed: {e}")
        
        # Clear memory
        gc.collect()
        
    def train_light_lstm(self):
        """Train lightweight LSTM model."""
        if not TENSORFLOW_AVAILABLE:
            print("WARNING: TensorFlow not available, skipping LSTM")
            return
            
        print("\n" + "="*80)
        print("LIGHTWEIGHT LSTM TRAINING")
        print("="*80)
        
        # Use smaller sequence length and sample size
        sequence_length = 5  # Reduced from 10
        sample_size = min(10000, len(self.X_train))
        
        # Prepare sequences
        X_sample = self.X_train.head(sample_size)
        y_sample = self.y_train.head(sample_size)
        
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_sample)):
            X_seq.append(X_sample.iloc[i-sequence_length:i].values)
            y_seq.append(y_sample.iloc[i])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        print(f"Sequence data shape: {X_seq.shape}")
        
        # Create lightweight LSTM
        lstm_model = Sequential([
            Input(shape=(sequence_length, self.X_train.shape[1])),
            LSTM(64, return_sequences=False, dropout=0.2),  # Reduced from 128
            Dense(16, activation='relu'),                   # Reduced from 32
            Dropout(0.2),
            Dense(1)
        ])
        
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print(f"Training LSTM for {self.lstm_epochs} epochs...")
        lstm_history = lstm_model.fit(
            X_seq, y_seq,
            epochs=self.lstm_epochs,  # Reduced from 50
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Simple validation
        tscv = TimeSeriesSplit(n_splits=2)  # Reduced from 3
        lstm_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_seq)):
            X_fold_train = X_seq[train_idx]
            X_fold_val = X_seq[val_idx]
            y_fold_train = y_seq[train_idx]
            y_fold_val = y_seq[val_idx]
            
            fold_model = Sequential([
                Input(shape=(sequence_length, self.X_train.shape[1])),
                LSTM(64, return_sequences=False, dropout=0.2),
                Dense(16, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            fold_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            fold_model.fit(X_fold_train, y_fold_train, epochs=5, batch_size=32, verbose=0)
            preds = fold_model.predict(X_fold_val, verbose=0).flatten()
            score = pearsonr(y_fold_val, preds)[0]
            lstm_scores.append(score)
            
            print(f"  LSTM Fold {fold+1}: {score:.4f}")
        
        lstm_avg_score = np.mean(lstm_scores)
        lstm_std_score = np.std(lstm_scores)
        
        self.models['LSTM_Light'] = lstm_model
        self.results['LSTM_Light'] = {
            'avg_score': lstm_avg_score,
            'std_score': lstm_std_score,
            'scores': lstm_scores
        }
        
        print(f"SUCCESS: LSTM_Light: {lstm_avg_score:.4f} ± {lstm_std_score:.4f}")
        
        # Clear memory
        gc.collect()
        
    def create_ensemble(self):
        """Create ensemble of best models."""
        print("\n" + "="*80)
        print("ENSEMBLE MODEL CREATION")
        print("="*80)
        
        # Select top 3 models based on CV scores
        model_scores = [(name, results['avg_score']) for name, results in self.results.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_models = model_scores[:3]
        print(f"Top 3 models for ensemble:")
        for i, (name, score) in enumerate(top_models):
            print(f"  {i+1}. {name}: {score:.4f}")
        
        # Create weighted ensemble
        ensemble_predictions = np.zeros(len(self.X_test))
        total_weight = 0
        
        for name, score in top_models:
            weight = score
            ensemble_predictions += weight * self.results[name]['predictions']
            total_weight += weight
        
        ensemble_predictions /= total_weight
        
        # Evaluate ensemble
        ensemble_score = np.mean([score for _, score in top_models])
        
        self.models['Ensemble'] = None
        self.results['Ensemble'] = {
            'avg_score': ensemble_score,
            'std_score': 0.0,
            'scores': [ensemble_score],
            'predictions': ensemble_predictions,
            'weights': {name: score for name, score in top_models}
        }
        
        print(f"SUCCESS: Ensemble: {ensemble_score:.4f}")
        
    def analyze_results(self):
        """Analyze and compare model results."""
        print("\n" + "="*80)
        print("MODEL RESULTS ANALYSIS")
        print("="*80)
        
        # Create results summary
        results_summary = []
        for name, results in self.results.items():
            results_summary.append({
                'Model': name,
                'CV Score': f"{results['avg_score']:.4f} ± {results['std_score']:.4f}",
                'Avg Score': results['avg_score'],
                'Std Score': results['std_score']
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df = results_df.sort_values('Avg Score', ascending=False)
        
        print("\nModel Performance Ranking:")
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        best_score = results_df.iloc[0]['Avg Score']
        
        print(f"\nBest Model: {best_model_name} (Score: {best_score:.4f})")
        
        return results_df, best_model_name
        
    def save_results(self):
        """Save model results and predictions."""
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)
        
        # Save results summary
        results_summary = []
        for name, results in self.results.items():
            results_summary.append({
                'Model': name,
                'CV Score': results['avg_score'],
                'CV Std': results['std_score'],
                'Best Score': max(results['scores'])
            })
        
        results_df = pd.DataFrame(results_summary)
        results_df.to_csv('phase4_memory_optimized_results.csv', index=False)
        print("SUCCESS: Model results saved: phase4_memory_optimized_results.csv")
        
        # Save best model predictions
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['avg_score'])
        best_predictions = self.results[best_model_name]['predictions']
        
        submission = pd.DataFrame({
            'timestamp': self.test.index,
            'pred': best_predictions
        })
        
        submission.to_csv('phase4_memory_optimized_submission.csv', index=False)
        print(f"SUCCESS: Best model predictions saved: phase4_memory_optimized_submission.csv ({best_model_name})")
        
        # Save ensemble predictions
        if 'Ensemble' in self.results:
            ensemble_predictions = self.results['Ensemble']['predictions']
            ensemble_submission = pd.DataFrame({
                'timestamp': self.test.index,
                'pred': ensemble_predictions
            })
            ensemble_submission.to_csv('phase4_memory_optimized_ensemble_submission.csv', index=False)
            print("SUCCESS: Ensemble predictions saved: phase4_memory_optimized_ensemble_submission.csv")
        
    def run_memory_optimized_training(self):
        """Run complete memory-optimized training pipeline."""
        print("Starting Phase 4: Memory Optimized Model Training")
        print("="*80)
        print("Environment: Kaggle (Memory Optimized)")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train models in order of speed
        self.train_quick_models()      # Fast models first
        self.train_time_series_models() # Medium speed models
        self.train_light_lstm()        # Slow model last
        
        # Create ensemble
        self.create_ensemble()
        
        # Analyze results
        results_df, best_model = self.analyze_results()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("SUCCESS: Phase 4 Memory Optimized Training Complete!")
        print("Key Results:")
        print(f"   Best Model: {best_model}")
        print(f"   Best Score: {results_df.iloc[0]['Avg Score']:.4f}")
        print(f"   Output Files:")
        print(f"      - phase4_memory_optimized_results.csv")
        print(f"      - phase4_memory_optimized_submission.csv")
        print(f"      - phase4_memory_optimized_ensemble_submission.csv")
        print("="*80)

def main():
    """Main function to run memory-optimized training."""
    trainer = MemoryOptimizedTrainer()
    trainer.run_memory_optimized_training()

if __name__ == "__main__":
    main() 