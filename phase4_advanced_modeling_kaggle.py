#!/usr/bin/env python3
"""
Phase 4: Advanced Modeling & Ensembling (Kaggle Version)
========================================================

Advanced modeling with deep learning, Bayesian methods, and sophisticated ensembling.

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

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not available, skipping deep learning models")
    TENSORFLOW_AVAILABLE = False

# Bayesian Methods
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
    BAYESIAN_AVAILABLE = True
except ImportError:
    print("WARNING: Gaussian Process not available, skipping Bayesian methods")
    BAYESIAN_AVAILABLE = False

# Advanced ML
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedModelerKaggle:
    def __init__(self):
        """Initialize advanced modeling for Kaggle environment."""
        self.train = None
        self.test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        
        # Model storage
        self.models = {}
        self.results = {}
        self.ensemble_predictions = None
        
        # File paths
        self.train_file = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        self.test_file = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
        self.phase3_results = '/kaggle/working/phase3_model_results.csv'
        
    def load_data(self):
        """Load data and Phase 3 results."""
        print("Loading data and Phase 3 results...")
        
        # Load data
        self.train = pd.read_parquet(self.train_file)
        self.test = pd.read_parquet(self.test_file)
        
        # Load Phase 3 results if available
        try:
            phase3_results = pd.read_csv(self.phase3_results)
            print(f"SUCCESS: Phase 3 results loaded: {len(phase3_results)} models")
        except:
            print("WARNING: Phase 3 results not found, will use default models")
            phase3_results = None
        
        print(f"SUCCESS: Train data: {self.train.shape}")
        print(f"SUCCESS: Test data: {self.test.shape}")
        
        return phase3_results
        
    def prepare_features(self):
        """Prepare features for advanced modeling."""
        print("\nPreparing features for advanced modeling...")
        
        # Get target variable
        self.y_train = self.train['label']
        
        # Select features (use top features for computational efficiency)
        feature_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != 'label' and col != 'timestamp']
        
        # Use top 100 features for advanced modeling
        correlations = []
        for col in feature_cols:
            corr = abs(self.y_train.corr(self.train[col]))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:100]]
        
        self.X_train = self.train[top_features].fillna(method='bfill').fillna(0)
        self.X_test = self.test[top_features].fillna(method='bfill').fillna(0)
        
        print(f"Using top {len(top_features)} features")
        print(f"Training samples: {self.X_train.shape[0]}")
        
    def create_lstm_model(self, input_shape):
        """Create LSTM model for time series prediction."""
        model = Sequential([
            Input(shape=input_shape),
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def create_gru_model(self, input_shape):
        """Create GRU model for time series prediction."""
        model = Sequential([
            Input(shape=input_shape),
            GRU(128, return_sequences=True, dropout=0.2),
            GRU(64, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
        
    def prepare_sequences(self, X, y, sequence_length=10):
        """Prepare sequences for RNN models."""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X.iloc[i-sequence_length:i].values)
            y_seq.append(y.iloc[i])
            
        return np.array(X_seq), np.array(y_seq)
        
    def train_deep_learning_models(self):
        """Train deep learning models."""
        if not TENSORFLOW_AVAILABLE:
            print("WARNING: Skipping deep learning models (TensorFlow not available)")
            return
            
        print("\n" + "="*80)
        print("DEEP LEARNING MODELS TRAINING")
        print("="*80)
        
        # Prepare sequences
        sequence_length = 10
        X_train_seq, y_train_seq = self.prepare_sequences(self.X_train, self.y_train, sequence_length)
        
        print(f"Sequence data shape: {X_train_seq.shape}")
        
        # Train LSTM
        print("\nTraining LSTM model...")
        lstm_model = self.create_lstm_model((sequence_length, self.X_train.shape[1]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        lstm_history = lstm_model.fit(
            X_train_seq, y_train_seq,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate LSTM
        tscv = TimeSeriesSplit(n_splits=3)
        lstm_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_seq)):
            X_fold_train = X_train_seq[train_idx]
            X_fold_val = X_train_seq[val_idx]
            y_fold_train = y_train_seq[train_idx]
            y_fold_val = y_train_seq[val_idx]
            
            # Train on fold
            fold_model = self.create_lstm_model((sequence_length, self.X_train.shape[1]))
            fold_model.fit(X_fold_train, y_fold_train, epochs=5, batch_size=32, verbose=0)
            
            # Predict
            preds = fold_model.predict(X_fold_val, verbose=0).flatten()
            score = pearsonr(y_fold_val, preds)[0]
            lstm_scores.append(score)
            
            print(f"  LSTM Fold {fold+1}: {score:.4f}")
        
        lstm_avg_score = np.mean(lstm_scores)
        lstm_std_score = np.std(lstm_scores)
        
        self.models['LSTM'] = lstm_model
        self.results['LSTM'] = {
            'avg_score': lstm_avg_score,
            'std_score': lstm_std_score,
            'scores': lstm_scores
        }
        
        print(f"SUCCESS: LSTM: {lstm_avg_score:.4f} ± {lstm_std_score:.4f}")
        
        # Train GRU
        print("\nTraining GRU model...")
        gru_model = self.create_gru_model((sequence_length, self.X_train.shape[1]))
        
        gru_history = gru_model.fit(
            X_train_seq, y_train_seq,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate GRU
        gru_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_seq)):
            X_fold_train = X_train_seq[train_idx]
            X_fold_val = X_train_seq[val_idx]
            y_fold_train = y_train_seq[train_idx]
            y_fold_val = y_train_seq[val_idx]
            
            fold_model = self.create_gru_model((sequence_length, self.X_train.shape[1]))
            fold_model.fit(X_fold_train, y_fold_train, epochs=5, batch_size=32, verbose=0)
            
            preds = fold_model.predict(X_fold_val, verbose=0).flatten()
            score = pearsonr(y_fold_val, preds)[0]
            gru_scores.append(score)
            
            print(f"  GRU Fold {fold+1}: {score:.4f}")
        
        gru_avg_score = np.mean(gru_scores)
        gru_std_score = np.std(gru_scores)
        
        self.models['GRU'] = gru_model
        self.results['GRU'] = {
            'avg_score': gru_avg_score,
            'std_score': gru_std_score,
            'scores': gru_scores
        }
        
        print(f"SUCCESS: GRU: {gru_avg_score:.4f} ± {gru_std_score:.4f}")
        
    def train_bayesian_models(self):
        """Train Bayesian models."""
        if not BAYESIAN_AVAILABLE:
            print("WARNING: Skipping Bayesian models (Gaussian Process not available)")
            return
            
        print("\n" + "="*80)
        print("BAYESIAN MODELS TRAINING")
        print("="*80)
        
        # Scale features for Gaussian Process
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Different kernels to try
        kernels = {
            'RBF': RBF(length_scale=1.0),
            'Matern': Matern(length_scale=1.0, nu=1.5),
            'RationalQuadratic': RationalQuadratic(length_scale=1.0, alpha=1.0)
        }
        
        for kernel_name, kernel in kernels.items():
            print(f"\nTraining Gaussian Process with {kernel_name} kernel...")
            
            # Use smaller sample for computational efficiency
            sample_size = min(5000, len(X_train_scaled))
            sample_idx = np.random.choice(len(X_train_scaled), sample_size, replace=False)
            
            X_sample = X_train_scaled[sample_idx]
            y_sample = self.y_train.iloc[sample_idx]
            
            gp_model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                random_state=42
            )
            
            # Simple train/test split for GP
            split_idx = int(len(X_sample) * 0.8)
            X_train_split = X_sample[:split_idx]
            X_val_split = X_sample[split_idx:]
            y_train_split = y_sample[:split_idx]
            y_val_split = y_sample[split_idx:]
            
            gp_model.fit(X_train_split, y_train_split)
            val_preds = gp_model.predict(X_val_split)
            score = pearsonr(y_val_split, val_preds)[0]
            
            # Train on full sample
            gp_model.fit(X_sample, y_sample)
            predictions = gp_model.predict(X_test_scaled)
            
            self.models[f'GP_{kernel_name}'] = gp_model
            self.results[f'GP_{kernel_name}'] = {
                'avg_score': score,
                'std_score': 0.0,
                'scores': [score],
                'predictions': predictions
            }
            
            print(f"SUCCESS: GP_{kernel_name}: {score:.4f}")
            
    def create_advanced_ensembles(self):
        """Create advanced ensemble methods."""
        print("\n" + "="*80)
        print("ADVANCED ENSEMBLE METHODS")
        print("="*80)
        
        # Base models for ensemble
        base_models = [
            ('lgb', lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)),
            ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]
        
        # 1. Stacking Ensemble
        print("\nCreating Stacking Ensemble...")
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=LinearRegression(),
            cv=TimeSeriesSplit(n_splits=3)
        )
        
        # Evaluate stacking
        tscv = TimeSeriesSplit(n_splits=3)
        stacking_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            stacking_model.fit(X_fold_train, y_fold_train)
            preds = stacking_model.predict(X_fold_val)
            score = pearsonr(y_fold_val, preds)[0]
            stacking_scores.append(score)
            
            print(f"  Stacking Fold {fold+1}: {score:.4f}")
        
        stacking_avg_score = np.mean(stacking_scores)
        stacking_std_score = np.std(stacking_scores)
        
        # Train final stacking model
        stacking_model.fit(self.X_train, self.y_train)
        stacking_predictions = stacking_model.predict(self.X_test)
        
        self.models['Stacking'] = stacking_model
        self.results['Stacking'] = {
            'avg_score': stacking_avg_score,
            'std_score': stacking_std_score,
            'scores': stacking_scores,
            'predictions': stacking_predictions
        }
        
        print(f"SUCCESS: Stacking: {stacking_avg_score:.4f} ± {stacking_std_score:.4f}")
        
        # 2. Voting Ensemble
        print("\nCreating Voting Ensemble...")
        voting_model = VotingRegressor(
            estimators=base_models,
            weights=[0.4, 0.4, 0.2]  # Give more weight to boosting models
        )
        
        # Evaluate voting
        voting_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X_train)):
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            
            voting_model.fit(X_fold_train, y_fold_train)
            preds = voting_model.predict(X_fold_val)
            score = pearsonr(y_fold_val, preds)[0]
            voting_scores.append(score)
            
            print(f"  Voting Fold {fold+1}: {score:.4f}")
        
        voting_avg_score = np.mean(voting_scores)
        voting_std_score = np.std(voting_scores)
        
        # Train final voting model
        voting_model.fit(self.X_train, self.y_train)
        voting_predictions = voting_model.predict(self.X_test)
        
        self.models['Voting'] = voting_model
        self.results['Voting'] = {
            'avg_score': voting_avg_score,
            'std_score': voting_std_score,
            'scores': voting_scores,
            'predictions': voting_predictions
        }
        
        print(f"SUCCESS: Voting: {voting_avg_score:.4f} ± {voting_std_score:.4f}")
        
    def post_process_predictions(self):
        """Apply post-processing techniques."""
        print("\n" + "="*80)
        print("POST-PROCESSING PREDICTIONS")
        print("="*80)
        
        # Get all predictions
        all_predictions = {}
        for name, results in self.results.items():
            if 'predictions' in results:
                all_predictions[name] = results['predictions']
        
        if not all_predictions:
            print("WARNING: No predictions available for post-processing")
            return
        
        # 1. Outlier handling
        print("\nApplying outlier handling...")
        for name, preds in all_predictions.items():
            # Remove extreme outliers (beyond 3 standard deviations)
            mean_pred = np.mean(preds)
            std_pred = np.std(preds)
            lower_bound = mean_pred - 3 * std_pred
            upper_bound = mean_pred + 3 * std_pred
            
            preds_cleaned = np.clip(preds, lower_bound, upper_bound)
            all_predictions[f"{name}_cleaned"] = preds_cleaned
            
            print(f"  {name}: clipped {np.sum((preds < lower_bound) | (preds > upper_bound))} outliers")
        
        # 2. Smoothing
        print("\nApplying smoothing...")
        for name, preds in all_predictions.items():
            if not name.endswith('_cleaned'):
                continue
                
            # Simple moving average smoothing
            window_size = 5
            preds_smoothed = pd.Series(preds).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            all_predictions[f"{name}_smoothed"] = preds_smoothed.values
            
            print(f"  {name}: applied {window_size}-point moving average")
        
        # 3. Calibration
        print("\nApplying calibration...")
        for name, preds in all_predictions.items():
            if not name.endswith('_smoothed'):
                continue
                
            # Simple scaling to match target distribution
            target_mean = self.y_train.mean()
            target_std = self.y_train.std()
            pred_mean = np.mean(preds)
            pred_std = np.std(preds)
            
            preds_calibrated = (preds - pred_mean) / pred_std * target_std + target_mean
            all_predictions[f"{name}_calibrated"] = preds_calibrated
            
            print(f"  {name}: calibrated to target distribution")
        
        # Store post-processed predictions
        self.post_processed_predictions = all_predictions
        
    def create_super_ensemble(self):
        """Create a super ensemble of all models."""
        print("\n" + "="*80)
        print("SUPER ENSEMBLE CREATION")
        print("="*80)
        
        # Get all available predictions
        available_predictions = []
        available_weights = []
        
        for name, results in self.results.items():
            if 'predictions' in results:
                available_predictions.append(results['predictions'])
                available_weights.append(results['avg_score'])
        
        if not available_predictions:
            print("WARNING: No predictions available for super ensemble")
            return
        
        # Create weighted ensemble
        weights = np.array(available_weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        super_ensemble_pred = np.zeros(len(available_predictions[0]))
        for pred, weight in zip(available_predictions, weights):
            super_ensemble_pred += weight * pred
        
        # Calculate ensemble score (weighted average of individual scores)
        ensemble_score = np.average(available_weights, weights=weights)
        
        self.models['Super_Ensemble'] = None
        self.results['Super_Ensemble'] = {
            'avg_score': ensemble_score,
            'std_score': 0.0,
            'scores': [ensemble_score],
            'predictions': super_ensemble_pred,
            'weights': dict(zip([name for name, results in self.results.items() if 'predictions' in results], weights))
        }
        
        print(f"SUCCESS: Super Ensemble: {ensemble_score:.4f}")
        print("Model weights:")
        for name, weight in self.results['Super_Ensemble']['weights'].items():
            print(f"   {name}: {weight:.3f}")
        
    def analyze_results(self):
        """Analyze and compare all results."""
        print("\n" + "="*80)
        print("ADVANCED MODELING RESULTS ANALYSIS")
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
        
        print("\nAdvanced Model Performance Ranking:")
        print(results_df.to_string(index=False))
        
        # Find best model
        best_model_name = results_df.iloc[0]['Model']
        best_score = results_df.iloc[0]['Avg Score']
        
        print(f"\nBest Advanced Model: {best_model_name} (Score: {best_score:.4f})")
        
        return results_df, best_model_name
        
    def save_advanced_results(self):
        """Save advanced modeling results."""
        print("\n" + "="*80)
        print("SAVING ADVANCED MODELING RESULTS")
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
        results_df.to_csv('phase4_advanced_results.csv', index=False)
        print("SUCCESS: Advanced results saved: phase4_advanced_results.csv")
        
        # Save best model predictions
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['avg_score'])
        if 'predictions' in self.results[best_model_name]:
            best_predictions = self.results[best_model_name]['predictions']
            
            submission = pd.DataFrame({
                'timestamp': self.test.index,
                'pred': best_predictions
            })
            
            submission.to_csv('phase4_advanced_submission.csv', index=False)
            print(f"SUCCESS: Best advanced model predictions saved: phase4_advanced_submission.csv ({best_model_name})")
        
        # Save super ensemble predictions
        if 'Super_Ensemble' in self.results:
            ensemble_predictions = self.results['Super_Ensemble']['predictions']
            ensemble_submission = pd.DataFrame({
                'timestamp': self.test.index,
                'pred': ensemble_predictions
            })
            ensemble_submission.to_csv('phase4_super_ensemble_submission.csv', index=False)
            print("SUCCESS: Super ensemble predictions saved: phase4_super_ensemble_submission.csv")
        
    def run_advanced_modeling(self):
        """Run complete advanced modeling pipeline."""
        print("Starting Phase 4: Advanced Modeling & Ensembling")
        print("="*80)
        print("Environment: Kaggle (Full Dataset)")
        print("="*80)
        
        # Load and prepare data
        phase3_results = self.load_data()
        self.prepare_features()
        
        # Train advanced models
        self.train_deep_learning_models()
        self.train_bayesian_models()
        
        # Create advanced ensembles
        self.create_advanced_ensembles()
        
        # Post-process predictions
        self.post_process_predictions()
        
        # Create super ensemble
        self.create_super_ensemble()
        
        # Analyze results
        results_df, best_model = self.analyze_results()
        
        # Save results
        self.save_advanced_results()
        
        print("\n" + "="*80)
        print("SUCCESS: Phase 4 Advanced Modeling & Ensembling Complete!")
        print("Key Results:")
        print(f"   Best Advanced Model: {best_model}")
        print(f"   Best Score: {results_df.iloc[0]['Avg Score']:.4f}")
        print(f"   Output Files:")
        print(f"      - phase4_advanced_results.csv")
        print(f"      - phase4_advanced_submission.csv")
        print(f"      - phase4_super_ensemble_submission.csv")
        print("="*80)

def main():
    """Main function to run advanced modeling."""
    modeler = AdvancedModelerKaggle()
    modeler.run_advanced_modeling()

if __name__ == "__main__":
    main() 