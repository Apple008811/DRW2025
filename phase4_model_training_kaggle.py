#!/usr/bin/env python3
"""
Phase 4: Model Training & Optimization (Kaggle Version)
=======================================================

Comprehensive model training, comparison, and optimization based on Phase 3 features.

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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

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

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelTrainerKaggle:
    def __init__(self):
        """Initialize model training for Kaggle environment."""
        self.train = None
        self.test = None
        self.train_features = None
        self.test_features = None
        
        # Model results storage
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
        # File paths
        self.train_file = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        self.test_file = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
        self.features_file = '/kaggle/working/engineered_features.parquet'
        
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
        
        # Select features (exclude non-numeric and target)
        feature_cols = self.train_features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != 'label' and col != 'timestamp']
        
        # Handle missing values
        self.X_train = self.train_features[feature_cols].fillna(method='bfill').fillna(0)
        self.X_test = self.test_features[feature_cols].fillna(method='bfill').fillna(0)
        
        print(f"Final feature set: {self.X_train.shape[1]} features")
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        
        # Feature correlation analysis
        print(f"\nFeature correlation analysis:")
        correlations = []
        for col in self.X_train.columns:
            corr = abs(self.y_train.corr(self.X_train[col]))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        print(f"Top 10 feature correlations:")
        for i, (col, corr) in enumerate(correlations[:10]):
            print(f"  {i+1}. {col}: {corr:.4f}")
            
    def time_series_cv(self, model, X, y, n_splits=5):
        """Perform time series cross-validation."""
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
        
    def train_time_series_models(self):
        """Train time series models (ARIMA, SARIMA, Prophet)."""
        print("\n" + "="*80)
        print("TIME SERIES MODELS TRAINING")
        print("="*80)
        
        # Prepare time series data
        y_series = self.y_train.reset_index(drop=True)
        
        # ARIMA Model
        if ARIMA_AVAILABLE:
            print("\nTraining ARIMA model...")
            try:
                # Use a subset for computational efficiency
                sample_size = min(5000, len(y_series))
                y_sample = y_series.head(sample_size)
                
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
        
        # SARIMA Model (Skipped due to computational complexity)
        if ARIMA_AVAILABLE and False:  # Disabled for memory optimization
            print("\nTraining SARIMA model...")
            try:
                # Use a subset for computational efficiency
                sample_size = min(3000, len(y_series))
                y_sample = y_series.head(sample_size)
                
                # Fit SARIMA model
                sarima_model = SARIMAX(y_sample, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                sarima_fitted = sarima_model.fit(disp=False)
                
                # Simple validation
                train_size = int(len(y_sample) * 0.8)
                y_train_sarima = y_sample[:train_size]
                y_val_sarima = y_sample[train_size:]
                
                # Fit on training data
                sarima_train = SARIMAX(y_train_sarima, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
                sarima_fitted_train = sarima_train.fit(disp=False)
                
                # Predict validation set
                sarima_forecast = sarima_fitted_train.forecast(steps=len(y_val_sarima))
                score = pearsonr(y_val_sarima, sarima_forecast)[0]
                
                # Predict test set
                sarima_test_forecast = sarima_fitted.forecast(steps=len(self.X_test))
                
                self.models['SARIMA'] = sarima_fitted
                self.results['SARIMA'] = {
                    'avg_score': score,
                    'std_score': 0.0,
                    'scores': [score],
                    'predictions': sarima_test_forecast
                }
                
                print(f"SUCCESS: SARIMA: {score:.4f}")
                
            except Exception as e:
                print(f"ERROR: SARIMA training failed: {e}")
        
        # Prophet Model
        if PROPHET_AVAILABLE:
            print("\nTraining Prophet model...")
            try:
                # Prepare data for Prophet
                sample_size = min(2000, len(y_series))
                y_sample = y_series.head(sample_size)
                
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
    
    def train_linear_models(self):
        """Train linear models."""
        print("\n" + "="*80)
        print("LINEAR MODELS TRAINING")
        print("="*80)
        
        # Standardize features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        linear_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.01)
        }
        
        for name, model in linear_models.items():
            print(f"\nTraining {name}...")
            avg_score, std_score, scores = self.time_series_cv(model, X_train_scaled, self.y_train)
            
            # Train final model
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
            
    def train_tree_models(self):
        """Train tree-based models."""
        print("\n" + "="*80)
        print("TREE-BASED MODELS TRAINING")
        print("="*80)
        
        tree_models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                verbose=-1
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                verbosity=0
            )
        }
        
        for name, model in tree_models.items():
            print(f"\nTraining {name}...")
            avg_score, std_score, scores = self.time_series_cv(model, self.X_train, self.y_train)
            
            # Train final model
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            
            self.models[name] = model
            self.results[name] = {
                'avg_score': avg_score,
                'std_score': std_score,
                'scores': scores,
                'predictions': predictions
            }
            
            # Get feature importance for tree models
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            print(f"SUCCESS: {name}: {avg_score:.4f} ± {std_score:.4f}")
            
    def train_svm_model(self):
        """Train Support Vector Regression model."""
        print("\n" + "="*80)
        print("SUPPORT VECTOR REGRESSION TRAINING")
        print("="*80)
        
        # Use a subset of data for SVR (computationally expensive)
        sample_size = min(10000, len(self.X_train))
        sample_idx = np.random.choice(len(self.X_train), sample_size, replace=False)
        
        X_train_sample = self.X_train.iloc[sample_idx]
        y_train_sample = self.y_train.iloc[sample_idx]
        
        # Standardize features for SVR
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sample)
        X_test_scaled = scaler.transform(self.X_test)
        
        print(f"Using {sample_size} samples for SVR training...")
        
        # Train SVR model
        svr_model = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        # Simple train/test split for SVR (faster than CV)
        split_idx = int(len(X_train_scaled) * 0.8)
        X_train_split = X_train_scaled[:split_idx]
        X_val_split = X_train_scaled[split_idx:]
        y_train_split = y_train_sample[:split_idx]
        y_val_split = y_train_sample[split_idx:]
        
        svr_model.fit(X_train_split, y_train_split)
        val_preds = svr_model.predict(X_val_split)
        score = pearsonr(y_val_split, val_preds)[0]
        
        # Train on full sample
        svr_model.fit(X_train_scaled, y_train_sample)
        predictions = svr_model.predict(X_test_scaled)
        
        self.models['SVR'] = svr_model
        self.results['SVR'] = {
            'avg_score': score,
            'std_score': 0.0,
            'scores': [score],
            'predictions': predictions
        }
        
        print(f"SUCCESS: SVR: {score:.4f}")
        
    def hyperparameter_tuning(self, model_name='LightGBM'):
        """Perform hyperparameter tuning for specified model."""
        print("\n" + "="*80)
        print(f"HYPERPARAMETER TUNING FOR {model_name.upper()}")
        print("="*80)
        
        if model_name == 'LightGBM':
            # LightGBM parameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [6, 8, 10],
                'num_leaves': [31, 63, 127],
                'feature_fraction': [0.8, 0.9, 1.0]
            }
            base_model = lgb.LGBMRegressor(random_state=42, verbose=-1)
            
        elif model_name == 'XGBoost':
            # XGBoost parameter grid
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [6, 8, 10],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            base_model = xgb.XGBRegressor(random_state=42, verbosity=0)
        
        print(f"Parameter grid: {len(param_grid)} parameters")
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Grid search with time series CV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"SUCCESS: Best parameters: {grid_search.best_params_}")
        print(f"SUCCESS: Best CV score: {-grid_search.best_score_:.4f}")
        
        # Train best model
        best_model = grid_search.best_estimator_
        best_model.fit(self.X_train, self.y_train)
        predictions = best_model.predict(self.X_test)
        
        # Evaluate with time series CV
        avg_score, std_score, scores = self.time_series_cv(best_model, self.X_train, self.y_train)
        
        self.models[f'{model_name}_Tuned'] = best_model
        self.results[f'{model_name}_Tuned'] = {
            'avg_score': avg_score,
            'std_score': std_score,
            'scores': scores,
            'predictions': predictions,
            'best_params': grid_search.best_params_
        }
        
        print(f"SUCCESS: {model_name}_Tuned: {avg_score:.4f} ± {std_score:.4f}")
        
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
            weight = score  # Use CV score as weight
            ensemble_predictions += weight * self.results[name]['predictions']
            total_weight += weight
        
        ensemble_predictions /= total_weight
        
        # Evaluate ensemble (use average of top models' CV scores)
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
        
    def create_visualizations(self):
        """Create visualizations for model comparison."""
        print("\n" + "="*80)
        print("CREATING MODEL COMPARISON VISUALIZATIONS")
        print("="*80)
        
        # Prepare data for plotting
        model_names = list(self.results.keys())
        scores = [self.results[name]['avg_score'] for name in model_names]
        std_scores = [self.results[name]['std_score'] for name in model_names]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Model performance comparison
        bars = ax1.bar(range(len(model_names)), scores, yerr=std_scores, capsize=5)
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Pearson Correlation')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        
        # Color bars based on performance
        colors = ['green' if score > 0.5 else 'orange' if score > 0.3 else 'red' for score in scores]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Feature importance plot (if available)
        if 'LightGBM' in self.feature_importance:
            top_features = self.feature_importance['LightGBM'].head(10)
            ax2.barh(range(len(top_features)), top_features['importance'])
            ax2.set_yticks(range(len(top_features)))
            ax2.set_yticklabels(top_features['feature'])
            ax2.set_xlabel('Feature Importance')
            ax2.set_title('Top 10 Feature Importance (LightGBM)')
        
        plt.tight_layout()
        plt.show()
        
        # Save plot
        plt.savefig('phase4_model_comparison.png', dpi=300, bbox_inches='tight')
        print("SUCCESS: Model comparison plot saved: phase4_model_comparison.png")
        
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
        results_df.to_csv('phase4_model_results.csv', index=False)
        print("SUCCESS: Model results saved: phase4_model_results.csv")
        
        # Save best model predictions
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['avg_score'])
        best_predictions = self.results[best_model_name]['predictions']
        
        submission = pd.DataFrame({
            'timestamp': self.test.index,
            'pred': best_predictions
        })
        
        submission.to_csv('phase4_best_submission.csv', index=False)
        print(f"SUCCESS: Best model predictions saved: phase4_best_submission.csv ({best_model_name})")
        
        # Save ensemble predictions
        if 'Ensemble' in self.results:
            ensemble_predictions = self.results['Ensemble']['predictions']
            ensemble_submission = pd.DataFrame({
                'timestamp': self.test.index,
                'pred': ensemble_predictions
            })
            ensemble_submission.to_csv('phase4_ensemble_submission.csv', index=False)
            print("SUCCESS: Ensemble predictions saved: phase4_ensemble_submission.csv")
        
    def run_complete_training(self):
        """Run complete model training pipeline."""
        print("Starting Phase 4: Model Training & Optimization")
        print("="*80)
        print("Environment: Kaggle (Full Dataset)")
        print("="*80)
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
        # Train all model types
        self.train_time_series_models()  # ARIMA, Prophet (SARIMA disabled)
        self.train_linear_models()       # Linear, Ridge, Lasso
        self.train_tree_models()         # Random Forest, LightGBM, XGBoost
        self.train_svm_model()           # SVR
        
        # Hyperparameter tuning for best model
        self.hyperparameter_tuning('LightGBM')
        
        # Create ensemble
        self.create_ensemble()
        
        # Analyze results
        results_df, best_model = self.analyze_results()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save results
        self.save_results()
        
        print("\n" + "="*80)
        print("SUCCESS: Phase 4 Complete Model Training Finished!")
        print("Key Results:")
        print(f"   Best Model: {best_model}")
        print(f"   Best Score: {results_df.iloc[0]['Avg Score']:.4f}")
        print(f"   Output Files:")
        print(f"      - phase4_model_results.csv")
        print(f"      - phase4_best_submission.csv")
        print(f"      - phase4_ensemble_submission.csv")
        print(f"      - phase4_model_comparison.png")
        print("="*80)

def main():
    """Main function to run the complete model training."""
    trainer = ModelTrainerKaggle()
    trainer.run_complete_training()

if __name__ == "__main__":
    main() 