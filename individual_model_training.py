#!/usr/bin/env python3
"""
Individual Model Training Script
Train models one by one to avoid memory issues
"""

import pandas as pd
import numpy as np
import warnings
import time
import gc
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class IndividualModelTrainer:
    def __init__(self, data_path=None):
        # Try different possible data paths
        if data_path is None:
            self.possible_paths = [
                '/kaggle/working/quick_features.parquet',      # Quick feature engineering output
                '/kaggle/working/engineered_features.parquet', # Kaggle working directory
                '/kaggle/working/train_features.parquet',      # Phase 3 output
                '/kaggle/input/drw-crypto-market-prediction/train.parquet',  # Original data
                'data/engineered_features.parquet',            # Local path
                'data/train.parquet'                          # Local train data
            ]
        else:
            self.possible_paths = [data_path]
        
        self.data_path = None
        self.results = {}
        self.models = {
            '1': ('ARIMA', self.train_arima),
            '2': ('Prophet', self.train_prophet),
            '3': ('Linear Regression', self.train_linear),
            '4': ('Ridge', self.train_ridge),
            '5': ('Lasso', self.train_lasso),
            '6': ('Random Forest', self.train_random_forest),
            '7': ('LightGBM', self.train_lightgbm),
            '8': ('XGBoost', self.train_xgboost),
            '9': ('SVR', self.train_svr),
            '10': ('All Models', self.train_all_models)
        }
        
    def load_data(self):
        """Load engineered features data"""
        print("Searching for data files...")
        
        for path in self.possible_paths:
            print(f"Trying: {path}")
            try:
                if os.path.exists(path):
                    df = pd.read_parquet(path)
                    print(f"✅ Successfully loaded: {path}")
                    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
                    self.data_path = path
                    return df
                else:
                    print(f"   ❌ File not found")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n❌ No processed data file found.")
        print("Attempting to process raw data automatically...")
        return self.process_raw_data()
    
    def process_raw_data(self):
        """Process raw data to create features and target"""
        print("Processing raw data...")
        
        # Try to load raw data
        raw_paths = [
            '/kaggle/input/drw-crypto-market-prediction/train.parquet',
            'data/train.parquet'
        ]
        
        df = None
        for path in raw_paths:
            try:
                if os.path.exists(path):
                    print(f"Loading raw data from: {path}")
                    df = pd.read_parquet(path)
                    print(f"✅ Loaded raw data: {len(df)} rows, {len(df.columns)} columns")
                    break
            except Exception as e:
                print(f"❌ Error loading {path}: {e}")
        
        if df is None:
            print("❌ No raw data found")
            return None
        
        # Create target column
        df = self.create_target(df)
        if df is None:
            return None
        
        # Create basic features
        df = self.create_basic_features(df)
        if df is None:
            return None
        
        # Save processed data
        output_path = '/kaggle/working/quick_features.parquet'
        try:
            df.to_parquet(output_path)
            print(f"✅ Saved processed data to: {output_path}")
            self.data_path = output_path
            return df
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")
            return None
    
    def create_target(self, df):
        """Create target column from raw data"""
        print("Creating target column...")
        
        # Check if we have price columns
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        print(f"Found price columns: {price_cols}")
        
        if len(price_cols) >= 2:
            # Use price difference as target
            price1 = price_cols[0]
            price2 = price_cols[1]
            df['target'] = df[price2] - df[price1]
            print(f"Created target from {price1} and {price2}")
        elif 'close' in df.columns:
            # Use close price as target
            df['target'] = df['close']
            print("Using close price as target")
        else:
            # Use first numeric column as target
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                target_col = numeric_cols[0]
                df['target'] = df[target_col]
                print(f"Using {target_col} as target")
            else:
                print("❌ No suitable target column found")
                return None
        
        print(f"Target range: {df['target'].min():.4f} to {df['target'].max():.4f}")
        return df
    
    def create_basic_features(self, df):
        """Create basic features"""
        print("Creating basic features...")
        
        # Get numeric columns (excluding target)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'target']
        
        print(f"Using {len(feature_cols)} numeric columns as features")
        
        # Create some basic derived features
        for col in feature_cols[:10]:  # Limit to first 10 columns to avoid memory issues
            try:
                # Rolling mean
                df[f'{col}_rolling_mean_5'] = df[col].rolling(window=5, min_periods=1).mean()
                
                # Rolling std
                df[f'{col}_rolling_std_5'] = df[col].rolling(window=5, min_periods=1).std()
                
                # Lag features
                df[f'{col}_lag_1'] = df[col].shift(1)
                df[f'{col}_lag_2'] = df[col].shift(2)
                
                # Fill NaN values
                df[f'{col}_rolling_mean_5'].fillna(df[col], inplace=True)
                df[f'{col}_rolling_std_5'].fillna(0, inplace=True)
                df[f'{col}_lag_1'].fillna(df[col], inplace=True)
                df[f'{col}_lag_2'].fillna(df[col], inplace=True)
                
            except Exception as e:
                print(f"Warning: Could not create features for {col}: {e}")
                continue
        
        # Remove rows with NaN in target
        df = df.dropna(subset=['target'])
        
        print(f"Final dataset: {len(df)} rows with {len(df.columns)} columns")
        return df
    
    def prepare_data(self, df, target_col='target', sample_size=None):
        """Prepare data for training"""
        print("Preparing data...")
        
        # Check if this is raw data (no target column)
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found in raw data")
            print("Available columns:", list(df.columns)[:10], "...")
            
            # Try to find target column with different names
            possible_targets = ['target', 'Target', 'TARGET', 'y', 'Y', 'label', 'Label']
            found_target = None
            for possible in possible_targets:
                if possible in df.columns:
                    found_target = possible
                    break
            
            if found_target:
                print(f"Found target column: {found_target}")
                target_col = found_target
            else:
                print("❌ No target column found. This appears to be raw data.")
                print("You need to run Phase 3 (Feature Engineering) first to create target column.")
                return None, None, None
        
        # Sample data if specified
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {len(df)} rows")
        
        # Sort by time if available
        if 'time_id' in df.columns:
            df = df.sort_values('time_id')
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, 'time_id']]
        X = df[feature_cols].values
        y = df[target_col].values
        
        print(f"Features: {X.shape[1]}, Target range: {y.min():.4f} to {y.max():.4f}")
        
        return X, y, feature_cols
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance"""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        corr, _ = pearsonr(y_true, y_pred)
        
        result = {
            'model': model_name,
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'correlation': corr
        }
        
        print(f"{model_name} Results:")
        print(f"  MAE: {mae:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Correlation: {corr:.4f}")
        
        return result
    
    def train_arima(self, X, y, feature_cols):
        """Train ARIMA model"""
        print("Training ARIMA model...")
        start_time = time.time()
        
        try:
            # Use first 80% for training
            train_size = int(len(y) * 0.8)
            y_train = y[:train_size]
            y_test = y[train_size:]
            
            # Fit ARIMA model
            model = ARIMA(y_train, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Make predictions
            train_pred = fitted_model.predict(start=0, end=len(y_train)-1)
            test_pred = fitted_model.forecast(steps=len(y_test))
            
            training_time = time.time() - start_time
            print(f"ARIMA training completed in {training_time:.2f} seconds")
            
            # Evaluate
            train_result = self.evaluate_model(y_train, train_pred, "ARIMA (Train)")
            test_result = self.evaluate_model(y_test, test_pred, "ARIMA (Test)")
            
            return {
                'train': train_result,
                'test': test_result,
                'training_time': training_time
            }
            
        except Exception as e:
            print(f"ARIMA training failed: {e}")
            return None
    
    def train_prophet(self, X, y, feature_cols):
        """Train Prophet model"""
        print("Training Prophet model...")
        start_time = time.time()
        
        try:
            # Prepare data for Prophet
            df_prophet = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
                'y': y
            })
            
            # Use first 80% for training
            train_size = int(len(df_prophet) * 0.8)
            train_df = df_prophet[:train_size]
            test_df = df_prophet[train_size:]
            
            # Fit Prophet model
            model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
            model.fit(train_df)
            
            # Make predictions
            train_forecast = model.predict(train_df)
            test_forecast = model.predict(test_df)
            
            training_time = time.time() - start_time
            print(f"Prophet training completed in {training_time:.2f} seconds")
            
            # Evaluate
            train_result = self.evaluate_model(train_df['y'].values, train_forecast['yhat'].values, "Prophet (Train)")
            test_result = self.evaluate_model(test_df['y'].values, test_forecast['yhat'].values, "Prophet (Test)")
            
            return {
                'train': train_result,
                'test': test_result,
                'training_time': training_time
            }
            
        except Exception as e:
            print(f"Prophet training failed: {e}")
            return None
    
    def train_linear(self, X, y, feature_cols):
        """Train Linear Regression"""
        print("Training Linear Regression...")
        start_time = time.time()
        
        # Use first 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        print(f"Linear Regression training completed in {training_time:.2f} seconds")
        
        # Evaluate
        train_result = self.evaluate_model(y_train, train_pred, "Linear Regression (Train)")
        test_result = self.evaluate_model(y_test, test_pred, "Linear Regression (Test)")
        
        return {
            'train': train_result,
            'test': test_result,
            'training_time': training_time
        }
    
    def train_ridge(self, X, y, feature_cols):
        """Train Ridge Regression"""
        print("Training Ridge Regression...")
        start_time = time.time()
        
        # Use first 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        print(f"Ridge training completed in {training_time:.2f} seconds")
        
        # Evaluate
        train_result = self.evaluate_model(y_train, train_pred, "Ridge (Train)")
        test_result = self.evaluate_model(y_test, test_pred, "Ridge (Test)")
        
        return {
            'train': train_result,
            'test': test_result,
            'training_time': training_time
        }
    
    def train_lasso(self, X, y, feature_cols):
        """Train Lasso Regression"""
        print("Training Lasso Regression...")
        start_time = time.time()
        
        # Use first 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = Lasso(alpha=0.1)
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        print(f"Lasso training completed in {training_time:.2f} seconds")
        
        # Evaluate
        train_result = self.evaluate_model(y_train, train_pred, "Lasso (Train)")
        test_result = self.evaluate_model(y_test, test_pred, "Lasso (Test)")
        
        return {
            'train': train_result,
            'test': test_result,
            'training_time': training_time
        }
    
    def train_random_forest(self, X, y, feature_cols):
        """Train Random Forest"""
        print("Training Random Forest...")
        start_time = time.time()
        
        # Use first 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        print(f"Random Forest training completed in {training_time:.2f} seconds")
        
        # Evaluate
        train_result = self.evaluate_model(y_train, train_pred, "Random Forest (Train)")
        test_result = self.evaluate_model(y_test, test_pred, "Random Forest (Test)")
        
        return {
            'train': train_result,
            'test': test_result,
            'training_time': training_time
        }
    
    def train_lightgbm(self, X, y, feature_cols):
        """Train LightGBM"""
        print("Training LightGBM...")
        start_time = time.time()
        
        # Use first 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        print(f"LightGBM training completed in {training_time:.2f} seconds")
        
        # Evaluate
        train_result = self.evaluate_model(y_train, train_pred, "LightGBM (Train)")
        test_result = self.evaluate_model(y_test, test_pred, "LightGBM (Test)")
        
        return {
            'train': train_result,
            'test': test_result,
            'training_time': training_time
        }
    
    def train_xgboost(self, X, y, feature_cols):
        """Train XGBoost"""
        print("Training XGBoost...")
        start_time = time.time()
        
        # Use first 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        print(f"XGBoost training completed in {training_time:.2f} seconds")
        
        # Evaluate
        train_result = self.evaluate_model(y_train, train_pred, "XGBoost (Train)")
        test_result = self.evaluate_model(y_test, test_pred, "XGBoost (Test)")
        
        return {
            'train': train_result,
            'test': test_result,
            'training_time': training_time
        }
    
    def train_svr(self, X, y, feature_cols):
        """Train SVR"""
        print("Training SVR...")
        start_time = time.time()
        
        # Use first 80% for training
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model
        model = SVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        training_time = time.time() - start_time
        print(f"SVR training completed in {training_time:.2f} seconds")
        
        # Evaluate
        train_result = self.evaluate_model(y_train, train_pred, "SVR (Train)")
        test_result = self.evaluate_model(y_test, test_pred, "SVR (Test)")
        
        return {
            'train': train_result,
            'test': test_result,
            'training_time': training_time
        }
    
    def train_all_models(self, X, y, feature_cols):
        """Train all models sequentially"""
        print("Training all models sequentially...")
        
        all_results = {}
        models_to_train = [
            ('ARIMA', self.train_arima),
            ('Prophet', self.train_prophet),
            ('Linear Regression', self.train_linear),
            ('Ridge', self.train_ridge),
            ('Lasso', self.train_lasso),
            ('Random Forest', self.train_random_forest),
            ('LightGBM', self.train_lightgbm),
            ('XGBoost', self.train_xgboost),
            ('SVR', self.train_svr)
        ]
        
        for model_name, train_func in models_to_train:
            print(f"\n{'='*50}")
            print(f"Training {model_name}")
            print(f"{'='*50}")
            
            try:
                result = train_func(X, y, feature_cols)
                if result:
                    all_results[model_name] = result
                    print(f"✅ {model_name} completed successfully")
                else:
                    print(f"❌ {model_name} failed")
                    
                # Clean up memory
                gc.collect()
                
            except Exception as e:
                print(f"❌ {model_name} failed with error: {e}")
                continue
        
        return all_results
    
    def save_results(self, results, model_name):
        """Save results to file"""
        try:
            os.makedirs('results', exist_ok=True)
            
            if isinstance(results, dict) and 'train' in results:
                # Single model result
                df = pd.DataFrame([{
                    'model': model_name,
                    'train_mae': results['train']['mae'],
                    'train_mse': results['train']['mse'],
                    'train_r2': results['train']['r2'],
                    'train_correlation': results['train']['correlation'],
                    'test_mae': results['test']['mae'],
                    'test_mse': results['test']['mse'],
                    'test_r2': results['test']['r2'],
                    'test_correlation': results['test']['correlation'],
                    'training_time': results['training_time']
                }])
            else:
                # Multiple model results
                df = pd.DataFrame()
                for name, result in results.items():
                    if result and 'test' in result:
                        row = {
                            'model': name,
                            'train_mae': result['train']['mae'],
                            'train_mse': result['train']['mse'],
                            'train_r2': result['train']['r2'],
                            'train_correlation': result['train']['correlation'],
                            'test_mae': result['test']['mae'],
                            'test_mse': result['test']['mse'],
                            'test_r2': result['test']['r2'],
                            'test_correlation': result['test']['correlation'],
                            'training_time': result['training_time']
                        }
                        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
            filename = f'results/{model_name.lower().replace(" ", "_")}_results.csv'
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def show_menu(self):
        """Show model selection menu"""
        print("\n" + "="*60)
        print("INDIVIDUAL MODEL TRAINING")
        print("="*60)
        print("Select a model to train:")
        for key, (name, _) in self.models.items():
            print(f"{key}. {name}")
        print("0. Exit")
        print("="*60)
    
    def run(self):
        """Main run function"""
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare data (use smaller sample to avoid memory issues)
        sample_size = min(50000, len(df))  # Use 50k samples or less
        X, y, feature_cols = self.prepare_data(df, sample_size=sample_size)
        if X is None:
            return
        
        # Step-by-step training mode
        print("\n" + "="*60)
        print("STEP-BY-STEP TRAINING MODE")
        print("="*60)
        print("Training one model at a time to avoid memory issues...")
        
        # Define training sequence (fastest to slowest)
        training_sequence = [
            ('3', 'Linear Regression'),
            ('4', 'Ridge'),
            ('5', 'Lasso'),
            ('6', 'Random Forest'),
            ('7', 'LightGBM'),
            ('8', 'XGBoost'),
            ('9', 'SVR'),
            ('1', 'ARIMA'),
            ('2', 'Prophet')
        ]
        
        all_results = {}
        
        for i, (choice, model_name) in enumerate(training_sequence, 1):
            print(f"\n{'='*50}")
            print(f"MODEL {i}/{len(training_sequence)}: {model_name}")
            print(f"{'='*50}")
            
            # Show estimated time
            time_estimates = {
                'Linear Regression': '1-2 minutes',
                'Ridge': '1-2 minutes',
                'Lasso': '1-2 minutes',
                'Random Forest': '3-5 minutes',
                'LightGBM': '5-10 minutes',
                'XGBoost': '5-10 minutes',
                'SVR': '10-15 minutes',
                'ARIMA': '10-20 minutes',
                'Prophet': '15-30 minutes'
            }
            
            print(f"Estimated time: {time_estimates.get(model_name, 'Unknown')}")
            print(f"Starting training...")
            
            try:
                train_func = self.models[choice][1]
                result = train_func(X, y, feature_cols)
                
                if result:
                    self.save_results(result, model_name)
                    all_results[model_name] = result
                    print(f"✅ {model_name} completed successfully")
                    
                    # Show detailed results immediately
                    print(f"\n📊 {model_name} RESULTS:")
                    print(f"{'='*40}")
                    
                    if 'train' in result and 'test' in result:
                        train_result = result['train']
                        test_result = result['test']
                        
                        print(f"TRAINING SET:")
                        print(f"  MAE: {train_result['mae']:.4f}")
                        print(f"  MSE: {train_result['mse']:.4f}")
                        print(f"  R²:  {train_result['r2']:.4f}")
                        print(f"  Correlation: {train_result['correlation']:.4f}")
                        
                        print(f"\nTEST SET:")
                        print(f"  MAE: {test_result['mae']:.4f}")
                        print(f"  MSE: {test_result['mse']:.4f}")
                        print(f"  R²:  {test_result['r2']:.4f}")
                        print(f"  Correlation: {test_result['correlation']:.4f}")
                        
                        print(f"\nTraining Time: {result['training_time']:.2f} seconds")
                        
                        # Show best metric
                        best_metric = test_result['correlation']
                        print(f"🎯 Best Metric (Test Correlation): {best_metric:.4f}")
                        
                    elif 'test' in result:
                        test_result = result['test']
                        print(f"  MAE: {test_result['mae']:.4f}")
                        print(f"  MSE: {test_result['mse']:.4f}")
                        print(f"  R²:  {test_result['r2']:.4f}")
                        print(f"  Correlation: {test_result['correlation']:.4f}")
                    
                    print(f"{'='*40}")
                else:
                    print(f"❌ {model_name} failed")
                    
                # Clean up memory
                gc.collect()
                
            except Exception as e:
                print(f"❌ {model_name} failed with error: {e}")
            
            # Ask user if they want to continue
            if i < len(training_sequence):
                print(f"\n{'='*50}")
                print(f"Next model: {training_sequence[i][1]}")
                print(f"Continue to next model? (y/n): ", end="")
                
                # For Kaggle notebook, we'll auto-continue after a delay
                import time
                print("Auto-continuing in 5 seconds...")
                time.sleep(5)
                print("Continuing...")
        
        # Save combined results and show summary
        if all_results:
            self.save_results(all_results, "all_models")
            
            print(f"\n{'='*60}")
            print("FINAL RESULTS SUMMARY")
            print(f"{'='*60}")
            
            # Create summary table
            summary_data = []
            for model_name, result in all_results.items():
                if 'test' in result:
                    test_result = result['test']
                    summary_data.append({
                        'Model': model_name,
                        'Test Correlation': f"{test_result['correlation']:.4f}",
                        'Test R²': f"{test_result['r2']:.4f}",
                        'Test MAE': f"{test_result['mae']:.4f}",
                        'Training Time (s)': f"{result.get('training_time', 0):.1f}"
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                print(summary_df.to_string(index=False))
                
                # Find best model
                best_model = max(all_results.items(), 
                               key=lambda x: x[1]['test']['correlation'] if 'test' in x[1] else -1)
                best_name = best_model[0]
                best_corr = best_model[1]['test']['correlation']
                
                print(f"\n🏆 BEST MODEL: {best_name}")
                print(f"   Test Correlation: {best_corr:.4f}")
            
            print(f"\n✅ Training completed! {len(all_results)} models trained successfully.")
            print(f"📁 Results saved in /kaggle/working/results/")
        else:
            print("\n❌ No models trained successfully.")

def main():
    """Main function"""
    trainer = IndividualModelTrainer()
    trainer.run()

if __name__ == "__main__":
    main() 