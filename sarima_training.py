#!/usr/bin/env python3
"""
Standalone SARIMA Training Script
Run this separately to test SARIMA model performance
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
import time
import gc
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SARIMATrainer:
    def __init__(self, data_path=None):
        # Try different possible data paths
        if data_path is None:
            self.possible_paths = [
                '/kaggle/working/engineered_features.parquet',  # Kaggle working directory
                '/kaggle/working/train_features.parquet',      # Phase 3 output
                '/kaggle/input/drw-crypto-market-prediction/train.parquet',  # Original data
                'data/engineered_features.parquet',            # Local path
                'data/train.parquet'                          # Local train data
            ]
        else:
            self.possible_paths = [data_path]
        
        self.data_path = None
        self.results = {}
        
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
        
        print("\n❌ No data file found. Available options:")
        print("1. Run Phase 3 first to generate engineered features")
        print("2. Use original train.parquet from Kaggle dataset")
        print("3. Specify custom data path")
        return None
    
    def prepare_data(self, df, target_col='target'):
        """Prepare data for SARIMA training"""
        print("Preparing data for SARIMA...")
        
        # Ensure target column exists
        if target_col not in df.columns:
            print(f"Target column '{target_col}' not found. Available columns: {list(df.columns)[:10]}...")
            return None, None
            
        # Sort by time if time column exists
        if 'time_id' in df.columns:
            df = df.sort_values('time_id')
            
        # Extract target and time series
        y = df[target_col].values
        print(f"Target series length: {len(y)}")
        print(f"Target range: {y.min():.4f} to {y.max():.4f}")
        print(f"Target mean: {y.mean():.4f}")
        
        return y, df
    
    def train_sarima(self, y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        """Train SARIMA model with specified parameters"""
        print(f"Training SARIMA{order}{seasonal_order}...")
        print(f"Data length: {len(y)}")
        
        start_time = time.time()
        
        try:
            # Fit SARIMA model
            model = SARIMAX(y, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit(disp=False)
            
            training_time = time.time() - start_time
            print(f"SARIMA training completed in {training_time:.2f} seconds")
            
            return fitted_model, training_time
            
        except Exception as e:
            print(f"SARIMA training failed: {e}")
            return None, 0
    
    def evaluate_sarima(self, model, y, test_size=0.2):
        """Evaluate SARIMA model performance"""
        if model is None:
            return None
            
        print("Evaluating SARIMA model...")
        
        # Split data
        split_idx = int(len(y) * (1 - test_size))
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        
        # Make predictions
        try:
            # Get in-sample predictions for training
            train_pred = model.predict(start=0, end=len(y_train)-1)
            
            # Get out-of-sample predictions for test
            test_pred = model.forecast(steps=len(y_test))
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, train_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            train_r2 = r2_score(y_train, train_pred)
            train_corr, _ = pearsonr(y_train, train_pred)
            
            test_mae = mean_absolute_error(y_test, test_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            test_corr, _ = pearsonr(y_test, test_pred)
            
            results = {
                'train_mae': train_mae,
                'train_mse': train_mse,
                'train_r2': train_r2,
                'train_correlation': train_corr,
                'test_mae': test_mae,
                'test_mse': test_mse,
                'test_r2': test_r2,
                'test_correlation': test_corr,
                'train_predictions': train_pred,
                'test_predictions': test_pred
            }
            
            print("SARIMA Evaluation Results:")
            print(f"Train - MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, R²: {train_r2:.4f}, Corr: {train_corr:.4f}")
            print(f"Test  - MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, R²: {test_r2:.4f}, Corr: {test_corr:.4f}")
            
            return results
            
        except Exception as e:
            print(f"SARIMA evaluation failed: {e}")
            return None
    
    def test_different_orders(self, y):
        """Test different SARIMA orders to find best parameters"""
        print("Testing different SARIMA orders...")
        
        # Define different orders to test
        orders_to_test = [
            ((1, 1, 1), (1, 1, 1, 12)),  # Basic SARIMA
            ((1, 1, 0), (1, 1, 0, 12)),  # Simpler seasonal
            ((0, 1, 1), (0, 1, 1, 12)),  # MA-based
            ((1, 1, 1), (0, 1, 1, 12)),  # No seasonal AR
            ((1, 1, 1), (1, 0, 1, 12)),  # No seasonal differencing
        ]
        
        best_result = None
        best_correlation = -1
        best_order = None
        
        for order, seasonal_order in orders_to_test:
            print(f"\nTesting SARIMA{order}{seasonal_order}...")
            
            try:
                model, training_time = self.train_sarima(y, order, seasonal_order)
                if model is not None:
                    result = self.evaluate_sarima(model, y)
                    if result and result['test_correlation'] > best_correlation:
                        best_correlation = result['test_correlation']
                        best_result = result
                        best_order = (order, seasonal_order)
                        print(f"New best: SARIMA{order}{seasonal_order} with correlation {best_correlation:.4f}")
                        
                # Clean up memory
                gc.collect()
                
            except Exception as e:
                print(f"Failed to test SARIMA{order}{seasonal_order}: {e}")
                continue
        
        if best_result:
            print(f"\nBest SARIMA model: SARIMA{best_order[0]}{best_order[1]}")
            print(f"Best test correlation: {best_correlation:.4f}")
            return best_result, best_order
        else:
            print("No SARIMA model succeeded")
            return None, None
    
    def run_sarima_analysis(self):
        """Run complete SARIMA analysis"""
        print("=" * 50)
        print("SARIMA STANDALONE TRAINING")
        print("=" * 50)
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare data
        y, df = self.prepare_data(df)
        if y is None:
            return
        
        # Test different SARIMA orders
        result, best_order = self.test_different_orders(y)
        
        if result:
            print("\n" + "=" * 50)
            print("SARIMA TRAINING COMPLETED SUCCESSFULLY")
            print("=" * 50)
            print(f"Best model: SARIMA{best_order[0]}{best_order[1]}")
            print(f"Test correlation: {result['test_correlation']:.4f}")
            print(f"Test R²: {result['test_r2']:.4f}")
            print(f"Test MAE: {result['test_mae']:.4f}")
            
            # Save results
            self.save_results(result, best_order)
        else:
            print("\nSARIMA training failed for all tested configurations")
    
    def save_results(self, result, best_order):
        """Save SARIMA results"""
        try:
            # Create results directory if it doesn't exist
            os.makedirs('results', exist_ok=True)
            
            # Save metrics
            metrics_df = pd.DataFrame([{
                'model': f'SARIMA{best_order[0]}{best_order[1]}',
                'train_mae': result['train_mae'],
                'train_mse': result['train_mse'],
                'train_r2': result['train_r2'],
                'train_correlation': result['train_correlation'],
                'test_mae': result['test_mae'],
                'test_mse': result['test_mse'],
                'test_r2': result['test_r2'],
                'test_correlation': result['test_correlation']
            }])
            
            metrics_df.to_csv('results/sarima_results.csv', index=False)
            print("Results saved to results/sarima_results.csv")
            
        except Exception as e:
            print(f"Error saving results: {e}")

def main():
    """Main function to run SARIMA training"""
    trainer = SARIMATrainer()
    trainer.run_sarima_analysis()

if __name__ == "__main__":
    main() 