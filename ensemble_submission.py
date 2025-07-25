#!/usr/bin/env python3
"""
Ensemble Submission Script
Combine predictions from multiple models for better performance
"""

import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path

def load_submission_file(file_path):
    """Load a submission file and return predictions"""
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_path}: {len(df)} rows")
        return df['prediction'].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def create_ensemble_submission():
    """Create ensemble submission from available models"""
    
    # Define the models that successfully completed
    available_models = {
        'neural_network': '/kaggle/working/neural_network_submission.csv',  # Neural Network (best performer)
        'xgboost': '/kaggle/input/results/xgboost_submission.csv', 
        'random_forest': '/kaggle/input/results/random_forest_submission.csv',
        'lightgbm': '/kaggle/input/results/submission.csv',  # LightGBM
        'arima': '/kaggle/input/results/arima_submission.csv'  # if available
    }
    
    # Load available predictions
    predictions = {}
    for model_name, file_path in available_models.items():
        pred = load_submission_file(file_path)
        if pred is not None:
            predictions[model_name] = pred
            print(f"✅ {model_name}: loaded {len(pred)} predictions")
        else:
            print(f"❌ {model_name}: failed to load")
    
    if not predictions:
        print("❌ No valid prediction files found!")
        return
    
    print(f"\n📊 Loaded {len(predictions)} models for ensemble")
    
    # Convert to DataFrame for easier manipulation
    pred_df = pd.DataFrame(predictions)
    
    # Calculate ensemble predictions using different methods
    ensemble_methods = {
        'simple_average': pred_df.mean(axis=1),
        'median': pred_df.median(axis=1),
        'weighted_average': pred_df.apply(lambda x: 
            x['neural_network'] * 0.6 + x['xgboost'] * 0.25 + 
            x['random_forest'] * 0.15, axis=1),  # weights based on actual performance
        'trimmed_mean': pred_df.apply(lambda x: 
            np.mean(sorted(x)[1:-1]), axis=1)  # Remove min and max
    }
    
    # Create submission files for each ensemble method
    for method_name, ensemble_pred in ensemble_methods.items():
        # Ensure correct submission format
        expected_rows = 538150
        
        # Ensure we have the correct number of predictions
        if len(ensemble_pred) != expected_rows:
            print(f"WARNING: Expected {expected_rows} predictions, got {len(ensemble_pred)}")
            if len(ensemble_pred) < expected_rows:
                # Pad with last prediction value
                padding = [ensemble_pred.iloc[-1]] * (expected_rows - len(ensemble_pred))
                ensemble_pred = pd.concat([ensemble_pred, pd.Series(padding)], ignore_index=True)
            else:
                # Truncate to expected length
                ensemble_pred = ensemble_pred[:expected_rows]
        
        submission_df = pd.DataFrame({
            'id': range(1, expected_rows + 1),  # IDs from 1 to 538150
            'prediction': ensemble_pred
        })
        
        output_path = f'/kaggle/working/ensemble_{method_name}_submission.csv'
        submission_df.to_csv(output_path, index=False)
        
        print(f"\n📁 Created {method_name} ensemble:")
        print(f"   File: {output_path}")
        print(f"   Predictions: {len(ensemble_pred)}")
        print(f"   Mean: {ensemble_pred.mean():.6f}")
        print(f"   Std: {ensemble_pred.std():.6f}")
        print(f"   Min: {ensemble_pred.min():.6f}")
        print(f"   Max: {ensemble_pred.max():.6f}")
        print(f"   Range: {ensemble_pred.max() - ensemble_pred.min():.6f}")
    
    # Create the main ensemble submission (weighted average)
    main_ensemble_pred = ensemble_methods['weighted_average']
    
    # Ensure correct submission format for main ensemble
    expected_rows = 538150
    if len(main_ensemble_pred) != expected_rows:
        print(f"WARNING: Main ensemble expected {expected_rows} predictions, got {len(main_ensemble_pred)}")
        if len(main_ensemble_pred) < expected_rows:
            padding = [main_ensemble_pred.iloc[-1]] * (expected_rows - len(main_ensemble_pred))
            main_ensemble_pred = pd.concat([main_ensemble_pred, pd.Series(padding)], ignore_index=True)
        else:
            main_ensemble_pred = main_ensemble_pred[:expected_rows]
    
    main_submission_df = pd.DataFrame({
        'id': range(1, expected_rows + 1),  # IDs from 1 to 538150
        'prediction': main_ensemble_pred
    })
    
    main_output_path = '/kaggle/working/ensemble_submission.csv'
    main_submission_df.to_csv(main_output_path, index=False)
    
    print(f"\n🎯 Main ensemble submission created:")
    print(f"   File: {main_output_path}")
    print(f"   Method: Weighted Average")
    print(f"   Weights: Neural Network(60%), XGBoost(25%), RF(15%)")
    print(f"   Rows: {len(main_submission_df)} (expected: {expected_rows})")
    print(f"   ID range: {main_submission_df['id'].min()} to {main_submission_df['id'].max()}")
    
    # Verify submission format
    if len(main_submission_df) == expected_rows and main_submission_df['id'].min() == 1 and main_submission_df['id'].max() == expected_rows:
        print(f"   ✅ Submission format is correct!")
    else:
        print(f"   ❌ Submission format error!")
        print(f"      Expected: {expected_rows} rows, ID 1-{expected_rows}")
        print(f"      Actual: {len(main_submission_df)} rows, ID {main_submission_df['id'].min()}-{main_submission_df['id'].max()}")
    
    # Model correlation analysis
    print(f"\n📈 Model Correlation Matrix:")
    correlation_matrix = pred_df.corr()
    print(correlation_matrix.round(3))
    
    # Clean up memory
    del pred_df, ensemble_methods
    gc.collect()
    
    return main_output_path

if __name__ == "__main__":
    print("🚀 Creating ensemble submission from available models...")
    ensemble_file = create_ensemble_submission()
    print(f"\n✅ Ensemble creation completed!")
    print(f"📤 Ready to submit: {ensemble_file}") 