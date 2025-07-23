#!/usr/bin/env python3
"""
Ensemble Submission Script
Combine predictions from multiple models for better performance
"""

import pandas as pd
import numpy as np
import os
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
        'lightgbm': '/kaggle/working/results/lightgbm_submission.csv',
        'xgboost': '/kaggle/working/results/xgboost_submission.csv', 
        'random_forest': '/kaggle/working/results/random_forest_submission.csv',
        'arima': '/kaggle/working/results/arima_submission.csv'
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
            x['lightgbm'] * 0.3 + x['xgboost'] * 0.3 + 
            x['random_forest'] * 0.25 + x['arima'] * 0.15, axis=1),
        'trimmed_mean': pred_df.apply(lambda x: 
            np.mean(sorted(x)[1:-1]), axis=1)  # Remove min and max
    }
    
    # Create submission files for each ensemble method
    for method_name, ensemble_pred in ensemble_methods.items():
        submission_df = pd.DataFrame({
            'id': range(len(ensemble_pred)),
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
    main_submission_df = pd.DataFrame({
        'id': range(len(ensemble_methods['weighted_average'])),
        'prediction': ensemble_methods['weighted_average']
    })
    
    main_output_path = '/kaggle/working/ensemble_submission.csv'
    main_submission_df.to_csv(main_output_path, index=False)
    
    print(f"\n🎯 Main ensemble submission created:")
    print(f"   File: {main_output_path}")
    print(f"   Method: Weighted Average")
    print(f"   Weights: LightGBM(30%), XGBoost(30%), RF(25%), ARIMA(15%)")
    
    # Model correlation analysis
    print(f"\n📈 Model Correlation Matrix:")
    correlation_matrix = pred_df.corr()
    print(correlation_matrix.round(3))
    
    return main_output_path

if __name__ == "__main__":
    print("🚀 Creating ensemble submission from available models...")
    ensemble_file = create_ensemble_submission()
    print(f"\n✅ Ensemble creation completed!")
    print(f"📤 Ready to submit: {ensemble_file}") 