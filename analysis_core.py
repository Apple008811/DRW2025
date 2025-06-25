"""
Core analysis functions shared between quick and full analysis scripts.
This module contains the main logic for data processing, model training, and evaluation.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(train_file, test_file, submission_file, top_n=100):
    """
    Load and prepare data with feature selection.
    
    Args:
        train_file (str): Path to training data file
        test_file (str): Path to test data file  
        submission_file (str): Path to submission template file
        top_n (int): Number of top features to select
        
    Returns:
        tuple: (X_train, y_train, X_test, top_features, submission_template)
    """
    print("📊 Loading data...")
    
    # Load data (support both CSV and parquet formats)
    if train_file.endswith('.parquet'):
        train_data = pd.read_parquet(train_file)
    else:
        train_data = pd.read_csv(train_file)
        
    if test_file.endswith('.parquet'):
        test_data = pd.read_parquet(test_file)
    else:
        test_data = pd.read_csv(test_file)
        
    submission_template = pd.read_csv(submission_file)
    
    print(f"✅ Training data: {train_data.shape}")
    print(f"✅ Test data: {test_data.shape}")
    
    # Feature selection based on correlation with target
    print(f"\n🔍 Selecting top {top_n} features...")
    
    # Optimize: only calculate correlation for numeric columns, skip non-numeric columns
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != 'label' and col != 'time']
    
    # For large datasets, use more efficient correlation calculation
    if len(train_data) > 100000:
        # Sample for correlation calculation to improve speed
        sample_size = min(50000, len(train_data))
        sample_data = train_data.sample(n=sample_size, random_state=42)
        print(f"   Using {sample_size} samples for correlation calculation...")
        
        correlations = []
        for col in feature_cols:
            corr = abs(sample_data[col].corr(sample_data['label']))
            correlations.append((col, corr))
    else:
        # Direct calculation for small datasets
        correlations = []
        for col in feature_cols:
            corr = abs(train_data[col].corr(train_data['label']))
            correlations.append((col, corr))
    
    # Sort by correlation and select top features
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [col for col, _ in correlations[:top_n]]
    
    print(f"✅ Selected {len(top_features)} features")
    print(f"   Top 5 features: {top_features[:5]}")
    
    # Prepare data
    X_train = train_data[top_features]
    y_train = train_data['label']
    X_test = test_data[top_features]
    
    return X_train, y_train, X_test, top_features, submission_template

def engineer_features(X_train, X_test, top_features, lag_feature_count, lag_periods=[1, 2, 3, 5]):
    """
    Engineer features including lag features.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        top_features (list): List of top features
        lag_feature_count (int): Number of features to create lag features for
        lag_periods (list): List of lag periods to use
        
    Returns:
        tuple: (X_train_engineered, X_test_engineered)
    """
    print(f"📈 Feature engineering with {lag_feature_count} lag features...")
    
    # Select features for lag calculation
    lag_features = top_features[:lag_feature_count]
    
    # Calculate lag features
    for feat in lag_features:
        for lag in lag_periods:
            X_train[f'{feat}_lag_{lag}'] = X_train[feat].shift(lag)
            X_test[f'{feat}_lag_{lag}'] = X_test[feat].shift(lag)
    
    # Fill missing values
    X_train = X_train.fillna(method='bfill').fillna(0)
    X_test = X_test.fillna(method='bfill').fillna(0)
    
    print(f"✅ Final feature count: {X_train.shape[1]}")
    print(f"   - Original features: {len(top_features)}")
    print(f"   - Lag features: {len(lag_features) * len(lag_periods)}")
    
    return X_train, X_test

def train_and_evaluate_model(X_train, y_train, n_splits=3, n_estimators=100, learning_rate=0.1, max_depth=6):
    """
    Train model with time series cross-validation and evaluate.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_splits (int): Number of CV splits
        n_estimators (int): Number of estimators for LightGBM
        learning_rate (float): Learning rate for LightGBM
        max_depth (int): Max depth for LightGBM
        
    Returns:
        tuple: (final_model, scores, avg_score, score_std)
    """
    print(f"\n🤖 Training LightGBM model with {n_splits}-fold CV...")
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_fold_train, y_fold_train)
        
        # Predict and evaluate
        preds = model.predict(X_fold_val)
        score = pearsonr(y_fold_val, preds)[0]
        scores.append(score)
        
        print(f"  Fold {fold+1}: Pearson correlation = {score:.4f}")
    
    avg_score = np.mean(scores)
    score_std = np.std(scores)
    print(f"\n📊 Average Pearson correlation: {avg_score:.4f} ± {score_std:.4f}")
    
    # Train final model on full data
    final_model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42,
        verbose=-1
    )
    final_model.fit(X_train, y_train)
    
    return final_model, scores, avg_score, score_std

def get_feature_importance(model, X_train, top_n=10):
    """
    Get and display feature importance.
    
    Args:
        model: Trained LightGBM model
        X_train (pd.DataFrame): Training features
        top_n (int): Number of top features to display
        
    Returns:
        pd.DataFrame: Feature importance dataframe
    """
    print(f"\n🏆 Feature Importance (Top {top_n}):")
    
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df

def generate_predictions(model, X_test, submission_template, output_file):
    """
    Generate predictions and save to file.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        submission_template (pd.DataFrame): Submission template
        output_file (str): Output file path
        
    Returns:
        np.array: Predictions
    """
    print(f"\n📤 Generating predictions...")
    
    predictions = model.predict(X_test)
    
    # Update submission template
    submission = submission_template.copy()
    submission['pred'] = predictions
    
    # Save results
    submission.to_csv(output_file, index=False)
    print(f"✅ Predictions saved to {output_file}")
    
    # Print prediction statistics
    print(f"\n📈 Prediction Statistics:")
    print(f"- Mean prediction: {predictions.mean():.4f}")
    print(f"- Std prediction: {predictions.std():.4f}")
    print(f"- Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    return predictions

def run_analysis_stage(stage, train_file, test_file, submission_file, 
                      top_n, lag_feature_count, output_file, 
                      n_splits=3, n_estimators=100, learning_rate=0.1, max_depth=6):
    """
    Run complete analysis for a given stage.
    
    Args:
        stage (str): Analysis stage ('quick', 'medium', 'full')
        train_file (str): Path to training data
        test_file (str): Path to test data
        submission_file (str): Path to submission template
        top_n (int): Number of top features to select
        lag_feature_count (int): Number of features to create lag features for
        output_file (str): Output file path
        n_splits (int): Number of CV splits
        n_estimators (int): Number of estimators
        learning_rate (float): Learning rate
        max_depth (int): Max depth
        
    Returns:
        dict: Analysis results
    """
    print(f"\n🚀 Running {stage.upper()} Analysis")
    print("=" * 50)
    
    # 1. Load and prepare data
    X_train, y_train, X_test, top_features, submission_template = load_and_prepare_data(
        train_file, test_file, submission_file, top_n
    )
    
    # 2. Feature engineering
    X_train, X_test = engineer_features(X_train, X_test, top_features, lag_feature_count)
    
    # 3. Train and evaluate model
    final_model, scores, avg_score, score_std = train_and_evaluate_model(
        X_train, y_train, n_splits, n_estimators, learning_rate, max_depth
    )
    
    # 4. Feature importance
    importance_df = get_feature_importance(final_model, X_train)
    
    # 5. Generate predictions
    predictions = generate_predictions(final_model, X_test, submission_template, output_file)
    
    print(f"\n🎉 {stage.upper()} analysis completed!")
    
    return {
        'stage': stage,
        'avg_score': avg_score,
        'score_std': score_std,
        'feature_count': X_train.shape[1],
        'top_features': top_features[:10],
        'importance': importance_df.head(10),
        'predictions': predictions
    } 