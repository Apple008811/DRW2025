#!/usr/bin/env python3
"""
Quick test script to verify environment and basic functionality.
Uses the existing train_sample.csv file for testing.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

def test_environment():
    """Test if the environment is working correctly."""
    print("🔧 Testing Environment")
    print("=" * 40)
    
    try:
        # Test data loading
        print("📊 Loading train_sample.csv...")
        data = pd.read_csv('train_sample.csv')
        print(f"✅ Data loaded successfully: {data.shape}")
        
        # Test basic data processing
        print("\n🔍 Basic data analysis...")
        print(f"   - Columns: {len(data.columns)}")
        print(f"   - Rows: {len(data)}")
        print(f"   - Target column 'label' exists: {'label' in data.columns}")
        
        # Test feature selection
        print("\n🎯 Feature selection test...")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != 'label' and col != 'timestamp']
        print(f"   - Numeric features: {len(feature_cols)}")
        
        # Test correlation calculation
        print("\n📈 Correlation test...")
        correlations = []
        for col in feature_cols[:5]:  # Test first 5 features
            corr = abs(data[col].corr(data['label']))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        print(f"   - Top feature: {correlations[0][0]} (corr: {correlations[0][1]:.4f})")
        
        # Test LightGBM
        print("\n🤖 LightGBM test...")
        X = data[feature_cols[:10]]  # Use first 10 features
        y = data['label']
        
        # Simple train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        model = lgb.LGBMRegressor(
            n_estimators=20,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = pearsonr(y_test, predictions)[0]
        
        print(f"   - Model trained successfully")
        print(f"   - Test score (Pearson): {score:.4f}")
        
        print("\n🎉 Environment test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        return False

def quick_analysis_test():
    """Run a quick analysis test using the sample data."""
    print("\n🚀 Quick Analysis Test")
    print("=" * 40)
    
    try:
        # Load data
        data = pd.read_csv('train_sample.csv')
        
        # Prepare features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != 'label' and col != 'timestamp']
        
        # Select top 5 features by correlation
        correlations = []
        for col in feature_cols:
            corr = abs(data[col].corr(data['label']))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:5]]
        
        print(f"📊 Using top 5 features: {top_features}")
        
        # Prepare data
        X = data[top_features]
        y = data['label']
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=2)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = lgb.LGBMRegressor(
                n_estimators=30,
                learning_rate=0.15,
                max_depth=4,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            score = pearsonr(y_fold_val, preds)[0]
            scores.append(score)
            
            print(f"  Fold {fold+1}: Pearson correlation = {score:.4f}")
        
        avg_score = np.mean(scores)
        score_std = np.std(scores)
        print(f"\n📊 Average Pearson correlation: {avg_score:.4f} ± {score_std:.4f}")
        
        # Generate sample predictions
        final_model = lgb.LGBMRegressor(
            n_estimators=30,
            learning_rate=0.15,
            max_depth=4,
            random_state=42,
            verbose=-1
        )
        final_model.fit(X, y)
        
        # Create sample submission
        sample_submission = pd.DataFrame({
            'timestamp': data['timestamp'],
            'pred': final_model.predict(X)
        })
        
        sample_submission.to_csv('test_sample_submission.csv', index=False)
        print(f"✅ Sample submission saved: test_sample_submission.csv")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during quick analysis: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 Quick Test Script")
    print("=" * 50)
    
    # Test environment
    env_ok = test_environment()
    
    if env_ok:
        # Run quick analysis test
        analysis_ok = quick_analysis_test()
        
        if analysis_ok:
            print("\n🎉 All tests passed! Your environment is ready for analysis.")
        else:
            print("\n⚠️  Environment test passed but analysis test failed.")
    else:
        print("\n❌ Environment test failed. Please check your setup.") 