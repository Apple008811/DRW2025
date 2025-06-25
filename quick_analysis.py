#!/usr/bin/env python3
"""
Quick Analysis Script for Local Mac Usage
Optimized for fast execution with smaller datasets and fewer features.
Use this for rapid prototyping and testing on your local machine.
"""

import sys
import os
from analysis_core import run_analysis_stage

# Configuration for quick analysis (optimized for local Mac)
QUICK_CONFIG = {
    'train_file': '/Users/yixuan/DRW/data/train.parquet',
    'test_file': '/Users/yixuan/DRW/data/test.parquet', 
    'submission_file': '/Users/yixuan/DRW/data/sample_submission.csv',
    'top_n': 10,  # Fewer features for speed, reduced from 20 to 10
    'lag_feature_count': 3,  # Fewer lag features, reduced from 5 to 3
    'n_splits': 2,  # Fewer CV splits, reduced from 3 to 2
    'n_estimators': 50,  # Fewer estimators, reduced from 100 to 50
    'learning_rate': 0.15,  # Slightly higher learning rate for faster convergence
    'max_depth': 4,  # Shallower tree depth, reduced from 6 to 4
    'output_file': 'quick_submission.csv'
}

def run_quick_analysis():
    """Run quick analysis optimized for local Mac usage."""
    print("🚀 QUICK ANALYSIS - Local Mac Version")
    print("=" * 50)
    print("📱 Optimized for fast execution on local machine")
    print("🔧 Features: 10 top features + 3 lag features")
    print("⚡ CV: 2-fold time series split")
    print("=" * 50)
    
    # Check if data files exist
    for file_path in [QUICK_CONFIG['train_file'], QUICK_CONFIG['test_file'], QUICK_CONFIG['submission_file']]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            print("Please ensure data files are in the correct location.")
            return None
    
    # Run quick analysis
    results = run_analysis_stage(
        stage='quick',
        train_file=QUICK_CONFIG['train_file'],
        test_file=QUICK_CONFIG['test_file'],
        submission_file=QUICK_CONFIG['submission_file'],
        top_n=QUICK_CONFIG['top_n'],
        lag_feature_count=QUICK_CONFIG['lag_feature_count'],
        output_file=QUICK_CONFIG['output_file'],
        n_splits=QUICK_CONFIG['n_splits'],
        n_estimators=QUICK_CONFIG['n_estimators'],
        learning_rate=QUICK_CONFIG['learning_rate'],
        max_depth=QUICK_CONFIG['max_depth']
    )
    
    print(f"\n📊 Quick Analysis Summary:")
    print(f"   - Features used: {results['feature_count']}")
    print(f"   - CV Score: {results['avg_score']:.4f} ± {results['score_std']:.4f}")
    print(f"   - Output: {QUICK_CONFIG['output_file']}")
    
    return results

def run_quick_with_sampling(sample_size=10000):
    """Run quick analysis on a sample of the data for even faster testing."""
    print(f"🚀 QUICK ANALYSIS WITH SAMPLING ({sample_size} samples)")
    print("=" * 50)
    print("📱 Ultra-fast testing with data sampling")
    print("=" * 50)
    
    import pandas as pd
    
    # Load and sample data
    print(f"📊 Loading and sampling {sample_size} rows...")
    
    # Support both parquet and csv formats
    if QUICK_CONFIG['train_file'].endswith('.parquet'):
        train_data = pd.read_parquet(QUICK_CONFIG['train_file'])
        test_data = pd.read_parquet(QUICK_CONFIG['test_file'])
    else:
    train_data = pd.read_csv(QUICK_CONFIG['train_file'])
    test_data = pd.read_csv(QUICK_CONFIG['test_file'])
    
    submission_template = pd.read_csv(QUICK_CONFIG['submission_file'])
    
    # Sample data
    if len(train_data) > sample_size:
        train_data = train_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"✅ Sampled {sample_size} training rows")
    
    if len(test_data) > sample_size:
        test_data = test_data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"✅ Sampled {sample_size} test rows")
    
    # Save sampled data temporarily
    temp_train = 'temp_train_sample.csv'
    temp_test = 'temp_test_sample.csv'
    temp_submission = 'temp_submission_sample.csv'
    
    train_data.to_csv(temp_train, index=False)
    test_data.to_csv(temp_test, index=False)
    submission_template.to_csv(temp_submission, index=False)
    
    try:
        # Adjust parameters based on sample size
        if sample_size <= 5000:
            # Ultra-lightweight configuration
            n_splits = 2
            n_estimators = 30
            top_n = 8
            lag_feature_count = 2
        else:
            # Lightweight configuration
            n_splits = 2
            n_estimators = 50
            top_n = 10
            lag_feature_count = 3
        
        # Run analysis on sampled data
        results = run_analysis_stage(
            stage='quick_sample',
            train_file=temp_train,
            test_file=temp_test,
            submission_file=temp_submission,
            top_n=top_n,
            lag_feature_count=lag_feature_count,
            output_file='quick_sample_submission.csv',
            n_splits=n_splits,
            n_estimators=n_estimators,
            learning_rate=0.15,
            max_depth=4
        )
        
        print(f"\n📊 Quick Sample Analysis Summary:")
        print(f"   - Sample size: {sample_size}")
        print(f"   - Features used: {results['feature_count']}")
        print(f"   - CV Score: {results['avg_score']:.4f} ± {results['score_std']:.4f}")
        print(f"   - Output: quick_sample_submission.csv")
        
        return results
        
    finally:
        # Clean up temporary files
        for temp_file in [temp_train, temp_test, temp_submission]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"🧹 Cleaned up {temp_file}")

def run_ultra_quick_analysis():
    """Run ultra-quick analysis with minimal resources."""
    print("🚀 ULTRA-QUICK ANALYSIS - Minimal Resource Version")
    print("=" * 50)
    print("⚡ Ultra-fast testing with minimal features and data")
    print("🔧 Features: 5 top features + 2 lag features")
    print("⚡ CV: 2-fold time series split")
    print("🌳 Model: 20 estimators, shallow trees")
    print("=" * 50)
    
    # Check if data files exist
    for file_path in [QUICK_CONFIG['train_file'], QUICK_CONFIG['test_file'], QUICK_CONFIG['submission_file']]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            print("Please ensure data files are in the correct location.")
            return None
    
    # Run ultra-quick analysis
    results = run_analysis_stage(
        stage='ultra_quick',
        train_file=QUICK_CONFIG['train_file'],
        test_file=QUICK_CONFIG['test_file'],
        submission_file=QUICK_CONFIG['submission_file'],
        top_n=5,  # Minimal features
        lag_feature_count=2,  # Minimal lag features
        output_file='ultra_quick_submission.csv',
        n_splits=2,
        n_estimators=20,  # Minimal estimators
        learning_rate=0.2,  # Higher learning rate
        max_depth=3  # Shallowest tree depth
    )
    
    print(f"\n📊 Ultra-Quick Analysis Summary:")
    print(f"   - Features used: {results['feature_count']}")
    print(f"   - CV Score: {results['avg_score']:.4f} ± {results['score_std']:.4f}")
    print(f"   - Output: ultra_quick_submission.csv")
    
    return results

if __name__ == "__main__":
    print("🔧 Quick Analysis Script for Local Mac")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'ultra':
            # 超轻量级分析
            results = run_ultra_quick_analysis()
        elif command == 'sample':
            try:
                sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
                results = run_quick_with_sampling(sample_size)
            except (ValueError, IndexError):
                print("❌ Invalid sample size. Using default 10000.")
                results = run_quick_with_sampling(10000)
        elif command == 'help':
            print("\nUsage:")
            print("  python3 quick_analysis.py          # Run quick analysis (10 features)")
            print("  python3 quick_analysis.py ultra    # Run ultra-quick analysis (5 features)")
            print("  python3 quick_analysis.py sample   # Run with 10k sample")
            print("  python3 quick_analysis.py sample 5000  # Run with 5k sample")
            print("  python3 quick_analysis.py help     # Show this help")
            print("\nPerformance levels:")
            print("  ultra:   5 features, 20 estimators, ~30 seconds")
            print("  default: 10 features, 50 estimators, ~1-2 minutes")
            print("  sample:  Based on sample size, optimized for speed")
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python3 quick_analysis.py help' for usage information")
    else:
        # Default: run quick analysis
        results = run_quick_analysis() 