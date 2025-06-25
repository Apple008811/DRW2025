#!/usr/bin/env python3
"""
Full Analysis Script for Kaggle Notebooks
Optimized for comprehensive analysis with full datasets and maximum features.
Use this on Kaggle for production-ready analysis and submissions.
"""

import sys
import os
from analysis_core import run_analysis_stage

# Configuration for full analysis (optimized for Kaggle)
FULL_CONFIG = {
    'train_file': '/kaggle/input/drw-crypto-market-prediction/train.csv',  # Kaggle path
    'test_file': '/kaggle/input/drw-crypto-market-prediction/test.csv',    # Kaggle path
    'submission_file': '/kaggle/input/drw-crypto-market-prediction/sample_submission.csv',  # Kaggle path
    'top_n': 100,  # More features for better performance
    'lag_feature_count': 20,  # More lag features
    'n_splits': 5,  # More CV splits for better validation
    'n_estimators': 200,  # More estimators for better performance
    'learning_rate': 0.05,  # Lower learning rate for more stable training
    'max_depth': 8,  # Slightly deeper trees
    'output_file': 'full_submission.csv'
}

# Alternative configuration for local testing (if needed)
LOCAL_CONFIG = {
    'train_file': '/Users/yixuan/DRW/data/train.parquet',
    'test_file': '/Users/yixuan/DRW/data/test.parquet',
    'submission_file': '/Users/yixuan/DRW/data/sample_submission.csv',
    'top_n': 100,
    'lag_feature_count': 20,
    'n_splits': 5,
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 8,
    'output_file': 'full_submission.csv'
}

def run_full_analysis(use_local=False):
    """Run full analysis optimized for Kaggle notebooks."""
    config = LOCAL_CONFIG if use_local else FULL_CONFIG
    
    print("🚀 FULL ANALYSIS - Kaggle Production Version")
    print("=" * 50)
    print("🖥️  Optimized for comprehensive analysis on Kaggle")
    print("🔧 Features: 100 top features + 20 lag features")
    print("⚡ CV: 5-fold time series split")
    print("🌳 Model: 200 estimators, deeper trees")
    print("=" * 50)
    
    # Check if data files exist
    for file_path in [config['train_file'], config['test_file'], config['submission_file']]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            if not use_local:
                print("This script is designed for Kaggle notebooks.")
                print("For local testing, use: python full_analysis.py local")
            return None
    
    # Run full analysis
    results = run_analysis_stage(
        stage='full',
        train_file=config['train_file'],
        test_file=config['test_file'],
        submission_file=config['submission_file'],
        top_n=config['top_n'],
        lag_feature_count=config['lag_feature_count'],
        output_file=config['output_file'],
        n_splits=config['n_splits'],
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth']
    )
    
    print(f"\n📊 Full Analysis Summary:")
    print(f"   - Features used: {results['feature_count']}")
    print(f"   - CV Score: {results['avg_score']:.4f} ± {results['score_std']:.4f}")
    print(f"   - Output: {config['output_file']}")
    
    return results

def run_comprehensive_analysis():
    """Run comprehensive analysis with multiple stages for comparison."""
    print("🚀 COMPREHENSIVE ANALYSIS - Multi-Stage Comparison")
    print("=" * 60)
    print("🔄 Running multiple analysis stages for comparison")
    print("=" * 60)
    
    # Check if we're on Kaggle
    is_kaggle = os.path.exists('/kaggle/input')
    config = FULL_CONFIG if is_kaggle else LOCAL_CONFIG
    
    # Check if data files exist
    for file_path in [config['train_file'], config['test_file'], config['submission_file']]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return None
    
    results = {}
    
    # Stage 1: Quick analysis (for baseline)
    print("\n" + "="*30)
    print("STAGE 1: QUICK BASELINE")
    print("="*30)
    results['quick'] = run_analysis_stage(
        stage='quick',
        train_file=config['train_file'],
        test_file=config['test_file'],
        submission_file=config['submission_file'],
        top_n=20,
        lag_feature_count=5,
        output_file='comprehensive_quick_submission.csv',
        n_splits=3,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    
    # Stage 2: Medium analysis
    print("\n" + "="*30)
    print("STAGE 2: MEDIUM ANALYSIS")
    print("="*30)
    results['medium'] = run_analysis_stage(
        stage='medium',
        train_file=config['train_file'],
        test_file=config['test_file'],
        submission_file=config['submission_file'],
        top_n=50,
        lag_feature_count=10,
        output_file='comprehensive_medium_submission.csv',
        n_splits=4,
        n_estimators=150,
        learning_rate=0.07,
        max_depth=7
    )
    
    # Stage 3: Full analysis
    print("\n" + "="*30)
    print("STAGE 3: FULL ANALYSIS")
    print("="*30)
    results['full'] = run_analysis_stage(
        stage='full',
        train_file=config['train_file'],
        test_file=config['test_file'],
        submission_file=config['submission_file'],
        top_n=config['top_n'],
        lag_feature_count=config['lag_feature_count'],
        output_file='comprehensive_full_submission.csv',
        n_splits=config['n_splits'],
        n_estimators=config['n_estimators'],
        learning_rate=config['learning_rate'],
        max_depth=config['max_depth']
    )
    
    # Summary comparison
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE RESULTS COMPARISON")
    print("="*60)
    
    for stage, result in results.items():
        print(f"\n{stage.upper()}:")
        print(f"  - Features: {result['feature_count']}")
        print(f"  - Score: {result['avg_score']:.4f} ± {result['score_std']:.4f}")
        print(f"  - Output: comprehensive_{stage}_submission.csv")
    
    # Find best performing stage
    best_stage = max(results.keys(), key=lambda x: results[x]['avg_score'])
    print(f"\n🏆 Best performing stage: {best_stage.upper()}")
    print(f"   Score: {results[best_stage]['avg_score']:.4f} ± {results[best_stage]['score_std']:.4f}")
    
    return results

def run_hyperparameter_tuning():
    """Run hyperparameter tuning for the best model."""
    print("🚀 HYPERPARAMETER TUNING")
    print("=" * 50)
    print("🔍 Testing different hyperparameter combinations")
    print("=" * 50)
    
    # Check if we're on Kaggle
    is_kaggle = os.path.exists('/kaggle/input')
    config = FULL_CONFIG if is_kaggle else LOCAL_CONFIG
    
    # Check if data files exist
    for file_path in [config['train_file'], config['test_file'], config['submission_file']]:
        if not os.path.exists(file_path):
            print(f"❌ Error: {file_path} not found!")
            return None
    
    # Different hyperparameter combinations to test
    param_combinations = [
        {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
        {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8},
        {'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 10},
        {'n_estimators': 150, 'learning_rate': 0.07, 'max_depth': 7},
        {'n_estimators': 250, 'learning_rate': 0.04, 'max_depth': 9},
    ]
    
    tuning_results = {}
    
    for i, params in enumerate(param_combinations):
        print(f"\n🔧 Testing combination {i+1}: {params}")
        
        result = run_analysis_stage(
            stage=f'tuning_{i+1}',
            train_file=config['train_file'],
            test_file=config['test_file'],
            submission_file=config['submission_file'],
            top_n=config['top_n'],
            lag_feature_count=config['lag_feature_count'],
            output_file=f'tuning_submission_{i+1}.csv',
            n_splits=config['n_splits'],
            **params
        )
        
        tuning_results[f'combo_{i+1}'] = {
            'params': params,
            'score': result['avg_score'],
            'std': result['score_std']
        }
    
    # Find best parameters
    best_combo = max(tuning_results.keys(), key=lambda x: tuning_results[x]['score'])
    best_params = tuning_results[best_combo]['params']
    best_score = tuning_results[best_combo]['score']
    
    print(f"\n🏆 Best hyperparameters: {best_params}")
    print(f"   Score: {best_score:.4f} ± {tuning_results[best_combo]['std']:.4f}")
    
    return tuning_results

if __name__ == "__main__":
    print("🔧 Full Analysis Script for Kaggle")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'local':
            results = run_full_analysis(use_local=True)
        elif command == 'comprehensive':
            results = run_comprehensive_analysis()
        elif command == 'tuning':
            results = run_hyperparameter_tuning()
        elif command == 'help':
            print("\nUsage:")
            print("  python3 full_analysis.py              # Run full analysis (Kaggle)")
            print("  python3 full_analysis.py local        # Run full analysis (local)")
            print("  python3 full_analysis.py comprehensive # Run multi-stage comparison")
            print("  python3 full_analysis.py tuning       # Run hyperparameter tuning")
            print("  python3 full_analysis.py help         # Show this help")
            print("\nNote: This script is optimized for Kaggle notebooks.")
            print("Use 'local' flag for local testing.")
        else:
            print(f"❌ Unknown command: {command}")
            print("Use 'python3 full_analysis.py help' for usage information")
    else:
        # Default: run full analysis (assumes Kaggle environment)
        results = run_full_analysis() 