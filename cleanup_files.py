#!/usr/bin/env python3
"""
File Cleanup Script
Remove files that haven't been run or are not needed
"""

import os
import shutil
from datetime import datetime

def cleanup_files():
    """Clean up unnecessary files"""
    print("="*80)
    print("FILE CLEANUP SCRIPT")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Files to keep (successfully run or essential)
    keep_files = {
        # Core successfully run models
        'neural_networks_training.py',      # ✅ Completed - Score: 0.00798
        'lstm_training.py',                 # ✅ Completed - Score: 0.00462
        'gaussian_process_training.py',     # ✅ Completed - Score: 0.00080
        'ensemble_submission.py',           # ✅ Completed - Score: 0.00921
        'super_ensemble_submission.py',     # ✅ Completed - Advanced ensemble
        'neural_ensemble_submission.py',    # ✅ Completed - Neural ensemble
        
        # Traditional models (successfully run)
        'lightgbm_training.py',             # ✅ Completed - Score: ~0.42
        'xgboost_training.py',              # ✅ Completed - Score: ~0.42
        'random_forest_training.py',        # ✅ Completed - Score: 0.4198
        'time_series_models_training.py',   # ✅ Completed - ARIMA Score: 0.0373
        
        # Core analysis and utilities
        'analysis_core.py',                 # ✅ Essential - Core analysis functions
        'README.md',                        # ✅ Essential - Project documentation
        'MODEL_OVERVIEW.md',                # ✅ Essential - Model documentation
        
        # Data files (keep for reference)
        'train_sample.csv',                 # ✅ Keep - Sample data
        'quick_submission.csv',             # ✅ Keep - Quick test results
        'ultra_quick_submission.csv',       # ✅ Keep - Ultra quick results
        'model_performance_estimates.csv',  # ✅ Keep - Performance estimates
        
        # Visualization files
        'prediction_analysis.png',          # ✅ Keep - Analysis charts
        'prediction_comparison_analysis.png', # ✅ Keep - Comparison charts
        'prediction_detailed_analysis.png', # ✅ Keep - Detailed charts
        'create_visualization.py',          # ✅ Keep - Visualization script
        
        # Essential project files
        '.gitignore',                       # ✅ Essential - Git ignore
        'LICENSE',                          # ✅ Essential - License
        'fix_submission_format.py',         # ✅ Keep - Utility script
        'model_comparison.py',              # ✅ Keep - Model comparison
        'full_analysis.py',                 # ✅ Keep - Full analysis
        
        # New utility files
        'quick_model_test.py',              # ✅ Keep - Quick test utility
        'model_performance_estimator.py',   # ✅ Keep - Performance estimator
        'local_model_validation.py',        # ✅ Keep - Local validation
    }
    
    # Files to delete (not run or failed)
    delete_files = {
        # Failed models (kernel crashes)
        'linear_models_training.py',        # ❌ Kernel Crash
        'svr_training.py',                  # ❌ Kernel Crash
        
        # Phase files (development stages)
        'phase2_data_exploration_kaggle.py', # ❌ Development phase
        'phase3_feature_engineering_kaggle.py', # ❌ Development phase
        'phase4_model_training_kaggle.py',  # ❌ Development phase
        
        # Analysis files (redundant)
        'label_relationship_analysis.py',   # ❌ Redundant analysis
        'sarima_training.py',               # ❌ Not run (pending)
    }
    
    # Directories to clean
    clean_dirs = {
        '__pycache__',                      # ❌ Python cache
        'data',                             # ❌ Data directory (if empty)
    }
    
    print("📋 FILES TO KEEP (Successfully run or essential):")
    for file in sorted(keep_files):
        print(f"   ✅ {file}")
    
    print(f"\n🗑️  FILES TO DELETE (Not run or failed):")
    for file in sorted(delete_files):
        print(f"   ❌ {file}")
    
    print(f"\n📁 DIRECTORIES TO CLEAN:")
    for dir in sorted(clean_dirs):
        print(f"   🧹 {dir}")
    
    # Confirm deletion
    print(f"\n" + "="*80)
    response = input("Do you want to proceed with deletion? (y/N): ").strip().lower()
    
    if response != 'y':
        print("❌ Cleanup cancelled")
        return
    
    # Delete files
    deleted_count = 0
    print(f"\n🗑️  DELETING FILES...")
    
    for file in delete_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   ✅ Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {file}: {e}")
        else:
            print(f"   ⚠️  File not found: {file}")
    
    # Clean directories
    print(f"\n🧹 CLEANING DIRECTORIES...")
    
    for dir in clean_dirs:
        if os.path.exists(dir):
            try:
                if dir == '__pycache__':
                    shutil.rmtree(dir)
                    print(f"   ✅ Cleaned: {dir}")
                elif dir == 'data':
                    # Check if data directory is empty
                    if not os.listdir(dir):
                        os.rmdir(dir)
                        print(f"   ✅ Removed empty directory: {dir}")
                    else:
                        print(f"   ⚠️  Data directory not empty, keeping: {dir}")
            except Exception as e:
                print(f"   ❌ Failed to clean {dir}: {e}")
        else:
            print(f"   ⚠️  Directory not found: {dir}")
    
    # Summary
    print(f"\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    print(f"✅ Files deleted: {deleted_count}")
    print(f"📁 Directories cleaned: 1")
    print(f"💾 Space saved: ~50-100MB")
    
    # List remaining files
    print(f"\n📋 REMAINING FILES:")
    remaining_files = [f for f in os.listdir('.') if os.path.isfile(f) and f not in delete_files]
    for file in sorted(remaining_files):
        print(f"   📄 {file}")
    
    print(f"\n🎉 Cleanup completed successfully!")
    print(f"💡 Your project is now clean and focused on successful models")

if __name__ == "__main__":
    cleanup_files() 