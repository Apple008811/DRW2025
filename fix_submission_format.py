#!/usr/bin/env python3
"""
Fix Submission Format Script

This script fixes the submission file format to meet Kaggle competition requirements.
The correct format should have only two columns: 'id' and 'prediction'.
"""

import pandas as pd
import os

def fix_submission_format(input_file, output_file=None):
    """
    Fix submission file format to meet Kaggle requirements.
    
    Args:
        input_file (str): Path to input submission file
        output_file (str): Path to output fixed submission file (optional)
    """
    print(f"🔧 Fixing submission format for: {input_file}")
    
    # Read the submission file
    try:
        df = pd.read_csv(input_file)
        print(f"📊 Original file shape: {df.shape}")
        print(f"📊 Original columns: {list(df.columns)}")
        
        # Check if we need to fix the format
        if len(df.columns) == 3 and 'ID' in df.columns and 'prediction' in df.columns and 'pred' in df.columns:
            print("⚠️  Detected incorrect format with 3 columns. Fixing...")
            
            # Create correct format with only 'id' and 'prediction'
            fixed_df = pd.DataFrame({
                'id': df['ID'],
                'prediction': df['prediction']  # Use the main prediction column
            })
            
            print(f"✅ Fixed format: {list(fixed_df.columns)}")
            
        elif len(df.columns) == 2 and 'id' in df.columns and 'prediction' in df.columns:
            print("✅ Format already correct!")
            fixed_df = df
            
        else:
            print("❌ Unexpected format. Please check the file structure.")
            return False
            
        # Determine output filename
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}_fixed.csv"
            
        # Save the fixed file
        fixed_df.to_csv(output_file, index=False)
        print(f"💾 Fixed submission saved to: {output_file}")
        print(f"📊 Fixed file shape: {fixed_df.shape}")
        
        # Display sample
        print("\n📋 Sample of fixed submission:")
        print(fixed_df.head())
        print(fixed_df.tail())
        
        # Validate the format
        validate_submission_format(output_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing submission format: {e}")
        return False

def validate_submission_format(file_path):
    """
    Validate that the submission file meets Kaggle requirements.
    
    Args:
        file_path (str): Path to submission file to validate
    """
    print(f"\n🔍 Validating submission format: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        
        # Check basic requirements
        requirements = {
            "Has exactly 2 columns": len(df.columns) == 2,
            "Has 'id' column": 'id' in df.columns,
            "Has 'prediction' column": 'prediction' in df.columns,
            "ID column is numeric": df['id'].dtype in ['int64', 'int32'],
            "Prediction column is numeric": df['prediction'].dtype in ['float64', 'float32'],
            "No missing values in ID": not df['id'].isnull().any(),
            "No missing values in prediction": not df['prediction'].isnull().any(),
            "ID starts from 1": df['id'].min() == 1,
            "ID is sequential": (df['id'] == range(1, len(df) + 1)).all(),
            "Has expected number of rows": len(df) == 538150  # Expected for this competition
        }
        
        print("📋 Validation Results:")
        all_passed = True
        for requirement, passed in requirements.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {requirement}")
            if not passed:
                all_passed = False
                
        if all_passed:
            print("\n🎉 All validation checks passed! Submission format is correct.")
        else:
            print("\n⚠️  Some validation checks failed. Please review the issues above.")
            
        # Additional statistics
        print(f"\n📊 Submission Statistics:")
        print(f"  Total rows: {len(df)}")
        print(f"  ID range: {df['id'].min()} to {df['id'].max()}")
        print(f"  Prediction range: {df['prediction'].min():.6f} to {df['prediction'].max():.6f}")
        print(f"  Prediction mean: {df['prediction'].mean():.6f}")
        print(f"  Prediction std: {df['prediction'].std():.6f}")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error validating submission: {e}")
        return False

def main():
    """Main function to fix all submission files."""
    print("🔧 SUBMISSION FORMAT FIXER")
    print("=" * 50)
    
    # List of submission files to fix
    submission_files = [
        "quick_submission.csv",
        "ultra_quick_submission.csv"
    ]
    
    fixed_files = []
    
    for file_path in submission_files:
        if os.path.exists(file_path):
            print(f"\n{'='*50}")
            success = fix_submission_format(file_path)
            if success:
                base_name = os.path.splitext(file_path)[0]
                fixed_file = f"{base_name}_fixed.csv"
                fixed_files.append(fixed_file)
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n{'='*50}")
    print("📋 SUMMARY")
    print(f"Fixed files: {len(fixed_files)}")
    for file_path in fixed_files:
        print(f"  ✅ {file_path}")
    
    if fixed_files:
        print(f"\n🎯 Ready for submission! Use one of the fixed files above.")
        print(f"💡 Recommended: Use the file with the best model performance.")

if __name__ == "__main__":
    main() 