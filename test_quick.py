#!/usr/bin/env python3
"""
Test script to verify quick_analysis.py optimization
"""

import time
import os
import sys

def test_ultra_quick():
    """Test ultra-quick analysis"""
    print("🧪 Testing Ultra-Quick Analysis...")
    start_time = time.time()
    
    try:
        # Import and run ultra-quick analysis
        from quick_analysis import run_ultra_quick_analysis
        results = run_ultra_quick_analysis()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if results:
            print(f"✅ Ultra-quick test completed in {duration:.1f} seconds")
            print(f"   Score: {results['avg_score']:.4f} ± {results['score_std']:.4f}")
            return True
        else:
            print("❌ Ultra-quick test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in ultra-quick test: {e}")
        return False

def test_quick_sample():
    """Test quick analysis with sampling"""
    print("🧪 Testing Quick Analysis with Sampling...")
    start_time = time.time()
    
    try:
        # Import and run quick analysis with sampling
        from quick_analysis import run_quick_with_sampling
        results = run_quick_with_sampling(5000)  # 5k samples
        
        end_time = time.time()
        duration = end_time - start_time
        
        if results:
            print(f"✅ Quick sample test completed in {duration:.1f} seconds")
            print(f"   Score: {results['avg_score']:.4f} ± {results['score_std']:.4f}")
            return True
        else:
            print("❌ Quick sample test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in quick sample test: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("🔍 Checking data files...")
    
    required_files = [
        '/Users/yixuan/DRW/data/train.parquet',
        '/Users/yixuan/DRW/data/test.parquet', 
        '/Users/yixuan/DRW/data/sample_submission.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease ensure data files are in the correct location.")
        return False
    else:
        print("✅ All data files found")
        return True

if __name__ == "__main__":
    print("🧪 Quick Analysis Optimization Test")
    print("=" * 40)
    
    # Check data files first
    if not check_data_files():
        sys.exit(1)
    
    print("\n" + "="*40)
    
    # Test ultra-quick analysis
    ultra_success = test_ultra_quick()
    
    print("\n" + "="*40)
    
    # Test quick analysis with sampling
    sample_success = test_quick_sample()
    
    print("\n" + "="*40)
    print("📊 Test Results Summary:")
    print(f"   Ultra-quick: {'✅ PASS' if ultra_success else '❌ FAIL'}")
    print(f"   Quick sample: {'✅ PASS' if sample_success else '❌ FAIL'}")
    
    if ultra_success and sample_success:
        print("\n🎉 All tests passed! Quick analysis is optimized for local Mac.")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.") 