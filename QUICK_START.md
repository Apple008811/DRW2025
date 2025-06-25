# Quick Start Guide - Local Mac Optimization

## 🚀 Quick Start

This guide helps you quickly run cryptocurrency market prediction analysis on your local Mac, avoiding performance issues.

## 📋 Prerequisites

1. **Data Files**: The following files are already available and configured:
   - `train.parquet` (training data) - located at `/Users/yixuan/DRW/data/`
   - `test.parquet` (test data) - located at `/Users/yixuan/DRW/data/`  
   - `sample_submission.csv` (submission template) - located at `/Users/yixuan/DRW/data/`

2. **Python Packages**: Install required packages:
   ```bash
   pip install pandas numpy scipy lightgbm scikit-learn pyarrow
   ```

## ⚡ Running Options

### 1. Ultra-Lightweight Test (Recommended for first run)
```bash
python3 quick_analysis.py ultra
```
- **Features**: 5 top features + 2 lag features
- **Model**: 20 estimators, shallow trees
- **Time**: ~30 seconds
- **Purpose**: Verify environment setup, quick testing

### 2. Standard Quick Analysis
```bash
python3 quick_analysis.py
```
- **Features**: 10 top features + 3 lag features
- **Model**: 50 estimators
- **Time**: ~1-2 minutes
- **Purpose**: Standard quick analysis

### 3. Sampling Analysis (for large datasets)
```bash
# Use 10k samples
python3 quick_analysis.py sample

# Use 5k samples (faster)
python3 quick_analysis.py sample 5000
```
- **Purpose**: Quick testing for large datasets
- **Time**: Adjusted based on sample size

## 🧪 Test Optimization Results

Run the test script to verify optimization:
```bash
python3 test_quick.py
```

## 📊 Performance Comparison

| Mode | Features | Estimators | Expected Time | Use Case |
|------|----------|------------|---------------|----------|
| ultra | 5 | 20 | ~30 seconds | Environment verification |
| default | 10 | 50 | ~1-2 minutes | Standard testing |
| sample 5k | 8 | 30 | ~45 seconds | Large dataset testing |
| sample 10k | 10 | 50 | ~1 minute | Large dataset testing |

## 🔧 Troubleshooting

### Issue 1: File not found
```
❌ Error: data/train.parquet not found!
```
**Solution**: Check data file paths, ensure files are in correct location.

### Issue 2: Insufficient memory
```
MemoryError: Unable to allocate array
```
**Solution**: Use lighter options:
```bash
python3 quick_analysis.py ultra
# or
python3 quick_analysis.py sample 3000
```

### Issue 3: Slow execution
**Solution**: 
1. Use `ultra` mode
2. Reduce sample size
3. Close other applications to free memory

## 📈 Output Files

- `ultra_quick_submission.csv` - Ultra-lightweight analysis results
- `quick_submission.csv` - Standard quick analysis results  
- `quick_sample_submission.csv` - Sampling analysis results

## 🎯 Next Steps

1. **Local Testing**: Use `ultra` mode to verify environment
2. **Standard Analysis**: Use default mode for complete analysis
3. **Kaggle Deployment**: Use `full_analysis.py` on Kaggle for production-level analysis

## 💡 Tips

- Recommended to use `ultra` mode for first run
- If still slow, try reducing sample size
- Ensure sufficient disk space for result files
- Monitor system resource usage 