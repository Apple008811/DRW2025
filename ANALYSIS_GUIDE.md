# Analysis Scripts Guide

This guide explains how to use the split analysis scripts for different environments and use cases.

## 📁 File Structure

```
DRW/
├── analysis_core.py      # Shared core logic (used by both scripts)
├── quick_analysis.py     # Quick analysis for local Mac
├── full_analysis.py      # Full analysis for Kaggle notebooks
└── ANALYSIS_GUIDE.md     # This guide
```

## 🚀 Quick Analysis (Local Mac)

**Purpose**: Fast prototyping and testing on your local machine
**Target**: Mac with limited resources
**Features**: 20 top features + 5 lag features
**CV**: 3-fold time series split
**Model**: 100 estimators, learning rate 0.1

### Usage

```bash
# Basic quick analysis
python quick_analysis.py

# Quick analysis with data sampling (10k samples)
python quick_analysis.py sample

# Quick analysis with custom sample size (5k samples)
python quick_analysis.py sample 5000

# Show help
python quick_analysis.py help
```

### Output Files
- `quick_submission.csv` - Basic quick analysis results
- `quick_sample_submission.csv` - Sampled analysis results

### When to Use
- ✅ Testing new ideas quickly
- ✅ Debugging code changes
- ✅ Local development and prototyping
- ✅ When you have limited time/resources
- ❌ Final submissions (use full analysis instead)

---

## 🖥️ Full Analysis (Kaggle Notebooks)

**Purpose**: Production-ready analysis with maximum performance
**Target**: Kaggle notebooks with full resources
**Features**: 100 top features + 20 lag features
**CV**: 5-fold time series split
**Model**: 200 estimators, learning rate 0.05, deeper trees

### Usage

```python
# In Kaggle notebook
!python full_analysis.py                    # Basic full analysis
!python full_analysis.py comprehensive      # Multi-stage comparison
!python full_analysis.py tuning             # Hyperparameter tuning
!python full_analysis.py help               # Show help
```

### Output Files
- `full_submission.csv` - Basic full analysis results
- `comprehensive_quick_submission.csv` - Quick stage results
- `comprehensive_medium_submission.csv` - Medium stage results
- `comprehensive_full_submission.csv` - Full stage results
- `tuning_submission_*.csv` - Hyperparameter tuning results

### When to Use
- ✅ Final model training and submissions
- ✅ Comprehensive analysis and comparison
- ✅ Hyperparameter optimization
- ✅ Production-ready results
- ❌ Quick testing (use quick analysis instead)

---

## 🔧 Shared Core Module

The `analysis_core.py` module contains all the shared logic:

### Key Functions

1. **`load_and_prepare_data()`** - Load data and select top features
2. **`engineer_features()`** - Create lag features and handle missing values
3. **`train_and_evaluate_model()`** - Train LightGBM with time series CV
4. **`get_feature_importance()`** - Extract and display feature importance
5. **`generate_predictions()`** - Generate and save predictions
6. **`run_analysis_stage()`** - Complete analysis pipeline

### Benefits of Shared Core
- ✅ Consistent logic across all scripts
- ✅ Easy to maintain and update
- ✅ Reusable functions
- ✅ No code duplication

---

## 📊 Analysis Stages Comparison

| Stage | Features | Lag Features | CV Splits | Estimators | Use Case |
|-------|----------|--------------|-----------|------------|----------|
| Quick | 20 | 5 | 3 | 100 | Local testing |
| Medium | 50 | 10 | 4 | 150 | Intermediate |
| Full | 100 | 20 | 5 | 200 | Production |

---

## 🛠️ Configuration

### Quick Analysis Config (Local Mac)
```python
QUICK_CONFIG = {
    'train_file': 'data/train.csv',
    'test_file': 'data/test.csv', 
    'submission_file': 'data/sample_submission.csv',
    'top_n': 20,  # Fewer features for speed
    'lag_feature_count': 5,  # Fewer lag features
    'n_splits': 3,  # Fewer CV splits
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'output_file': 'quick_submission.csv'
}
```

### Full Analysis Config (Kaggle)
```python
FULL_CONFIG = {
    'train_file': '/kaggle/input/drw-crypto-market-prediction/train.csv',
    'test_file': '/kaggle/input/drw-crypto-market-prediction/test.csv',
    'submission_file': '/kaggle/input/drw-crypto-market-prediction/sample_submission.csv',
    'top_n': 100,  # More features for better performance
    'lag_feature_count': 20,  # More lag features
    'n_splits': 5,  # More CV splits for better validation
    'n_estimators': 200,  # More estimators for better performance
    'learning_rate': 0.05,  # Lower learning rate for more stable training
    'max_depth': 8,  # Slightly deeper trees
    'output_file': 'full_submission.csv'
}
```

---

## 🔄 Workflow Recommendations

### Development Workflow
1. **Local Development**: Use `quick_analysis.py` for rapid prototyping
2. **Testing**: Use `quick_analysis.py sample` for ultra-fast testing
3. **Production**: Use `full_analysis.py` on Kaggle for final results

### Kaggle Workflow
1. **Baseline**: Run `full_analysis.py` for baseline results
2. **Comparison**: Run `full_analysis.py comprehensive` for multi-stage comparison
3. **Optimization**: Run `full_analysis.py tuning` for hyperparameter optimization
4. **Submission**: Use the best performing model's output file

---

## 📈 Performance Expectations

### Quick Analysis (Local Mac)
- ⏱️ Execution time: 2-5 minutes
- 💾 Memory usage: ~2-4 GB
- 🎯 Expected score: Baseline performance
- 📱 Suitable for: MacBook Air/Pro

### Full Analysis (Kaggle)
- ⏱️ Execution time: 10-30 minutes
- 💾 Memory usage: ~8-16 GB
- 🎯 Expected score: Production performance
- 🖥️ Suitable for: Kaggle GPU/TPU notebooks

---

## 🚨 Troubleshooting

### Common Issues

1. **File not found errors**
   - Check data file paths in config
   - Ensure data files exist in specified locations

2. **Memory issues on local Mac**
   - Use `python quick_analysis.py sample` for smaller datasets
   - Reduce `top_n` or `lag_feature_count` in config

3. **Slow execution on Kaggle**
   - Use GPU acceleration if available
   - Reduce `n_splits` or `n_estimators` for faster testing

4. **Import errors**
   - Ensure all required packages are installed
   - Check Python version compatibility

### Getting Help
```bash
# Quick analysis help
python quick_analysis.py help

# Full analysis help
python full_analysis.py help
```

---

## 📝 Best Practices

1. **Always start with quick analysis** for new ideas
2. **Use sampling** for ultra-fast testing
3. **Run comprehensive analysis** on Kaggle for final results
4. **Save intermediate results** for comparison
5. **Document your findings** in notebooks or comments
6. **Version control your scripts** and results

---

## 🔮 Future Enhancements

Potential improvements for the analysis scripts:

1. **Additional models**: XGBoost, CatBoost, Neural Networks
2. **Advanced features**: Technical indicators, sentiment analysis
3. **Ensemble methods**: Stacking, blending multiple models
4. **Automated tuning**: Bayesian optimization, Optuna
5. **Visualization**: Feature importance plots, prediction plots
6. **Monitoring**: Training progress, resource usage tracking 