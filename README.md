# DRW Crypto Market Prediction

This project implements a machine learning pipeline for predicting cryptocurrency market movements using various forecasting methods.

## Project Overview

The goal is to predict cryptocurrency market movements using historical market data. The project implements multiple forecasting methods and evaluates them using the Pearson correlation coefficient.

### Key Features

- Multiple prediction methods:
  - LightGBM (Gradient Boosting)
  - Gaussian Process (Bayesian approach)
  - Ensemble of multiple models
- Time series cross-validation
- Feature engineering with technical indicators
- Model evaluation using Pearson correlation coefficient

---

## Methodology & Execution Plan

This project follows a systematic, end-to-end approach to time series forecasting, combining both theoretical methodology and practical execution. Each phase below includes both the conceptual focus and the concrete steps to implement it, with all required models and methods explicitly listed.

| Phase | Theoretical Focus | Practical Steps |
|-------|-------------------|-----------------|
| 1. Environment & Baseline | Environment setup, data loading, baseline model | - Install dependencies<br>- Load and validate data<br>- Run quick_test.py for baseline LightGBM model<br>- Generate ultra_quick_submission.csv and check Pearson correlation |
| 2. Data Exploration & Feature Engineering | Exploratory Data Analysis (EDA), feature construction, feature selection | - Analyze data structure and distributions<br>- Visualize key features<br>- Engineer new features (technical indicators, rolling stats, lag features, time-based features)<br>- Select features using correlation, importance, and statistical tests |
| 3. Model Training & Optimization | Comprehensive model testing: classical, ML, and hyperparameter tuning | - Train and compare the following models:<br> • ARIMA/SARIMA<br> • Prophet<br> • Linear Regression, Ridge, Lasso<br> • Random Forest<br> • XGBoost<br> • LightGBM<br> • Support Vector Regression (SVR)<br>- Use time series cross-validation<br>- Tune hyperparameters (e.g., LightGBM: num_leaves, learning_rate, feature_fraction)<br>- Compare model performance using Pearson correlation |
| 4. Advanced Modeling & Ensembling | Deep learning, Bayesian methods, model ensembling | - Implement and test:<br> • LSTM, GRU, Transformer neural networks<br> • Gaussian Process (Bayesian approach)<br>- Ensemble models (stacking, blending, weighted averaging)<br>- Apply post-processing (calibration, outlier handling, smoothing) |
| 5. Validation & Submission | Final validation, result analysis, submission preparation | - Perform cross-validation and stability checks<br>- Analyze prediction distributions, errors, and feature importance<br>- Prepare and verify submission files<br>- Backup results |

### Phase Details

#### Phase 1: Environment & Baseline
- **Theory:** Ensure reproducibility and a working baseline for comparison.
- **Steps:**
  - Install all required packages (see Setup section)
  - Load training, testing, and sample submission data
  - Run `python quick_test.py` to verify environment and generate a baseline LightGBM model
  - Check that Pearson correlation > 0.1 and ultra_quick_submission.csv is created

#### Phase 2: Data Exploration & Feature Engineering
- **Theory:** Understand data structure, identify patterns, and create informative features.
- **Steps:**
  - Use pandas and visualization tools to explore data (e.g., `train.describe()`, `train.info()`, histograms)
  - Engineer features:
    - Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
    - Statistical features: rolling mean, std, min, max
    - Lag features: previous time steps
    - Time-based features: hour, weekday, month, cyclical encoding
    - Interaction features: bid/ask volume ratios, activity intensity
  - Select features using:
    - Correlation analysis (Pearson, Spearman)
    - Feature importance (from tree models)
    - Recursive feature elimination
    - Mutual information and filter methods

#### Phase 3: Model Training & Optimization
- **Theory:** Build and optimize predictive models using both classical and machine learning approaches.
- **Steps:**
  - Train and compare the following models:
    - Classical: ARIMA/SARIMA, Prophet
    - Machine Learning: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, LightGBM, SVR
  - Use time series cross-validation for robust evaluation
  - Tune hyperparameters (e.g., grid search for LightGBM: num_leaves, learning_rate, feature_fraction)
  - Select best models based on Pearson correlation and other metrics (MAE, R²)

#### Phase 4: Advanced Modeling & Ensembling
- **Theory:** Leverage advanced models and combine predictions for improved accuracy and robustness.
- **Steps:**
  - Implement and test deep learning models:
    - LSTM (Long Short-Term Memory)
    - GRU (Gated Recurrent Unit)
    - Transformer neural networks
  - Apply Bayesian methods:
    - Gaussian Process Regression
  - Ensemble multiple models:
    - Stacking
    - Blending
    - Weighted averaging
  - Post-process predictions:
    - Calibration
    - Outlier handling
    - Smoothing techniques

#### Phase 5: Validation & Submission
- **Theory:** Ensure model generalization, analyze results, and prepare for competition submission.
- **Steps:**
  - Perform time series cross-validation and multi-fold validation
  - Analyze prediction distributions, errors, and feature importance
  - Prepare submission files in required format and verify file size
  - Backup and document results

---

## Setup and Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Installation Steps

1. Clone the repository:
```bash
git clone [repository-url]
cd DRW
```

2. Install dependencies:
```bash
# For Apple Silicon Macs
pip install -r requirements_apple_silicon.txt

# For other systems
pip install -r requirements.txt
```

3. Verify installation:
```bash
python check_setup.py
```

### Data Setup

1. Place the following data files in the `data/` directory:
- train.parquet (3.3GB)
- test.parquet (3.4GB)
- sample_submission.csv (14MB)

2. Run data validation:
     ```bash
python test_data_loading.py
     ```

## Project Structure

```
DRW/
├── analysis_core.py      # Shared core logic (used by all scripts)
├── quick_analysis.py     # Quick analysis for local Mac
├── full_analysis.py      # Full analysis for Kaggle notebooks
├── create_visualization.py # Visualization and analysis charts
├── data/                 # Data directory
├── models/               # Saved models
└── README.md            # This comprehensive guide
```

---

## Usage and Analysis Scripts

This project provides different analysis scripts optimized for different environments and use cases.

### 📁 File Structure

```
DRW/
├── analysis_core.py      # Shared core logic (used by both scripts)
├── quick_analysis.py     # Quick analysis for local Mac
├── full_analysis.py      # Full analysis for Kaggle notebooks
└── create_visualization.py # Visualization and analysis charts
```

### 🚀 Quick Analysis (Local Mac)

**Purpose**: Fast prototyping and testing on your local machine
**Target**: Mac with limited resources
**Features**: 20 top features + 5 lag features
**CV**: 3-fold time series split
**Model**: 100 estimators, learning rate 0.1

#### Usage

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

#### Output Files
- `ultra_quick_submission.csv` - Basic quick analysis results
- `quick_sample_submission.csv` - Sampled analysis results

#### When to Use
- ✅ Testing new ideas quickly
- ✅ Debugging code changes
- ✅ Local development and prototyping
- ✅ When you have limited time/resources
- ❌ Final submissions (use full analysis instead)

### 🖥️ Full Analysis (Kaggle Notebooks)

**Purpose**: Production-ready analysis with maximum performance
**Target**: Kaggle notebooks with full resources
**Features**: 100 top features + 20 lag features
**CV**: 5-fold time series split
**Model**: 200 estimators, learning rate 0.05, deeper trees

#### Usage

```python
# In Kaggle notebook
!python full_analysis.py                    # Basic full analysis
!python full_analysis.py comprehensive      # Multi-stage comparison
!python full_analysis.py tuning             # Hyperparameter tuning
!python full_analysis.py help               # Show help
```

#### Output Files
- `full_submission.csv` - Basic full analysis results
- `comprehensive_quick_submission.csv` - Quick stage results
- `comprehensive_medium_submission.csv` - Medium stage results
- `comprehensive_full_submission.csv` - Full stage results
- `tuning_submission_*.csv` - Hyperparameter tuning results

#### When to Use
- ✅ Final model training and submissions
- ✅ Comprehensive analysis and comparison
- ✅ Hyperparameter optimization
- ✅ Production-ready results
- ❌ Quick testing (use quick analysis instead)

### 🔧 Shared Core Module

The `analysis_core.py` module contains all the shared logic:

#### Key Functions

1. **`load_and_prepare_data()`** - Load data and select top features
2. **`engineer_features()`** - Create lag features and handle missing values
3. **`train_and_evaluate_model()`** - Train LightGBM with time series CV
4. **`get_feature_importance()`** - Extract and display feature importance
5. **`generate_predictions()`** - Generate and save predictions
6. **`run_analysis_stage()`** - Complete analysis pipeline

#### Benefits of Shared Core
- ✅ Consistent logic across all scripts
- ✅ Easy to maintain and update
- ✅ Reusable functions
- ✅ No code duplication

### 📊 Analysis Stages Comparison

| Stage | Features | Lag Features | CV Splits | Estimators | Use Case |
|-------|----------|--------------|-----------|------------|----------|
| Quick | 20 | 5 | 3 | 100 | Local testing |
| Medium | 50 | 10 | 4 | 150 | Intermediate |
| Full | 100 | 20 | 5 | 200 | Production |

### 🛠️ Configuration

#### Quick Analysis Config (Local Mac)
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
    'output_file': 'ultra_quick_submission.csv'
}
```

#### Full Analysis Config (Kaggle)
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

### 🔄 Workflow Recommendations

#### Development Workflow
1. **Local Development**: Use `quick_analysis.py` for rapid prototyping
2. **Testing**: Use `quick_analysis.py sample` for ultra-fast testing
3. **Production**: Use `full_analysis.py` on Kaggle for final results

#### Kaggle Workflow
1. **Baseline**: Run `full_analysis.py` for baseline results
2. **Comparison**: Run `full_analysis.py comprehensive` for multi-stage comparison
3. **Optimization**: Run `full_analysis.py tuning` for hyperparameter optimization
4. **Submission**: Use the best performing model's output file

### 📈 Performance Expectations

#### Quick Analysis (Local Mac)
- ⏱️ Execution time: 2-5 minutes
- 💾 Memory usage: ~2-4 GB
- 🎯 Expected score: Baseline performance
- 📱 Suitable for: MacBook Air/Pro

#### Full Analysis (Kaggle)
- ⏱️ Execution time: 10-30 minutes
- 💾 Memory usage: ~8-16 GB
- 🎯 Expected score: Production performance
- 🖥️ Suitable for: Kaggle GPU/TPU notebooks

### 🚨 Troubleshooting

#### Common Issues

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

#### Getting Help
```bash
# Quick analysis help
python quick_analysis.py help

# Full analysis help
python full_analysis.py help
```

### 📝 Best Practices

1. **Always start with quick analysis** for new ideas
2. **Use sampling** for ultra-fast testing
3. **Run comprehensive analysis** on Kaggle for final results
4. **Save intermediate results** for comparison
5. **Document your findings** in notebooks or comments
6. **Version control your scripts** and results

---

## Model Architecture

### 1. Data Processing
- Loading and cleaning data
- Feature engineering
- Technical indicator calculation

### 2. Feature Engineering
- Rolling statistics
- Technical indicators
- Volume-based features
- Price action features

### 3. Model Training
- LightGBM with custom Pearson correlation metric
- Gaussian Process (Bayesian approach)
- Ensemble prediction

### 4. Cross-Validation
- Time series cross-validation
- Multiple prediction methods validation
- Ensemble model validation

### 5. LightGBM Regressor Theory

LightGBM Regressor is an implementation of Gradient Boosting Decision Trees (GBDT) for regression tasks. The core idea is to build an ensemble of decision trees, where each new tree is trained to correct the errors (residuals) of the combined previous trees. The final prediction is the sum of the outputs from all trees:

    ŷ = f₁(x) + f₂(x) + ... + f_M(x)

where each fₘ(x) is a regression tree, and M is the total number of trees. At each iteration, the model fits a new tree to the negative gradient (residual) of the loss function (e.g., mean squared error) with respect to the current prediction. LightGBM optimizes this process with efficient histogram-based algorithms and leaf-wise tree growth, making it fast and scalable for large datasets.

Key properties:
- Each tree is trained to minimize the loss function by focusing on the errors of the previous ensemble.
- The model uses a learning rate to control the contribution of each tree.
- Feature importance can be extracted from the trained model to interpret which features are most influential.

For more details, see the [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/).

### 6. Pearson Correlation Coefficient Theory

The Pearson correlation coefficient (often denoted as **r**) is a statistical measure that quantifies the linear relationship between two continuous variables. In the context of this project, it is used to evaluate how well the model's predictions align with the actual market movements.

#### Formula

Given two variables, $X$ (true values) and $Y$ (predicted values), the Pearson correlation coefficient is calculated as:

$$
r = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^n (Y_i - \bar{Y})^2}}
$$

where:
- $X_i$, $Y_i$ are the individual sample points
- $\bar{X}$, $\bar{Y}$ are the means of $X$ and $Y$
- $n$ is the number of samples

#### Interpretation
- **r = 1**: Perfect positive linear correlation (predictions increase exactly as true values increase)
- **r = 0**: No linear correlation
- **r = -1**: Perfect negative linear correlation (predictions decrease exactly as true values increase)

#### Why Use Pearson Correlation for Evaluation?
- **Scale-invariant**: Measures the strength of the relationship regardless of the scale of the variables.
- **Sensitive to direction**: Captures whether predictions move in the same direction as the true values.
- **Widely used in finance and time series**: Especially useful for evaluating models where the direction and relative magnitude of changes are more important than absolute values.

In this project, a higher Pearson correlation indicates that the model's predictions more closely follow the actual market trends, making it a suitable metric for performance evaluation.

## Evaluation

The primary evaluation metric is the Pearson correlation coefficient between predictions and actual values. This measures the linear correlation between the predicted and actual market movements.

### Model Performance

Current model performance metrics:
- LightGBM: [Pearson correlation score]
- Gaussian Process: [Pearson correlation score]
- Ensemble: [Pearson correlation score]

## Experiment Results

| Date       | Mode  | Train Shape   | Test Shape    | Features | CV Score         | Output File                | Notes         |
|------------|-------|--------------|--------------|----------|------------------|----------------------------|---------------|
| 2024-06-23 | ultra | (525887,896) | (538150,896) | 13 (5 original + 8 lag) | - | ultra_quick_submission.csv | Baseline run, feature engineering completed |

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 