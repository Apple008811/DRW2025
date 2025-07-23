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

## Experiment Results

### Model Prediction Data Statistics

| Model | Mean | Std | Min | Max | Range | Training Time | Test Correlation | Status |
|-------|------|-----|-----|-----|-------|---------------|------------------|--------|
| **LightGBM** | -0.000828 | 0.504797 | -2.274801 | 2.380469 | 4.655270 | ~5-10 min | ~0.42 | ✅ Completed |
| **XGBoost** | 0.000286 | 0.486328 | -2.136424 | 2.279565 | 4.415988 | ~5-10 min | ~0.42 | ✅ Completed |
| **Random Forest** | -0.000373 | 0.453784 | -2.252822 | 2.071551 | 4.324373 | ~11 min | 0.4198 | ✅ Completed |
| Linear Regression | - | - | - | - | - | ~5-10 min | - | ⏳ Pending |
| Ridge Regression | - | - | - | - | - | ~5-10 min | - | ⏳ Pending |
| Lasso Regression | - | - | - | - | - | ~5-10 min | - | ⏳ Pending |
| ARIMA | - | - | - | - | - | ~10-30 min | - | ⏳ Pending |
| Prophet | - | - | - | - | - | ~10-30 min | - | ⏳ Pending |
| SARIMA | - | - | - | - | - | ~10-30 min | - | ⏳ Pending |
| SVR | - | - | - | - | - | ~10-20 min | - | ⏳ Pending |
| LSTM | - | - | - | - | - | ~30-60 min | - | ⏳ Pending |
| GRU | - | - | - | - | - | ~30-60 min | - | ⏳ Pending |
| Transformer | - | - | - | - | - | ~30-60 min | - | ⏳ Pending |
| Gaussian Process | - | - | - | - | - | ~20-40 min | - | ⏳ Pending |

**Key Observations:**
- **LightGBM**: Largest prediction range, most symmetric distribution, fastest training
- **XGBoost**: Similar performance to LightGBM, slightly positive bias
- **Random Forest**: Most conservative predictions, smallest range, slowest training
- **Best Submission Candidate**: LightGBM (largest range suitable for volatile crypto market)

### Phase 1: Ultra-Quick Test Results

| Date       | Mode  | Train Shape   | Test Shape    | Features | CV Score         | Output File                | Notes         |
|------------|-------|--------------|--------------|----------|------------------|----------------------------|---------------|
| 2025-06-15 | ultra | (525887,896) | (538150,896) | 13 (5 original + 8 lag) | 0.0474 ± 0.0034 | ultra_quick_submission.csv | Baseline run, feature engineering completed |
| 2025-06-15 | test  | (10000,896)  | (10000,896)  | 13 (5 original + 8 lag) | 0.0474 ± 0.0034 | quick_sample_submission.csv | Test script validation, 10k sample |

### Phase 2: Data Exploration & Analysis Results

| Date       | Analysis Type | Data Size | Key Findings | Output Files | Notes |
|------------|---------------|-----------|--------------|--------------|-------|
| 2025-07-20 | Complete EDA | 525,887 rows × 896 cols | • Label distribution: concentrated near 0<br>• Market data correlations: very weak (0.005-0.01)<br>• Technical indicators: 890 features, most with low correlation<br>• Label autocorrelation: very strong (0.97 at lag 1, 0.71 at lag 20) | phase2_analysis_charts.png<br>phase2_results.json | Full dataset analysis, comprehensive insights for feature engineering |

#### Phase 2 Key Insights

**Label Analysis**:
- **Distribution**: Highly concentrated near 0, symmetric distribution
- **Range**: Most values between -5 and 5
- **Autocorrelation**: Very strong time dependency (0.97 at lag 1, 0.71 at lag 20)

**Market Data Analysis**:
- **Volume Features**: bid_qty, ask_qty, buy_qty, sell_qty, volume
- **Correlations**: All very weak (0.005 to 0.01 absolute correlation)
- **Derived Features**: bid_ask_ratio, buy_sell_ratio, volume_intensity show potential

**Technical Indicators Analysis**:
- **Total Count**: 890 technical indicators (X1-X890)
- **Correlation Distribution**: 
  - High correlation (|corr| > 0.1): ~5-10% of features
  - Medium correlation (0.05 < |corr| ≤ 0.1): ~15-20% of features
  - Low correlation (|corr| ≤ 0.05): ~70-80% of features
- **Feature Selection**: Need to focus on high-correlation indicators

**Feature Engineering Strategy**:
1. **Lag Features**: Critical importance based on strong autocorrelation
2. **Market Imbalance Features**: bid_ask_ratio, buy_sell_ratio, volume_intensity
3. **Technical Indicator Selection**: Focus on top 100-200 high-correlation features
4. **Time-based Features**: Cyclical encoding of time components
5. **Interaction Features**: Ratios and combinations of market data

### Test Script Results

| Test Function | Status | Execution Time | Sample Size | Features | CV Score | Notes |
|---------------|--------|----------------|-------------|----------|----------|-------|
| `test_ultra_quick()` | ✅ PASS | ~30-60 seconds | Full dataset | 13 (5+8) | 0.0474 ± 0.0034 | Ultra-quick analysis validation |
| `test_quick_sample()` | ✅ PASS | ~30-60 seconds | 10,000 rows | 13 (5+8) | 0.0474 ± 0.0034 | Sampling analysis validation |
| `check_data_files()` | ✅ PASS | <1 second | N/A | N/A | N/A | Data file validation |

---

## Methodology & Execution Plan

This project follows a systematic, end-to-end approach to time series forecasting, combining both theoretical methodology and practical execution. Each phase below includes both the conceptual focus and the concrete steps to implement it, with all required models and methods explicitly listed.

| Phase | Theoretical Focus | Practical Steps |
|-------|-------------------|-----------------|
| 1. Environment & Baseline | Environment setup, data loading, baseline model | - Install dependencies<br>- Load and validate data<br>- Run quick_test.py for baseline LightGBM model<br>- Generate ultra_quick_submission.csv and check Pearson correlation |
| 2. Data Exploration & Feature Engineering | Exploratory Data Analysis (EDA), feature construction, feature selection | - Analyze data structure and distributions<br>- Visualize key features<br>- Engineer new features (technical indicators, rolling stats, lag features, time-based features)<br>- Select features using correlation, importance, and statistical tests |
| 3. Model Training & Optimization | Comprehensive model testing: classical, ML, and hyperparameter tuning | **Phase 3 (Basic Models):**<br>• **Linear Models**: Linear, Ridge, Lasso<br>  - Script: `lightweight_models.py` (5-10 minutes)<br>• **Tree Models**: LightGBM, XGBoost<br>  - Script: `lightgbm_training.py` (5-10 minutes)<br>  - Script: `xgboost_training.py` (5-10 minutes)<br>• **Time Series Models**: ARIMA, SARIMA, Prophet<br>  - Script: `sarima_training.py` (10-30 minutes)<br>• **Other Models**: SVR, Random Forest<br>  - Script: `individual_model_training.py` (可选择性训练)<br>• **Hyperparameter Tuning**: LightGBM optimization<br>• **Model Ensemble**: Weighted averaging |
| 4. Advanced Modeling & Ensembling | Deep learning, Bayesian methods, model ensembling | **Phase 4 (Advanced Models):**<br>• Deep Learning: LSTM, GRU<br>• Bayesian Methods: Gaussian Process Regression<br>• Advanced Ensembles: Stacking, Voting<br>• Post-processing: Outlier handling, Smoothing, Calibration<br><br>**Files:**<br>• `phase4_model_training_kaggle.py` - Phase 3 implementation (basic models)<br>• `phase4_advanced_modeling_kaggle.py` - Phase 4 implementation (advanced models) |
| 5. Validation & Submission | Final validation, result analysis, submission preparation | - Perform cross-validation and stability checks<br>- Analyze prediction distributions, errors, and feature importance<br>- Prepare and verify submission files<br>- Backup results |

## Script Usage Guide

### Model Training Scripts

#### **Phase 3: Basic Models Training**

**1. Linear Models (Fast & Lightweight)**
```bash
python linear_models_training.py
```
- **Models**: Linear Regression, Ridge, Lasso
- **Time**: 5-10 minutes
- **Memory**: Low usage
- **Output**: `/kaggle/working/results/linear_models_results.csv`

**2. Tree Models (Individual Training)**
```bash
# LightGBM
python lightgbm_training.py

# XGBoost  
python xgboost_training.py

# Random Forest
python random_forest_training.py
```
- **Time**: 5-15 minutes each
- **Memory**: Medium usage
- **Output**: `/kaggle/working/results/lightgbm_results.csv`, `/kaggle/working/results/xgboost_results.csv`, `/kaggle/working/results/random_forest_results.csv`

**3. Time Series Models**
```bash
# ARIMA and Prophet
python time_series_models_training.py

# SARIMA (standalone)
python sarima_training.py
```
- **Time**: 10-30 minutes
- **Memory**: Medium usage
- **Output**: `/kaggle/working/results/time_series_models_results.csv`, `/kaggle/working/results/sarima_results.csv`

**4. Support Vector Regression**
```bash
python svr_training.py
```
- **Models**: SVR with RBF kernel
- **Time**: 10-20 minutes
- **Memory**: Medium usage
- **Output**: `/kaggle/working/results/svr_results.csv`

#### **Phase 4: Advanced Models**

**1. Neural Networks (Deep Learning)**
```bash
python neural_networks_training.py
```
- **Models**: LSTM, GRU, Transformer
- **Time**: 30-60 minutes
- **Memory**: High usage
- **Output**: `/kaggle/working/results/neural_networks_results.csv`

**2. Gaussian Process Regression**
```bash
python gaussian_process_training.py
```
- **Models**: Gaussian Process with multiple kernels (RBF, Matern, RationalQuadratic)
- **Time**: 20-40 minutes
- **Memory**: High usage
- **Output**: `/kaggle/working/results/gaussian_process_results.csv`

### Training Strategy

1. **Start with Linear Models** (`linear_models_training.py`)
   - Quick baseline results
   - Low memory usage
   - Fast validation

2. **Continue with Tree Models** (`lightgbm_training.py`, `xgboost_training.py`, `random_forest_training.py`)
   - Better performance expected
   - Moderate memory usage
   - Individual training to avoid conflicts

3. **Time Series Models** (`time_series_models_training.py`, `sarima_training.py`)
   - Classical time series approaches
   - Good for capturing temporal patterns
   - Moderate computational cost

4. **Support Vector Regression** (`svr_training.py`)
   - Non-linear modeling capability
   - Moderate memory usage
   - Good for complex patterns

5. **Advanced Models** (if needed)
   - **Neural Networks** (`neural_networks_training.py`): LSTM, GRU, Transformer
   - **Gaussian Process** (`gaussian_process_training.py`): Bayesian approach
   - High computational cost, run individually

### Results Location
All results are saved in `/kaggle/working/results/` with the following format:

**Phase 3 Models:**
- `linear_models_results.csv` - Linear Regression, Ridge, Lasso results
- `lightgbm_results.csv` - LightGBM results  
- `xgboost_results.csv` - XGBoost results
- `random_forest_results.csv` - Random Forest results
- `time_series_models_results.csv` - ARIMA, Prophet results
- `sarima_results.csv` - SARIMA results
- `svr_results.csv` - Support Vector Regression results

**Phase 4 Models:**
- `neural_networks_results.csv` - LSTM, GRU, Transformer results
- `gaussian_process_results.csv` - Gaussian Process Regression results

**Submission Files:**
- `*_submission.csv` - Individual model submission files
- `ensemble_submission.csv` - Ensemble model submission

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
**Features**: 10 top features + 3 lag features (optimized for speed)
**CV**: 2-fold time series split
**Model**: 50 estimators, learning rate 0.15

#### Usage

```bash
# Basic quick analysis
python quick_analysis.py

# Ultra-quick analysis (5 features, 20 estimators)
python quick_analysis.py ultra

# Quick analysis with data sampling (10k samples)
python quick_analysis.py sample

# Quick analysis with custom sample size (5k samples)
python quick_analysis.py sample 5000

# Show help
python quick_analysis.py help
```

#### Output Files
- `quick_submission.csv` - Standard quick analysis results
- `ultra_quick_submission.csv` - Ultra-quick analysis results
- `quick_sample_submission.csv` - Sampled analysis results

#### When to Use
- ✅ Testing new ideas quickly
- ✅ Debugging code changes
- ✅ Local development and prototyping
- ✅ When you have limited time/resources
- ❌ Final submissions (use full analysis instead)

### 🧪 Test Script (test_quick.py)

**Purpose**: Verify and test the functionality of quick_analysis.py
**Target**: Validation and testing of analysis scripts

#### Usage

```bash
# Run all tests
python test_quick.py
```

#### Test Functions
- **`test_ultra_quick()`**: Tests ultra-quick analysis functionality
- **`test_quick_sample()`**: Tests quick analysis with sampling
- **`check_data_files()`**: Verifies required data files exist

#### Test Output
- ⏱️ Execution time measurement
- ✅/❌ Success/failure status
- 📊 Performance metrics (Pearson correlation)
- 🔍 Data file validation

#### When to Use
- ✅ After making changes to quick_analysis.py
- ✅ Before committing code changes
- ✅ To verify environment setup
- ✅ To validate data file locations

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
| Ultra-Quick | 5 | 2 | 2 | 20 | Ultra-fast testing |
| Quick | 10 | 3 | 2 | 50 | Local testing |
| Medium | 50 | 10 | 4 | 150 | Intermediate |
| Full | 100 | 20 | 5 | 200 | Production |

### 📈 Data Sampling Logic

#### Dataset Information
- **Training Data**: 525,887 rows (2023-03-01 to 2024-02-29)
- **Test Data**: Similar size and time range
- **Features**: 896 columns (including target variable)

#### Sampling Strategy

**1. Correlation Calculation Sampling**
- **Trigger**: When dataset > 100,000 rows
- **Sample Size**: 50,000 rows (fixed)
- **Purpose**: Speed up correlation calculation for feature selection
- **Random State**: 42 (reproducible)

**2. Quick Analysis Sampling**
- **Default Sample Size**: 10,000 rows
- **Custom Sample Size**: Configurable (e.g., 5,000)
- **Random State**: 42 (reproducible)
- **Purpose**: Ultra-fast testing on local machines

**3. Sampling Behavior**
```python
# Example: 10,000 sample with random_state=42
sampled_data = train_data.sample(n=10000, random_state=42).reset_index(drop=True)
```

**4. Sampled Data Characteristics**
- **Time Range**: Covers entire dataset period (2023-03-01 to 2024-02-29)
- **Distribution**: Random sampling preserves temporal distribution
- **Reproducibility**: Same random_state ensures consistent results

**5. Detailed Sampling Analysis (10,000 samples with random_state=42)**

**Sampling Parameters**:
- Original dataset: 525,887 rows
- Sample size: 10,000 rows (1.90%)
- Random seed: 42
- Row range: Row 3 to Row 525,887

**Specific Sampled Row Positions**:
- **First 50 row numbers**: [458379, 233213, 231699, 482014, 34207, 320024, 160123, 497749, 89980, 442198, 149762, 137622, 172870, 421094, 508049, 422810, 243504, 280441, 414464, 7068, 114069, 221533, 446158, 133439, 506082, 149449, 447546, 157679, 4741, 169534, 459322, 360487, 349794, 201367, 152555, 379094, 415104, 317439, 51883, 438567, 193040, 405515, 78390, 394700, 410373, 251761, 264516, 167218, 85172, 8010]

- **Last 50 row numbers**: [494547, 233779, 23110, 423742, 224526, 76059, 297856, 88690, 205005, 178937, 294850, 259954, 180808, 309905, 303169, 443646, 514078, 167822, 464932, 168490, 502285, 51760, 353070, 274142, 272633, 382127, 471050, 315333, 227278, 461072, 463579, 25385, 175938, 383435, 123755, 142717, 162031, 315166, 179380, 9261, 493509, 289683, 454213, 232512, 363096, 400668, 308635, 490080, 306569, 138760]

**Sampling Distribution Statistics**:
- Minimum row number: 2 (Row 3)
- Maximum row number: 525886 (Row 525,887)
- Average interval: 52.6 rows
- Time span: 365 days

**⚠️ Important Sampling Limitations and Considerations**

**Data Loss Analysis**:
- **Missing rows**: 515,887 rows (98.10% of original data)
- **Time continuity loss**: Random sampling breaks continuous time series
- **Average time gap**: 52.6 minutes between sampled points (vs. 1 minute in original)

**Potential Issues with Random Sampling**:

1. **Time Series Continuity Loss**:
   - Original data: Continuous 1-minute intervals
   - Sampled data: Random jumps with 52.6-minute average gaps
   - Impact: Loss of sequential time patterns

2. **Feature Engineering Impact**:
   - **Lag features**: May lose temporal continuity (e.g., t-1 might jump from t-52)
   - **Rolling statistics**: May be calculated on discontinuous time windows
   - **Time-based features**: May not reflect actual time relationships

3. **Cross-Validation Issues**:
   - **Time series CV**: Random sampling may include future data in validation sets
   - **Data leakage**: Risk of using future information to predict past
   - **Validation reliability**: May not reflect real-world performance

4. **Model Training Limitations**:
   - **Pattern learning**: Model may learn from discontinuous time patterns
   - **Generalization**: May not generalize well to continuous time series
   - **Prediction accuracy**: May be less accurate on full continuous data

**Recommendations for Sampling**:
- ✅ Use sampling only for **ultra-fast testing** and **prototyping**
- ✅ For **production models**, use full dataset or systematic sampling
- ✅ Consider **time-based sampling** (e.g., every Nth minute) for better continuity
- ✅ Validate results on full dataset before final submission

**6. Sample Size Recommendations**
- **5,000 samples**: Ultra-fast testing (~30 seconds)
- **10,000 samples**: Standard quick testing (~1-2 minutes)
- **50,000 samples**: Correlation calculation (automatic)
- **Full dataset**: Production analysis (525,887 rows)

### 🛠️ Configuration

#### Quick Analysis Config (Local Mac)
```python
QUICK_CONFIG = {
    'train_file': '/Users/yixuan/DRW/data/train.parquet',
    'test_file': '/Users/yixuan/DRW/data/test.parquet', 
    'submission_file': '/Users/yixuan/DRW/data/sample_submission.csv',
    'top_n': 10,  # Fewer features for speed
    'lag_feature_count': 3,  # Fewer lag features
    'n_splits': 2,  # Fewer CV splits
    'n_estimators': 50,  # Fewer estimators
    'learning_rate': 0.15,  # Higher learning rate for faster convergence
    'max_depth': 4,  # Shallower tree depth
    'output_file': 'quick_submission.csv'
}
```

#### Ultra-Quick Config (Minimal Resources)
```python
ULTRA_QUICK_CONFIG = {
    'top_n': 5,  # Minimal features
    'lag_feature_count': 2,  # Minimal lag features
    'n_splits': 2,
    'n_estimators': 20,  # Minimal estimators
    'learning_rate': 0.2,  # Higher learning rate
    'max_depth': 3,  # Shallowest tree depth
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

#### Ultra-Quick Analysis (Local Mac)
- ⏱️ Execution time: 30-60 seconds
- 💾 Memory usage: ~1-2 GB
- 🎯 Expected score: Baseline performance
- 📱 Suitable for: Any Mac

#### Quick Analysis (Local Mac)
- ⏱️ Execution time: 1-2 minutes
- 💾 Memory usage: ~2-3 GB
- 🎯 Expected score: Baseline performance
- 📱 Suitable for: MacBook Air/Pro

#### Quick Analysis with Sampling (Local Mac)
- ⏱️ Execution time: 30 seconds - 2 minutes (depending on sample size)
- 💾 Memory usage: ~1-2 GB
- 🎯 Expected score: Baseline performance
- 📱 Suitable for: Any Mac

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

## Target Variable (Label) Analysis

### Label Definition and Characteristics

The target variable `label` represents **cryptocurrency price movement percentage** that we aim to predict. This section provides a comprehensive analysis of the label based on actual data verification.

#### Label Verification Results

**Data Source**: train.parquet (525,887 rows, 2023-03-01 to 2024-02-29)

**Basic Statistics**:
- **Range**: -24.42% to +20.74%
- **Mean**: 0.036% (close to zero, indicating balanced up/down movements)
- **Standard Deviation**: 1.01%
- **Distribution**: 48.83% negative, 51.17% positive (nearly symmetric)

**Time Series Characteristics**:
- **High Autocorrelation**: Lag 1 = 0.981, Lag 2 = 0.963, Lag 3 = 0.945
- **Smooth Transitions**: Continuous time series with gradual changes
- **No Zero Values**: 0.00% of values are exactly zero

**Distribution Analysis**:
- **Concentration**: 96.0% of values fall within [-1.84%, 2.68%)
- **Extreme Values**: Only 0.1% of values exceed ±6.35%
- **Outliers**: Very few extreme values (1 sample at -24.42%, 1 sample at +20.74%)

#### Label Calculation Hypothesis

Based on the data characteristics, the label likely represents:

```
label = (Future_Price - Current_Price) / Current_Price × 100%
```

**Evidence Supporting This Hypothesis**:
1. **Percentage Range**: ±20% is typical for cryptocurrency price movements
2. **Symmetric Distribution**: Price movements are naturally symmetric around zero
3. **Time Series Properties**: High autocorrelation suggests price continuity
4. **Market Context**: DRW is a quantitative trading firm focused on price prediction

#### Label vs. Market Data Correlation

**Raw Market Data Correlation** (Pearson coefficient):
- `bid_qty`: -0.013 (very weak negative)
- `ask_qty`: -0.016 (very weak negative)  
- `buy_qty`: 0.006 (very weak positive)
- `sell_qty`: 0.011 (very weak positive)
- `volume`: 0.009 (very weak positive)

**Interpretation**:
- **Low Linear Correlation**: Raw market data shows minimal linear relationship with label
- **Nonlinear Relationships**: Important relationships may be nonlinear or time-lagged
- **Feature Engineering Required**: Simple features need transformation for predictive power
- **Sample Size Effect**: With 525K samples, even small correlations may be statistically significant

#### Label Prediction Challenges

**1. Time Series Nature**:
- Label represents future price movement (forward-looking)
- Requires time series models and lag features
- Cross-validation must respect temporal order

**2. Nonlinear Relationships**:
- Simple linear correlations are insufficient
- Complex feature interactions needed
- Advanced models (LightGBM, neural networks) required

**3. Market Microstructure**:
- Price movements influenced by order book dynamics
- Bid/ask spreads, volume profiles, market depth
- High-frequency trading patterns

**4. Feature Engineering Strategy**:
- **Lag Features**: Previous time steps of important features
- **Ratio Features**: bid_qty/ask_qty, buy_qty/sell_qty, volume ratios
- **Technical Indicators**: Moving averages, momentum, volatility measures
- **Interaction Features**: Combinations of market variables

#### Label Analysis Summary

**✅ Confirmed Characteristics**:
- Label is price movement percentage
- Symmetric distribution around zero
- High temporal autocorrelation
- Extreme values are rare

**⚠️ Prediction Challenges**:
- Low correlation with raw market data
- Requires sophisticated feature engineering
- Time series modeling complexity
- Nonlinear relationship patterns

**🎯 Modeling Implications**:
- Use time series cross-validation
- Implement lag feature engineering
- Focus on feature interactions and ratios
- Consider ensemble methods for robustness

#### Data Verification Process

**Verification Script**: `verify_label.py` (temporary script, deleted after use)

**Verification Steps**:
1. **Basic Statistics**: Calculated min, max, mean, std of label values
2. **Distribution Analysis**: Analyzed positive/negative/zero value proportions
3. **Time Series Analysis**: Examined consecutive time points and autocorrelation
4. **Market Data Correlation**: Calculated Pearson correlation with raw market features
5. **Distribution Histogram**: Analyzed value distribution across ranges
6. **Change Rate Analysis**: Examined label-to-label changes over time

**Key Findings from Verification**:
- **Temporal Continuity**: High autocorrelation (0.981 at lag 1) confirms time series nature
- **Symmetric Distribution**: Nearly equal positive/negative proportions (51.17%/48.83%)
- **Concentration Pattern**: 96% of values within ±2.68% range
- **Market Relationship**: Very weak linear correlation with raw market data
- **Change Patterns**: Smooth transitions between consecutive time points

**Verification Code Example**:
```python
# Label basic statistics
print(f"Range: {train['label'].min():.2f} to {train['label'].max():.2f}")
print(f"Mean: {train['label'].mean():.6f}")
print(f"Distribution: {(train['label'] < 0).mean():.2%} negative, {(train['label'] > 0).mean():.2%} positive")

# Autocorrelation analysis
for lag in range(1, 11):
    corr = train['label'].autocorr(lag=lag)
    print(f"Lag {lag}: {corr:.6f}")

# Market data correlation
for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
    corr = train['label'].corr(train[col])
    print(f"{col}: {corr:.6f}")
```

---

## Model Architecture

### 1. Data Processing
- Loading and cleaning data
- Feature engineering
- Technical indicator calculation

### 2. Feature Engineering

#### Current Implementation
The project uses **correlation-based feature selection** from the original 896 features:

- **Selection Method**: Pearson correlation with target variable
- **Feature Types**: Market data (bid_qty, ask_qty, volume), technical indicators (X1-X15+)
- **Lag Features**: Generated for top features with periods [1, 2, 3, 5]

#### Feature Count by Mode
- **Ultra-Quick**: 5 original + 8 lag = 13 total features
- **Quick**: 10 original + 12 lag = 22 total features  
- **Full**: 100 original + 80 lag = 180 total features

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

### 7. Cross-Validation Score (CV Score) Theory

**CV Score** stands for **Cross-Validation Score**, which is the average performance metric across all folds in k-fold cross-validation.

#### CV Score Formula

For k-fold cross-validation, the CV Score is calculated as:

$$\text{CV Score} = \frac{1}{k} \sum_{i=1}^{k} r_i$$

where:
- $k$ = number of folds (in our project, $k = 2$)
- $r_i$ = Pearson correlation coefficient for fold $i$

#### CV Score Standard Deviation

The standard deviation of CV Score measures the stability of model performance:

$$\text{CV Std} = \sqrt{\frac{1}{k-1} \sum_{i=1}^{k} (r_i - \text{CV Score})^2}$$

#### Time Series Cross-Validation Implementation

In our project, we use **2-fold time series cross-validation**:

```
Data: [t₁, t₂, t₃, ..., tₙ]

Fold 1: Train [t₁, t₂, ..., t_{n/2}] → Validate [t_{n/2+1}, ..., tₙ]
Fold 2: Train [t₁, t₂, ..., t_{n-1}] → Validate [tₙ]
```

**CV Score Calculation**:
$$\text{CV Score} = \frac{r_1 + r_2}{2}$$

where:
- $r_1$ = Pearson correlation for Fold 1
- $r_2$ = Pearson correlation for Fold 2

#### Project CV Score Results

**Current Performance**:
- **CV Score**: 0.0474 ± 0.0034
- **Interpretation**: Average Pearson correlation of 4.74% across 2 folds
- **Stability**: Standard deviation of 0.34% indicates consistent performance

**Correlation Strength Classification**:
- $|r| < 0.1$: Very weak correlation
- $0.1 \leq |r| < 0.3$: Weak correlation
- $0.3 \leq |r| < 0.5$: Moderate correlation
- $0.5 \leq |r| < 0.7$: Strong correlation
- $|r| \geq 0.7$: Very strong correlation

**Our Result (0.0474)**: Very weak correlation, indicating significant room for improvement.

#### Why Use CV Score?

1. **Model Selection**: Compare different models and hyperparameters
2. **Overfitting Detection**: Ensure model generalizes well
3. **Performance Estimation**: Estimate how well model will perform on unseen data
4. **Development Guidance**: Guide feature engineering and model improvements

#### CV Score vs Competition Score

- **CV Score**: Internal development metric (our estimate)
- **Competition Score**: Official evaluation by competition organizers
- **Relationship**: CV Score ≈ Competition Score (in ideal cases)
- **Strategy**: Use CV Score for development, but don't over-optimize

## Evaluation

The primary evaluation metric is the Pearson correlation coefficient between predictions and actual values. This measures the linear correlation between the predicted and actual market movements.

### Model Performance

Current model performance metrics:
- LightGBM: 0.0474 ± 0.0034 (Pearson correlation)
- Gaussian Process: [Not yet implemented]
- Ensemble: [Not yet implemented]

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 