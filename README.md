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
| 1. Environment & Baseline | Environment setup, data loading, baseline model | - Install dependencies<br>- Load and validate data<br>- Run quick_test.py for baseline LightGBM model<br>- Generate quick_submission.csv and check Pearson correlation |
| 2. Data Exploration & Feature Engineering | Exploratory Data Analysis (EDA), feature construction, feature selection | - Analyze data structure and distributions<br>- Visualize key features<br>- Engineer new features (technical indicators, rolling stats, lag features, time-based features)<br>- Select features using correlation, importance, and statistical tests |
| 3. Model Training & Optimization | Comprehensive model testing: classical, ML, and hyperparameter tuning | - Train and compare the following models:<br> • ARIMA/SARIMA<br> • Prophet<br> • Linear Regression, Ridge, Lasso<br> • Random Forest<br> • XGBoost<br> • LightGBM<br> • Support Vector Regression (SVR)<br>- Use time series cross-validation<br>- Tune hyperparameters (e.g., LightGBM: num_leaves, learning_rate, feature_fraction)<br>- Compare model performance using Pearson correlation |
| 4. Advanced Modeling & Ensembling | Deep learning, Bayesian methods, model ensembling | - Implement and test:<br> • LSTM, GRU, Transformer neural networks<br> • Gaussian Process (Bayesian approach)<br>- Ensemble models (stacking, blending, weighted averaging)<br>- Apply post-processing (calibration, outlier handling, smoothing) |
| 5. Validation & Submission | Final validation, result analysis, submission preparation | - Perform cross-validation and stability checks<br>- Analyze prediction distributions and feature importance<br>- Prepare and verify submission files<br>- Backup results |

### Phase Details

#### Phase 1: Environment & Baseline
- **Theory:** Ensure reproducibility and a working baseline for comparison.
- **Steps:**
  - Install all required packages (see Setup section)
  - Load training, testing, and sample submission data
  - Run `python quick_test.py` to verify environment and generate a baseline LightGBM model
  - Check that Pearson correlation > 0.1 and quick_submission.csv is created

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
├── data/                    # Data directory
├── models/                  # Saved models
├── config.py               # Configuration settings
├── data_processing.py      # Data preprocessing
├── feature_exploration.py  # Feature analysis
├── model.py               # Model implementations
├── drw_pipeline.py        # Main pipeline
└── quick_test.py          # Quick testing script
```

## Usage

### Quick Start

For a quick test of the pipeline:
```bash
python quick_test.py
```

### Full Pipeline

To run the complete pipeline:
```bash
python main.py
```

### Feature Exploration

To explore and analyze features:
```bash
python feature_exploration.py
```

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

This project is licensed under the terms of the LICENSE file included in the repository. 