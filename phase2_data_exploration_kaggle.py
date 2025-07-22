#!/usr/bin/env python3
"""
Phase 2: Data Exploration & Feature Engineering (Kaggle Version)
================================================================

Complete data exploration for Kaggle environment with full dataset.

Author: AI Assistant
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataExplorerKaggle:
    def __init__(self):
        """Initialize for Kaggle environment."""
        self.train = None
        self.test = None
        self.train_file = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        self.test_file = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
        
    def load_data(self):
        """Load complete datasets."""
        print("📊 Loading complete datasets...")
        
        self.train = pd.read_parquet(self.train_file)
        self.test = pd.read_parquet(self.test_file)
        
        print(f"✅ Train data loaded: {self.train.shape}")
        print(f"✅ Test data loaded: {self.test.shape}")
        
        # Display memory usage
        train_memory = self.train.memory_usage(deep=True).sum() / 1024**3
        test_memory = self.test.memory_usage(deep=True).sum() / 1024**3
        print(f"💾 Memory usage - Train: {train_memory:.2f}GB, Test: {test_memory:.2f}GB")
        
    def basic_info(self):
        """Display comprehensive basic information."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE DATA INFORMATION")
        print("="*80)
        
        print(f"\n📈 Dataset Shapes:")
        print(f"   Train: {self.train.shape}")
        print(f"   Test: {self.test.shape}")
        
        print(f"\n⏰ Time Range:")
        print(f"   Train: {self.train.index.min()} to {self.train.index.max()}")
        print(f"   Test: {self.test.index.min()} to {self.test.index.max()}")
        
        print(f"\n🏷️ Target Variable (label) Analysis:")
        label_stats = self.train['label'].describe()
        print(f"   Count: {label_stats['count']:,.0f}")
        print(f"   Mean: {label_stats['mean']:.8f}")
        print(f"   Std: {label_stats['std']:.8f}")
        print(f"   Min: {label_stats['min']:.8f}")
        print(f"   Max: {label_stats['max']:.8f}")
        print(f"   25%: {label_stats['25%']:.8f}")
        print(f"   50%: {label_stats['50%']:.8f}")
        print(f"   75%: {label_stats['75%']:.8f}")
        
        # Check for missing values
        print(f"\n❓ Missing Values:")
        train_missing = self.train.isnull().sum().sum()
        test_missing = self.test.isnull().sum().sum()
        print(f"   Train missing values: {train_missing}")
        print(f"   Test missing values: {test_missing}")
        
        # Column types
        print(f"\n📊 Column Types:")
        print(self.train.dtypes.value_counts())
        
    def market_data_analysis(self):
        """Comprehensive market data analysis."""
        print("\n" + "="*80)
        print("💰 COMPREHENSIVE MARKET DATA ANALYSIS")
        print("="*80)
        
        market_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        
        print(f"\n📊 Market Data Statistics:")
        market_stats = self.train[market_cols].describe()
        print(market_stats)
        
        print(f"\n🔗 Market Data Correlations with Label:")
        correlations = []
        for col in market_cols:
            corr = self.train['label'].corr(self.train[col])
            correlations.append((col, corr))
            print(f"   {col:12s}: {corr:8.6f}")
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n🏆 Top Market Features by |Correlation|:")
        for i, (col, corr) in enumerate(correlations):
            print(f"   {i+1}. {col:12s}: {corr:8.6f}")
            
        # Market imbalance analysis
        print(f"\n⚖️ Market Imbalance Analysis:")
        self.train['bid_ask_ratio'] = self.train['bid_qty'] / (self.train['ask_qty'] + 1e-8)
        self.train['buy_sell_ratio'] = self.train['buy_qty'] / (self.train['sell_qty'] + 1e-8)
        self.train['volume_intensity'] = self.train['volume'] / (self.train['bid_qty'] + self.train['ask_qty'] + 1e-8)
        
        imbalance_cols = ['bid_ask_ratio', 'buy_sell_ratio', 'volume_intensity']
        for col in imbalance_cols:
            corr = self.train['label'].corr(self.train[col])
            print(f"   {col:18s}: {corr:8.6f}")
            
    def technical_indicators_analysis(self):
        """Comprehensive technical indicators analysis."""
        print("\n" + "="*80)
        print("📈 COMPREHENSIVE TECHNICAL INDICATORS ANALYSIS")
        print("="*80)
        
        # Get technical indicator columns
        tech_cols = [col for col in self.train.columns if col.startswith('X')]
        
        print(f"\n📊 Technical Indicators Overview:")
        print(f"   Total count: {len(tech_cols)}")
        print(f"   Range: {tech_cols[0]} to {tech_cols[-1]}")
        
        # Basic statistics for all technical indicators
        print(f"\n📊 Technical Indicators Statistics:")
        tech_stats = self.train[tech_cols].describe()
        print(tech_stats)
        
        # Correlation analysis with label
        print(f"\n🔗 Computing correlations with label...")
        correlations = []
        for col in tech_cols:
            corr = self.train['label'].corr(self.train[col])
            correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🏆 Top 20 Technical Indicators by |Correlation|:")
        for i, (col, corr) in enumerate(correlations[:20]):
            print(f"   {i+1:2d}. {col:8s}: {corr:8.6f}")
            
        print(f"\n📉 Bottom 10 Technical Indicators by |Correlation|:")
        for i, (col, corr) in enumerate(correlations[-10:]):
            print(f"   {len(correlations)-9+i:2d}. {col:8s}: {corr:8.6f}")
            
        # Correlation distribution
        corr_values = [abs(corr) for _, corr in correlations]
        print(f"\n📊 Correlation Distribution:")
        print(f"   Mean |correlation|: {np.mean(corr_values):.6f}")
        print(f"   Std |correlation|: {np.std(corr_values):.6f}")
        print(f"   Max |correlation|: {np.max(corr_values):.6f}")
        print(f"   Min |correlation|: {np.min(corr_values):.6f}")
        
        # Count high correlation features
        high_corr_count = sum(1 for corr in corr_values if corr > 0.1)
        print(f"   Features with |correlation| > 0.1: {high_corr_count}/{len(corr_values)} ({high_corr_count/len(corr_values)*100:.1f}%)")
        
    def time_series_analysis(self):
        """Time series specific analysis."""
        print("\n" + "="*80)
        print("⏰ TIME SERIES ANALYSIS")
        print("="*80)
        
        # Convert index to datetime if not already
        if not isinstance(self.train.index, pd.DatetimeIndex):
            self.train.index = pd.to_datetime(self.train.index)
            
        print(f"\n📅 Time Series Properties:")
        print(f"   Start time: {self.train.index.min()}")
        print(f"   End time: {self.train.index.max()}")
        print(f"   Duration: {self.train.index.max() - self.train.index.min()}")
        print(f"   Total observations: {len(self.train)}")
        
        # Time intervals
        time_diffs = self.train.index.to_series().diff().dropna()
        print(f"\n⏱️ Time Intervals:")
        print(f"   Mean interval: {time_diffs.mean()}")
        print(f"   Std interval: {time_diffs.std()}")
        print(f"   Min interval: {time_diffs.min()}")
        print(f"   Max interval: {time_diffs.max()}")
        
        # Check for missing time points
        expected_times = pd.date_range(start=self.train.index.min(), 
                                     end=self.train.index.max(), 
                                     freq=time_diffs.mode().iloc[0])
        missing_times = expected_times.difference(self.train.index)
        print(f"\n❓ Missing Time Points: {len(missing_times)}")
        
        # Label autocorrelation
        print(f"\n🔄 Label Autocorrelation Analysis:")
        for lag in [1, 2, 3, 5, 10, 20, 50, 100]:
            autocorr = self.train['label'].autocorr(lag=lag)
            print(f"   Lag {lag:3d}: {autocorr:8.6f}")
            
    def feature_engineering_plan(self):
        """Detailed feature engineering plan based on analysis."""
        print("\n" + "="*80)
        print("🔧 DETAILED FEATURE ENGINEERING PLAN")
        print("="*80)
        
        print(f"\n📋 Based on Analysis, Here's the Feature Engineering Strategy:")
        
        print(f"\n1. 🎯 High-Priority Features (Based on Correlation Analysis):")
        print(f"   - Focus on technical indicators with |correlation| > 0.1")
        print(f"   - Market imbalance features (bid_ask_ratio, buy_sell_ratio)")
        print(f"   - Volume intensity features")
        
        print(f"\n2. 📈 Technical Indicators Enhancement:")
        print(f"   - Moving averages (SMA, EMA) for high-correlation features")
        print(f"   - Momentum indicators (RSI, MACD) for trend features")
        print(f"   - Volatility indicators (ATR, Bollinger Bands)")
        print(f"   - Cross-over signals between indicators")
        
        print(f"\n3. ⏰ Time-based Features:")
        print(f"   - Cyclical encoding of time components")
        print(f"   - Time since market open/close")
        print(f"   - Day of week, month, quarter effects")
        print(f"   - Holiday and weekend indicators")
        
        print(f"\n4. 🔄 Lag Features (Based on Autocorrelation):")
        print(f"   - Previous label values [1, 2, 3, 5, 10, 20]")
        print(f"   - Rolling statistics of high-correlation features")
        print(f"   - Exponential weighted features")
        
        print(f"\n5. 🔗 Interaction Features:")
        print(f"   - Ratio features between correlated indicators")
        print(f"   - Polynomial features for high-correlation variables")
        print(f"   - Market regime indicators")
        
        print(f"\n6. 📊 Statistical Features:")
        print(f"   - Rolling mean, std, min, max, quantiles")
        print(f"   - Z-score normalization")
        print(f"   - Percentile ranks")
        
        print(f"\n7. 🎯 Implementation Priority:")
        print(f"   - Phase 1: High-correlation features + basic lags")
        print(f"   - Phase 2: Time-based + interaction features")
        print(f"   - Phase 3: Advanced technical indicators")
        print(f"   - Phase 4: Feature selection and optimization")
        
    def create_visualizations(self):
        """Create key visualizations for analysis."""
        print("\n" + "="*80)
        print("📊 CREATING KEY VISUALIZATIONS")
        print("="*80)
        
        # Set up the plotting
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DRW Crypto Market Prediction - Data Analysis', fontsize=16)
        
        # 1. Label distribution
        axes[0, 0].hist(self.train['label'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Label Distribution')
        axes[0, 0].set_xlabel('Label Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Market data correlations
        market_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        market_corrs = [self.train['label'].corr(self.train[col]) for col in market_cols]
        axes[0, 1].bar(market_cols, market_corrs, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Market Data Correlations with Label')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Technical indicators correlation distribution
        tech_cols = [col for col in self.train.columns if col.startswith('X')]
        tech_corrs = [abs(self.train['label'].corr(self.train[col])) for col in tech_cols]
        axes[1, 0].hist(tech_corrs, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Technical Indicators |Correlation| Distribution')
        axes[1, 0].set_xlabel('|Correlation|')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Label autocorrelation
        lags = range(1, 21)
        autocorrs = [self.train['label'].autocorr(lag=lag) for lag in lags]
        axes[1, 1].plot(lags, autocorrs, marker='o', color='purple', alpha=0.7)
        axes[1, 1].set_title('Label Autocorrelation')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_exploration(self):
        """Run complete data exploration."""
        print("🚀 Starting Phase 2: Complete Data Exploration & Feature Engineering")
        print("="*80)
        print("📍 Environment: Kaggle (Full Dataset)")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run comprehensive analyses
        self.basic_info()
        self.market_data_analysis()
        self.technical_indicators_analysis()
        self.time_series_analysis()
        self.feature_engineering_plan()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*80)
        print("✅ Phase 2 Complete Data Exploration Finished!")
        print("📝 Next Steps:")
        print("   1. Implement feature engineering based on findings")
        print("   2. Create lag features and time-based features")
        print("   3. Build enhanced models with new features")
        print("   4. Cross-validate and optimize")
        print("="*80)

def main():
    """Main function to run the complete data exploration."""
    explorer = DataExplorerKaggle()
    explorer.run_complete_exploration()

if __name__ == "__main__":
    main() 