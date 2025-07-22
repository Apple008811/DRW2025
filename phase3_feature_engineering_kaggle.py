#!/usr/bin/env python3
"""
Phase 3: Feature Engineering Implementation (Kaggle Version)
============================================================

Implement feature engineering based on Phase 2 analysis results.

Author: Yixuan
Date: 2025-07-20
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

class FeatureEngineerKaggle:
    def __init__(self):
        """Initialize feature engineering for Kaggle environment."""
        self.train = None
        self.test = None
        self.train_file = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        self.test_file = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
        
        # Feature engineering results
        self.train_features = None
        self.test_features = None
        self.feature_names = []
        
    def load_data(self):
        """Load datasets."""
        print("📊 Loading datasets for feature engineering...")
        
        self.train = pd.read_parquet(self.train_file)
        self.test = pd.read_parquet(self.test_file)
        
        print(f"✅ Train data loaded: {self.train.shape}")
        print(f"✅ Test data loaded: {self.test.shape}")
        
    def create_market_imbalance_features(self):
        """Create market imbalance features based on Phase 2 analysis."""
        print("\n" + "="*80)
        print("⚖️ CREATING MARKET IMBALANCE FEATURES")
        print("="*80)
        
        # Market imbalance features
        for df in [self.train, self.test]:
            # Basic ratios
            df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
            df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
            df['volume_intensity'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-8)
            
            # Market pressure indicators
            df['buy_pressure'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
            df['sell_pressure'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
            
            # Spread indicators
            df['bid_ask_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)
            
            # Volume analysis
            df['volume_per_trade'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
            
        market_features = ['bid_ask_ratio', 'buy_sell_ratio', 'volume_intensity', 
                          'buy_pressure', 'sell_pressure', 'bid_ask_spread', 'volume_per_trade']
        
        print(f"✅ Created {len(market_features)} market imbalance features:")
        for feature in market_features:
            print(f"   - {feature}")
            
        self.feature_names.extend(market_features)
        
    def create_lag_features(self):
        """Create lag features based on autocorrelation analysis."""
        print("\n" + "="*80)
        print("🔄 CREATING LAG FEATURES")
        print("="*80)
        
        # Based on Phase 2 autocorrelation analysis, create lag features
        lag_periods = [1, 2, 3, 5, 10, 20]  # Based on autocorrelation analysis
        
        for df in [self.train, self.test]:
            # Lag features for label (if available)
            if 'label' in df.columns:
                for lag in lag_periods:
                    df[f'label_lag_{lag}'] = df['label'].shift(lag)
                    
            # Lag features for market data
            market_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
            for col in market_cols:
                for lag in [1, 2, 3, 5]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    
            # Lag features for market imbalance features
            imbalance_cols = ['bid_ask_ratio', 'buy_sell_ratio', 'volume_intensity']
            for col in imbalance_cols:
                for lag in [1, 2, 3]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        # Count lag features
        lag_features = []
        for lag in lag_periods:
            lag_features.append(f'label_lag_{lag}')
        
        for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
            for lag in [1, 2, 3, 5]:
                lag_features.append(f'{col}_lag_{lag}')
                
        for col in ['bid_ask_ratio', 'buy_sell_ratio', 'volume_intensity']:
            for lag in [1, 2, 3]:
                lag_features.append(f'{col}_lag_{lag}')
        
        print(f"✅ Created {len(lag_features)} lag features")
        print(f"   - Label lags: {lag_periods}")
        print(f"   - Market data lags: [1, 2, 3, 5]")
        print(f"   - Imbalance feature lags: [1, 2, 3]")
        
        self.feature_names.extend(lag_features)
        
    def create_rolling_features(self):
        """Create rolling window features for high-correlation indicators."""
        print("\n" + "="*80)
        print("📊 CREATING ROLLING WINDOW FEATURES")
        print("="*80)
        
        # Rolling windows based on autocorrelation analysis
        windows = [5, 10, 20, 50]
        
        for df in [self.train, self.test]:
            # Rolling features for market data
            market_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
            for col in market_cols:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
                    
            # Rolling features for market imbalance
            imbalance_cols = ['bid_ask_ratio', 'buy_sell_ratio', 'volume_intensity']
            for col in imbalance_cols:
                for window in windows[:2]:  # Use smaller windows for imbalance features
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
        
        # Count rolling features
        rolling_features = []
        for col in market_cols:
            for window in windows:
                for stat in ['mean', 'std', 'min', 'max']:
                    rolling_features.append(f'{col}_rolling_{stat}_{window}')
                    
        for col in imbalance_cols:
            for window in windows[:2]:
                for stat in ['mean', 'std']:
                    rolling_features.append(f'{col}_rolling_{stat}_{window}')
        
        print(f"✅ Created {len(rolling_features)} rolling window features")
        print(f"   - Windows: {windows}")
        print(f"   - Statistics: mean, std, min, max")
        
        self.feature_names.extend(rolling_features)
        
    def create_time_features(self):
        """Create time-based features."""
        print("\n" + "="*80)
        print("⏰ CREATING TIME-BASED FEATURES")
        print("="*80)
        
        for df in [self.train, self.test]:
            # Convert index to datetime if not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Time components
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time since start
            df['time_since_start'] = (df.index - df.index.min()).total_seconds()
            
            # Market session indicators
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['is_market_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        time_features = ['hour', 'day_of_week', 'day_of_month', 'month', 'quarter',
                        'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                        'month_sin', 'month_cos', 'time_since_start', 'is_weekend', 'is_market_hours']
        
        print(f"✅ Created {len(time_features)} time-based features:")
        for feature in time_features:
            print(f"   - {feature}")
            
        self.feature_names.extend(time_features)
        
    def create_interaction_features(self):
        """Create interaction features between correlated variables."""
        print("\n" + "="*80)
        print("🔗 CREATING INTERACTION FEATURES")
        print("="*80)
        
        for df in [self.train, self.test]:
            # Market interaction features
            df['volume_bid_ratio'] = df['volume'] / (df['bid_qty'] + 1e-8)
            df['volume_ask_ratio'] = df['volume'] / (df['ask_qty'] + 1e-8)
            df['buy_sell_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
            
            # Pressure interaction features
            df['buy_pressure_volume'] = df['buy_pressure'] * df['volume']
            df['sell_pressure_volume'] = df['sell_pressure'] * df['volume']
            
            # Ratio interactions
            df['bid_ask_volume_ratio'] = df['bid_ask_ratio'] * df['volume_intensity']
            
            # Time interaction features
            df['hour_volume_intensity'] = df['hour'] * df['volume_intensity']
            df['weekend_volume_ratio'] = df['is_weekend'] * df['volume_per_trade']
        
        interaction_features = ['volume_bid_ratio', 'volume_ask_ratio', 'buy_sell_imbalance',
                              'buy_pressure_volume', 'sell_pressure_volume', 'bid_ask_volume_ratio',
                              'hour_volume_intensity', 'weekend_volume_ratio']
        
        print(f"✅ Created {len(interaction_features)} interaction features:")
        for feature in interaction_features:
            print(f"   - {feature}")
            
        self.feature_names.extend(interaction_features)
        
    def select_technical_indicators(self):
        """Select high-correlation technical indicators based on Phase 2 analysis."""
        print("\n" + "="*80)
        print("📈 SELECTING HIGH-CORRELATION TECHNICAL INDICATORS")
        print("="*80)
        
        # Get technical indicator columns
        tech_cols = [col for col in self.train.columns if col.startswith('X')]
        
        # Calculate correlations with label
        correlations = []
        for col in tech_cols:
            corr = abs(self.train['label'].corr(self.train[col]))
            correlations.append((col, corr))
        
        # Sort by correlation and select top features
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 100 technical indicators (or adjust based on memory constraints)
        top_tech_features = [col for col, _ in correlations[:100]]
        
        print(f"✅ Selected {len(top_tech_features)} high-correlation technical indicators")
        print(f"   - Total available: {len(tech_cols)}")
        print(f"   - Selection criteria: |correlation| > {correlations[99][1]:.6f}")
        
        # Add selected technical indicators to feature names
        self.feature_names.extend(top_tech_features)
        
        return top_tech_features
        
    def prepare_final_features(self):
        """Prepare final feature dataset."""
        print("\n" + "="*80)
        print("🎯 PREPARING FINAL FEATURE DATASET")
        print("="*80)
        
        # Select features for training
        all_features = self.feature_names + ['label']  # Include label for training
        
        # Prepare train features
        self.train_features = self.train[all_features].copy()
        
        # Prepare test features (without label)
        test_features = [f for f in self.feature_names if f in self.test.columns]
        self.test_features = self.test[test_features].copy()
        
        # Handle missing values
        self.train_features = self.train_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        self.test_features = self.test_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"✅ Final feature dataset prepared:")
        print(f"   - Train features: {self.train_features.shape}")
        print(f"   - Test features: {self.test_features.shape}")
        print(f"   - Total engineered features: {len(self.feature_names)}")
        
        # Feature summary
        feature_summary = {
            "market_imbalance": len([f for f in self.feature_names if any(x in f for x in ['ratio', 'pressure', 'intensity'])]),
            "lag_features": len([f for f in self.feature_names if 'lag' in f]),
            "rolling_features": len([f for f in self.feature_names if 'rolling' in f]),
            "time_features": len([f for f in self.feature_names if any(x in f for x in ['hour', 'day', 'month', 'weekend', 'time'])]),
            "interaction_features": len([f for f in self.feature_names if any(x in f for x in ['ratio', 'pressure', 'imbalance'])]),
            "technical_indicators": len([f for f in self.feature_names if f.startswith('X')])
        }
        
        print(f"\n📊 Feature Engineering Summary:")
        for category, count in feature_summary.items():
            print(f"   - {category}: {count} features")
            
    def save_features(self):
        """Save engineered features."""
        print("\n" + "="*80)
        print("💾 SAVING ENGINEERED FEATURES")
        print("="*80)
        
        # Save to parquet files
        self.train_features.to_parquet('/kaggle/working/train_features.parquet')
        self.test_features.to_parquet('/kaggle/working/test_features.parquet')
        
        # Save feature names
        import json
        feature_info = {
            "feature_names": self.feature_names,
            "train_shape": self.train_features.shape,
            "test_shape": self.test_features.shape,
            "total_features": len(self.feature_names)
        }
        
        with open('/kaggle/working/feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print("✅ Features saved:")
        print("   - /kaggle/working/train_features.parquet")
        print("   - /kaggle/working/test_features.parquet")
        print("   - /kaggle/working/feature_info.json")
        
    def run_feature_engineering(self):
        """Run complete feature engineering pipeline."""
        print("🚀 Starting Phase 3: Feature Engineering Implementation")
        print("="*80)
        print("📍 Environment: Kaggle (Full Dataset)")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Create features based on Phase 2 analysis
        self.create_market_imbalance_features()
        self.create_lag_features()
        self.create_rolling_features()
        self.create_time_features()
        self.create_interaction_features()
        self.select_technical_indicators()
        
        # Prepare final dataset
        self.prepare_final_features()
        
        # Save features
        self.save_features()
        
        print("\n" + "="*80)
        print("✅ Phase 3 Feature Engineering Complete!")
        print("📝 Next Steps:")
        print("   1. Build enhanced models with new features")
        print("   2. Cross-validate and optimize")
        print("   3. Generate predictions")
        print("="*80)

def main():
    """Main function to run feature engineering."""
    engineer = FeatureEngineerKaggle()
    engineer.run_feature_engineering()

if __name__ == "__main__":
    main() 