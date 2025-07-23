#!/usr/bin/env python3
"""
Phase 4 Quick Test: Model Training & Comparison
===============================================

Quick model training and comparison for validation.

Author: Yixuan
Date: 2025-07-20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QuickModelTester:
    def __init__(self):
        """Initialize quick model testing."""
        self.train = None
        self.test = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        
        # Results storage
        self.results = {}
        
    def load_sample_data(self):
        """Load sample data for quick testing."""
        print("📊 Loading sample data for quick testing...")
        
        # Try to load from different possible locations
        possible_files = [
            'train_sample.csv',
            '/kaggle/input/drw-crypto-market-prediction/train.parquet',
            '/Users/yixuan/DRW 2/data/train.parquet'
        ]
        
        for file_path in possible_files:
            try:
                if file_path.endswith('.csv'):
                    self.train = pd.read_csv(file_path)
                else:
                    self.train = pd.read_parquet(file_path)
                print(f"✅ Data loaded from: {file_path}")
                break
            except:
                continue
        
        if self.train is None:
            print("❌ No data file found. Please ensure data is available.")
            return False
            
        print(f"📊 Data shape: {self.train.shape}")
        return True
        
    def prepare_features(self):
        """Prepare features for quick testing."""
        print("\n🔧 Preparing features...")
        
        # Get target variable
        if 'label' in self.train.columns:
            self.y_train = self.train['label']
        else:
            print("❌ No 'label' column found")
            return False
        
        # Select numeric features
        feature_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in feature_cols if col != 'label' and col != 'timestamp']
        
        # Use top 50 features by correlation for quick testing
        correlations = []
        for col in feature_cols:
            corr = abs(self.y_train.corr(self.train[col]))
            correlations.append((col, corr))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:50]]
        
        self.X_train = self.train[top_features].fillna(method='bfill').fillna(0)
        
        print(f"📊 Using top {len(top_features)} features")
        print(f"📊 Training samples: {self.X_train.shape[0]}")
        
        return True
        
    def quick_cv(self, model, X, y, n_splits=3):
        """Quick cross-validation."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            preds = model.predict(X_fold_val)
            score = pearsonr(y_fold_val, preds)[0]
            scores.append(score)
            
            print(f"  Fold {fold+1}: {score:.4f}")
        
        return np.mean(scores), np.std(scores)
        
    def test_models(self):
        """Test different models quickly."""
        print("\n" + "="*60)
        print("🤖 QUICK MODEL TESTING")
        print("="*60)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(
                n_estimators=50, max_depth=6, random_state=42, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.1, max_depth=6, 
                random_state=42, verbose=-1
            )
        }
        
        for name, model in models.items():
            print(f"\n🔍 Testing {name}...")
            avg_score, std_score = self.quick_cv(model, self.X_train, self.y_train)
            
            self.results[name] = {
                'avg_score': avg_score,
                'std_score': std_score
            }
            
            print(f"✅ {name}: {avg_score:.4f} ± {std_score:.4f}")
            
    def compare_results(self):
        """Compare model results."""
        print("\n" + "="*60)
        print("📊 MODEL COMPARISON")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'CV Score': f"{results['avg_score']:.4f} ± {results['std_score']:.4f}",
                'Avg Score': results['avg_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Avg Score', ascending=False)
        
        print("\n🏆 Model Performance Ranking:")
        print(comparison_df.to_string(index=False))
        
        # Find best model
        best_model = comparison_df.iloc[0]['Model']
        best_score = comparison_df.iloc[0]['Avg Score']
        
        print(f"\n🥇 Best Model: {best_model} (Score: {best_score:.4f})")
        
        return comparison_df, best_model
        
    def create_quick_plot(self):
        """Create quick comparison plot."""
        print("\n📊 Creating comparison plot...")
        
        model_names = list(self.results.keys())
        scores = [self.results[name]['avg_score'] for name in model_names]
        std_scores = [self.results[name]['std_score'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, scores, yerr=std_scores, capsize=5, alpha=0.7)
        
        # Color the best model
        best_idx = np.argmax(scores)
        bars[best_idx].set_color('gold')
        
        plt.title('Quick Model Comparison')
        plt.ylabel('Pearson Correlation')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('phase4_quick_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comparison plot saved: phase4_quick_comparison.png")
        
    def save_quick_results(self):
        """Save quick test results."""
        print("\n💾 Saving quick test results...")
        
        # Save results
        results_data = []
        for name, results in self.results.items():
            results_data.append({
                'Model': name,
                'CV Score': results['avg_score'],
                'CV Std': results['std_score']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv('phase4_quick_results.csv', index=False)
        
        print("✅ Quick test results saved: phase4_quick_results.csv")
        
    def run_quick_test(self):
        """Run complete quick test."""
        print("🚀 Phase 4 Quick Model Test")
        print("="*60)
        
        # Load and prepare data
        if not self.load_sample_data():
            return
            
        if not self.prepare_features():
            return
        
        # Test models
        self.test_models()
        
        # Compare results
        comparison_df, best_model = self.compare_results()
        
        # Create visualization
        self.create_quick_plot()
        
        # Save results
        self.save_quick_results()
        
        print("\n" + "="*60)
        print("✅ Phase 4 Quick Test Complete!")
        print(f"🥇 Best Model: {best_model}")
        print("📁 Output Files:")
        print("   - phase4_quick_results.csv")
        print("   - phase4_quick_comparison.png")
        print("="*60)

def main():
    """Main function."""
    tester = QuickModelTester()
    tester.run_quick_test()

if __name__ == "__main__":
    main() 