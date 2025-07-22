#!/usr/bin/env python3
"""
Label Relationship Analysis
==========================

Detailed analysis of label relationships with volume features and technical indicators.

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

class LabelRelationshipAnalyzer:
    def __init__(self):
        """Initialize analyzer."""
        self.train = None
        self.test = None
        self.train_file = '/kaggle/input/drw-crypto-market-prediction/train.parquet'
        self.test_file = '/kaggle/input/drw-crypto-market-prediction/test.parquet'
        
    def load_data(self):
        """Load datasets."""
        print("📊 Loading datasets for label relationship analysis...")
        
        self.train = pd.read_parquet(self.train_file)
        self.test = pd.read_parquet(self.test_file)
        
        print(f"✅ Train data loaded: {self.train.shape}")
        print(f"✅ Test data loaded: {self.test.shape}")
        
    def analyze_volume_label_relationships(self):
        """Analyze relationships between volume features and label."""
        print("\n" + "="*80)
        print("📊 LABEL vs VOLUME FEATURES ANALYSIS")
        print("="*80)
        
        # Volume-related features
        volume_features = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
        
        print(f"\n📈 Volume Features Statistics:")
        volume_stats = self.train[volume_features].describe()
        print(volume_stats)
        
        print(f"\n🔗 Volume Features vs Label Correlations:")
        correlations = []
        for feature in volume_features:
            corr = self.train['label'].corr(self.train[feature])
            correlations.append((feature, corr))
            print(f"   {feature:12s}: {corr:8.6f}")
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n🏆 Volume Features Ranked by |Correlation|:")
        for i, (feature, corr) in enumerate(correlations):
            print(f"   {i+1}. {feature:12s}: {corr:8.6f}")
            
        # Create volume-based features and analyze
        print(f"\n⚖️ Derived Volume Features vs Label:")
        
        # Calculate derived features
        self.train['volume_bid_ratio'] = self.train['volume'] / (self.train['bid_qty'] + 1e-8)
        self.train['volume_ask_ratio'] = self.train['volume'] / (self.train['ask_qty'] + 1e-8)
        self.train['buy_sell_ratio'] = self.train['buy_qty'] / (self.train['sell_qty'] + 1e-8)
        self.train['bid_ask_ratio'] = self.train['bid_qty'] / (self.train['ask_qty'] + 1e-8)
        self.train['volume_intensity'] = self.train['volume'] / (self.train['bid_qty'] + self.train['ask_qty'] + 1e-8)
        
        derived_features = ['volume_bid_ratio', 'volume_ask_ratio', 'buy_sell_ratio', 
                           'bid_ask_ratio', 'volume_intensity']
        
        derived_correlations = []
        for feature in derived_features:
            corr = self.train['label'].corr(self.train[feature])
            derived_correlations.append((feature, corr))
            print(f"   {feature:18s}: {corr:8.6f}")
            
        # Sort derived features
        derived_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"\n🏆 Derived Volume Features Ranked by |Correlation|:")
        for i, (feature, corr) in enumerate(derived_correlations):
            print(f"   {i+1}. {feature:18s}: {corr:8.6f}")
            
        return correlations, derived_correlations
        
    def analyze_technical_indicators_relationships(self):
        """Analyze relationships between technical indicators (X) and label."""
        print("\n" + "="*80)
        print("📈 LABEL vs TECHNICAL INDICATORS (X) ANALYSIS")
        print("="*80)
        
        # Get technical indicator columns
        tech_cols = [col for col in self.train.columns if col.startswith('X')]
        
        print(f"\n📊 Technical Indicators Overview:")
        print(f"   Total count: {len(tech_cols)}")
        print(f"   Range: {tech_cols[0]} to {tech_cols[-1]}")
        
        # Calculate correlations with label
        print(f"\n🔗 Computing correlations with label...")
        correlations = []
        for col in tech_cols:
            corr = self.train['label'].corr(self.train[col])
            correlations.append((col, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n🏆 Top 30 Technical Indicators by |Correlation|:")
        for i, (col, corr) in enumerate(correlations[:30]):
            print(f"   {i+1:2d}. {col:8s}: {corr:8.6f}")
            
        print(f"\n📉 Bottom 10 Technical Indicators by |Correlation|:")
        for i, (col, corr) in enumerate(correlations[-10:]):
            print(f"   {len(correlations)-9+i:2d}. {col:8s}: {corr:8.6f}")
            
        # Correlation distribution analysis
        corr_values = [abs(corr) for _, corr in correlations]
        print(f"\n📊 Correlation Distribution Statistics:")
        print(f"   Mean |correlation|: {np.mean(corr_values):.6f}")
        print(f"   Std |correlation|: {np.std(corr_values):.6f}")
        print(f"   Max |correlation|: {np.max(corr_values):.6f}")
        print(f"   Min |correlation|: {np.min(corr_values):.6f}")
        print(f"   Median |correlation|: {np.median(corr_values):.6f}")
        
        # Count features by correlation strength
        high_corr = sum(1 for corr in corr_values if corr > 0.1)
        medium_corr = sum(1 for corr in corr_values if 0.05 < corr <= 0.1)
        low_corr = sum(1 for corr in corr_values if corr <= 0.05)
        
        print(f"\n📊 Feature Count by Correlation Strength:")
        print(f"   High correlation (|corr| > 0.1): {high_corr} features ({high_corr/len(corr_values)*100:.1f}%)")
        print(f"   Medium correlation (0.05 < |corr| ≤ 0.1): {medium_corr} features ({medium_corr/len(corr_values)*100:.1f}%)")
        print(f"   Low correlation (|corr| ≤ 0.05): {low_corr} features ({low_corr/len(corr_values)*100:.1f}%)")
        
        return correlations
        
    def analyze_correlation_patterns(self):
        """Analyze patterns in correlations."""
        print("\n" + "="*80)
        print("🔍 CORRELATION PATTERN ANALYSIS")
        print("="*80)
        
        # Get technical indicators
        tech_cols = [col for col in self.train.columns if col.startswith('X')]
        
        # Calculate correlations
        correlations = []
        for col in tech_cols:
            corr = self.train['label'].corr(self.train[col])
            correlations.append((col, corr))
        
        # Extract X numbers and analyze patterns
        x_numbers = []
        corr_values = []
        
        for col, corr in correlations:
            try:
                x_num = int(col[1:])  # Extract number after 'X'
                x_numbers.append(x_num)
                corr_values.append(abs(corr))
            except:
                continue
        
        # Create DataFrame for analysis
        corr_df = pd.DataFrame({
            'x_number': x_numbers,
            'abs_correlation': corr_values
        })
        
        print(f"\n📊 X Number vs Correlation Analysis:")
        print(f"   X number range: {min(x_numbers)} to {max(x_numbers)}")
        
        # Analyze correlation by X number ranges
        ranges = [(1, 100), (101, 200), (201, 300), (301, 400), (401, 500), 
                 (501, 600), (601, 700), (701, 800), (801, 890)]
        
        print(f"\n📈 Average |Correlation| by X Number Ranges:")
        for start, end in ranges:
            mask = (corr_df['x_number'] >= start) & (corr_df['x_number'] <= end)
            if mask.sum() > 0:
                avg_corr = corr_df.loc[mask, 'abs_correlation'].mean()
                count = mask.sum()
                print(f"   X{start:3d}-X{end:3d}: {avg_corr:.6f} (avg) - {count} features")
        
        # Find high correlation clusters
        high_corr_features = corr_df[corr_df['abs_correlation'] > 0.1]
        if len(high_corr_features) > 0:
            print(f"\n🎯 High Correlation Features (|corr| > 0.1):")
            print(f"   Count: {len(high_corr_features)}")
            print(f"   X number range: {high_corr_features['x_number'].min()} to {high_corr_features['x_number'].max()}")
            print(f"   Average X number: {high_corr_features['x_number'].mean():.1f}")
            
            # Check if high correlation features are clustered
            x_numbers_high = high_corr_features['x_number'].values
            x_numbers_high.sort()
            
            # Find gaps in high correlation features
            gaps = []
            for i in range(1, len(x_numbers_high)):
                gap = x_numbers_high[i] - x_numbers_high[i-1]
                if gap > 10:  # Gap larger than 10
                    gaps.append((x_numbers_high[i-1], x_numbers_high[i], gap))
            
            if gaps:
                print(f"   Large gaps in high correlation features:")
                for start, end, gap in gaps[:5]:  # Show top 5 gaps
                    print(f"     Gap {gap} between X{start} and X{end}")
            else:
                print(f"   High correlation features are relatively continuous")
        
        return corr_df
        
    def create_visualizations(self):
        """Create visualizations for label relationships."""
        print("\n" + "="*80)
        print("📊 CREATING LABEL RELATIONSHIP VISUALIZATIONS")
        print("="*80)
        
        # Set up the plotting
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Label Relationship Analysis', fontsize=16)
        
        # 1. Volume features correlation
        volume_features = ['volume', 'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty']
        volume_corrs = [self.train['label'].corr(self.train[col]) for col in volume_features]
        
        axes[0, 0].bar(volume_features, volume_corrs, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Volume Features vs Label Correlation')
        axes[0, 0].set_ylabel('Correlation')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Derived volume features correlation
        derived_features = ['volume_bid_ratio', 'volume_ask_ratio', 'buy_sell_ratio', 
                           'bid_ask_ratio', 'volume_intensity']
        derived_corrs = [self.train['label'].corr(self.train[col]) for col in derived_features]
        
        axes[0, 1].bar(derived_features, derived_corrs, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Derived Volume Features vs Label Correlation')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Technical indicators correlation distribution
        tech_cols = [col for col in self.train.columns if col.startswith('X')]
        tech_corrs = [abs(self.train['label'].corr(self.train[col])) for col in tech_cols]
        
        axes[0, 2].hist(tech_corrs, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Technical Indicators |Correlation| Distribution')
        axes[0, 2].set_xlabel('|Correlation|')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Top 20 technical indicators correlation
        correlations = [(col, abs(self.train['label'].corr(self.train[col]))) for col in tech_cols]
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_20 = correlations[:20]
        
        x_labels = [item[0] for item in top_20]
        y_values = [item[1] for item in top_20]
        
        axes[1, 0].bar(range(len(x_labels)), y_values, color='purple', alpha=0.7)
        axes[1, 0].set_title('Top 20 Technical Indicators |Correlation|')
        axes[1, 0].set_ylabel('|Correlation|')
        axes[1, 0].set_xticks(range(len(x_labels)))
        axes[1, 0].set_xticklabels(x_labels, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. X number vs correlation scatter plot
        x_numbers = []
        corr_values = []
        for col, corr in correlations:
            try:
                x_num = int(col[1:])
                x_numbers.append(x_num)
                corr_values.append(corr)
            except:
                continue
        
        axes[1, 1].scatter(x_numbers, corr_values, alpha=0.5, s=10, color='orange')
        axes[1, 1].set_title('X Number vs |Correlation|')
        axes[1, 1].set_xlabel('X Number')
        axes[1, 1].set_ylabel('|Correlation|')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Correlation strength distribution
        high_corr = sum(1 for corr in tech_corrs if corr > 0.1)
        medium_corr = sum(1 for corr in tech_corrs if 0.05 < corr <= 0.1)
        low_corr = sum(1 for corr in tech_corrs if corr <= 0.05)
        
        categories = ['High (>0.1)', 'Medium (0.05-0.1)', 'Low (≤0.05)']
        counts = [high_corr, medium_corr, low_corr]
        colors = ['red', 'orange', 'green']
        
        axes[1, 2].pie(counts, labels=categories, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Technical Indicators by Correlation Strength')
        
        plt.tight_layout()
        plt.show()
        
    def run_complete_analysis(self):
        """Run complete label relationship analysis."""
        print("🚀 Starting Label Relationship Analysis")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run analyses
        volume_corrs, derived_corrs = self.analyze_volume_label_relationships()
        tech_corrs = self.analyze_technical_indicators_relationships()
        corr_df = self.analyze_correlation_patterns()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n" + "="*80)
        print("✅ Label Relationship Analysis Complete!")
        print("📝 Key Insights:")
        print("   1. Volume features correlation patterns")
        print("   2. Technical indicators importance ranking")
        print("   3. Correlation distribution and patterns")
        print("   4. Feature selection recommendations")
        print("="*80)
        
        return volume_corrs, derived_corrs, tech_corrs, corr_df

def main():
    """Main function to run label relationship analysis."""
    analyzer = LabelRelationshipAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main() 