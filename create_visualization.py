#!/usr/bin/env python3
"""
Create visualization charts for prediction results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set font for better display
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Load data
df = pd.read_csv('ultra_quick_submission.csv')

# Create Chart 1: Basic Distribution Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Ultra Quick Submission Prediction Results Analysis', fontsize=16, fontweight='bold')

# 1. Prediction Value Distribution Histogram
axes[0,0].hist(df['pred'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].axvline(df['pred'].mean(), color='red', linestyle='--', label=f'Mean: {df["pred"].mean():.4f}')
axes[0,0].axvline(df['pred'].median(), color='green', linestyle='--', label=f'Median: {df["pred"].median():.4f}')
axes[0,0].set_title('Prediction Value Distribution Histogram')
axes[0,0].set_xlabel('Prediction Value')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Prediction Value Box Plot
axes[0,1].boxplot(df['pred'], patch_artist=True, boxprops=dict(facecolor='lightblue'))
axes[0,1].set_title('Prediction Value Box Plot')
axes[0,1].set_ylabel('Prediction Value')
axes[0,1].grid(True, alpha=0.3)

# 3. Prediction Value Density Plot
axes[1,0].hist(df['pred'], bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
axes[1,0].axvline(df['pred'].mean(), color='red', linestyle='--', label=f'Mean: {df["pred"].mean():.4f}')
axes[1,0].axvline(df['pred'].median(), color='green', linestyle='--', label=f'Median: {df["pred"].median():.4f}')
axes[1,0].set_title('Prediction Value Density Distribution')
axes[1,0].set_xlabel('Prediction Value')
axes[1,0].set_ylabel('Density')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 4. Positive vs Negative Values Pie Chart
positive = (df['pred'] > 0).sum()
negative = (df['pred'] < 0).sum()
labels = ['Positive', 'Negative']
sizes = [positive, negative]
colors = ['lightcoral', 'lightblue']
axes[1,1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
axes[1,1].set_title('Positive vs Negative Values Distribution')

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
print('✅ Prediction results analysis chart saved as prediction_analysis.png')

# Create Chart 2: Detailed Analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Prediction Value Detailed Analysis', fontsize=16, fontweight='bold')

# 1. Prediction Strength Distribution
axes[0,0].hist(df['pred'].abs(), bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[0,0].axvline(df['pred'].abs().mean(), color='red', linestyle='--', label=f'Average Strength: {df["pred"].abs().mean():.4f}')
axes[0,0].set_title('Prediction Strength Distribution (Absolute Values)')
axes[0,0].set_xlabel('Prediction Strength |pred|')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Prediction Value Range Analysis
weak = (df['pred'].abs() < 0.05).sum()
medium = ((df['pred'].abs() >= 0.05) & (df['pred'].abs() < 0.1)).sum()
strong = (df['pred'].abs() >= 0.1).sum()
categories = ['Weak Prediction', 'Medium Prediction', 'Strong Prediction']
counts = [weak, medium, strong]
colors = ['lightblue', 'orange', 'red']
axes[0,1].bar(categories, counts, color=colors, alpha=0.7)
axes[0,1].set_title('Prediction Strength Classification')
axes[0,1].set_ylabel('Sample Count')
for i, v in enumerate(counts):
    axes[0,1].text(i, v + 1000, f'{v:,}', ha='center', va='bottom')
axes[0,1].grid(True, alpha=0.3)

# 3. Prediction Value Cumulative Distribution
axes[1,0].hist(df['pred'], bins=50, cumulative=True, density=True, alpha=0.7, color='purple', edgecolor='black')
axes[1,0].axvline(0, color='red', linestyle='--', label='Zero Line')
axes[1,0].set_title('Prediction Value Cumulative Distribution')
axes[1,0].set_xlabel('Prediction Value')
axes[1,0].set_ylabel('Cumulative Probability')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 4. Prediction Value Q-Q Plot
stats.probplot(df['pred'], dist='norm', plot=axes[1,1])
axes[1,1].set_title('Prediction Value Q-Q Plot (vs Normal Distribution)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_detailed_analysis.png', dpi=300, bbox_inches='tight')
print('✅ Prediction value detailed analysis chart saved as prediction_detailed_analysis.png')

# Create Chart 3: Comparison Analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Prediction Value Distribution Feature Comparison Analysis', fontsize=16, fontweight='bold')

# 1. Prediction Value Distribution vs Normal Distribution
axes[0].hist(df['pred'], bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black', label='Actual Distribution')
x = np.linspace(df['pred'].min(), df['pred'].max(), 100)
normal_dist = stats.norm.pdf(x, df['pred'].mean(), df['pred'].std())
axes[0].plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
axes[0].set_title('Prediction Value Distribution vs Normal Distribution')
axes[0].set_xlabel('Prediction Value')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Prediction Value Quantile Plot
axes[1].plot(range(1, len(df)+1), np.sort(df['pred']), 'b-', linewidth=1)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Zero Line')
axes[1].axhline(y=df['pred'].mean(), color='green', linestyle='--', alpha=0.7, label='Mean Line')
axes[1].set_title('Prediction Value Sorted Plot')
axes[1].set_xlabel('Sample Order')
axes[1].set_ylabel('Prediction Value')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Prediction Value Statistical Summary
axes[2].axis('off')
mean_val = df['pred'].mean()
median_val = df['pred'].median()
std_val = df['pred'].std()
skew_val = df['pred'].skew()
kurt_val = df['pred'].kurtosis()
positive_pct = (df['pred'] > 0).sum()/len(df)*100
negative_pct = (df['pred'] < 0).sum()/len(df)*100
weak_pct = (df['pred'].abs() < 0.05).sum()/len(df)*100
strong_pct = (df['pred'].abs() >= 0.1).sum()/len(df)*100

stats_text = f'''Prediction Value Statistical Summary

Total Samples: {len(df):,}
Mean: {mean_val:.4f}
Median: {median_val:.4f}
Standard Deviation: {std_val:.4f}
Skewness: {skew_val:.4f}
Kurtosis: {kurt_val:.4f}

Distribution Features:
• Positive Values: {positive_pct:.1f}%
• Negative Values: {negative_pct:.1f}%
• Weak Predictions: {weak_pct:.1f}%
• Strong Predictions: {strong_pct:.1f}%

Model Configuration:
• Features: 13 (5 main + 8 lag features)
• Estimators: 20
• Learning Rate: 0.2
• Tree Depth: 3'''

axes[2].text(0.1, 0.9, stats_text, transform=axes[2].transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('prediction_comparison_analysis.png', dpi=300, bbox_inches='tight')
print('✅ Prediction value comparison analysis chart saved as prediction_comparison_analysis.png')

print('\n🎉 All visualization charts have been generated successfully!')
print('📊 Generated chart files:')
print('   - prediction_analysis.png (Basic distribution analysis)')
print('   - prediction_detailed_analysis.png (Detailed analysis)')
print('   - prediction_comparison_analysis.png (Comparison analysis)') 