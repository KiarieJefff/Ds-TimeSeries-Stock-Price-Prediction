#!/usr/bin/env python3
"""
Run EDA analysis and save plots to Images folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

# Create Images directory if it doesn't exist
os.makedirs('Images', exist_ok=True)

# Set plotting style
plt.rcParams["figure.figsize"] = (12, 5)
sns.set_style("whitegrid")

def main():
    print("🔍 Running EDA Analysis...")
    
    # Load Data
    df = pd.read_csv('data/raw/NFLX.csv', parse_dates=['Date'])
    df = df.sort_values("Date").reset_index(drop=True)
    df.set_index("Date", inplace=True)
    
    print(f"Dataset shape: {df.shape}")
    
    # Create features
    df["log_return"] = np.log(df["Close"]).diff()
    df.dropna(inplace=True)
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["oc_return"] = (df["Close"] - df["Open"]) / df["Open"]
    df['log_vol'] = np.log(df['Volume'])
    
    # 1. Price Evolution Plot
    print("📊 Generating price evolution plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    df['Close'].plot(ax=ax1, title='Netflix Stock Price - Linear Scale')
    ax1.set_ylabel('Price ($)')
    
    np.log(df['Close']).plot(ax=ax2, title='Netflix Stock Price - Log Scale', color='orange')
    ax2.set_ylabel('Log Price')
    ax2.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('Images/price_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Volatility Patterns
    print("📈 Generating volatility patterns plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    df['log_return'].plot(ax=ax1, title='Daily Log Returns', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Log Return')
    
    rolling_vol_20 = df["log_return"].rolling(20).std()
    rolling_vol_60 = df["log_return"].rolling(60).std()
    rolling_vol_20.plot(ax=ax2, label="20-day Volatility", alpha=0.8)
    rolling_vol_60.plot(ax=ax2, label="60-day Volatility", alpha=0.8)
    ax2.set_title('Rolling Volatility (Volatility Clustering)')
    ax2.set_ylabel('Standard Deviation')
    ax2.legend()
    ax2.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig('Images/volatility_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Correlation Analysis
    print("🔗 Generating correlation analysis...")
    # OHLC correlations
    ohlc_corr = df[['Open', 'High', 'Low', 'Close']].corr()
    features_corr = df[['log_return','hl_range','oc_return','log_vol']].corr()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(ohlc_corr, annot=True, cmap='coolwarm', center=0, ax=ax1)
    ax1.set_title('OHLC Price Correlations')
    
    sns.heatmap(features_corr, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title('Engineered Features Correlations')
    
    plt.tight_layout()
    plt.savefig('Images/correlation_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Feature Relationships
    print("💝 Generating feature relationships plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(x=df['log_vol'], y=df['hl_range'], alpha=0.6, ax=ax1)
    ax1.set_title('Volume vs Intraday Range')
    ax1.set_xlabel('Log Volume')
    ax1.set_ylabel('High-Low Range')
    
    sns.scatterplot(x=df['oc_return'], y=df['log_return'], alpha=0.6, ax=ax2)
    ax2.set_title('Daily Return vs Open-to-Close Return')
    ax2.set_xlabel('Open-to-Close Return')
    ax2.set_ylabel('Log Return')
    
    plt.tight_layout()
    plt.savefig('Images/feature_relationships.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Time Series Properties
    print("⏰ Generating time series properties plot...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    df['log_return'].hist(bins=50, ax=ax1, alpha=0.7)
    ax1.set_title('Log Returns Distribution')
    ax1.set_xlabel('Log Return')
    ax1.axvline(df['log_return'].mean(), color='red', linestyle='--', label='Mean')
    ax1.legend()
    
    stats.probplot(df['log_return'], dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot vs Normal Distribution')
    
    plot_acf(df['log_return'], lags=40, ax=ax3, title='ACF of Log Returns')
    ax3.set_title('Autocorrelation of Returns')
    
    plot_acf(df['log_return']**2, lags=40, ax=ax4, title='ACF of Squared Returns')
    ax4.set_title('Autocorrelation of Squared Returns (Volatility Clustering)')
    
    plt.tight_layout()
    plt.savefig('Images/time_series_properties.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Summary Statistics
    print("📋 Generating summary statistics...")
    
    # Create summary dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Price trend
    axes[0,0].plot(df.index, df['Close'])
    axes[0,0].set_title('Price Trend')
    axes[0,0].set_ylabel('Price ($)')
    
    # Returns distribution
    axes[0,1].hist(df['log_return'], bins=50, alpha=0.7)
    axes[0,1].set_title('Returns Distribution')
    axes[0,1].set_xlabel('Log Return')
    
    # Volume trend
    axes[0,2].plot(df.index, df['Volume'])
    axes[0,2].set_title('Volume Trend')
    axes[0,2].set_ylabel('Volume')
    
    # Volatility clustering
    axes[1,0].plot(df.index, df['log_return'].rolling(20).std())
    axes[1,0].set_title('20-Day Rolling Volatility')
    axes[1,0].set_ylabel('Std Dev')
    
    # Intraday range
    axes[1,1].plot(df.index, df['hl_range'])
    axes[1,1].set_title('Intraday Range (H-L)/Close')
    axes[1,1].set_ylabel('Range')
    
    # Volume vs Returns scatter
    axes[1,2].scatter(df['log_vol'], df['log_return'], alpha=0.6)
    axes[1,2].set_title('Volume vs Returns')
    axes[1,2].set_xlabel('Log Volume')
    axes[1,2].set_ylabel('Log Return')
    
    plt.tight_layout()
    plt.savefig('Images/eda_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ EDA Analysis Complete!")
    print("📁 Plots saved to Images folder:")
    
    # List generated files
    import glob
    image_files = glob.glob('Images/*.png')
    for img_file in sorted(image_files):
        print(f"  • {img_file}")

if __name__ == "__main__":
    main()
