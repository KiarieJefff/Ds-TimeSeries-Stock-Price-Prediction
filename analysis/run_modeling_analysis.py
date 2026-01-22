#!/usr/bin/env python3
"""
Run modeling analysis and save plots to Images folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Create Images directory if it doesn't exist
import os
os.makedirs('Images', exist_ok=True)

# Set plotting style
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_style("whitegrid")

def create_features(df):
    """Create technical indicators and lagged features"""
    df_feat = df.copy()
    
    # Price-based features
    df_feat['hl_range'] = (df_feat['High'] - df_feat['Low']) / df_feat['Close']
    df_feat['oc_return'] = (df_feat['Close'] - df_feat['Open']) / df_feat['Open']
    df_feat['price_change'] = df_feat['Close'].pct_change()
    
    # Volume features
    df_feat['log_volume'] = np.log(df_feat['Volume'])
    df_feat['volume_change'] = df_feat['Volume'].pct_change()
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        df_feat[f'ma_{window}'] = df_feat['Close'].rolling(window).mean()
        df_feat[f'volume_ma_{window}'] = df_feat['Volume'].rolling(window).mean()
    
    # Volatility features
    df_feat['volatility_5'] = df_feat['log_return'].rolling(5).std()
    df_feat['volatility_20'] = df_feat['log_return'].rolling(20).std()
    
    # RSI
    delta = df_feat['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df_feat['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Momentum
    df_feat['momentum_5'] = df_feat['Close'] / df_feat['Close'].shift(5) - 1
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df_feat[f'lag_{lag}'] = df_feat['log_return'].shift(lag)
    
    # Time-based features
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    df_feat['quarter'] = df_feat.index.quarter
    
    return df_feat

def main():
    print("🤖 Running Modeling Analysis...")
    
    # Load and prepare data
    df = pd.read_csv('data/raw/NFLX.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)
    
    # Create target variable
    df['log_return'] = np.log(df['Close']).diff()
    df['target'] = df['log_return'].shift(-1)
    df.dropna(inplace=True)
    
    print(f"Dataset shape: {df.shape}")
    
    # Train-test split
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training set: {len(train_df)} observations")
    print(f"Test set: {len(test_df)} observations")
    
    # 1. Train-Test Split Visualization
    print("📊 Generating train-test split plot...")
    plt.figure(figsize=(15, 6))
    plt.plot(train_df.index, train_df['Close'], label='Training Data', color='blue', alpha=0.7)
    plt.plot(test_df.index, test_df['Close'], label='Test Data', color='red', alpha=0.7)
    plt.title('Netflix Stock Price - Train/Test Split')
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Images/train_test_split.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Target distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    train_df['target'].hist(bins=50, ax=ax1, alpha=0.7, label='Train')
    test_df['target'].hist(bins=50, ax=ax2, alpha=0.7, label='Test', color='orange')
    
    ax1.set_title('Training Set - Target Distribution')
    ax1.set_xlabel('Log Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    ax2.set_title('Test Set - Target Distribution')
    ax2.set_xlabel('Log Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('Images/target_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature engineering
    print("⚙️ Creating features...")
    train_features = create_features(train_df)
    test_features = create_features(test_df)
    
    # Prepare ML data
    feature_cols = [col for col in train_features.columns 
                   if col not in ['target', 'Close', 'High', 'Low', 'Open', 'Adj Close']]
    
    train_clean = train_features[feature_cols + ['target']].dropna()
    test_clean = test_features[feature_cols + ['target']].dropna()
    
    X_train = train_clean[feature_cols]
    y_train = train_clean['target']
    X_test = test_clean[feature_cols]
    y_test = test_clean['target']
    
    print(f"ML Models - Training: {X_train.shape}, Testing: {X_test.shape}")
    
    # Train Random Forest
    print("🌲 Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 2. Feature Importance Plot
    print("📈 Generating feature importance plot...")
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importance - Random Forest')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('Images/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Model Predictions Comparison
    print("🔮 Generating model predictions plot...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()
    
    # Random Forest predictions
    axes[0].plot(y_test.values[:100], label='Actual', alpha=0.7, linewidth=1)
    axes[0].plot(rf_pred[:100], label='RF Predicted', alpha=0.7, linewidth=1)
    axes[0].set_title('Random Forest - Predictions vs Actual (First 100 points)')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Log Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1].scatter(y_test.values, rf_pred, alpha=0.6, s=20)
    min_val = min(y_test.values.min(), rf_pred.min())
    max_val = max(y_test.values.max(), rf_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[1].set_title('Random Forest - Actual vs Predicted')
    axes[1].set_xlabel('Actual Log Return')
    axes[1].set_ylabel('Predicted Log Return')
    axes[1].grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_test.values - rf_pred
    axes[2].hist(residuals, bins=50, alpha=0.7, density=True)
    mu, sigma = 0, residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[2].plot(x, (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2), 
                'r-', alpha=0.8, label='Normal fit')
    axes[2].set_title('Random Forest - Residual Distribution')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Performance metrics
    mse = mean_squared_error(y_test, rf_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, rf_pred)
    r2 = r2_score(y_test, rf_pred)
    
    metrics_text = f'MSE: {mse:.6f}\\nRMSE: {rmse:.6f}\\nMAE: {mae:.6f}\\nR²: {r2:.4f}'
    axes[3].text(0.1, 0.5, metrics_text, fontsize=14, transform=axes[3].transAxes)
    axes[3].set_title('Model Performance Metrics')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('Images/model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Model Comparison Summary
    print("📊 Generating model comparison summary...")
    
    # Create a simple baseline model (predict mean)
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    # Compare models
    models_comparison = pd.DataFrame({
        'Model': ['Baseline (Mean)', 'Random Forest'],
        'RMSE': [np.sqrt(baseline_mse), rmse],
        'MAE': [mean_absolute_error(y_test, baseline_pred), mae],
        'R²': [baseline_r2, r2]
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # RMSE comparison
    axes[0].bar(models_comparison['Model'], models_comparison['RMSE'], color=['red', 'green'])
    axes[0].set_title('RMSE Comparison (Lower is Better)')
    axes[0].set_ylabel('RMSE')
    
    # MAE comparison
    axes[1].bar(models_comparison['Model'], models_comparison['MAE'], color=['red', 'green'])
    axes[1].set_title('MAE Comparison (Lower is Better)')
    axes[1].set_ylabel('MAE')
    
    # R² comparison
    axes[2].bar(models_comparison['Model'], models_comparison['R²'], color=['red', 'green'])
    axes[2].set_title('R² Comparison (Higher is Better)')
    axes[2].set_ylabel('R²')
    
    plt.tight_layout()
    plt.savefig('Images/model_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Modeling Analysis Complete!")
    print("📁 Additional plots saved to Images folder:")
    
    # List generated files
    import glob
    image_files = glob.glob('Images/*.png')
    modeling_files = [f for f in image_files if any(x in f for x in ['train_test', 'target', 'feature_importance', 'model_performance', 'model_comparison'])]
    for img_file in sorted(modeling_files):
        print(f"  • {img_file}")

if __name__ == "__main__":
    main()
