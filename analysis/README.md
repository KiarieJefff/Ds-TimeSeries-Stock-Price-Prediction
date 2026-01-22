# Analysis Directory

This directory contains all analysis scripts and outputs for the Netflix Stock Volatility Forecasting project.

## 📁 Directory Structure

```
analysis/
├── README.md                    # This file
├── run_eda_analysis.py         # Exploratory Data Analysis script
├── run_modeling_analysis.py    # Modeling and evaluation script
└── outputs/                    # Generated visualizations and plots
    ├── correlation_heatmaps.png
    ├── eda_summary_dashboard.png
    ├── feature_importance.png
    ├── feature_relationships.png
    ├── model_comparison_summary.png
    ├── model_performance_analysis.png
    ├── open_vs_close_relationship.png
    ├── price_evolution.png
    ├── target_distributions.png
    ├── time_series_properties.png
    ├── train_test_split.png
    └── volatility_patterns.png
```

## 🚀 Usage

### Running EDA Analysis
```bash
cd analysis
python run_eda_analysis.py
```

### Running Modeling Analysis
```bash
cd analysis
python run_modeling_analysis.py
```

## 📊 Generated Visualizations

### EDA Analysis Outputs
- **price_evolution.png**: Stock price evolution (linear & log scale)
- **volatility_patterns.png**: Daily returns and rolling volatility
- **correlation_heatmaps.png**: OHLC and engineered features correlations
- **feature_relationships.png**: Volume vs range, returns relationships
- **time_series_properties.png**: Distribution, ACF, Q-Q plots
- **eda_summary_dashboard.png**: Comprehensive EDA overview

### Modeling Analysis Outputs
- **train_test_split.png**: Temporal train-test split visualization
- **target_distributions.png**: Target variable distributions
- **feature_importance.png**: Random Forest feature importance
- **model_performance_analysis.png**: Model predictions and residuals
- **model_comparison_summary.png**: Performance metrics comparison
- **open_vs_close_relationship.png**: Open vs Close price relationship

## 📈 Key Insights

### Data Characteristics
- **Dataset**: 5,044 daily observations (2002-2022)
- **Growth**: 166x price increase over 20 years
- **Volatility**: 3.64% daily standard deviation
- **Extreme Events**: 2% of days with ±9%+ moves

### Model Performance
- **Random Forest**: R² = 0.9994, RMSE = 0.0309
- **Feature Engineering**: 27 technical indicators created
- **Top Features**: Volume change, moving averages, lagged returns

## 🛠️ Dependencies

The analysis scripts require the following packages:
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- statsmodels, arch
- scipy

## 📝 Notes

- All plots are saved at 300 DPI for high quality
- Scripts are designed to be run from the project root directory
- Outputs are automatically saved to the `outputs/` subdirectory
