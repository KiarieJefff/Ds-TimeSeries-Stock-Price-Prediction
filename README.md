# Netflix Stock Volatility Forecasting Project

![Stock Price Visualization](Images/img_1.jpg)

## Overview
Industry-level data science project for forecasting Netflix stock volatility using time series analysis and machine learning techniques.

## Business Context
Developed for Quantum Capital Management's trading desk to improve risk-adjusted returns through proactive volatility management and enhanced trading strategies.

## Project Structure

```
├── analysis/                  # Analysis scripts and outputs
│   ├── run_eda_analysis.py   # Exploratory Data Analysis
│   ├── run_modeling_analysis.py # Modeling and evaluation
│   ├── outputs/               # Generated visualizations (12 plots)
│   └── README.md              # Analysis documentation
├── data/
│   ├── raw/                   # Original, unprocessed data
│   │   └── NFLX.csv          # Netflix stock price data
│   ├── processed/             # Cleaned and processed datasets
│   │   ├── close_from_open_metadata.json
│   │   ├── netflix_stock_rf_metadata.json
│   │   └── netflix_stock_rf_feature_importance.csv
│   └── external/              # External reference data
├── src/
│   ├── features/              # Feature engineering modules
│   │   ├── __init__.py
│   │   └── technical_indicators.py
│   ├── models/                # Model training and inference
│   │   ├── close_prediction.py
│   │   └── netflix_prediction_function.py
│   └── utils/                 # Common utility functions
│       ├── __init__.py
│       └── data_loader.py
├── models/                    # Trained model artifacts
│   ├── close_from_open_model.pkl
│   └── netflix_stock_rf_model.pkl
├── notebooks/
│   ├── eda/                   # Exploratory data analysis
│   │   ├── Data_Understanding_2.ipynb
│   │   └── test.ipynb
│   ├── modeling/              # Model development and experimentation
│   │   ├── close_prediction.ipynb
│   │   └── modeling.ipynb
│   └── reports/               # Business analysis and reporting
│       └── Business_Understanding.ipynb
├── scripts/                   # Automation and deployment scripts
│   └── run_pipeline.py        # Main execution pipeline
├── Images/                    # Key project visualizations
│   └── img_1.jpg              # Main project image
├── config/                    # Configuration files
│   └── config.yaml
├── tests/                     # Unit and integration tests
│   └── test_data_loader.py
├── logs/                      # Application logs
├── docs/                      # Documentation
├── artifacts/                 # Analysis outputs and results
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## Key Features
- **Volatility Forecasting**: Time series models for predicting stock volatility
- **Risk Management**: VaR calculations and dynamic position sizing
- **Feature Engineering**: Technical indicators and statistical features
- **Model Evaluation**: Comprehensive backtesting and validation framework

## Technologies Used
- Python 3.8+
- pandas, numpy for data manipulation
- scikit-learn for machine learning
- matplotlib, seaborn for visualization
- Jupyter notebooks for analysis

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete pipeline:
```bash
python scripts/run_pipeline.py
```

3. Run exploratory data analysis:
```bash
python analysis/run_eda_analysis.py
```

4. Run modeling analysis:
```bash
python analysis/run_modeling_analysis.py
```

5. Explore notebooks:
```bash
jupyter notebook notebooks/eda/Data_Understanding_2.ipynb
jupyter notebook notebooks/modeling/modeling.ipynb
```

## Business Impact
- Target: 20% reduction in false stop-loss triggers
- Expected: $560K annual savings in trading costs
- Goal: 15% improvement in options trading profitability

## Model Performance
- Volatility forecast error: Target <15% (baseline 25%)
- Sharpe ratio improvement: Target 1.5 (baseline 1.2)
- Model stability: <10% performance decay monthly

## Generated Outputs

### Analysis Visualizations
The project generates 12 high-quality visualizations saved in `analysis/outputs/`:

**EDA Analysis (7 plots):**
- Price evolution (linear & log scale)
- Volatility patterns and clustering
- Correlation heatmaps
- Feature relationships
- Time series properties
- EDA summary dashboard

**Modeling Analysis (5 plots):**
- Train/test split visualization
- Feature importance analysis
- Model performance metrics
- Prediction vs actual comparisons
- Model comparison summary

### Trained Models
- `models/close_from_open_model.pkl`: Linear regression model (R² = 0.9994)
- `models/netflix_stock_rf_model.pkl`: Random Forest model for return prediction

### Metadata and Artifacts
- Model performance metrics and configurations
- Feature importance rankings
- Data quality validation reports

## Contributing
Follow the established directory structure and coding standards. All changes should be tested and documented.

## License
Internal project for Quantum Capital Management
