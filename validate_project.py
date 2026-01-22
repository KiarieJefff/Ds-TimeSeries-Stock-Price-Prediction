#!/usr/bin/env python3
"""
Comprehensive project validation script
Validates all components of the Netflix Stock Volatility Forecasting project
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

def check_file_structure():
    """Validate project file structure"""
    print("🔍 Checking Project Structure...")
    
    required_files = [
        'README.md',
        'requirements.txt',
        'config/config.yaml',
        'data/raw/NFLX.csv',
        'src/utils/data_loader.py',
        'src/features/technical_indicators.py',
        'src/models/close_prediction.py',
        'src/models/netflix_prediction_function.py',
        'scripts/run_pipeline.py',
        'analysis/run_eda_analysis.py',
        'analysis/run_modeling_analysis.py',
        'analysis/README.md'
    ]
    
    required_dirs = [
        'data/raw',
        'data/processed', 
        'models',
        'analysis/outputs',
        'notebooks/eda',
        'notebooks/modeling',
        'notebooks/reports',
        'tests'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    print("✅ Project structure is complete!")
    return True

def check_data_integrity():
    """Validate data files"""
    print("\n📊 Checking Data Integrity...")
    
    try:
        # Check main data file
        df = pd.read_csv('data/raw/NFLX.csv')
        
        # Validate columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        # Check data quality
        if df.empty:
            print("❌ Data file is empty")
            return False
        
        # Check for reasonable data ranges
        if df['Close'].min() <= 0 or df['Volume'].min() <= 0:
            print("❌ Invalid data values detected")
            return False
        
        print(f"✅ Data validation passed! {len(df)} rows, {len(df.columns)} columns")
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {str(e)}")
        return False

def check_models():
    """Validate trained models and metadata"""
    print("\n🤖 Checking Models...")
    
    model_checks = []
    
    # Check close prediction model
    if os.path.exists('models/close_from_open_model.pkl'):
        if os.path.exists('data/processed/close_from_open_metadata.json'):
            model_checks.append("✅ Close prediction model")
        else:
            model_checks.append("❌ Close prediction model metadata missing")
    else:
        model_checks.append("❌ Close prediction model missing")
    
    # Check Random Forest model
    if os.path.exists('models/netflix_stock_rf_model.pkl'):
        if os.path.exists('data/processed/netflix_stock_rf_metadata.json'):
            model_checks.append("✅ Random Forest model")
        else:
            model_checks.append("❌ Random Forest model metadata missing")
    else:
        model_checks.append("❌ Random Forest model missing")
    
    for check in model_checks:
        print(f"  {check}")
    
    return all("✅" in check for check in model_checks)

def check_visualizations():
    """Validate visualization outputs"""
    print("\n📈 Checking Visualizations...")
    
    expected_plots = [
        'correlation_heatmaps.png',
        'eda_summary_dashboard.png',
        'feature_importance.png',
        'feature_relationships.png',
        'model_comparison_summary.png',
        'model_performance_analysis.png',
        'open_vs_close_relationship.png',
        'price_evolution.png',
        'target_distributions.png',
        'time_series_properties.png',
        'train_test_split.png',
        'volatility_patterns.png'
    ]
    
    missing_plots = []
    for plot in expected_plots:
        if not os.path.exists(f'analysis/outputs/{plot}'):
            missing_plots.append(plot)
    
    if missing_plots:
        print(f"❌ Missing plots: {missing_plots}")
        return False
    
    print(f"✅ All {len(expected_plots)} visualization plots found!")
    return True

def check_code_imports():
    """Test code imports and basic functionality"""
    print("\n🔧 Testing Code Imports...")
    
    import_tests = []
    
    try:
        sys.path.append('src')
        from utils.data_loader import load_data, validate_data_quality
        import_tests.append("✅ Data loader imports")
    except Exception as e:
        import_tests.append(f"❌ Data loader import failed: {e}")
    
    try:
        from features.technical_indicators import calculate_sma, calculate_rsi
        import_tests.append("✅ Technical indicators imports")
    except Exception as e:
        import_tests.append(f"❌ Technical indicators import failed: {e}")
    
    try:
        from models.netflix_prediction_function import load_and_predict
        import_tests.append("✅ Prediction function imports")
    except Exception as e:
        import_tests.append(f"❌ Prediction function import failed: {e}")
    
    for test in import_tests:
        print(f"  {test}")
    
    return all("✅" in test for test in import_tests)

def check_documentation():
    """Validate documentation files"""
    print("\n📚 Checking Documentation...")
    
    doc_checks = []
    
    # Check main README
    if os.path.exists('README.md'):
        with open('README.md', 'r') as f:
            readme_content = f.read()
            if len(readme_content) > 1000:  # Reasonable length
                doc_checks.append("✅ Main README")
            else:
                doc_checks.append("❌ Main README too short")
    else:
        doc_checks.append("❌ Main README missing")
    
    # Check analysis README
    if os.path.exists('analysis/README.md'):
        doc_checks.append("✅ Analysis README")
    else:
        doc_checks.append("❌ Analysis README missing")
    
    # Check config
    if os.path.exists('config/config.yaml'):
        doc_checks.append("✅ Configuration file")
    else:
        doc_checks.append("❌ Configuration file missing")
    
    for check in doc_checks:
        print(f"  {check}")
    
    return all("✅" in check for check in doc_checks)

def main():
    """Run comprehensive project validation"""
    print("🚀 Netflix Stock Volatility Forecasting - Project Validation")
    print("=" * 60)
    
    validation_results = []
    
    # Run all validation checks
    validation_results.append(check_file_structure())
    validation_results.append(check_data_integrity())
    validation_results.append(check_models())
    validation_results.append(check_visualizations())
    validation_results.append(check_code_imports())
    validation_results.append(check_documentation())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed_checks = sum(validation_results)
    total_checks = len(validation_results)
    
    print(f"✅ Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 PROJECT VALIDATION SUCCESSFUL!")
        print("🚀 Ready for production use!")
        return True
    else:
        print("⚠️  PROJECT VALIDATION FAILED!")
        print("🔧 Please address the issues above before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
