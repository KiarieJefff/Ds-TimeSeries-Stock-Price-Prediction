"""
Tests for prediction function
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.netflix_prediction_function import load_and_predict

def test_load_and_predict():
    """Test the load_and_predict function with actual model files"""
    
    # Check if model files exist
    model_path = 'models/netflix_stock_rf_model.pkl'
    metadata_path = 'data/processed/netflix_stock_rf_metadata.json'
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        pytest.skip("Model files not available")
    
    # Create sample data with required features
    sample_data = pd.DataFrame({
        'Volume': [1000000, 1100000, 1200000],
        'log_return': [0.01, -0.005, 0.02],
        'hl_range': [0.03, 0.025, 0.04],
        'oc_return': [0.015, -0.01, 0.025],
        'price_change': [0.02, -0.015, 0.03],
        'log_volume': [13.8, 13.9, 14.0],
        'volume_change': [0.1, -0.05, 0.15],
        'ma_5': [100, 102, 105],
        'volume_ma_5': [1050000, 1080000, 1120000],
        'ma_10': [98, 100, 103],
        'volume_ma_10': [1030000, 1050000, 1080000],
        'ma_20': [95, 97, 100],
        'volume_ma_20': [1000000, 1020000, 1050000],
        'ma_50': [92, 94, 96],
        'volume_ma_50': [980000, 1000000, 1020000],
        'volatility_5': [0.02, 0.018, 0.022],
        'volatility_20': [0.025, 0.023, 0.027],
        'rsi_14': [55, 45, 60],
        'momentum_5': [0.05, -0.02, 0.08],
        'lag_1': [0.01, -0.005, 0.02],
        'lag_2': [0.015, -0.01, 0.025],
        'lag_3': [0.008, -0.008, 0.018],
        'lag_5': [0.02, -0.015, 0.03],
        'lag_10': [0.025, -0.02, 0.035],
        'day_of_week': [1, 2, 3],
        'month': [1, 2, 3],
        'quarter': [1, 1, 1]
    })
    
    try:
        # Test the prediction function
        predictions, metadata = load_and_predict(model_path, metadata_path, sample_data)
        
        # Assertions
        assert len(predictions) == 3, "Should return 3 predictions"
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        assert 'feature_columns' in metadata, "Metadata should contain feature columns"
        assert 'performance_metrics' in metadata, "Metadata should contain performance metrics"
        
        print("✅ Prediction function test passed!")
        
    except Exception as e:
        pytest.fail(f"Prediction function failed: {str(e)}")

def test_missing_features():
    """Test prediction function with missing features"""
    
    model_path = 'models/netflix_stock_rf_model.pkl'
    metadata_path = 'data/processed/netflix_stock_rf_metadata.json'
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        pytest.skip("Model files not available")
    
    # Create data with missing features
    incomplete_data = pd.DataFrame({
        'Volume': [1000000, 1100000],
        'log_return': [0.01, -0.005]
        # Missing many required features
    })
    
    try:
        load_and_predict(model_path, metadata_path, incomplete_data)
        pytest.fail("Should have raised ValueError for missing features")
    except ValueError as e:
        assert "Missing features" in str(e), "Should mention missing features"
        print("✅ Missing features test passed!")
    except Exception as e:
        pytest.fail(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_load_and_predict()
    test_missing_features()
    print("All prediction tests passed!")
