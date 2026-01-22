"""
Tests for data loading utilities
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.data_loader import load_data, validate_data_quality

@pytest.fixture
def sample_data():
    """Create sample stock data for testing"""
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    data = {
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    return df

def test_validate_data_quality(sample_data):
    """Test data quality validation"""
    quality_report = validate_data_quality(sample_data)
    
    assert 'shape' in quality_report
    assert 'missing_values' in quality_report
    assert 'duplicate_rows' in quality_report
    assert 'data_types' in quality_report
    assert quality_report['shape'] == (100, 5)

def test_ohlc_consistency_check():
    """Test OHLC data consistency validation"""
    # Create consistent OHLC data
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    data = {
        'Open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'High': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
        'Low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104],
        'Close': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111]
    }
    df = pd.DataFrame(data, index=dates)
    
    quality_report = validate_data_quality(df)
    assert quality_report['ohlc_consistency']['overall_valid'] == True
