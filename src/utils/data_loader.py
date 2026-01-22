"""
Data loading utilities for Netflix stock analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str, date_column: str = 'Date') -> pd.DataFrame:
    """
    Load stock data from CSV file with proper preprocessing
    
    Args:
        file_path: Path to the CSV file
        date_column: Name of the date column
        
    Returns:
        Preprocessed DataFrame with datetime index
    """
    try:
        df = pd.read_csv(file_path)
        
        # Convert date column to datetime
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.sort_values(date_column).reset_index(drop=True)
            df.set_index(date_column, inplace=True)
        
        logger.info(f"Loaded data with shape: {df.shape}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"Data file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return summary statistics
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing quality metrics
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    # Check for OHLC consistency
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    if all(col in df.columns for col in ohlc_columns):
        # High should be >= Open, Close, Low
        high_consistency = (df['High'] >= df['Open']).all() & \
                          (df['High'] >= df['Close']).all() & \
                          (df['High'] >= df['Low']).all()
        
        # Low should be <= Open, Close, High  
        low_consistency = (df['Low'] <= df['Open']).all() & \
                         (df['Low'] <= df['Close']).all() & \
                         (df['Low'] <= df['High']).all()
        
        quality_report['ohlc_consistency'] = {
            'high_valid': high_consistency,
            'low_valid': low_consistency,
            'overall_valid': high_consistency and low_consistency
        }
    
    return quality_report
