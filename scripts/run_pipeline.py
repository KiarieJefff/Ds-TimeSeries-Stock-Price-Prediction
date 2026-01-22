"""
Main pipeline script for Netflix stock volatility forecasting
"""

import sys
import os
import logging
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.data_loader import load_data, validate_data_quality
from features.technical_indicators import calculate_sma, calculate_rsi, calculate_bollinger_bands
from models.close_prediction import train_close_prediction_model

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/pipeline.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main pipeline execution"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Netflix Stock Volatility Forecasting Pipeline")
        
        # 1. Load and validate data
        logger.info("Loading raw data...")
        df = load_data('data/raw/NFLX.csv')
        
        logger.info("Validating data quality...")
        quality_report = validate_data_quality(df)
        logger.info(f"Data quality report: {quality_report}")
        
        # 2. Feature engineering
        logger.info("Creating technical indicators...")
        df['SMA_20'] = calculate_sma(df['Close'], 20)
        df['RSI_14'] = calculate_rsi(df['Close'], 14)
        bollinger = calculate_bollinger_bands(df['Close'])
        df = pd.concat([df, bollinger], axis=1)
        
        # 3. Train models
        logger.info("Training prediction models...")
        train_close_prediction_model()
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
