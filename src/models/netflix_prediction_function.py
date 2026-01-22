
import joblib
import pandas as pd
import numpy as np
import json

def load_and_predict(model_path, metadata_path, new_data):
    """
    Load the saved Random Forest model and make predictions

    Args:
        model_path: Path to the saved model file
        metadata_path: Path to the metadata file
        new_data: DataFrame with the same features as training data

    Returns:
        predictions: Array of predicted log returns
    """
    # Load model and metadata
    model = joblib.load(model_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Ensure new_data has the required features
    required_features = metadata['feature_columns']
    if not all(feature in new_data.columns for feature in required_features):
        missing = set(required_features) - set(new_data.columns)
        raise ValueError(f"Missing features: {missing}")

    # Make predictions
    predictions = model.predict(new_data[required_features])

    return predictions, metadata

# Example usage:
# predictions, metadata = load_and_predict('netflix_stock_rf_model.pkl', 
#                                         'netflix_stock_rf_metadata.json', 
#                                         new_data)
