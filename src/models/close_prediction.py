# Simple Close Price Prediction from Open Price
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import json
from datetime import datetime

def train_close_prediction_model():
    """Train a simple model to predict close price from open price"""
    
    # Load the data
    df = pd.read_csv('data/raw/NFLX.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Display basic statistics
    print("\nBasic statistics:")
    print(df[['Open', 'Close']].describe())
    
    # Prepare data for prediction
    # Features: Open price
    # Target: Close price
    
    X = df[['Open']].values  # Feature: Open price
    y = df['Close'].values    # Target: Close price
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Model performance
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"Training - MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
    print(f"Test - MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    print(f"\nModel Equation: Close = {model.coef_[0]:.4f} * Open + {model.intercept_:.4f}")
    
    # Visualize the relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.6, s=10, label='Training data')
    plt.scatter(X_test, y_test, alpha=0.6, s=10, color='red', label='Test data')
    plt.xlabel('Open Price ($)')
    plt.ylabel('Close Price ($)')
    plt.title('Netflix Stock: Open vs Close Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/outputs/open_vs_close_relationship.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Saved open vs close relationship plot to analysis/outputs/open_vs_close_relationship.png")
    
    return model, X_train, X_test, y_train, y_test, train_mse, test_mse, train_r2, test_r2

def predict_close(open_price, model):
    """
    Predict close price given open price
    
    Args:
        open_price: Single open price or array of open prices
        model: Trained linear regression model
    
    Returns:
        Predicted close price(s)
    """
    if isinstance(open_price, (int, float)):
        open_price = np.array([[open_price]])
    elif isinstance(open_price, list):
        open_price = np.array(open_price).reshape(-1, 1)
    else:
        open_price = np.array(open_price).reshape(-1, 1)
    
    return model.predict(open_price)

def save_model_and_metadata(model, train_mse, test_mse, train_r2, test_r2, df):
    """Save the trained model and metadata"""
    
    # Save the trained model
    model_filename = 'close_from_open_model.pkl'
    joblib.dump(model, model_filename)
    print(f"✅ Model saved as: {model_filename}")
    
    # Save model metadata
    model_metadata = {
        'model_type': 'LinearRegression',
        'feature': 'Open Price',
        'target': 'Close Price',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'performance_metrics': {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2)
        },
        'model_coefficients': {
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_)
        },
        'data_range': {
            'min_open': float(df['Open'].min()),
            'max_open': float(df['Open'].max()),
            'min_close': float(df['Close'].min()),
            'max_close': float(df['Close'].max())
        }
    }
    
    metadata_filename = 'close_from_open_metadata.json'
    with open(metadata_filename, 'w') as f:
        json.dump(model_metadata, f, indent=4)
    print(f"📋 Metadata saved as: {metadata_filename}")
    
    return model_filename, metadata_filename

def main():
    """Main function to run the complete prediction pipeline"""
    
    print("🚀 Netflix Close Price Prediction from Open Price")
    print("=" * 50)
    
    # Train the model
    model, X_train, X_test, y_train, y_test, train_mse, test_mse, train_r2, test_r2 = train_close_prediction_model()
    
    # Test the prediction function
    print("\n🔮 Testing Predictions:")
    test_prices = [100, 200, 300, 400, 500]
    predictions = predict_close(test_prices, model)
    
    print("Single Predictions:")
    for open_price, close_pred in zip(test_prices, predictions):
        print(f"Open: ${open_price:6.2f} -> Predicted Close: ${close_pred:6.2f}")
    
    # Calculate prediction accuracy on recent data
    df = pd.read_csv('data/raw/NFLX.csv', parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    recent_data = df.tail(50)  # Last 50 trading days
    recent_open = recent_data['Open'].values.reshape(-1, 1)
    recent_close_actual = recent_data['Close'].values
    recent_close_pred = predict_close(recent_open, model)
    
    recent_mse = mean_squared_error(recent_close_actual, recent_close_pred)
    recent_r2 = r2_score(recent_close_actual, recent_close_pred)
    
    print(f"\n📊 Recent 50 days performance:")
    print(f"MSE: {recent_mse:.4f}")
    print(f"R²: {recent_r2:.4f}")
    print(f"Average prediction error: ${np.sqrt(recent_mse):.2f}")
    
    # Save model and metadata
    model_filename, metadata_filename = save_model_and_metadata(
        model, train_mse, test_mse, train_r2, test_r2, df
    )
    
    print(f"\n🎯 Model Summary:")
    print(f"📈 Expected accuracy: R² = {test_r2:.4f}")
    print(f"📊 Average error: ${np.sqrt(test_mse):.2f}")
    print(f"🔧 Model saved: {model_filename}")
    print(f"📋 Metadata saved: {metadata_filename}")
    
    # Interactive example
    print(f"\n🎮 Interactive Prediction Example:")
    example_open = 350.00
    predicted_close = predict_close(example_open, model)[0]
    print(f"Example: Open ${example_open:.2f} -> Predicted Close ${predicted_close:.2f}")
    print(f"Expected change: ${predicted_close - example_open:.2f} ({((predicted_close/example_open - 1)*100):+.2f}%)")

if __name__ == "__main__":
    main()
