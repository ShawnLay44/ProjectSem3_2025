import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. Data Loading & Cleaning ====================
def load_and_clean_data(file_path):
    """Load and clean raw soil moisture data (handle outliers/invalid timestamps)"""
    try:
        df = pd.read_csv(file_path)
        print(f"📊 Raw data loaded: {len(df)} rows")
        
        # Process timestamp (convert to datetime, drop invalid)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Filter physical outliers (realistic sensor ranges)
        df = df[(df['soil_moisture_percent'] >= 0) & (df['soil_moisture_percent'] <= 100)]
        df = df[(df['temperature_c'] >= 0) & (df['temperature_c'] <= 50)]
        df = df[df['light_intensity_lux'] >= 0]
        
        # Sort by time (critical for time-series analysis)
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"✅ Cleaned data: {len(df)} valid rows")
        return df
        
    except Exception as e:
        print(f"❌ Data loading failed: {str(e)}")
        return pd.DataFrame()

# ==================== 2. Data Preprocessing ====================
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    """Data denoising (moving average) and normalization (0-1 scale)"""
    # Moving average denoising (window=3, center-aligned)
    df['moisture_denoised'] = df['soil_moisture_percent'].rolling(window=3, center=True).mean()
    # Fill edge NaNs (forward/backward fill)
    df['moisture_denoised'] = df['moisture_denoised'].fillna(method='bfill').fillna(method='ffill')
    
    # Moisture normalization (for model training)
    scaler_moisture = MinMaxScaler(feature_range=(0,1))
    df['moisture_norm'] = scaler_moisture.fit_transform(df[['moisture_denoised']])
    
    # Feature normalization (time/temperature/light for linear regression)
    scaler_features = MinMaxScaler(feature_range=(0,1))
    df['time_index'] = np.arange(len(df))  # Time sequence index
    features = ['time_index', 'temperature_c', 'light_intensity_lux']
    df[['time_norm', 'temp_norm', 'light_norm']] = scaler_features.fit_transform(df[features])
    
    return df, scaler_moisture, scaler_features

# ==================== 3. Visualization ====================
import matplotlib.pyplot as plt

def plot_moisture_trend(df):
    """Plot raw vs denoised moisture trend (mark watering event if exists)"""
    plt.rcParams['font.sans-serif'] = ['Arial']  # English font for report
    plt.figure(figsize=(12, 6))
    
    # Plot moisture trends
    plt.plot(df['timestamp'], df['soil_moisture_percent'], label='Raw Moisture', alpha=0.7)
    plt.plot(df['timestamp'], df['moisture_denoised'], label='Denoised Moisture', linewidth=2)
    
    # Mark watering event (fault-tolerant)
    if 'plant_health_status' in df.columns:
        watering_events = df[df['plant_health_status'] == "post_watering"]
        if len(watering_events) > 0:
            plt.axvline(
                watering_events['timestamp'].iloc[0],
                linestyle='--', color='red', label='Watering Event'
            )
    
    # Chart formatting (report-ready)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Soil Moisture (%)", fontsize=12)
    plt.title("Soil Moisture Trend Over Time", fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(loc='upper right')
    plt.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==================== 4. Linear Regression ====================
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_linear_regression(df):
    """Train multivariate linear regression model (time/temp/light features)"""
    # Prepare features (normalized) and target (normalized moisture)
    X = df[['time_norm', 'temp_norm', 'light_norm']]
    y = df['moisture_norm']
    
    # Time-based split (80% train / 20% test, no shuffle for time-series)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X)
    test_pred = model.predict(X_test)
    
    # Evaluate performance (test set only)
    mse = mean_squared_error(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    
    # Print results (report-friendly format)
    print(f"\n📈 Linear Regression Results:")
    print(f"   MSE (Test Set): {mse:.4f}")
    print(f"   MAE (Test Set): {mae:.4f}")
    print(f"   Coefficients (Time/Temp/Light): {np.round(model.coef_, 4)}")
    print(f"   Intercept: {model.intercept_:.4f}")
    
    # Save predictions to dataframe
    df['lr_pred_norm'] = y_pred
    
    return df, model, mse, mae

# ==================== 5. LSTM Model ====================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_sequences(data, window_size=5):
    """Create sliding window sequences for LSTM input"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])  # Past 5 time steps
        y.append(data[i+window_size])   # Next time step
    return np.array(X), np.array(y)

def train_lstm_model(df, scaler_moisture, window_size=5):
    """Train lightweight LSTM model for time-series prediction"""
    # Prepare sequence data (normalized moisture)
    data = df['moisture_norm'].values
    X, y = create_lstm_sequences(data, window_size)
    # Reshape for LSTM: (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Time-based split (80% train / 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Build lightweight LSTM (student-friendly architecture)
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(window_size, 1)),  # 32 units (not overfitting)
        Dense(1)  # Output layer (predict next moisture value)
    ])
    
    # Compile model (Adam optimizer + MSE loss)
    model.compile(optimizer='adam', loss='mse')
    
    # Train model (30 epochs, small batch size for stability)
    print("\n🔄 Training LSTM model (30 epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=8,
        verbose=1,
        validation_split=0.2  # 20% validation for overfitting check
    )
    
    # Predictions
    y_pred = model.predict(X)
    test_pred = model.predict(X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, test_pred)
    mae = mean_absolute_error(y_test, test_pred)
    
    print(f"\n🤖 LSTM Model Results:")
    print(f"   MSE (Test Set): {mse:.4f}")
    print(f"   MAE (Test Set): {mae:.4f}")
    
    # Align predictions with original dataframe (fill NaN for window size)
    full_pred = np.full(len(df), np.nan)
    full_pred[window_size:] = y_pred.flatten()
    df['lstm_pred_norm'] = full_pred
    
    return df, model, mse, mae, history

# ==================== 6. Model Comparison ====================
def compare_predictions(df, scaler_moisture):
    """Compare Linear Regression vs LSTM predictions (denormalized values)"""
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(14, 8))
    
    # Denormalize actual moisture (back to % scale)
    actual_values = scaler_moisture.inverse_transform(df[['moisture_norm']])
    plt.plot(df['timestamp'], actual_values, 
             label='Actual Moisture', color='black', linewidth=3, zorder=10)
    
    # Linear Regression predictions (denormalized)
    if 'lr_pred_norm' in df.columns:
        lr_pred = scaler_moisture.inverse_transform(df[['lr_pred_norm']])
        plt.plot(df['timestamp'], lr_pred, 
                 label='Linear Regression Prediction', linestyle='--', linewidth=2, color='blue')
    
    # LSTM predictions (denormalized, remove NaN)
    if 'lstm_pred_norm' in df.columns:
        valid_mask = ~df['lstm_pred_norm'].isna()
        lstm_pred = scaler_moisture.inverse_transform(df[['lstm_pred_norm']])
        plt.plot(df['timestamp'][valid_mask], lstm_pred[valid_mask], 
                 label='LSTM Prediction', linestyle='-.', linewidth=2, color='red')
    
    # Chart formatting
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Soil Moisture (%)", fontsize=12)
    plt.title("Model Prediction Comparison (Actual vs Predictions)", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plot LSTM training/validation loss (regular + log scale)"""
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.figure(figsize=(10, 4))
    
    # Regular scale loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.title('LSTM Training Process')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale loss (better for small values)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', alpha=0.7)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss (Log Scale)')
    plt.title('LSTM Training Process (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==================== 7. Main Program ====================
def main():
    """Main workflow (student-friendly step-by-step execution)"""
    print("=" * 60)
    print("🌱 Plant Soil Moisture Prediction System (Student Project)")
    print("=" * 60)
    
    # Step 1: Load and clean data
    print("\n1️⃣ Loading & cleaning raw data...")
    df = load_and_clean_data("soil_moisture.csv")
    if df.empty:
        print("❌ No valid data available, program exited")
        return
    
    # Step 2: Preprocess data (denoising + normalization)
    print("\n2️⃣ Preprocessing data (denoising + normalization)...")
    df, scaler_moisture, scaler_features = preprocess_data(df)
    
    # Step 3: Visualize moisture trend
    print("\n3️⃣ Generating moisture trend visualization...")
    plot_moisture_trend(df)
    
    # Step 4: Train Linear Regression model
    print("\n4️⃣ Training Multivariate Linear Regression model...")
    df, lr_model, lr_mse, lr_mae = train_linear_regression(df)
    
    # Step 5: Train LSTM model
    print("\n5️⃣ Training LSTM Time-Series model...")
    df, lstm_model, lstm_mse, lstm_mae, history = train_lstm_model(df, scaler_moisture)
    
    # Step 6: Visualize LSTM training history
    print("\n6️⃣ Visualizing LSTM training process...")
    plot_training_history(history)
    
    # Step 7: Compare model predictions
    print("\n7️⃣ Comparing model predictions (denormalized values)...")
    compare_predictions(df, scaler_moisture)
    
    # Step 8: Final results summary (report-ready)
    print("\n" + "=" * 60)
    print("📋 Final Project Results Summary")
    print("=" * 60)
    print(f"📈 Linear Regression: MSE={lr_mse:.4f}, MAE={lr_mae:.4f}")
    print(f"🤖 LSTM Model:        MSE={lstm_mse:.4f}, MAE={lstm_mae:.4f}")
    
    # Determine better model
    if lstm_mse < lr_mse:
        print("\n✅ LSTM model outperforms Linear Regression (lower MSE)")
        print("   Reason: LSTM captures time-series patterns in moisture data")
    else:
        print("\n✅ Linear Regression outperforms LSTM (lower MSE)")
        print("   Reason: Simple linear relationships dominate in simulated data")
    print("=" * 60)

# Run main program
if __name__ == "__main__":
    main()
