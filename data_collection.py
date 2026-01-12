import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. Load & Clean Raw Data
# ============================================================

def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV sensor data and perform basic cleaning:
    - Parse timestamps
    - Remove invalid rows
    - Filter physical outliers
    - Sort by time
    """
    df = pd.read_csv(file_path)
    print(f"[INFO] Raw rows loaded: {len(df)}")

    # Parse timestamp safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Physical range validation (sensor realism)
    df = df[(df["soil_moisture_percent"] >= 0) & (df["soil_moisture_percent"] <= 100)]
    df = df[(df["temperature_c"] >= 0) & (df["temperature_c"] <= 50)]
    df = df[df["light_intensity_lux"] >= 0]

    # Sort by time (critical for time-series models)
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"[INFO] Valid rows after cleaning: {len(df)}")
    return df


# ============================================================
# 2. Denoising & Feature Engineering
# ============================================================

def denoise_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply denoising and generate model-ready features:
    - Moving average smoothing
    - Time index
    """
    # Moving average denoising (light smoothing)
    df["moisture_denoised"] = (
        df["soil_moisture_percent"]
        .rolling(window=3, center=True)
        .mean()
    )

    # Fill boundary NaNs
    df["moisture_denoised"] = (
        df["moisture_denoised"]
        .bfill()
        .ffill()
    )

    # Time index (for regression models)
    df["time_index"] = np.arange(len(df))

    return df


# ============================================================
# 3. Normalization
# ============================================================

def normalize_features(df: pd.DataFrame):
    """
    Normalize numerical features into [0,1] range.
    Returns:
        - Processed DataFrame
        - Moisture scaler
        - Feature scaler
    """
    moisture_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    # Target normalization
    df["moisture_norm"] = moisture_scaler.fit_transform(
        df[["moisture_denoised"]]
    )

    # Feature normalization
    feature_cols = ["time_index", "temperature_c", "light_intensity_lux"]
    df[["time_norm", "temp_norm", "light_norm"]] = feature_scaler.fit_transform(
        df[feature_cols]
    )

    return df, moisture_scaler, feature_scaler


# ============================================================
# 4. Save Processed Dataset
# ============================================================

def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Save cleaned and processed dataset for modeling.
    """
    df.to_csv(output_path, index=False)
    print(f"[INFO] Processed data saved to: {output_path}")


# ============================================================
# 5. Main Execution Pipeline
# ============================================================

def main():
    input_file = "data/soil_moisture.csv"
    output_file = "data/soil_moisture_processed.csv"

    print("=" * 60)
    print("🌱 Soil Moisture Data Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Load & clean
    df = load_and_clean_data(input_file)

    # Step 2: Denoising & feature engineering
    df = denoise_and_engineer_features(df)

    # Step 3: Normalization
    df, moisture_scaler, feature_scaler = normalize_features(df)

    # Step 4: Save processed dataset
    save_processed_data(df, output_file)

    print("\n[INFO] Preprocessing completed successfully.")
    print("      Output file is ready for modeling and prediction.")
    print("=" * 60)


if __name__ == "__main__":
    main()

