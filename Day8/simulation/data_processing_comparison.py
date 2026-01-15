import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats


def process_and_visualize():
    # 1. Load the Data
    # We attempt to load the file generated in the previous step.
    try:
        df = pd.read_csv('synthetic_sensor_data.csv')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    except FileNotFoundError:
        print("Error: 'synthetic_sensor_data.csv' not found. Please generate the data first.")
        return

    # Create a copy for processing
    df_clean = df.copy()

    # ---------------------------------------------------------
    # 2. Handle Missing Values
    # ---------------------------------------------------------
    # Use Linear Interpolation to fill gaps based on the trend of neighboring points.
    df_clean['Temperature'] = df_clean['Temperature'].interpolate(method='linear')
    df_clean['Humidity'] = df_clean['Humidity'].interpolate(method='linear')

    # Drop any remaining NaNs (e.g., if the first row was NaN)
    df_clean.dropna(inplace=True)

    # ---------------------------------------------------------
    # 3. Detect & Handle Outliers (Z-Score Method)
    # ---------------------------------------------------------
    # If a value is > 3 standard deviations from the mean, replace it with the median.
    for col in ['Temperature', 'Humidity']:
        z_scores = np.abs(stats.zscore(df_clean[col]))
        outliers = z_scores > 3
        df_clean.loc[outliers, col] = df_clean[col].median()

    # ---------------------------------------------------------
    # 4. Noise Filtering / Smoothing
    # ---------------------------------------------------------
    # Apply a Rolling Mean (Moving Average) to smooth out high-frequency jitter.
    # Window size 4 corresponds to 1 hour (4 x 15 mins).
    df_clean['Temp_Smoothed'] = df_clean['Temperature'].rolling(window=4, center=True).mean().bfill().ffill()
    df_clean['Hum_Smoothed'] = df_clean['Humidity'].rolling(window=4, center=True).mean().bfill().ffill()

    # ---------------------------------------------------------
    # 5. Normalization (Min-Max Scaling)
    # ---------------------------------------------------------
    # Scales data to range [0, 1].
    scaler_minmax = MinMaxScaler()
    df_clean[['Temp_Norm', 'Hum_Norm']] = scaler_minmax.fit_transform(df_clean[['Temp_Smoothed', 'Hum_Smoothed']])

    # ---------------------------------------------------------
    # 6. Standardization (Z-Score Scaling)
    # ---------------------------------------------------------
    # Scales data to mean = 0, std = 1.
    scaler_std = StandardScaler()
    df_clean[['Temp_Std', 'Hum_Std']] = scaler_std.fit_transform(df_clean[['Temp_Smoothed', 'Hum_Smoothed']])

    # ---------------------------------------------------------
    # 7. Encoding Categorical Data
    # ---------------------------------------------------------
    # Convert 'Location' text labels into binary columns (One-Hot Encoding).
    df_clean = pd.get_dummies(df_clean, columns=['Location'], prefix='Loc')

    # Save to CSV
    df_clean.to_csv('processed_sensor_data.csv', index=False)
    print("Processed data saved to: processed_sensor_data.csv")

    # ---------------------------------------------------------
    # 8. Visualization
    # ---------------------------------------------------------
    plt.figure(figsize=(14, 12))

    # Plot 1: Temperature Comparison
    plt.subplot(3, 1, 1)
    plt.plot(df['Timestamp'], df['Temperature'], label='Raw (Noisy/Gaps)', color='lightgray', linewidth=1)
    plt.plot(df_clean['Timestamp'], df_clean['Temp_Smoothed'], label='Processed (Cleaned)', color='red', linewidth=1.5)
    plt.title('Temperature Processing: Raw vs Cleaned')
    plt.ylabel('Temperature (°C)')
    plt.legend()

    # Plot 2: Humidity Comparison
    plt.subplot(3, 1, 2)
    plt.plot(df['Timestamp'], df['Humidity'], label='Raw (Noisy/Gaps)', color='lightgray', linewidth=1)
    plt.plot(df_clean['Timestamp'], df_clean['Hum_Smoothed'], label='Processed (Cleaned)', color='blue', linewidth=1.5)
    plt.title('Humidity Processing: Raw vs Cleaned')
    plt.ylabel('Humidity (%)')
    plt.legend()

    # Plot 3: Scaling Comparison (First 48 hours only)
    subset = df_clean.iloc[:192]  # 48 hours * 4 readings/hr
    plt.subplot(3, 1, 3)
    plt.plot(subset['Timestamp'], subset['Temp_Norm'], label='Normalized [0-1]', color='green')
    plt.plot(subset['Timestamp'], subset['Temp_Std'], label='Standardized (Z-Score)', color='purple')
    plt.title('Data Scaling Techniques (Zoomed in 48h)')
    plt.ylabel('Scaled Value')
    plt.legend()

    plt.tight_layout()
    plt.savefig('data_processing_comparison.png')
    print("Visualization saved to: data_processing_comparison.png")

    return df_clean


# Run the function
processed_df = process_and_visualize()
print("\nFirst 5 rows of processed data:")
print(processed_df.head())