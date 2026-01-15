import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def generate_and_visualize_data():
    # ---------------------------------------------------------
    # 1. Setup Time Series
    # ---------------------------------------------------------
    # 7 days of data at 15-minute intervals
    dates = pd.date_range(start='2024-01-01', periods=24 * 4 * 7, freq='15min')
    n = len(dates)

    # ---------------------------------------------------------
    # 2. Generate Base Data (Signal + Noise)
    # ---------------------------------------------------------
    # Temperature: Daily cycle (sine wave) + random Gaussian noise
    temp_base = 20 + 10 * np.sin(np.linspace(0, 7 * 2 * np.pi, n))
    temp_noise = np.random.normal(0, 1.5, n)
    temperature = temp_base + temp_noise

    # Humidity: Inverse to temp (sine wave) + random Gaussian noise
    humidity_base = 60 - 20 * np.sin(np.linspace(0, 7 * 2 * np.pi, n))
    humidity_noise = np.random.normal(0, 3, n)
    humidity = humidity_base + humidity_noise

    # Random Locations
    locations = np.random.choice(['Lab_A', 'Server_Room', 'Outdoor_Unit'], size=n)

    # Create Initial DataFrame
    df = pd.DataFrame({
        'Timestamp': dates,
        'Location': locations,
        'Temperature': temperature,
        'Humidity': humidity
    })

    # ---------------------------------------------------------
    # 3. Introduce Artifacts (Bad Data)
    # ---------------------------------------------------------

    # A. Insert Missing Values (Gaps) - approx 5% of data
    nan_indices_t = np.random.choice(df.index, size=int(n * 0.05), replace=False)
    nan_indices_h = np.random.choice(df.index, size=int(n * 0.05), replace=False)
    df.loc[nan_indices_t, 'Temperature'] = np.nan
    df.loc[nan_indices_h, 'Humidity'] = np.nan

    # B. Insert Outliers (Extreme Spikes)
    # Temperature outliers (e.g., sensor malfunction reading high heat)
    outlier_indices_t = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices_t, 'Temperature'] = df.loc[outlier_indices_t, 'Temperature'] * 3

    # Humidity outliers (e.g., impossible negative values)
    outlier_indices_h = np.random.choice(df.index, size=5, replace=False)
    df.loc[outlier_indices_h, 'Humidity'] = -20

    # ---------------------------------------------------------
    # 4. Save to CSV
    # ---------------------------------------------------------
    csv_filename = 'synthetic_sensor_data.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Dataset successfully saved to: {csv_filename}")

    # ---------------------------------------------------------
    # 5. Visualize the Data
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 10))

    # Subplot 1: Temperature
    plt.subplot(2, 1, 1)
    plt.plot(df['Timestamp'], df['Temperature'], label='Temperature (°C)', color='#d62728', linewidth=1)
    # Highlight the specific outliers we induced (for visualization clarity)
    plt.scatter(df['Timestamp'].iloc[outlier_indices_t], df['Temperature'].iloc[outlier_indices_t],
                color='black', label='Induced Outliers', zorder=5, marker='x', s=50)
    plt.title('Synthetic Temperature Data (Raw: Noise, Outliers, Gaps)')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Subplot 2: Humidity
    plt.subplot(2, 1, 2)
    plt.plot(df['Timestamp'], df['Humidity'], label='Humidity (%)', color='#1f77b4', linewidth=1)
    plt.scatter(df['Timestamp'].iloc[outlier_indices_h], df['Humidity'].iloc[outlier_indices_h],
                color='black', label='Induced Outliers', zorder=5, marker='x', s=50)
    plt.title('Synthetic Humidity Data (Raw: Noise, Outliers, Gaps)')
    plt.ylabel('Humidity (%)')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('synthetic_data_visualization.png')
    print("Visualization saved to: synthetic_data_visualization.png")

    return df


# Run the function
df_generated = generate_and_visualize_data()

# Show first few rows
print("\nFirst 5 rows of generated data:")
print(df_generated.head())