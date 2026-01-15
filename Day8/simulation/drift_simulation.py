import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def generate_drift_dataset():
    """
    Generates a synthetic dataset with 3 phases:
    1. Baseline (Normal operation)
    2. Data Drift (Sensor calibration drift)
    3. Concept Drift (Physical relationship change)
    """
    print("Step 1: Generating Synthetic Data...")

    # Setup: 21 days of data, 15-min intervals
    dates = pd.date_range(start='2024-01-01', periods=24 * 4 * 21, freq='15min')
    n = len(dates)

    # Split into 3 equal phases (7 days each)
    phase_len = n // 3

    # --- Phase 1: Baseline ---
    # Temp: 20-30C, Humidity: Inverse to Temp (80 - 2*T)
    t_p1 = 25 + 5 * np.sin(np.linspace(0, 7 * 2 * np.pi, phase_len)) + np.random.normal(0, 0.5, phase_len)
    h_p1 = 80 - 2 * (t_p1 - 20) + np.random.normal(0, 1, phase_len)

    # --- Phase 2: Data Drift (Covariate Shift) ---
    # Temp sensor starts drifting upward by 5 degrees.
    # Concept remains valid (Humidity still reacts to the "true" temp, but sensor reads high)
    drift_ramp = np.linspace(0, 5, phase_len)
    t_p2 = (25 + 5 * np.sin(np.linspace(0, 7 * 2 * np.pi, phase_len))) + drift_ramp + np.random.normal(0, 0.5,
                                                                                                       phase_len)
    h_p2 = 80 - 2 * (t_p2 - 20) + np.random.normal(0, 1, phase_len)  # Relationship stays same

    # --- Phase 3: Concept Drift (Relationship Shift) ---
    # Relationship FLIPS. Now Humidity correlates positively (e.g. Humidifier logic: 40 + 1.5*T)
    t_p3 = 25 + 5 * np.sin(np.linspace(0, 7 * 2 * np.pi, phase_len)) + np.random.normal(0, 0.5, phase_len)
    h_p3 = 40 + 1.5 * (t_p3 - 20) + np.random.normal(0, 1, phase_len)

    # Concatenate
    temp = np.concatenate([t_p1, t_p2, t_p3])
    hum = np.concatenate([h_p1, h_p2, h_p3])

    # Create DataFrame with some noise/gaps for cleaning
    df = pd.DataFrame({'Timestamp': dates, 'Location': 'Server_Room_A', 'Temperature': temp, 'Humidity': hum})

    # Insert random missing values (to be cleaned later)
    mask = np.random.choice([True, False], size=n, p=[0.05, 0.95])
    df.loc[mask, 'Temperature'] = np.nan

    # Save Raw
    df.to_csv('drift_raw_data.csv', index=False)
    return df


def clean_data_chunk(df_chunk):
    """
    Performs standard cleaning (Interpolation, Outlier Removal, Smoothing)
    """
    df_clean = df_chunk.copy()

    # 1. Handle Missing Values
    df_clean['Temperature'] = df_clean['Temperature'].interpolate(method='linear').bfill().ffill()
    df_clean['Humidity'] = df_clean['Humidity'].interpolate(method='linear').bfill().ffill()

    # 2. Simple Smoothing (Rolling Mean)
    df_clean['Temp_Smooth'] = df_clean['Temperature'].rolling(window=4, min_periods=1, center=True).mean()
    df_clean['Hum_Smooth'] = df_clean['Humidity'].rolling(window=4, min_periods=1, center=True).mean()

    return df_clean


def process_pipeline(df):
    """
    Simulates streaming processing to detect and handle drifts.
    """
    print("Step 2: Processing Pipeline (Detecting & Handling Drifts)...")

    # Parameters
    window_size = 24 * 4  # 1 day window
    ks_alpha = 0.05  # Threshold for Data Drift (KS-Test)
    error_threshold = 15  # Threshold for Concept Drift (MAE)

    results = []
    drift_events = []  # Log (Index, Type)

    # Initial Training (Day 1)
    initial_window = clean_data_chunk(df.iloc[:window_size])

    # Initialize Scaler (Handles Data Drift)
    scaler = StandardScaler()
    scaler.fit(initial_window[['Temp_Smooth']])

    # Initialize Model (Handles Concept Drift) - Predicting Hum from Temp
    model = LinearRegression()
    model.fit(initial_window[['Temp_Smooth']], initial_window['Hum_Smooth'])

    # Reference distribution for Data Drift detection
    ref_distribution = initial_window['Temp_Smooth'].values

    # Iterate through data in daily chunks (Simulating Stream)
    num_chunks = len(df) // window_size

    for i in range(num_chunks):
        start_idx = i * window_size
        end_idx = (i + 1) * window_size

        # 1. Get Chunk & Clean
        raw_chunk = df.iloc[start_idx:end_idx]
        chunk = clean_data_chunk(raw_chunk)

        # -----------------------------------------------------------
        # A. Detect Data Drift (Covariate Shift) using KS-Test
        # -----------------------------------------------------------
        current_distribution = chunk['Temp_Smooth'].values
        stat, p_value = ks_2samp(ref_distribution, current_distribution)

        if p_value < ks_alpha:
            # Drift Detected!
            drift_events.append((chunk['Timestamp'].iloc[0], 'Data Drift (Scaler Updated)'))
            # HANDLE: Update Scaler to new distribution
            scaler.fit(chunk[['Temp_Smooth']])
            # Update reference to current to accept this as "new normal"
            ref_distribution = current_distribution

        # Apply Scaling
        chunk['Temp_Scaled'] = scaler.transform(chunk[['Temp_Smooth']])

        # -----------------------------------------------------------
        # B. Detect Concept Drift (Model Decay) using Error Rate
        # -----------------------------------------------------------
        # Predict using current model
        preds = model.predict(chunk[['Temp_Smooth']])
        mae = mean_absolute_error(chunk['Hum_Smooth'], preds)

        if mae > error_threshold:
            # Concept Drift Detected!
            drift_events.append((chunk['Timestamp'].iloc[0], 'Concept Drift (Model Retrained)'))
            # HANDLE: Retrain model on current relationship
            model.fit(chunk[['Temp_Smooth']], chunk['Hum_Smooth'])
            # Re-predict with new model
            chunk['Hum_Pred'] = model.predict(chunk[['Temp_Smooth']])
        else:
            chunk['Hum_Pred'] = preds

        results.append(chunk)

    final_df = pd.concat(results)
    final_df.to_csv('drift_processed_data.csv', index=False)
    return final_df, drift_events


def visualize_results(df, events):
    print("Step 3: Visualizing Results...")
    plt.figure(figsize=(15, 10))

    # Plot 1: Raw Temperature (showing Data Drift)
    plt.subplot(3, 1, 1)
    plt.plot(df['Timestamp'], df['Temperature'], color='orange', alpha=0.5, label='Raw Temp')
    # Mark events
    for ts, event_type in events:
        if "Data" in event_type:
            plt.axvline(ts, color='red', linestyle='--', linewidth=2, label='Data Drift Detected')

    # Dedup legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title('Phase 1 & 2: Data Drift (Temperature Shift)')
    plt.ylabel('Temp (°C)')

    # Plot 2: Prediction vs Actual (Showing Concept Drift Handling)
    plt.subplot(3, 1, 2)
    plt.plot(df['Timestamp'], df['Humidity'], color='blue', alpha=0.3, label='Actual Humidity')
    plt.plot(df['Timestamp'], df['Hum_Pred'], color='green', linestyle='--', label='Model Prediction')

    for ts, event_type in events:
        if "Concept" in event_type:
            plt.axvline(ts, color='purple', linestyle='-', linewidth=2, label='Concept Drift Detected')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
    plt.title('Phase 3: Concept Drift (Model Retraining)')
    plt.ylabel('Humidity (%)')

    # Plot 3: Error Rate (MAE) over time
    plt.subplot(3, 1, 3)
    df['Error'] = abs(df['Humidity'] - df['Hum_Pred'])
    # Rolling error to visualize spikes
    plt.plot(df['Timestamp'], df['Error'].rolling(96).mean(), color='black', label='Prediction Error (Rolling MAE)')
    plt.axhline(y=15, color='r', linestyle=':', label='Threshold')
    plt.title('Model Error Monitoring')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('drift_handling_report.png')
    print("Visualization saved to 'drift_handling_report.png'")


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Generate
    raw_df = generate_drift_dataset()

    # 2. Process
    processed_df, events_log = process_pipeline(raw_df)

    # 3. Visualize
    visualize_results(processed_df, events_log)

    print("\n--- Processing Log ---")
    for ts, event in events_log:
        print(f"{ts}: {event}")
    print("\nDone! Check the folder for CSVs and PNG.")