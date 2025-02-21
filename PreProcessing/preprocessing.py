import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import butter, filtfilt, iirnotch, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np
import random

# Load the data

filepath = "CombinedTemp.txt"
with open(filepath, "r") as f:
    lines = f.readlines()

start_idx = next(i for i, line in enumerate(lines) if not line.startswith('%')) # AI generated

dataframe = pd.read_csv(filepath, skiprows=start_idx) # Put data into pandas dataframe

dataframe.rename(columns={' EXG Channel 0': 'EXG Channel 0', ' EXG Channel 1': 'EXG Channel 1',' EXG Channel 2': 'EXG Channel 2',' EXG Channel 3': 'EXG Channel 3',' Timestamp': 'Timestamp', ' Marker Channel': 'Marker Channel',' Timestamp (Formatted)': 'Timestamp (Formatted)',}, inplace=True) # Rename columns as they have spaces in them

# Drop unneeded columns
columns_to_drop = [" Other", " Other.1", " Other.2", " Other.3", " Other.4", " Accel Channel 0", " Accel Channel 1", " Accel Channel 2"]
df_cleaned = dataframe.drop(columns=[col for col in columns_to_drop])

# Convert data types (keeping Timestamp (Formatted) and Marker Channel)
df_cleaned = df_cleaned.astype({
    "Sample Index": int,
    "EXG Channel 0": float,
    "EXG Channel 1": float,
    "EXG Channel 2": float,
    "EXG Channel 3": float,
    "Timestamp": float,
    "Marker Channel": float
})

# Define filter functions
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def apply_bandpass_filter(data, lowcut=1, highcut=50, fs=200, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def apply_notch_filter(data, notch_freq=50, fs=200, quality_factor=30):
    b, a = iirnotch(notch_freq / (0.5 * fs), quality_factor)
    return filtfilt(b, a, data)

# Apply filters to EEG channels
fs = 200  # Sample rate from OpenBCI metadata
for channel in ["EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3"]:
    df_cleaned[channel] = apply_bandpass_filter(df_cleaned[channel], lowcut=1, highcut=50, fs=fs)
    df_cleaned[channel] = apply_notch_filter(df_cleaned[channel], notch_freq=50, fs=fs)  # 50hz for uk
    df_cleaned[channel] = df_cleaned[channel] - df_cleaned[channel].mean() # DC offset removal, faster than high pass



# # Define time axis
# df_cleaned["Time (s)"] = (df_cleaned["Timestamp"] - df_cleaned["Timestamp"].min()) 

# # Plot EEG signals
# plt.figure(figsize=(12, 6))
# for i, channel in enumerate(["EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3"]):
#     plt.subplot(4, 1, i + 1)
#     plt.plot(df_cleaned["Time (s)"], df_cleaned[channel], label=channel)
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude (µV)")
#     plt.title(channel)
#     plt.legend()
#     plt.tight_layout()

# plt.show()

def epoch_data(df, window=(-0.5, 1.0), fs=200):
    """
    Epochs the EEG data based on event markers.

    Parameters:
        df (pd.DataFrame): The cleaned EEG data.
        event_column (str): The column indicating event markers.
        time_column (str): The column representing time in seconds.
        window (tuple): The time window around each event (start, end) in seconds.
        fs (int): Sampling rate in Hz (default 200 Hz for OpenBCI Ganglion).

    Returns:
        dict: A dictionary of epochs where keys are (timestamp, marker_value) and values are dataframes.
    """
    epochs = {}
    samples_before = int(abs(window[0]) * fs)  # Number of samples before event
    samples_after = int(window[1] * fs)        # Number of samples after event
    total_samples = samples_before + samples_after

    # Find event timestamps and corresponding marker values
    events = df[df["Marker Channel"].isin([2,3,4,5,6])][["Timestamp", "Marker Channel"]].values  # Get time + marker value

    print(f"Number of events: {len(events)}")

    for event_time, marker_value in events:
        # Define the epoch time range
        start_time = event_time + window[0]
        end_time = event_time + window[1]

        # Extract data within this time window
        epoch_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()
        
        # Ensure consistent(ish) length
        if len(epoch_df) >= total_samples * 0.9825:
            # Add marker column to indicate event type
            epoch_df["Event Type"] = marker_value  
            epochs[(event_time, marker_value)] = epoch_df

    return epochs


# Example usage
event_epochs = epoch_data(df_cleaned)
print(f"Created {len(event_epochs)} epochs with markers.")

def generate_idle_epochs(df, window=(-0.5, 1.0), fs=200):
    """
    Generates as many idle epochs as possible from EEG data where no event markers are present.

    Parameters:
        df (pd.DataFrame): The cleaned EEG data.
        window (tuple): The time window around each epoch (start, end) in seconds.
        fs (int): Sampling rate in Hz (default 200 Hz for OpenBCI Ganglion).

    Returns:
        dict: A dictionary of idle epochs where keys are (start_timestamp, -1) and values are dataframes.
    """
    idle_epochs = {}
    samples_before = int(abs(window[0]) * fs)
    samples_after = int(window[1] * fs)
    total_samples = samples_before + samples_after

    # Identify timestamps without events
    event_times = set(df[df["Marker Channel"] != 0]["Timestamp"].values)
    idle_df = df[~df["Timestamp"].isin(event_times)]  # Select non-event rows
    timestamps = idle_df["Timestamp"].values

    i = 0
    while i < len(timestamps) - total_samples:
        start_time = timestamps[i]
        end_time = start_time + (total_samples / fs)

        # Extract potential idle epoch
        epoch_df = df[(df["Timestamp"] >= start_time) & (df["Timestamp"] <= end_time)].copy()

        if len(epoch_df) >= total_samples * 0.9825:  # Ensure sufficient samples
            epoch_df["Event Type"] = -1  # Label as idle
            idle_epochs[(start_time, -1)] = epoch_df
            i += total_samples  # Move forward to avoid overlap
        else:
            i += 1  # Shift forward if the window is too short

    return idle_epochs

# Extract idle epochs
idle_epochs = generate_idle_epochs(df_cleaned)
print(f"Created {len(idle_epochs)} idle epochs")

all_epochs = {**event_epochs, **idle_epochs}

print(f"Created {len(all_epochs)} total epochs")

def extract_features(epochs, fs=200):
    """
    Extracts features from EEG epochs.

    Parameters:
        epochs (dict): Dictionary of epochs where keys are (timestamp, marker) and values are DataFrames.
        fs (int): Sampling rate in Hz (default 200 Hz).

    Returns:
        pd.DataFrame: DataFrame with extracted features and labels.
    """
    features = []
    
    for (timestamp, event_type), epoch_df in epochs.items():
        feature_dict = {"Timestamp": timestamp, "Event Type": event_type}
        
        for ch in range(4):  # Assuming 4 EEG channels
            signal = epoch_df[f"EXG Channel {ch}"].values
            
            # Time-Domain Features
            feature_dict[f"Ch{ch}_Mean"] = np.mean(signal)
            feature_dict[f"Ch{ch}_Variance"] = np.var(signal)
            feature_dict[f"Ch{ch}_Skewness"] = skew(signal)
            feature_dict[f"Ch{ch}_Kurtosis"] = kurtosis(signal)
            feature_dict[f"Ch{ch}_RMS"] = np.sqrt(np.mean(signal ** 2))
            
            # Frequency-Domain Features
            freqs, psd = welch(signal, fs=fs, nperseg=len(signal))
            feature_dict[f"Ch{ch}_Delta"] = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
            feature_dict[f"Ch{ch}_Theta"] = np.sum(psd[(freqs >= 4) & (freqs < 8)])
            feature_dict[f"Ch{ch}_Alpha"] = np.sum(psd[(freqs >= 8) & (freqs < 13)])
            feature_dict[f"Ch{ch}_Beta"] = np.sum(psd[(freqs >= 13) & (freqs < 30)])
            feature_dict[f"Ch{ch}_Gamma"] = np.sum(psd[(freqs >= 30)])
        
        features.append(feature_dict)

    return pd.DataFrame(features)

df_features = extract_features(all_epochs)

# Extract features (everything except Timestamp & Event Type)
X = df_features.drop(columns=["Timestamp", "Event Type"])
y = df_features["Event Type"]  # Labels

# Normalize feature values (important for consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training (90%) and testing (10%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# Initialize model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

print("Model training complete!")

# Predict on test set
y_pred = rf_model.predict(X_test)

# Print accuracy and detailed report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))