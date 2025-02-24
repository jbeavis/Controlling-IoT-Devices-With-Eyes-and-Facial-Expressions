import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import numpy as np

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
