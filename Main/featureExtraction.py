import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import numpy as np

def extract_features(epochs, fs=200):
    features = []


    for (timestamp, event_type), epoch_df in epochs.items():
        feature_dict = {"Timestamp": timestamp, "Event Type": event_type}
        
        for ch in [0,1,3]:  # TODO Check if I need to remove unused channel or not
            signal = epoch_df[f"EXG Channel {ch}"].values
            
            # Time domain features
            feature_dict[f"Ch{ch}_Mean"] = np.mean(signal)
            feature_dict[f"Ch{ch}_Variance"] = np.var(signal)
            feature_dict[f"Ch{ch}_Skewness"] = skew(signal)
            feature_dict[f"Ch{ch}_Kurtosis"] = kurtosis(signal)
            feature_dict[f"Ch{ch}_RMS"] = np.sqrt(np.mean(signal ** 2))
            
            # Frequency domain features
            freqs, psd = welch(signal, fs=fs, nperseg=200)
            feature_dict[f"Ch{ch}_Delta"] = np.sum(psd[(freqs >= 0.5) & (freqs < 4)])
            feature_dict[f"Ch{ch}_Theta"] = np.sum(psd[(freqs >= 4) & (freqs < 8)])
            feature_dict[f"Ch{ch}_Alpha"] = np.sum(psd[(freqs >= 8) & (freqs < 13)])
            feature_dict[f"Ch{ch}_Beta"] = np.sum(psd[(freqs >= 13) & (freqs < 30)])
            feature_dict[f"Ch{ch}_Gamma"] = np.sum(psd[(freqs >= 30)])
        
        features.append(feature_dict)
    return pd.DataFrame(features)
