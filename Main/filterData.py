from scipy.signal import butter, filtfilt, iirnotch


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

def applyFilters(df):
    fs = 200  # Sample rate from OpenBCI metadata
    for channel in ["EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3"]:
        df[channel] = apply_bandpass_filter(df[channel], lowcut=1, highcut=50, fs=fs)
        df[channel] = apply_notch_filter(df[channel], notch_freq=50, fs=fs)  # 50hz for uk
        df[channel] = df[channel] - df[channel].mean() # DC offset removal, faster than high pass
    return df
