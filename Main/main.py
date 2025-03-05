import receiveData
from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import threading
import filterData
import featureExtraction
import pickle
from collections import deque 
from sklearn.preprocessing import StandardScaler

markers = {-1:"Idle", 1: "Blink", 2: "Left", 3: "Right", 4: "Jaw", 5: "Up", 6:"Down", 7: "Happy", 8: "Frustrated"}

with open("model.pkl", "rb") as f:
    new_model = pickle.load(f)
with open("scaler.pkl", "rb") as f: # Must use the same scaler as the one used for creating the model
    scaler = pickle.load(f)

# Need python threading. 
# Always be receiving new data
# Additionally, analyse the data every x seconds
# Return guessed event

df_columns = ["EXG Channel 0", "EXG Channel 1","EXG Channel 2", "EXG Channel 3", "Timestamp"]

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_byprop("type", "EEG")

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])


# Define parameters
window_size = 200  # 1 second of data ( 200 Hz)
step_size = 50     # overlap
buffer = deque(maxlen=window_size)  # Rolling buffer, maxlen means when a new thing is added, the oldest item is popped if it's over the max len

df_columns = ["EXG Channel 0", "EXG Channel 1", "EXG Channel 2", "EXG Channel 3", "Timestamp"]

# Streaming loop
while True:
    # Collect new samples
    for _ in range(step_size):  # Only collect step_size samples at a time
        sample, timestamp = inlet.pull_sample()
        buffer.append(sample + [timestamp])  # Append data with timestamp

    # Convert buffer to DataFrame
    sample_df = pd.DataFrame(buffer, columns=df_columns)

    # Clean data
    cleaned_df = filterData.applyFilters(sample_df)

    # Prepare format for feature extraction
    formatForExtraction = {(sample_df["Timestamp"].iloc[0], -1): cleaned_df}
    features = featureExtraction.extract_features(formatForExtraction)

    # Preprocess & predict
    features = features.drop(columns=["Timestamp", "Event Type"], errors="ignore")
    features = scaler.transform(features)
    prediction = new_model.predict(features)

    # Print results
    if prediction[0] != -1:
        print(markers[prediction[0]], new_model.predict_proba(features))