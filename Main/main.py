import receiveData
from pylsl import StreamInlet, resolve_byprop
import pandas as pd
import threading
import filterData
import featureExtraction
import pickle

with open("model.pkl", "rb") as f:
    new_model = pickle.load(f)

# Need python threading. 
# Always be receiving new data
# Additionally, analyse the data every x seconds
# Return guessed event

df_columns = ["EXG Channel 0", "EXG Channel 1","EXG Channel 2", "EXG Channel 3", "Timestamp"]
df_data = {}

for x in range(len(df_columns)):
    df_data[df_columns[x-1]] = []
    
# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_byprop("type", "EEG")

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

while True:
    count = 0
    timestampStart = 0
    while count < 300:
        sample, timestamp = inlet.pull_sample()
        #print(timestamp, sample)
        if count == 0:
            timestampStart = timestamp

        for x in range(4):
            df_data[df_columns[x]].append(sample[x])
        df_data["Timestamp"].append(timestamp)
        count = count+1
        
    sample_df = pd.DataFrame(data = df_data)

    # print(sample_df)

    # Clean data
    cleaned_df =filterData.applyFilters(sample_df)

    formatForExtraction = {}
    formatForExtraction[(timestampStart, -1)] = cleaned_df

    features = featureExtraction.extract_features(formatForExtraction)

    features = features.drop(columns = ["Timestamp", "Event Type"])
    new_predict = new_model.predict(features)

    print(new_predict)