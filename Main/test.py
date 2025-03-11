from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA
import pickle
import pandas as pd

import loadData
import filterData
import generateEpochs
import featureExtraction
import trainModel

# Function to classify based on already extracted features (added thresholding logic)
def classify_epoch(features_df, model, signal_threshold=0.01):
    # Ensure features_df is a row (not a string)
    if isinstance(features_df, pd.Series):
        rms_values = []
        
        # Check if any channel's RMS value is above the threshold
        for ch in [0, 1, 3]:  # channels you have used for feature extraction
            # Access the RMS value correctly
            rms = features_df[f"Ch{ch}_RMS"]
            rms_values.append(rms)
        
        # If any RMS value is above the threshold, classify as an event
        if any(rms > signal_threshold for rms in rms_values):
            # Drop Timestamp and Event Type columns and pass to model
            features_to_predict = features_df.drop(["Timestamp", "Event Type"])
            # Convert to a 2D array as model.predict expects 2D data
            predicted_label = model.predict([features_to_predict.values])
            return predicted_label[0]  # Return the prediction (single value)
        else:
            # If all RMS values are low, classify as idle
            return -1
    else:
        # Handle the case where features_df is not a Series
        raise ValueError("features_df should be a pandas Series (a single row of the DataFrame)")

# Function to classify a batch of epochs
def classify_epochs(epochs_features, model, signal_threshold=0.01):
    predictions = []
    
    for _, features_df in epochs_features.iterrows():
        # Call classify_epoch on each row of the DataFrame
        predicted_label = classify_epoch(features_df, model, signal_threshold)
        predictions.append(predicted_label)
    
    return predictions


# Load the data
df = loadData.loadData()

# Filter Data
df_cleaned = filterData.applyFilters(df)

# Extract event epochs
event_epochs = generateEpochs.generate_event_epochs(df_cleaned)
print(f"Created {len(event_epochs)} epochs with markers.")

# Extract idle epochs
idle_epochs = generateEpochs.generate_idle_epochs(df_cleaned, len(event_epochs))
print(f"Created {len(idle_epochs)} idle epochs")

# Combine epochs
all_epochs = {**event_epochs, **idle_epochs}
print(f"Created {len(all_epochs)} total epochs\n")

# Extract Features From epochs
df_features = featureExtraction.extract_features(all_epochs)

# Train Model
rf_model, X_test, y_test = trainModel.trainModel(df_features, len(all_epochs))

# Predict on the test set using the updated logic for idle detection
y_pred = classify_epochs(df_features, rf_model, signal_threshold=0.01)

# Print accuracy and detailed report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot it using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# with open("model.pkl", "rb") as f:
#     new_model = pickle.load(f)

# new_predict = new_model.predict(X_test)

# print(accuracy_score(y_test, new_predict))
