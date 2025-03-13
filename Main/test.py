from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


import pickle

import loadData
import filterData
import generateEpochs
import featureExtraction
import trainModel

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

# # DEBUG: Save features for fast running later
# with open("features.pkl", "wb") as f:
#     pickle.dump(df_features, f)

# # DEBUG: Load features if saved
# with open("features.pkl", "rb") as f:
#     df_features = pickle.load(f)


# # TESTING: Binary model
# df_features_copy = df_features.copy()
# for i in range(len(df_features_copy)):
#     if df_features_copy.loc[i, "Event Type"] != -1:
#         df_features_copy.loc[i, "Event Type"] = 1

# binaryModel, X_test_binary, y_test_binary = trainModel.trainModel(df_features_copy, len(all_epochs))

# y_pred_binary = binaryModel.predict(X_test_binary)

# # Print accuracy and detailed report
# print(f"Accuracy for binary model: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
# print(classification_report(y_test_binary, y_pred_binary))


# Multi class model
df_no_idle = df_features[df_features["Event Type"] != -1]
# Train Model
rf_model, X_test, y_test = trainModel.trainModel(df_no_idle, len(all_epochs))

# Predict on test set
y_pred = rf_model.predict(X_test)

# Print accuracy and detailed report
print(f"Accuracy for multi class model: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# X = df_features.drop(columns=["Timestamp", "Event Type"])
# y = df_features["Event Type"]  # Labels

# # Undersample -- I blink far more often than I do anything else during recordings (I can't help it) but this skews things
# #sampling_strategy={1: totalEpochs//9}, 
# undersampler = RandomUnderSampler(random_state=42) 
# X, y = undersampler.fit_resample(X, y)

# # Normalize feature values (important for consistency)
# with open("scaler.pkl", "rb") as f: # Must use the same scaler as the one used for creating the model
#     scaler = pickle.load(f)
# X_scaled = scaler.fit_transform(X)

# # Split into training (80%) and testing (20%) sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Random seed

# finalPredictions = []
# for x in X_test:
#     x = x.reshape(1,-1)
#     binary_pred = binaryModel.predict(x)
#     if binary_pred == 1: 
#         pred = int(rf_model.predict(x)[0])  # Ensure proper format
#     else:
#         pred = -1  # No event detected
#     finalPredictions.append(pred)

# print(f"Accuracy overall: {accuracy_score(y_test, finalPredictions):.4f}")
# print(classification_report(y_test, finalPredictions))

# # Generate the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# # Plot it using seaborn
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.show()

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# with open("model.pkl", "rb") as f:
#     new_model = pickle.load(f)

# new_predict = new_model.predict(X_test)

# print(accuracy_score(y_test, new_predict))
