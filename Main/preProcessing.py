from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import FastICA

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

# Train Model
rf_model, X_test, y_test = trainModel.trainModel(df_features, len(all_epochs))

# Predict on test set
y_pred = rf_model.predict(X_test)

# Print accuracy and detailed report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

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
