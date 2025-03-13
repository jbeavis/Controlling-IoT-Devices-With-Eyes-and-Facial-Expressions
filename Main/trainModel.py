from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

import pickle

def trainModel(df_features, totalEpochs):
    # Separate features from markers for machine learning
    X = df_features.drop(columns=["Timestamp", "Event Type"])
    y = df_features["Event Type"]  # Labels

    # Undersample -- I blink far more often than I do anything else during recordings (I can't help it) but this skews things
    #sampling_strategy={1: totalEpochs//9}, 
    undersampler = RandomUnderSampler(random_state=42) 
    X, y = undersampler.fit_resample(X, y)

    # Normalize feature values (important for consistency)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler (for real time prediction laters)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # Random seed

    print(f"Training samples: {len(X_train)}, testing samples: {len(X_test)}")

    # Initialize model
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_features="log2", max_depth=10, min_samples_split=2, class_weight={-1:5,1:1,2:1,3:1,4:1,5:1,6:1,7:1,8:1,9:1}) # More estimators = more accurate? but slower

    # Train the model
    rf_model.fit(X_train, y_train)
    print("Model training complete! :)\n")
    
    # Perform Grid Search with 5-fold cross-validation
    grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the model on your training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

    return rf_model, X_test, y_test
