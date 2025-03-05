from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def trainModel(df_features):
    # Separate features from markers for machine learning
    X = df_features.drop(columns=["Timestamp", "Event Type"])
    y = df_features["Event Type"]  # Labels

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
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42,max_features="log2") # More estimators = more accurate? but slower

    # Train the model
    rf_model.fit(X_train, y_train)
    train_acc = rf_model.score(X_train, y_train)
    test_acc = rf_model.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.2f}")
    print(f"Testing Accuracy: {test_acc:.2f}")

    print("Model training complete! :)")
    
    return rf_model, X_test, y_test
