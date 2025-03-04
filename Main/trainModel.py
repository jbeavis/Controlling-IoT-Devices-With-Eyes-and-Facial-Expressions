from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def trainModel(df_features):
    # Separate features from markers for machine learning
    X = df_features.drop(columns=["Timestamp", "Event Type"])
    y = df_features["Event Type"]  # Labels

    # Normalize feature values (important for consistency)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training (90%) and testing (10%) sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Initialize model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    print("Model training complete!")
    
    return rf_model, X_test, y_test
