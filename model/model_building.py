import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os


# 1. Load the Wine dataset
def load_data():
    wine = load_wine()
    data = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    data["cultivar"] = wine.target
    return data


# 2. Preprocessing & Feature Selection
def preprocess_and_train(data):
    # Selected features based on the prompt instructions (any 6)
    selected_features = [
        "alcohol",
        "flavanoids",
        "color_intensity",
        "hue",
        "od280/od315_of_diluted_wines",
        "proline",
    ]

    X = data[selected_features]
    y = data["cultivar"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline with Scaling and Random Forest
    # Scaling is mandatory as per requirements
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    # Train the model
    print("Training Random Forest Model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return pipeline


def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"\nModel saved to {filepath}")


if __name__ == "__main__":
    # Ensure model directory exists
    if not os.path.exists("model"):
        os.makedirs("model")

    df = load_data()
    model = preprocess_and_train(df)
    save_model(model, "model/wine_cultivar_model.pkl")
