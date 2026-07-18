import os
import joblib
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "dataset/flood.csv"


def load_dataset():
    """Load the flood dataset."""
    return pd.read_csv(DATASET_PATH)


def preprocess_data(df):
    """
    Convert FloodProbability into binary classes.
    0 = No Flood
    1 = Flood
    """

    # Create target column
    df["FloodClass"] = (df["FloodProbability"] >= 0.50).astype(int)

    # Features
    X = df.drop(["FloodProbability", "FloodClass"], axis=1)

    # Target
    y = df["FloodClass"]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(scaler, "saved_models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test