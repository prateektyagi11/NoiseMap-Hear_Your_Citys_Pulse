import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Expect a CSV prepared with columns: mfcc_0..mfcc_12, rms, zcr, label
DATA_CSV = "data/noise_features.csv"
MODEL_OUT = "../api/app/models/rf_noise_classifier.joblib"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    # find mfcc columns automatically
    feat_cols = [c for c in df.columns if c.startswith("mfcc_")] + ["rms","zcr"]
    X = df[feat_cols]
    y = df['label']
    return X, y

def train():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found. Prepare dataset CSV with features.")
    X, y = load_data(DATA_CSV)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(classification_report(y_test, pred))
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    print("Model saved to", MODEL_OUT)

if __name__ == "__main__":
    train()
