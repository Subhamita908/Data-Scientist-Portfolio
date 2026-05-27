# =========================================================
# CUSTOMER CHURN PREDICTION PIPELINE
# =========================================================
# Professional Modular ML Training Pipeline
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================

import pandas as pd
import numpy as np

from pathlib import Path

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# CONFIGURATION
# =========================

DATA_PATH = Path("data/churn.csv")

TARGET_COLUMN = "Churn"

RANDOM_STATE = 42

TEST_SIZE = 0.2

# =========================================================
# LOAD DATA
# =========================================================

def load_data(path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """

    df = pd.read_csv(path)

    return df


# =========================================================
# CLEAN DATA
# =========================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataset and fix datatypes.
    """

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    return df


# =========================================================
# PREPROCESS DATA
# =========================================================

def preprocess_data(df: pd.DataFrame):

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Encode target variable
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({
        "Yes": 1,
        "No": 0
    })

    # Features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Separate column types
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    # Numerical pipeline
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    # Column transformer
    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # One-hot encoding
    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y


# =========================================================
# SPLIT DATA
# =========================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# =========================================================
# SCALE DATA
# =========================================================

def scale_data(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    return model


# =========================================================
# TRAIN XGBOOST
# =========================================================

def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    return model


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    f1 = f1_score(y_test, predictions)

    print("\n==============================")
    print("MODEL EVALUATION")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # Load dataset
    df = load_data(DATA_PATH)

    print("Dataset Loaded Successfully")
    print(df.shape)

    # Clean dataset
    df = clean_data(df)

    # Preprocess
    X, y = preprocess_data(df)

    print("\nPreprocessing Completed")
    print(X.shape)

    # Train/Test Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTrain/Test Split Completed")

    # Scaling
    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )

    print("\nScaling Completed")

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    print("\n==============================")
    print("RANDOM FOREST")
    print("==============================")

    rf_model = train_random_forest(
        X_train_scaled,
        y_train
    )

    evaluate_model(
        rf_model,
        X_test_scaled,
        y_test
    )

    # =====================================================
    # XGBOOST
    # =====================================================

    print("\n==============================")
    print("XGBOOST")
    print("==============================")

    xgb_model = train_xgboost(
        X_train_scaled,
        y_train
    )

    evaluate_model(
        xgb_model,
        X_test_scaled,
        y_test
    )


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":
    main()