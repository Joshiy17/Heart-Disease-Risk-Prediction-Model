# ----------------------------------------------------------------------------------
# Author: Yash Joshi
# Student no: 2016AB001096
# Description:
#   Script to train a Random Forest and Logistic Regression model on heart disease
#   data using SMOTE for class imbalance handling.
# ----------------------------------------------------------------------------------

import pandas as pd
import numpy as np ##For future use
import joblib

# Sklearn Imports
from sklearn.model_selection import train_test_split ##For future use
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# For handling class imbalance
from imblearn.over_sampling import SMOTE


def load_data(csv_file):
    """
    This will Load the dataset from the specified CSV file path.

    :param csv_file: Path to the CSV file containing the heart disease data.
    :return: Pandas DataFrame containing the loaded dataset.

    """
    print("Loading dataset from:", csv_file)
    df = pd.read_csv(csv_file)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def clean_data(df):
    """
    USing this we will Clean the dataset by filling missing values and correcting minor data-entry errors.

    :param df: Pandas DataFrame to clean.
    :return: Cleaned Pandas DataFrame.

    """
    print("Cleaning data...")
    # Fill numeric columns' NaNs with their respective means
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Fill non-numeric columns' NaNs with mode
    df.fillna(df.mode().iloc[0], inplace=True)

    # Fix common typos in 'Gender' column
    df["Gender"] = df["Gender"].replace({"Malee": "Male", "Femal": "Female"})

    print("Data cleaning complete.")
    return df


def preprocess_data(df):
    """
    Now we will preprocesses the data by creating dummy variables and separating features from labels.

    :param df: Pandas DataFrame.
    :return: Tuple (X, y) where X is feature matrix and y is the label vector.
    """

    print("Preprocessing data (creating dummies and splitting X/y)...")

    # One-hot encode categorical variables
    df = pd.get_dummies(
        df,
        columns=["Gender", "Smoking_History", "Hypertension", "Diabetes"],
        drop_first=True
    )

    # Separate target variable and drop irrelevant columns
    X = df.drop(columns=["Patient_ID", "Heart_Disease"])
    y = df["Heart_Disease"].apply(lambda val: 1 if val == "Yes" else 0)

    print("Preprocessing complete.")
    return X, y


def train_models(X, y):
    """
    We will balance the dataset with SMOTE, then trains both a Random Forest
    and a Logistic Regression model.

    :param X: Feature matrix.
    :param y: Label vector.
    :return: Tuple (rf_model, log_model) of trained models.
    """

    print("Balancing dataset using SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("Training Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_resampled, y_resampled)

    print("Training Logistic Regression model...")
    log_model = LogisticRegression(max_iter=500, random_state=42)
    log_model.fit(X_resampled, y_resampled)

    return rf_model, log_model


def main():
    # File path to CSV
    csv_path = "heart_disease_risk.csv"

    # Load, clean, and preprocess the data
    df_data = load_data(csv_path)
    df_data = clean_data(df_data)
    X, y = preprocess_data(df_data)

    # Train both Random Forest and Logistic Regression
    rf_model, log_model = train_models(X, y)

    # Save the trained models and the column names for future inference
    print("Saving models and column metadata...")
    joblib.dump(rf_model, "random_forest_model.pkl")
    joblib.dump(log_model, "logistic_regression_model.pkl")
    joblib.dump(X.columns, "columns.pkl")
    print("Models and metadata saved successfully.")


if __name__ == "__main__":
    main()


## '''''Model development and training to be continued''''