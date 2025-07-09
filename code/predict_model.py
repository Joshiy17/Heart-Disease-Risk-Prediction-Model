# --------------------------------------------------------------------------------
# Description:
#   Loads pre-trained models for heart disease prediction, preprocesses a new
#   patient's data to match the training schema, and visualizes probability
#   comparisons with a radar chart. Also, tried to keep it friendly
#   (Users don't judge too harshly, it's a code file, after all :) ).
# --------------------------------------------------------------------------------


import pandas as pd
import joblib
import matplotlib.pyplot as plt
from math import pi


def preprocess_new_data(df_new_patient, columns):
    """
    Preprocesses new data by:
      1. Applying one-hot encoding to specific categorical features
      2. Ensuring the dataframe matches the training columns format

    :param df_new_patient: Pandas DataFrame containing the new patient data
    :param columns: List (or Index) of columns used in model training
    :return: DataFrame aligned to the model's expected features
    """
    print("Preprocessing new data... (The data is about to be squeaky clean!)")

    # One-hot encode certain columns
    df_new_patient = pd.get_dummies(
        df_new_patient,
        columns=["Gender", "Smoking_History", "Hypertension", "Diabetes"],
        drop_first=True
    )

    # Ensure all columns from training are present, adding missing ones as 0
    missing_cols = set(columns) - set(df_new_patient.columns)
    for col in missing_cols:
        df_new_patient[col] = 0

    # Reorder columns to exactly match the training set
    df_new_patient = df_new_patient[columns]

    return df_new_patient


###Note: Initially bar chart was also used,but due to coding limits, removed
def plot_algorithm_comparison_radar_chart(rf_probs, log_probs):
    """
    Plots a radar chart comparing model probabilities for "No Heart Disease" vs. "Heart Disease."

    :param rf_probs: Probability output from the Random Forest model for a single patient
    :param log_probs: Probability output from the Logistic Regression model for the same patient
    """
    print("Creating a fancy radar chart... (No stealth technology required!)")

    labels = ["No Heart Disease", "Heart Disease"]

    # Convert from NumPy arrays to lists and close the loop for radar visualization
    values_rf = rf_probs[0].tolist() + [rf_probs[0][0]]
    values_log = log_probs[0].tolist() + [log_probs[0][0]]

    # Calculate the angle for each axis in the radar chart
    angles = [n / float(len(labels)) * 2 * pi for n in range(len(labels))]
    angles += angles[:1]

    # Initialize a figure with polar coordinates
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111, polar=True)

    # Random Forest's line
    ax.plot(angles, values_rf, linewidth=2, linestyle='solid', label="Random Forest", color="blue")
    ax.fill(angles, values_rf, alpha=0.25, color="blue")

    # Logistic Regression's line
    ax.plot(angles, values_log, linewidth=2, linestyle='solid', label="Logistic Regression", color="red")
    ax.fill(angles, values_log, alpha=0.25, color="orange")

    # Formatting the chart
    plt.xticks(angles[:-1], labels, fontsize=10)
    ax.set_title("Radar Chart: Probability Comparison", fontsize=14, pad=20)
    ax.set_ylim(0, 1)
    plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2), fontsize=10)
    plt.tight_layout()
    plt.show()

    print(
        "Radar chart displayed! If you find yourself thinking, 'Hmm, this looks suspiciously scientific,' then our job here is done.")



def main():
    """
    Main function that will:
      - Load the saved Random Forest & Logistic Regression models
      - Load the column names used in training
      - Construct new patient data in a DataFrame
      - Preprocesses this new data
      - Predict heart disease risk (including probability)
      - Display a radar chart to compare the two models
    """
    print("Initializing the heart disease predictor... (Relax! No actual heart surgery happening, promise)")

    # Load models and column metadata
    rf_model = joblib.load("random_forest_model.pkl")
    log_model = joblib.load("logistic_regression_model.pkl")
    columns = joblib.load("columns.pkl")

    # Define new patient data — watch out values may look ominous...But Relax
    new_patient = pd.DataFrame({
        "Patient_ID": [81],
        "Age": [44],
        "Cholesterol_Level": [270],
        "Blood_Pressure": [120],
        "Heart_Rate": [90],
        "Gender": ["Female"],
        "Smoking_History": ["Yes"],
        "Hypertension": ["No"],
        "Diabetes": ["No"]
    })

    print("\nHere's the new patient's raw data. (Patient, let's hope they fare better than others...):")
    print(new_patient)

    # Preprocess new patient data to match model training format
    preprocessed_patient = preprocess_new_data(new_patient, columns)

    # Model predictions
    rf_prediction = rf_model.predict(preprocessed_patient)
    log_prediction = log_model.predict(preprocessed_patient)

    # Probability estimates
    rf_probs = rf_model.predict_proba(preprocessed_patient)
    log_probs = log_model.predict_proba(preprocessed_patient)

    # Output predictions
    print("\nPredictions:")
    print(f"Random Forest Prediction: {'Heart Disease' if rf_prediction[0] == 1 else 'No Heart Disease'}")
    print(f"Logistic Regression Prediction: {'Heart Disease' if log_prediction[0] == 1 else 'No Heart Disease'}")

    # Output probabilities
    print("\nProbabilities (Hey, at least computers are honest about their uncertainty!):")
    print(f"Random Forest Probabilities: {rf_probs[0]}")
    print(f"Logistic Regression Probabilities: {log_probs[0]}")

    # Compare probability distributions using a radar chart
    plot_algorithm_comparison_radar_chart(rf_probs, log_probs)

    print(
        "\nAll done! If this suggests 'Heart Disease,' remember I'm just a bunch of code. Always see a real healthcare professional.")


if __name__ == "__main__":
    main()


