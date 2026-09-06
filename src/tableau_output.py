"""
Create the model output in a Tableau compatible form for dashboard visualizations

"""
# Import packages

import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import numpy as np

# load data
engine = create_engine("sqlite:///../data/processed/obesity_levels.db")

obesity_levels_sql = pd.read_sql("obesity_levels", con=engine)

obesity_levels_sql.columns = [
    str(col) for col in obesity_levels_sql.columns
]
# load model
model = joblib.load(
    "../models/obesity_levels_rf_model.pkl"
)

# inspect feature names and importance
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

feature_names = preprocessor.get_feature_names_out()
feature_importances = classifier.feature_importances_

print(feature_names)
print(len(feature_names))
print(feature_importances)
print(len(feature_importances))

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})

# clean prefixes
feature_importance_df["Feature"] = (
    feature_importance_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
)

# sort highest to lowest importance
feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

# inspect
print(feature_importance_df)

# export to csv for tableau
feature_importance_df.to_csv(
    "../data/processed/obesity_feature_importance.csv",
    index=False
)

# separate variables from target
X = obesity_levels_sql.drop(["NObeyesdad", "Height", "Weight"], axis=1)
y = obesity_levels_sql["NObeyesdad"]

# inspect
print(X.shape)
print(y.shape)


# split data in to 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# use fitted pipeline on held-out test set
y_pred = model.predict(X_test)

# build Tableau output DataFrame from test rows
tableau_output = X_test.copy()

tableau_output["Actual_Class"] = y_test
tableau_output["Predicted_Class"] = y_pred

tableau_output["Correct_Prediction"] = (
    tableau_output["Actual_Class"] == tableau_output["Predicted_Class"]
)

print(tableau_output.shape)
print(tableau_output.head())

# prediction confidence
prediction_probabilities = model.predict_proba(X_test)

tableau_output["Prediction_Confidence"] = (
    prediction_probabilities.max(axis=1)
)

print(tableau_output[[
    "Actual_Class",
    "Predicted_Class",
    "Correct_Prediction",
    "Prediction_Confidence"
]].head())

# save to CSV
tableau_output.to_csv(
    "../data/processed/obesity_tableau_predictions.csv",
    index=False
)

print("Tableau output shape: ", tableau_output.shape)
print("Missing values: ", tableau_output.isna().sum())



