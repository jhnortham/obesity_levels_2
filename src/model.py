"""
Create model for Obesity Levels ETL Pipeline

"""

#Import packages
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV
import joblib

# load from SQLite for model
engine = create_engine("sqlite:///../data/processed/obesity_levels.db")

# read sql db
obesity_levels_sql = pd.read_sql("obesity_levels", con=engine)

obesity_levels_sql.columns = obesity_levels_sql.columns = [
    str(col) for col in obesity_levels_sql.columns
]

print(obesity_levels_sql.shape)


# separate variables from target
X = obesity_levels_sql.drop(["NObeyesdad", "Height", "Weight"], axis=1)
y = obesity_levels_sql["NObeyesdad"]

# inspect
print(X.shape)
print(y.shape)

# define features

numeric_features = [
    "Age",
    "NCP",
    "CH2O",
    "FAF",
    "TUE",
    "FCVC",
]

categorical_features = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]

# convert categorical features to object type (due to compatibility error with python version and sickit-learn)
#X[categorical_features] = X[categorical_features].astype("object")

#print(type(X))
#print(X.dtypes)
#print(X.columns.tolist())


# complete train_test split with 80% train and 20% split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=123,
    stratify=y
)

# inspect
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# create dummy variables for categorical features
#X_train_encoded = pd.get_dummies(
#    X_train,
#    columns=categorical_features
#)

#X_test_encoded = pd.get_dummies(
#    X_test,
#    columns=categorical_features
#)

# align test columns to training columns
#X_test_encoded = X_test_encoded.reindex(
#    columns=X_train_encoded.columns,
#    fill_value=0
#)

#X_train_encoded.columns = [str(col) for col in X_train_encoded.columns]
#X_test_encoded.columns = [str(col) for col in X_test_encoded.columns]

#print(set(type(col).__name__ for col in X_train_encoded.columns))

# inspect
#print(X_train_encoded.shape)
#print(X_test_encoded.shape)
#print(X_train_encoded.columns.equals(X_test_encoded.columns))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

#print("Testing preprocessor directly...")
#X_train_transformed = preprocessor.fit_transform(X_train)
#print(X_train_transformed.shape)

baseline_rf_model = RandomForestClassifier(
    random_state=123
)

baseline_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", baseline_rf_model)
    ]
)



baseline_model.fit(X_train, y_train)

baseline_y_pred = baseline_model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, baseline_y_pred)) 

print("Precision: ", precision_score(y_test, baseline_y_pred, average="macro")) 

print("Recall: ", recall_score(y_test, baseline_y_pred, average="macro")) 

print("F1: ", f1_score(y_test, baseline_y_pred, average="macro")) 

print("Confusion Matrix: ") 

print(confusion_matrix(y_test, baseline_y_pred)) 

print(baseline_rf_model.classes_)

# tune parameters

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=baseline_model,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Macro F1:", grid_search.best_score_)

# use tuned model on the test set

tuned_model = grid_search.best_estimator_
tuned_pred = tuned_model.predict(X_test)

# calculate metrics on tuned model


print("Tuned Accuracy: ", accuracy_score(y_test, tuned_pred)) 

print("Tuned Precision: ", precision_score(y_test, tuned_pred, average="macro")) 

print("Tuned Recall: ", recall_score(y_test, tuned_pred, average="macro")) 

print("Tuned F1: ", f1_score(y_test, tuned_pred, average="macro")) 

print("Tuned Confusion Matrix: ") 

print(confusion_matrix(y_test, tuned_pred)) 

# save tuned model serialized

joblib.dump(
    tuned_model,
    "../models/obesity_levels_rf_model.pkl"
)
    
