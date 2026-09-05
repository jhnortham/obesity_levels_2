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

# load from SQLite for model
engine = create_engine("sqlite:///../data/processed/obesity_levels.db")

# read sql db
obesity_levels_sql = pd.read_sql("obesity_levels", con=engine)

obesity_levels_sql.columns = obesity_levels_sql.columns.astype(str)

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
X[categorical_features] = X[categorical_features].astype("object")

print(type(X))
print(X.dtypes)
print(X.columns.tolist())


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
X_train_encoded = pd.get_dummies(
    X_train,
    columns=categorical_features
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=categorical_features
)

# align test columns to training columns
X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)

X_train_encoded.columns = [str(col) for col in X_train_encoded.columns]
X_test_encoded.columns = [str(col) for col in X_test_encoded.columns]

print(set(type(col).__name__ for col in X_train_encoded.columns))

# inspect
print(X_train_encoded.shape)
print(X_test_encoded.shape)
print(X_train_encoded.columns.equals(X_test_encoded.columns))

# preprocessor = ColumnTransformer(
#    transformers=[
#        ("num", "passthrough", numeric_features),
#        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
#    ]
#)

#print("Testing preprocessor directly...")
#X_train_transformed = preprocessor.fit_transform(X_train)
#print(X_train_transformed.shape)

rf_model = RandomForestClassifier(
    random_state=123
)

#model = Pipeline(
#    steps=[
#        ("preprocessor", preprocessor),
#        ("classifier", rf_model)
#    ]
#)



rf_model.fit(X_train_encoded, y_train)

y_pred = rf_model.predict(X_test_encoded)

print("Accuracy: ", accuracy_score(y_test, y_pred)) 

print("Precision: ", precision_score(y_test, y_pred, average="macro")) 

print("Recall: ", recall_score(y_test, y_pred, average="macro")) 

print("F1: ", f1_score(y_test, y_pred, average="macro")) 

print("Confusion Matrix: ") 

print(confusion_matrix(y_test, y_pred)) 

    
