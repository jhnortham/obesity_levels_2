"""
Import in Python Estimation of Obesity Levels Based On Eating Habits and Physical Condition

* Use .py file to import dataset into code
* view data using .py file

"""

#Import packages
import pandas as pd
import numpy as np
import ucimlrepo

#Import dataset into code from UCI respository
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)

  
# data (as pandas dataframes) 
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 

# inspect
print(type(X))
print(type(y))

print(X.shape)
print(y.shape)
  
# metadata 
#print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.metadata) 
  
# variable information 
print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.variables)

# Concatenate X and y for deeper inspection
obesity_levels = pd.concat([X, y], axis=1)

# Inspect
print(obesity_levels.shape)
print(obesity_levels.info())
print(obesity_levels["NObeyesdad"].value_counts())


# Inspect numeric features
print(obesity_levels.describe())

# Inspect categorical features
print(obesity_levels["Gender"].value_counts())

print(obesity_levels["family_history_with_overweight"].value_counts())

print(obesity_levels["FAVC"].value_counts())

print(obesity_levels["CAEC"].value_counts())

print(obesity_levels["SMOKE"].value_counts())

print(obesity_levels["SCC"].value_counts())

print(obesity_levels["CALC"].value_counts())

print(obesity_levels["MTRANS"].value_counts())

# Inspect for duplicated rows

print(obesity_levels.duplicated().sum())
print(obesity_levels[obesity_levels.duplicated(keep=False)])

# Clean df - remove duplicate rows to reduce risk of data leakage and overfitting and inspect

obesity_levels_cleaned = obesity_levels.drop_duplicates()
print(obesity_levels_cleaned.shape)
print(obesity_levels_cleaned.duplicated().sum())

# Inspect cleaned df for missingness

print(obesity_levels_cleaned.isna().sum())
print(obesity_levels_cleaned.describe)



