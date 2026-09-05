"""
Create ETL pipeline for UCI Obesity Levels dataset

"""


#Import packages
import pandas as pd
import numpy as np
import ucimlrepo
from sqlalchemy import create_engine

#Import dataset into code from UCI respository
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)

  
# data (as pandas dataframes) 
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets

# combine X and y
obesity_levels = pd.concat([X, y], axis=1)

# inspect
print(obesity_levels.shape)

# clean and inspect
obesity_levels_cleaned = obesity_levels.drop_duplicates()
print(obesity_levels_cleaned.shape)

# load
engine = create_engine("sqlite:///../data/processed/obesity_levels.db")

obesity_levels_cleaned.to_sql(
    "obesity_levels",
    con=engine,
    if_exists="replace",
    index=False
)

# verify successful load reading df with pandas

obesity_levels_sql = pd.read_sql("obesity_levels", con=engine)
print(obesity_levels_sql.shape)

# verify successful drop of duplicate rows
print(obesity_levels_sql.duplicated().sum())