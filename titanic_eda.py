# -*- coding: utf-8 -*-

# Below is an example of Exploratory Data Analysis (EDA) using the publicly available Titanic dataset. The goal is to show how to perform a comprehensive EDA process and, ultimately, how to prepare data for Machine Learning models.


# Import pandas and numpy and matplotlib and seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the csv, change encoding, set first column as index
titanic = pd.read_csv("Titanic-Dataset.csv", encoding = 'ISO-8859-1')

#Preliminary Analysis

# Check the general shape of the dataframe
print(titanic.head())
print(titanic.shape)

# Get the number and names of columns in the dataframe"""
print(titanic.columns)
# Get to know types of data stored in each column. Can be either a number (int64, float64) or string (object)"""
print(titanic.info())

# Basic statistics for each coulumn's data: count, mean, min, max, standard deviation and quartiles"""
titanic.describe()
# Check for missing values in each column"""
titanic.isna().sum()
# Transform the above into percentage values"""
titanic.isna().sum() / len(titanic)*100
# Check for duplicates"""
titanic.duplicated().sum()
# Show the distribution of the target variable (the most important one in the dataframe)"""
titanic["Survived"].value_counts()

# Some more complex analyses. linking the target variable (survival) to other columns"""
age = sns.FacetGrid(titanic, col="Survived")
age.map(plt.hist, "Age", bins=10)
sex = sns.FacetGrid(titanic, col="Survived")
sex.map(plt.hist, "Sex", bins=10)
sex = sns.FacetGrid(titanic, col="Survived")
sex.map(plt.hist, "Pclass", bins=10)
sex = sns.FacetGrid(titanic, col="Survived")
sex.map(plt.hist, "Fare", bins=10)

# A bit more complex correlations are also possible"""
age = sns.FacetGrid(titanic, col="Survived", row="Sex")
age.map(plt.hist, "Age", bins=10)

# Distinguish between different types of columns"""
string_columns = titanic.select_dtypes(include="object").columns
print("String columns: ", string_columns)
numerical_columns = titanic.select_dtypes(include=["int64","float64"]).columns
print("Numerical columns: ", numerical_columns)

# Check the distribution of numerical columns"""
# automate ploting: for each numerical column run the same countplot instructions
plt.figure(figsize=(40,30), dpi=100)
for num, col in enumerate(list(numerical_columns), 1):
  sns.countplot(x=titanic[col])
  plt.title(col)
  plt.show()

# Same for categorical variables"""
# automate ploting: for each categorical column run the same countplot instructions
plt.figure(figsize=(40,30), dpi=100)
for num, col in enumerate(list(string_columns), 1):
  sns.countplot(x=titanic[col])
  plt.title(col)
  plt.show()

# Prepare the Data for Machine Learning purposes
# Remove irrelevant columns
titanic = titanic.drop(columns=["PassengerId", "Name", "Cabin", "Ticket"], axis=1)
titanic.head()
# Replace categorical values with numerical values"""
titanic["Sex"] = titanic["Sex"].map({"male": 1, "female": 0})
# alternative: titanic["Sex"] = titanic["Sex"].replace({"male": 1, "female": 0})
titanic.head()
titanic["Embarked"] = titanic["Embarked"].replace({"S": 1, "C": 2, "Q":3})
titanic.head()

# Show Pearson's Correlations"""
titanic.corr()

# Exclude less rellevant correlations"""
corr_matrix = titanic.corr()
corr_matrix[abs(corr_matrix) < 0.5] = np.nan
corr_matrix

# Negative correlation indicates a relationship between an increase in one value and a decrease in another.
# For data modeling purposes, variables that are highly correlated should be removed. A correlation of 0.5 is generally okay.

# Finding extreme values (Outliers)
outliers = ["Age", "Fare", "Pclass", "SibSp", "Parch", "Embarked"]

# Seaborn comes with a ready-made outlier visualization tool"""
plt.figure(figsize=(10,6))
for num, col in enumerate(list(outliers), 1):
  sns.boxplot(x=titanic[col])
  plt.title(col)
  plt.show()

"""A few words of commentary on the above graphs:
1. The upper quartile Q3 is the upper edge of the box
2. The lower quartile Q1 is the lower edge of the box
3. The median (i.e. the second quartile) is the line inside the box. This box is the interquartile range called the IQR (25%-75%)

Whiskers:
1. The upper whisker is 1.5*IQR from the upper quartile. Values above this range are potential outliers
2. The lower whisker is 1.5*IGR from the lower quartile. Values ​​below this range are potential outliers

Data cleaning for the titanic dataframe may look as below:
"""

# Filling the gaps in the Age and Embarked columns with median values
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].median())

# Removing Outliers
numerical_cols = titanic.select_dtypes(include=[np.number])

def detect_outlier(dataset, n, features):
  outlier_indices = []
  for col in features:
    Q1 = np.percentile(dataset[col], 25)
    Q3 = np.percentile(dataset[col], 75)
    IQR = Q3 - Q1
    outlier_step = 1.5 * IQR
    lower_bound = Q1 - outlier_step
    upper_bound = Q3 + outlier_step
    outlier_index_list = dataset[(dataset[col] < lower_bound) | (titanic[col] > upper_bound)].index
    outlier_indices.extend(outlier_index_list)
  outlier_indices = pd.Series(outlier_indices)
  # Return the n most frequently occurring outlier indexes
  return outlier_indices.value_counts().index[:n]

outliers_to_drop = detect_outlier(numerical_cols, 5, numerical_cols.columns)
print("Indexes of outliers to remove: ", outliers_to_drop)

# Remove the outliers and reset idexes of the remaining dataframe elements
titanic = titanic.drop(outliers_to_drop, axis=0).reset_index(drop=True)

"""# Conclusions

A data cleaning operation consists of:

1. Elimination of missing data. This can be done by removing rows/columns or by supplementing them with average 
values (or otherwise calculated)
2. Transforming categorical data (e.g. into numerical values)
3. Removing outliers (because they can mess up modeling)

Proper data cleaning is crucial to obtaining correct data analysis and modeling results. Properly cleaned data (as shown in the example above) can increase the effectiveness of predictive models and improve the quality of conclusions.
"""