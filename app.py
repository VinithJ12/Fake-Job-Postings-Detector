#importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the dataset
df= pd.read_csv('fake_job_postings.csv')
# Displaying the first few rows of the dataset
print(df.head())
# Displaying the shape of the dataset
print("Shape of Dataset:", df.shape)
# Displaying the columns of the dataset
print("Columns of the Dataset:", df.columns)
# Displaying the data types of the columns
print("Data Types of the Columns:", df.dtypes)
# Displaying the summary statistics of the dataset
print("Summary Statistics of the Dataset:\n", df.describe()) # this won't be too helpful because the values in the columns are either 0 or 1
# Displaying the percentage of missing values in each column
print("Missing Values", df.isnull().sum()/len(df)*100)
# It's important to note that having good input data is crucial for building a good model (Garbage in, garbage out).
# Data Cleaning

# Let's first drop the job id column as it is not useful for our analysis ( let's copy the dataset first)
df_cleaned = df.copy()
# Now let's check for duplicate rows
duplicates = df_cleaned.duplicated().sum()
print("Number of duplicate rows:", duplicates)
# Dropping duplicate rows
# We will drop duplicates based on all columns except the 'job_id' column
df_cleaned = df.drop_duplicates(subset=df.columns.difference(['job_id']))
print(df_cleaned)
# Need to remember that the target variable is 'fraudulent' and it is a binary classification problem