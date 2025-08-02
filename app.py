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

#Let's impute missing values

# First let's see if there's a connection between the missing values and the target variable- if the missing values are random or not ( MNAR)
cols_with_missing = df_cleaned.columns[df_cleaned.isnull().any()].tolist()
for col in cols_with_missing:
        df_cleaned[f"{col}_missing"] = df_cleaned[col].isna().astype(int)

missing_flags= [col for col in df_cleaned.columns if col.endswith('_missing')]
df_cleaned.groupby('fraudulent')[missing_flags].mean().plot(kind='bar', figsize=(10, 5)) # This code will plot/ organize the missing values rate grouped by the fraudulent status
plt.xlabel("Fraudulent Status")
plt.ylabel("Missing Values Rate")
plt.xticks(rotation=0)
plt.legend(title="Columns with Missing Values")
plt.title("Missing Values by Fraudulent Status")
plt.show()


# Need to remember that the target variable is 'fraudulent' and it is a binary classification problem