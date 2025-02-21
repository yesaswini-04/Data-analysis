import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = sns.load_dataset("iris")

# Display the first few rows
print("First five rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Basic statistics
print("\nBasic statistics of the dataset:")
print(df.describe())

# Filtering data (example: filtering 'setosa' species)
setosa_df = df[df["species"] == "setosa"]
print("\nFiltered dataset (Setosa species only):")
print(setosa_df.head())

# Grouping and aggregating data
grouped_data = df.groupby("species").agg({"sepal_length": ["mean", "max"], "petal_length": ["mean", "max"]})
print("\nGrouped and Aggregated Data:")
print(grouped_data)

# Data Visualization - Histogram
plt.figure(figsize=(6, 4))
sns.histplot(df["sepal_length"], bins=20, kde=True)
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Count")
plt.show()

# Data Visualization - Scatter Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
