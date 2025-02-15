import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed dataset
df = pd.read_csv("data/Processed_Dataset.csv")
print(df.columns)

# Basic information
print("Dataset Overview:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Plot Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], bins=50, kde=True)
plt.title("Flight Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["price"])
plt.title("Boxplot of Flight Prices")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Price Variation by Airline
sns.boxplot(x="airline_Indigo", y="price", data=df) # Change 'airline_IndiGo' to actual column name
plt.title("Price Variation by Airline")
plt.xticks(rotation=45)
plt.show()

# Price Trends by Source City
plt.figure(figsize=(10, 6))
sns.boxplot(x="source_city_Delhi", y="price", data=df) # Change 'source_city_Bangalore' to actual column
plt.title("Price Trends by Source City")
plt.xticks(rotation=45)
plt.show()

# Price based on Class
plt.figure(figsize=(8, 5))
sns.boxplot(x="class_Economy", y="price", data=df)  # Change 'class_Economy' to actual column name
plt.title("Price Variation by Class")
plt.show()
