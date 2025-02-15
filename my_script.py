import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/Clean_Dataset.csv")

# Print column names to verify
print("Columns in dataset:", df.columns.tolist())

# Drop unnecessary columns (only if they exist)
columns_to_drop = ["Unnamed: 0", "flight"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Convert categorical columns using One-Hot Encoding
df = pd.get_dummies(df, columns=["airline", "source_city", "destination_city", "class", "stops"], drop_first=True)

# Define a mapping for categorical time values to numerical hours
time_mapping = {
    "Early Morning": 5,
    "Morning": 9,
    "Afternoon": 14,
    "Evening": 18,
    "Night": 21,
    "Late Night": 23
}

# Map categorical times to numbers
df["departure_hour"] = df["departure_time"].map(time_mapping)
df["arrival_hour"] = df["arrival_time"].map(time_mapping)

# Fill missing values (if any category was not mapped)
df["departure_hour"] = df["departure_hour"].fillna(0)
df["arrival_hour"] = df["arrival_hour"].fillna(0)

# Drop original time columns (only if they exist)
time_columns = ["departure_time", "arrival_time"]
df.drop(columns=[col for col in time_columns if col in df.columns], inplace=True)

# Save the processed dataset
df.to_csv("data/Processed_Dataset.csv", index=False)

print("✅ Feature Engineering completed! Processed dataset saved.")
