import pandas as pd

# Load dataset
df = pd.read_csv("dataset/cars.csv")

#1. Clean AskPrice(Target)
df["AskPrice"] = df["AskPrice"].str.replace("₹","")
df["AskPrice"] = df["AskPrice"].str.replace(",","")
df["AskPrice"] = df["AskPrice"].astype(float)

#2. Clean kmDriven
df["kmDriven"] = df["kmDriven"].str.replace("km","")
df["kmDriven"] = df["kmDriven"].str.replace(",","")
df["kmDriven"] = df["kmDriven"].astype(float)

#3. Drop useless columns
df.drop(["PostedDate","AdditionInfo","model"], axis=1, inplace=True)

#4 Handle missing values
df.dropna(inplace=True)

#5 Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Final Check
print(df.head())
print(df.info())