import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle

print("Starting model training...")

# Load dataset
df = pd.read_csv("dataset/cars.csv")

# ---------------- CLEANING ----------------
df["AskPrice"] = df["AskPrice"].str.replace("₹", "")
df["AskPrice"] = df["AskPrice"].str.replace(",", "")
df["AskPrice"] = df["AskPrice"].astype(float)

df["kmDriven"] = df["kmDriven"].str.replace("[^0-9.]", "", regex=True)
df["kmDriven"] = df["kmDriven"].astype(float)

# Remove unnecessary columns
df.drop(["PostedDate", "AdditionInfo", "model", "Brand"], axis=1, inplace=True)

# Remove outliers
df = df[df["AskPrice"] < df["AskPrice"].quantile(0.95)]
df = df[df["kmDriven"] < df["kmDriven"].quantile(0.95)]

# Handle missing values
df.dropna(inplace=True)

# Convert categorical → numeric
df = pd.get_dummies(df, drop_first=True)

# ---------------- SPLIT ----------------
X = df.drop("AskPrice", axis=1)
y = df["AskPrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- LINEAR REGRESSION ----------------
lr = LinearRegression()
lr.fit(X_train, y_train)

lr_score = lr.score(X_test, y_test)
print("Linear Regression Score:", lr_score)

# ---------------- RANDOM FOREST ----------------
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
rf_score = r2_score(y_test, y_pred)
print("Random Forest Accuracy (R2 Score):", rf_score)

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")
print("Training complete!")