print("Starting Flask App...")

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        year = int(request.form["year"])
        km = float(request.form["km"])
        fuel_petrol = 1 if request.form["fuel"] == "Petrol" else 0
        transmission_manual = 1 if request.form["transmission"] == "Manual" else 0
        owner_second = 1 if request.form["owner"] == "Second" else 0

        # IMPORTANT: match training feature order
        input_data = np.array([[year, km, transmission_manual, owner_second, fuel_petrol, 0, 0]])

        prediction = model.predict(input_data)[0]

        return render_template("index.html", result=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    print("Running Flask Server...")
    app.run(debug=True)
