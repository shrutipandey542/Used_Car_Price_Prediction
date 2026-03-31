# 🚗 Used Car Price Estimator

## 📌 Project Overview
The Used Car Price Estimator is a machine learning-based web application that predicts the selling price of a used car based on user inputs such as year, kilometers driven, fuel type, transmission, and ownership.

---

## ⚙️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (Random Forest Regression)
- Flask (Web Framework)
- HTML, CSS

---

## 🧠 Machine Learning Model
- Algorithm: Random Forest Regressor
- Data preprocessing:
  - Removed missing values
  - Cleaned price and km data
  - Removed outliers
  - Converted categorical data to numeric
- Model accuracy: ~0.56 R² Score

---

## 🌐 Features
- Predict car price instantly
- User-friendly web interface
- Real-world dataset used
- Fast and responsive

---

## 📂 Project Structure

car-price-estimator/
│── dataset/
│ └── cars.csv
│── templates/
│ └── index.html
│── app.py
│── train_model.py
│── model.pkl
│── README.md


---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn flask
2. Train model (optional)
python train_model.py
3. Run Flask app
python app.py
4. Open in browser
http://127.0.0.1:5000/
