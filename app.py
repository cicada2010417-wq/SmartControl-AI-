
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load("smartcontrol_ai_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    input_data = pd.DataFrame({
        "facility_type": [data.get("facility_type", 0)],
        "floor_area": [data.get("floor_area", 500)],
        "num_employees": [data.get("num_employees", 20)],
        "operating_hours": [data.get("operating_hours", 12)],
        "avg_temperature": [data.get("avg_temperature", 25)],
        "month": [data.get("month", 6)],
        "monthly_consumption": [data.get("monthly_consumption", 5000)]
    })
    
    reduction_rate = model.predict(input_data)[0]
    monthly_kwh = data.get("monthly_consumption", 5000)
    monthly_cost = data.get("monthly_cost", 150000)
    
    saved_kwh = monthly_kwh * (reduction_rate / 100)
    saved_cost = monthly_cost * (reduction_rate / 100)
    saved_co2 = saved_kwh * 0.5
    
    return jsonify({
        "reduction_rate": round(reduction_rate, 1),
        "saved_kwh": round(saved_kwh, 0),
        "saved_cost": round(saved_cost, 0),
        "saved_co2": round(saved_co2, 0),
        "annual_savings": round(saved_cost * 12, 0)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "SmartControl AI v1.0"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
