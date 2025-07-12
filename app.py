from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import json

app = Flask(__name__)

# Load model and columns
model = joblib.load("price_predictor.pkl")
with open("model_columns.json") as f:
    model_columns = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return "OK", 200


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        input_data = {
            "original_price": float(request.form["original_price"]),
            "inventory_level": int(request.form["inventory_level"]),
            "days_to_expiry": int(request.form["days_to_expiry"]),
            "historical_demand": float(request.form["historical_demand"]),
            "product_type": request.form["product_type"],
            "day_of_week": request.form["day_of_week"],
            "season": request.form["season"]
        }

        # Create zeroed DataFrame with correct columns
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

        # Set values
        input_df.at[0, 'original_price'] = input_data["original_price"]
        input_df.at[0, 'inventory_level'] = input_data["inventory_level"]
        input_df.at[0, 'days_to_expiry'] = input_data["days_to_expiry"]
        input_df.at[0, 'historical_demand'] = input_data["historical_demand"]

        # One-hot encode flags
        for col in model_columns:
            if col.endswith(input_data["product_type"]):
                input_df.at[0, col] = 1
            if col.endswith(input_data["day_of_week"]):
                input_df.at[0, col] = 1
            if col.endswith(input_data["season"]):
                input_df.at[0, col] = 1

        # Predict
        prediction = model.predict(input_df)[0]
        return render_template('index.html', prediction_text=f"Predicted Final Price: ₹{prediction:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
