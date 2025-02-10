import numpy as np
import tensorflow as tf
import joblib  # For loading preprocessing objects
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained deep learning model
model = tf.keras.models.load_model('descripp.h5')

# Load preprocessing objects (scaler & label encoder)
scaler_Y = joblib.load("scaler_Y.pkl")  # Used for inverse scaling
le_Description = joblib.load("le_Description.pkl")  # Used for inverse label encoding

# Feature names as per the model input
FEATURE_NAMES = [
    "CustSpecID", "RegReqID", "ROHS", "ProductSafety", "OrderFormat", "BOARD_Category", "SUPPLYPANELS", 
    "NOOFPIECES_PERSUPPLYPANEL", "PCB_MATERIAL_ID", "FINISH", "HardGoldReq", "IMPEDANCE_CONTROL", "LAYER_COUNT", 
    "COPPER_THICKNESSDETAILS", "PCB_THICKNESS", "SINGLEDIM_X", "SINGLEDIM_Y", "SUPPLYDIM_X", "SUPPLYDIM_Y", 
    "MINHOLE_DIAMETER", "EdgeCate", "EdgeMilling", "EdgePlating", "Press", "Panel"
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        input_values = []

        for feature in FEATURE_NAMES:
            if feature in input_data:
                input_values.append(float(input_data[feature]))  # Convert to float
            else:
                return f"Error: Missing feature {feature} in input", 400
        
        # Convert to NumPy array and reshape
        final_features = np.array([input_values])

        # Make prediction
        predicted_value = model.predict(final_features)

        # (i) Apply inverse scaling to get an integer
        predicted_integer = int(np.round(scaler_Y.inverse_transform(predicted_value)[0][0]))

        # (ii) Convert integer to label (inverse label encoding)
        predicted_description = le_Description.inverse_transform([predicted_integer])[0]

        return render_template('home.html', prediction_text=f"Predicted Description: {predicted_description}")

    except Exception as e:
        return str(e), 500

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)

        input_values = []
        for feature in FEATURE_NAMES:
            if feature in data:
                input_values.append(float(data[feature]))
            else:
                return jsonify({"error": f"Missing feature: {feature}"}), 400

        final_features = np.array([input_values])
        predicted_value = model.predict(final_features)

        # (i) Apply inverse scaling
        predicted_integer = int(np.round(scaler_Y.inverse_transform(predicted_value)[0][0]))

        # (ii) Convert integer to label (inverse label encoding)
        predicted_description = le_Description.inverse_transform([predicted_integer])[0]

        return jsonify({"predicted_description": predicted_description})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
