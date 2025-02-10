import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import joblib  # For loading preprocessing objects

app = Flask(__name__)

# Load the trained deep learning model
model = tf.keras.models.load_model('descripp.h5')

# Load any necessary preprocessing steps (if applicable)
# Example: label_encoders = joblib.load("label_encoders.pkl")

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
                return "Error: Missing feature {} in input".format(feature), 400
        
        # Convert to NumPy array and reshape
        final_features = np.array([input_values])
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # Format output
        output = prediction[0].tolist()  # Convert NumPy array to list
        return render_template('home.html', prediction_text="Predicted Description: {}".format(output))
    
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
                return jsonify({"error": "Missing feature: {}".format(feature)}), 400
        
        final_features = np.array([input_values])
        prediction = model.predict(final_features)
        output = prediction[0].tolist()
        
        return jsonify({"predicted_description": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
