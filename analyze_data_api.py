from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
from flask_cors import CORS




app = Flask(__name__)

CORS(app)
@app.route('/')
def home():
    return "Flask server is running successfully!"


   
# Load pre-trained Random Forest model
model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    Analyze mental health survey data and return classification.
    """
    try:
        data = request.json  # Expecting JSON input
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        categories = ['Normal', 'Needs Monitoring', 'At Risk']
        result = categories[prediction[0]]
        return jsonify({"status": "success", "classification": result}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
     app.run(debug=True, host="0.0.0.0", port=5001)
  