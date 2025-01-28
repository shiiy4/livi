from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib
from flask_cors import CORS

# إعداد Flask
app = Flask(__name__)
CORS(app)

# تحميل النموذج المدرب
model = joblib.load('random_forest_model.pkl')
print("✅ Model loaded successfully")

# وظيفة معالجة البيانات (نفس ما استخدمته أثناء التدريب)
def preprocess_data(input_data):
    """
    تحويل البيانات النصية إلى قيم عددية بنفس الطريقة المستخدمة أثناء التدريب.
    """
    df = pd.DataFrame([input_data])
    for column in df.columns:
        df[column] = df[column].astype("category").cat.codes
    return df

@app.route('/')
def home():
    return "Flask server is running successfully!"

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    تحليل بيانات استبيان الصحة النفسية وإرجاع التصنيفات الثلاثة مع احتمالاتها.
    """
    try:
        # قراءة البيانات من الطلب
        data = request.json
        print(f"Received data: {data}")

        # معالجة البيانات
        processed_data = preprocess_data(data)

        # توقع الاحتمالات باستخدام النموذج
        probabilities = model.predict_proba(processed_data)[0]
        print(f"Predicted probabilities: {probabilities}")

        # التصنيفات الثلاثة
        categories = ['Normal', 'Needs Monitoring', 'At Risk']
        result = {categories[i]: round(probabilities[i], 2) for i in range(len(categories))}

        # إرجاع التصنيفات مع الاحتمالات
        return jsonify({
            "status": "success",
            "classification": result
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
  
