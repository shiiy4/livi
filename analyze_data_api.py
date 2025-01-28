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
    تحليل بيانات استبيان الصحة النفسية وإرجاع التصنيف النهائي.
    """
    try:
        # قراءة البيانات من الطلب
        data = request.json
        print(f"Received data: {data}")

        # معالجة البيانات
        processed_data = preprocess_data(data)
        print(f"Processed data columns: {processed_data.columns.tolist()}")
        print(f"Expected model columns: {model.feature_names_in_.tolist()}")

        # التحقق من الأعمدة
        missing_columns = set(model.feature_names_in_) - set(processed_data.columns)
        if missing_columns:
            return jsonify({
                "status": "error",
                "message": f"Missing columns in input data: {missing_columns}"
            }), 400

        # توقع الاحتمالات باستخدام النموذج
        probabilities = model.predict_proba(processed_data)[0]
        print(f"Predicted probabilities: {probabilities}")

        # التصنيفات الثلاثة
        categories = ['Normal', 'Needs Monitoring', 'At Risk']

        # اختيار التصنيف بناءً على أعلى احتمال
        max_index = probabilities.argmax()
        final_classification = categories[max_index]

        # إرجاع التصنيف النهائي
        return jsonify({
            "status": "success",
            "classification": final_classification
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
