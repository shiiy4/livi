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

# قائمة الكلمات المفتاحية (الإجابات السلبية)
negative_keywords = [
    "Mentally drained", "Exhausted", "Overwhelmed",
    "Always a struggle", "Too much effort", "Overwhelming",
    "Never rested", "Constantly tired", "Drained",
    "Heavy body", "Worn out", "Drained quickly",
    "No motivation", "Dreading it", "Low energy",
    "Lost drive", "Can’t keep up", "Lacking energy",
    "Easily fatigued", "No stamina", "Emotionally strained",
    "Burned out", "Unmotivated", "No passion", "Disengaged",
    "Detached", "Uncaring", "Apathetic", "Cynical",
    "Doesn’t matter", "Unimportant", "Distracted",
    "Unfocused", "Easily sidetracked", "Foggy mind",
    "Confused", "Unclear", "Forgetful", "Lose focus",
    "Easily distracted", "Frequent errors", "Not focused",
    "Emotional outbursts", "Irritable", "Unstable",
    "Not myself", "Changed", "Easily frustrated",
    "Quick-tempered", "Unexplained sadness", "Emotional", "Low mood"
]

# وظيفة حساب الإجابات السلبية والإيجابية
def calculate_answers(input_data):
    """
    حساب عدد الإجابات السلبية والإيجابية بناءً على الكلمات المفتاحية.
    """
    negative_count = 0
    positive_count = 0

    for answer in input_data.values():
        if answer in negative_keywords:
            negative_count += 1
        else:
            positive_count += 1

    return negative_count, positive_count

# وظيفة تحديد التشخيص بناءً على عدد الإجابات السلبية
def determine_diagnosis(negative_count):
    """
    تحديد التشخيص النهائي بناءً على عدد الإجابات السلبية.
    """
    if negative_count <= 7:
        return "Normal"
    elif 8 <= negative_count <= 10:
        return "Needs Monitoring"
    else:
        return "At Risk"

@app.route('/')
def home():
    return "Flask server is running successfully!"

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """
    تحليل بيانات استبيان الصحة النفسية وإرجاع التشخيص النهائي بناءً على الحساب.
    """
    try:
        # قراءة البيانات من الطلب
        data = request.json
        print(f"Received data: {data}")

        # حساب الإجابات السلبية والإيجابية
        negative_count, positive_count = calculate_answers(data)

        # تحديد التشخيص بناءً على عدد الإجابات السلبية
        diagnosis = determine_diagnosis(negative_count)

        # معالجة البيانات للنموذج (الجزء الخاص بالخوارزمية - لن يتم استخدامه في المخرجات)
        processed_data = pd.DataFrame([data])
        for column in processed_data.columns:
            processed_data[column] = processed_data[column].astype("category").cat.codes
        model.predict_proba(processed_data)  # فقط لاستخدام النموذج دون عرض النتيجة

        # إرجاع التشخيص النهائي
        return jsonify({
            "status": "success",
            "diagnosis": diagnosis,
            "negative_count": negative_count,
            "positive_count": positive_count
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
