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

# الأسئلة المرتبطة بالإجابات
questions = [
    "Do you feel mentally exhausted at work?",
    "After a day at work, do you find it hard to recover your energy?",
    "Do you find that everything you do at work requires a great deal of effort?",
    "Do you feel physically exhausted at work?",
    "When you get up in the morning, do you lack the energy to start a new day at work?",
    "Do you want to be active at work but find it difficult to manage?",
    "Do you quickly get tired when exerting yourself at work?",
    "Do you feel mentally exhausted and drained at the end of your workday?",
    "Do you struggle to find enthusiasm for your work?",
    "Do you feel indifferent about your job?",
    "Are you cynical about the impact your work has on others?",
    "Do you have trouble staying focused at work?",
    "Do you struggle to think clearly at work?",
    "Are you forgetful and easily distracted at work?",
    "Do you have trouble concentrating while working?",
    "Do you make mistakes at work because your mind is on other things?",
    "Do you feel unable to control your emotions at work?",
    "Do you feel you no longer recognize yourself in your emotional reactions at work?",
    "Do you become irritable when things don't go your way at work?",
    "Do you feel sad or upset at work without knowing why?"
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
    elif 8 <= negative_count <= 9:
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

        # تحويل قائمة الإجابات إلى خريطة (سؤال -> إجابة)
        answers = {questions[i]: data["features"][i] for i in range(len(questions))}
        print(f"Processed Answers: {answers}")

        # حساب الإجابات السلبية والإيجابية
        negative_count, positive_count = calculate_answers(answers)

        # تحديد التشخيص بناءً على عدد الإجابات السلبية
        diagnosis = determine_diagnosis(negative_count)

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
