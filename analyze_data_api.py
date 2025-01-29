import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from flask import Flask, request, jsonify
import joblib

# إعداد Flask
app = Flask(__name__)

# 1. إعداد البيانات وتدريب النموذج
data_file = "answers_classification_dataset.csv"  # ملف الإجابات المصنفة مسبقًا
df = pd.read_csv(data_file)

# التحقق من شكل البيانات
print("First 5 rows of the dataset:")
print(df.head())

# فصل الميزات (الإجابات) والهدف (التصنيف)
X = df["Answer"]  # الإجابة النصية
y = df["Label"]   # التصنيف (Positive/Negative)

# تحويل النصوص إلى ميزات عددية باستخدام TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# تدريب نموذج Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# تقييم النموذج
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# حفظ النموذج و TF-IDF Vectorizer لاستخدامهما لاحقًا
joblib.dump(model, "answer_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\nModel and vectorizer saved successfully!")

# 2. إنشاء API باستخدام Flask
@app.route('/analyze', methods=['POST'])
def analyze_answers():
    try:
        # تحميل النموذج والمدخلات
        model = joblib.load("answer_classifier.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        
        # قراءة البيانات من الطلب
        data = request.json
        answers = data.get("answers")  # قائمة الإجابات من المستخدم

        if not answers or len(answers) != 20:
            return jsonify({
                "status": "error",
                "message": "Exactly 20 answers are required."
            }), 400

        # تحويل الإجابات إلى ميزات باستخدام TF-IDF
        features = vectorizer.transform(answers)

        # تصنيف كل إجابة
        predictions = model.predict(features)

        # حساب عدد الإجابات الإيجابية والسلبية
        positive_count = sum(1 for p in predictions if p == "Positive")
        negative_count = sum(1 for p in predictions if p == "Negative")

        # تحديد التشخيص بناءً على عدد الإجابات السلبية
        if negative_count <= 7:
            diagnosis = "Normal"
        elif 8 <= negative_count <= 10:
            diagnosis = "Needs Monitoring"
        else:
            diagnosis = "At Risk"

        # إعداد النتيجة
        result = {
            "status": "success",
            "positive_count": positive_count,
            "negative_count": negative_count,
            "diagnosis": diagnosis  # إضافة التشخيص النهائي
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
