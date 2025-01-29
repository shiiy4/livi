import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1. قراءة البيانات
data_file = "answers_classification_dataset.csv"  # ملف الإجابات المصنفة مسبقًا
df = pd.read_csv(data_file)

# التحقق من شكل البيانات
print("First 5 rows of the dataset:")
print(df.head())

# 2. فصل الميزات (الإجابات) والهدف (التصنيف)
X = df["Answer"]  # الإجابة النصية
y = df["Label"]   # التصنيف (Positive/Negative)

# 3. تحويل النصوص إلى ميزات عددية باستخدام TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# 4. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)

# 5. تدريب نموذج Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# 6. تقييم النموذج
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. حفظ النموذج و TF-IDF Vectorizer لاستخدامهما لاحقًا في API
joblib.dump(model, "random_forest_model.pkl")  # حفظ النموذج في ملف جديد
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")  # حفظ TF-IDF Vectorizer

print("\n✅ Model and vectorizer saved successfully as 'random_forest_model.pkl' and 'tfidf_vectorizer.pkl'!")
