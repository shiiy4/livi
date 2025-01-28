import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib





file_path = "/Users/shahad./Desktop/livi/custom_classification_dataset.csv"  # مسار الملف
df = pd.read_csv(file_path)  # قراءة ملف CSV

# عرض أسماء الأعمدة
print("أسماء الأعمدة في الملف:")
print(df.columns)




# 2. فصل الميزات والهدف
X = df.drop(columns=["Classification"])  # الميزات
y = df["Classification"]  # الهدف

# 3. معالجة البيانات
def preprocess_data(df):
    """
    تحويل البيانات النصية إلى قيم عددية بنفس الطريقة المستخدمة أثناء التدريب.
    """
    for column in df.columns:
        df[column] = df[column].astype("category").cat.codes
    return df

X = preprocess_data(X)  # معالجة الميزات
y = y.astype("category").cat.codes  # تحويل الهدف إلى أرقام

# 4. تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. إنشاء نموذج الغابة العشوائية
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

# 6. تدريب النموذج
model.fit(X_train, y_train)

# 7. تقييم النموذج
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Needs Monitoring", "At Risk"]))

# 8. حفظ النموذج
joblib.dump(model, "random_forest_model.pkl")
print("✅ Model trained and saved as 'random_forest_model.pkl'")
