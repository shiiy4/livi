import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# تحميل البيانات
test_file = "test_unified.csv"
test_data = pd.read_csv(test_file)

# تحديد الميزات والهدف
features = test_data.columns[:-1]
target = test_data.columns[-1]

X_test = test_data[features]
y_test = test_data[target]

# معالجة البيانات
def preprocess_data(df):
    for column in df.columns:
        df[column] = df[column].astype("category").cat.codes
    return df

X_test = preprocess_data(X_test)
y_test = y_test.astype("category").cat.codes

# تحميل النموذج
rf_model = joblib.load("random_forest_model.pkl")

# تقييم النموذج
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# تعريف الفئات بناءً على البيانات
unique_classes = sorted(set(y_test))
categories = {i: f"Class {i}" for i in unique_classes}

# طباعة تقرير التصنيف
print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=[categories[i] for i in unique_classes],
    labels=unique_classes
))

# حفظ التقرير
report = []
for i, prediction in enumerate(y_pred):
    report.append({
        "Employee_ID": i + 1,
        "Mental_Health_State": categories.get(prediction, "Unknown")
    })

with open("mental_health_report_rf.json", "w") as report_file:
    json.dump(report, report_file, indent=4)
print("\nComprehensive report saved as 'mental_health_report_rf.json'")
