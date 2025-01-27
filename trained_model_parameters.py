from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# إنشاء بيانات عشوائية
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# إنشاء نموذج الغابة العشوائية وتدريبه
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# طباعة أهمية الميزات
print("Feature Importances:", model.feature_importances_)

# طباعة عدد الأشجار
print("Number of Trees:", len(model.estimators_))
