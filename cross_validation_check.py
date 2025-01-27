from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# إنشاء بيانات عشوائية
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# إنشاء نموذج الغابة العشوائية
model = RandomForestClassifier(n_estimators=100, random_state=42)

# التحقق المتقاطع
cv_scores = cross_val_score(model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Average CV score:", cv_scores.mean())
