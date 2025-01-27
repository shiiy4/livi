from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# إنشاء بيانات عشوائية
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# إنشاء نموذج الغابة العشوائية
model = RandomForestClassifier(n_estimators=100, random_state=42)

# تدريب النموذج
model.fit(X, y)

# التحقق من انتهاء التدريب
print("Training Completed!")  # إذا ظهرت هذه الرسالة، فالتدريب انتهى بنجاح.
