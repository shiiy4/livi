import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import joblib

# 1. تحميل بيانات التدريب
train_file = "train_unified.csv"
train_data = pd.read_csv(train_file)

# 2. تحديد الميزات والهدف
features = train_data.columns[:-1]
target = train_data.columns[-1]

X_train = train_data[features]
y_train = train_data[target]

# 3. معالجة البيانات
def preprocess_data(df):
    for column in df.columns:
        df[column] = df[column].astype("category").cat.codes
    return df

X_train = preprocess_data(X_train)
y_train = y_train.astype("category").cat.codes

# 4. معالجة اختلال التوازن في البيانات باستخدام Oversampling
def manual_oversample(X, y):
    if y.name is None:
        y.name = "target"
    combined = pd.concat([X, y], axis=1)
    oversampled_data = []
    for label in y.unique():
        class_data = combined[combined[y.name] == label]
        if len(class_data) < 10:  # عدد أدنى من العينات
            class_data = resample(class_data, replace=True, n_samples=10, random_state=42)
        oversampled_data.append(class_data)
    oversampled = pd.concat(oversampled_data)
    return oversampled.drop(columns=[y.name]), oversampled[y.name]

X_train_balanced, y_train_balanced = manual_oversample(X_train, y_train)

# 5. إنشاء نموذج الغابة العشوائية
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# 6. تدريب النموذج
rf_model.fit(X_train_balanced, y_train_balanced)

# 7. حفظ النموذج
joblib.dump(rf_model, "random_forest_model.pkl")
print("\nRandom Forest model saved as 'random_forest_model.pkl'")
