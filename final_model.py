import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data.csv")

# Features & target
X = df[["marks", "attendance", "study_hours", "assignments_score", "sleep_hours"]]
y = df["action"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

results = {}

print("\nModel Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    results[name] = (acc, f1)

    print(f"{name}: Accuracy = {acc:.2f}, F1 = {f1:.2f}")

# Graphs
model_names = list(results.keys())
accuracy_values = [results[m][0] for m in model_names]
f1_values = [results[m][1] for m in model_names]

plt.figure()
plt.bar(model_names, accuracy_values)
plt.title("Model Accuracy Comparison")
plt.show()

plt.figure()
plt.bar(model_names, f1_values)
plt.title("Model F1 Score Comparison")
plt.show()