import os
import joblib
import matplotlib.pyplot as plt # type: ignore
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier # type: ignore
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocessing import load_dataset, preprocess_data

# Create output folders
os.makedirs("saved_models", exist_ok=True)
os.makedirs("static/graphs", exist_ok=True)

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = load_dataset()

X_train, X_test, y_train, y_test = preprocess_data(df)

# ------------------------------
# Models
# ------------------------------

models = {

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "XGBoost":
        XGBClassifier(
            eval_metric='logloss',
            random_state=42
        )
}

accuracy_scores = {}

best_accuracy = 0
best_model = None
best_name = ""

print("\nTraining Models...\n")

for name, model in models.items():

    print("-" * 50)
    print(f"Training {name}")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    accuracy_scores[name] = accuracy

    print(f"Accuracy : {accuracy:.4f}\n")

    print("Classification Report\n")

    print(classification_report(y_test, prediction))

    cm = confusion_matrix(y_test, prediction)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot()

    plt.title(name)

    plt.savefig(f"static/graphs/{name}.png")

    plt.close()

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_name = name

# ------------------------------
# Save Best Model
# ------------------------------

joblib.dump(
    best_model,
    "saved_models/flood_model.pkl"
)

print("\nBest Model Saved Successfully!")

print(f"Best Model : {best_name}")

print(f"Accuracy : {best_accuracy:.4f}")

# ------------------------------
# Accuracy Comparison Graph
# ------------------------------

plt.figure(figsize=(8,5))

plt.bar(
    accuracy_scores.keys(),
    accuracy_scores.values()
)

plt.ylabel("Accuracy")

plt.title("Model Comparison")

plt.savefig(
    "static/graphs/model_comparison.png"
)

plt.show()

print("\nTraining Completed Successfully.")