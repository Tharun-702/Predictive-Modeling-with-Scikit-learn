# ---------------------------
# Import Libraries
# ---------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import shap
import os

sns.set_theme(style="whitegrid")

# ---------------------------
# 1. Load and Prepare Dataset
# ---------------------------
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

print("\n🔍 Dataset Shape:", df.shape)
print("\n📌 First 5 Rows:")
print(df.head())

# ---------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------
def plot_eda(df):
    # Class Distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x='species', data=df, palette='Set2')
    plt.title("Class Distribution")
    plt.savefig("class_distribution.png")
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(6,4))
    sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.show()

    # Pairplot
    sns.pairplot(df, hue='species', palette='Set2', diag_kind='kde')
    plt.suptitle("Iris Feature Distribution", fontsize=14, y=1.02)
    plt.savefig("pairplot.png")
    plt.show()

plot_eda(df)

# ---------------------------
# 3. Feature and Target
# ---------------------------
X = df.drop('species', axis=1)
y = df['species']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 4. Model Training with GridSearchCV
# ---------------------------
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5],
    'min_samples_split': [2, 4]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"\n✅ Best Parameters: {grid_search.best_params_}")

# Cross-Validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print(f"📌 Cross-Validation Accuracy: {cv_scores.mean():.4f}")

# ---------------------------
# 5. Predictions and Evaluation
# ---------------------------
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# 6. Confusion Matrix (Absolute & Normalized)
# ---------------------------
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.show()

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Greens", xticklabels=classes, yticklabels=classes)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix_normalized.png")
    plt.show()

plot_confusion_matrix(y_test, y_pred, iris.target_names)

# ---------------------------
# 7. ROC Curve & AUC (One-vs-Rest)
# ---------------------------
def plot_roc_auc(model, X_test, y_test, classes):
    y_bin = label_binarize(y_test, classes=classes)
    n_classes = y_bin.shape[1]
    y_score = model.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(7, 6))
    colors = cycle(['blue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multi-class')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.show()

plot_roc_auc(best_model, X_test, y_test, iris.target_names)

# ---------------------------
# 8. Feature Importance
# ---------------------------
importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.savefig("feature_importance.png")
plt.show()

# ---------------------------
# 9. SHAP for Explainability
# ---------------------------
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)
plt.savefig("shap_summary.png")

