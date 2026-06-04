import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# ============================================================
# LOAD DATASET
# ============================================================
train_df = pd.read_csv('MLProject/iris_preprocessing/iris_train.csv')
test_df  = pd.read_csv('MLProject/iris_preprocessing/iris_test.csv')

feature_cols = ['sepal length (cm)', 'sepal width (cm)',
                'petal length (cm)', 'petal width (cm)']

X_train = train_df[feature_cols]
y_train = train_df['species']
X_test  = test_df[feature_cols]
y_test  = test_df['species']

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
param_grid = {
    'C'     : [0.1, 1, 10],
    'gamma' : ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

print("Memulai GridSearchCV...")
base_model  = SVC(random_state=42, probability=True)
grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model  = grid_search.best_estimator_
print(f"Best Params: {best_params}")

# ============================================================
# EVALUASI
# ============================================================
y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
y_bin     = label_binarize(y_test, classes=[0, 1, 2])
roc_auc   = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='weighted')

# ============================================================
# ARTEFAK
# ============================================================
os.makedirs('outputs', exist_ok=True)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['setosa', 'versicolor', 'virginica'],
            yticklabels=['setosa', 'versicolor', 'virginica'])
plt.title('Confusion Matrix – SVM Iris')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()

# Classification report
report = classification_report(
    y_test, y_pred,
    target_names=['setosa', 'versicolor', 'virginica'],
    output_dict=True
)
with open('outputs/classification_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# ============================================================
# MLFLOW LOGGING
# ============================================================
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment("Iris_SVM_CI")

with mlflow.start_run(run_name="SVM_CI_Run"):
    mlflow.log_param("kernel",    best_params['kernel'])
    mlflow.log_param("C",         best_params['C'])
    mlflow.log_param("gamma",     best_params['gamma'])
    mlflow.log_param("cv_folds",  5)

    mlflow.log_metric("accuracy",           accuracy)
    mlflow.log_metric("precision_weighted", precision)
    mlflow.log_metric("recall_weighted",    recall)
    mlflow.log_metric("f1_weighted",        f1)
    mlflow.log_metric("roc_auc_weighted",   roc_auc)
    mlflow.log_metric("cv_mean_score",      cv_scores.mean())
    mlflow.log_metric("cv_std_score",       cv_scores.std())

    mlflow.sklearn.log_model(best_model, artifact_path="svm_model")
    mlflow.log_artifact('outputs/confusion_matrix.png')
    mlflow.log_artifact('outputs/classification_report.json')

    mlflow.set_tag("model_type", "SVM")
    mlflow.set_tag("dataset",    "Iris")

    print(f"\nRun ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")

print("\nCI Modelling selesai!")
