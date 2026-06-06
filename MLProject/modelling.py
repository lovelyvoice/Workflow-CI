import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# LOAD DATASET
# ============================================================
train_df = pd.read_csv('banknote_preprocessing/banknote_train.csv')
test_df  = pd.read_csv('banknote_preprocessing/banknote_test.csv')

feature_cols = ['variance', 'skewness', 'curtosis', 'entropy']

X_train = train_df[feature_cols]
y_train = train_df['class']
X_test  = test_df[feature_cols]
y_test  = test_df['class']

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ============================================================
# MLFLOW AUTOLOG
# ============================================================
mlflow.set_experiment("Banknote_SVM_Basic")
mlflow.autolog()  # Kriteria 2 Basic: WAJIB MENGGUNAKAN AUTOLOG

with mlflow.start_run(run_name="SVM_Basic_Autolog"):
    print("Memulai pelatihan model SVM Basic (tanpa hyperparameter tuning)...")
    
    # Model sederhana tanpa tuning
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nRun ID: {mlflow.active_run().info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nModelling Basic selesai! Cek MLflow UI.")