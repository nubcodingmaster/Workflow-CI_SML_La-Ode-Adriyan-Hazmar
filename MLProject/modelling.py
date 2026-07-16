import os
import shutil
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Hapus folder model lama jika sudah ada agar tidak error saat ditimpa
if os.path.exists("model_artifact"):
    shutil.rmtree("model_artifact")

# Baca data
train_df = pd.read_csv('rice_dataset_preprocessing/train.csv')
X_train = train_df.drop('Class', axis=1)
y_train = train_df['Class']

# Latih model
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Simpan model menggunakan MLflow
mlflow.sklearn.save_model(rf, "model_artifact")