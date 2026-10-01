from pathlib import Path

import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# =========================================================
# 1. 경로
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "output"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 전처리 데이터 불러오기
# =========================================================

data = np.load(
    OUTPUT_DIR / "preprocessed_data.npz",
    allow_pickle=True
)

X = data["X"]
y = data["y"]

print("학습 데이터 수:", len(X))
print("특징 수:", X.shape[1])


# =========================================================
# 3. Pipeline 구성
# =========================================================

logistic_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000
        )
    )
])


svm_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        SVC(
            kernel="rbf",
            probability=True
        )
    )
])


# =========================================================
# 4. 전체 데이터로 모델 학습
# =========================================================

print("\nLogistic Regression 학습 중...")
logistic_model.fit(X, y)

print("SVM 학습 중...")
svm_model.fit(X, y)


# =========================================================
# 5. 모델 저장
# =========================================================

joblib.dump(
    logistic_model,
    MODEL_DIR / "logistic_regression.pkl"
)

joblib.dump(
    svm_model,
    MODEL_DIR / "svm.pkl"
)


print("\n모델 학습 완료")
print(
    "Logistic Regression:",
    MODEL_DIR / "logistic_regression.pkl"
)

print(
    "SVM:",
    MODEL_DIR / "svm.pkl"
)