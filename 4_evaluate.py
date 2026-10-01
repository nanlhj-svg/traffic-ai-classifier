from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.model_selection import LeaveOneOut

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


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
filenames = data["filenames"]


CLASS_NAMES = [
    "motorcycle",
    "bicycle",
    "kickboard"
]


# =========================================================
# 3. 평가할 모델
# =========================================================

models = {

    "Logistic Regression": Pipeline([
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
    ]),

    "SVM": Pipeline([
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
}


# =========================================================
# 4. Leave-One-Out 평가
# =========================================================

loo = LeaveOneOut()

results = []
prediction_dict = {}


for model_name, model in models.items():

    y_true = []
    y_pred = []

    for train_index, test_index in loo.split(X):

        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        model.fit(
            X_train,
            y_train
        )

        pred = model.predict(
            X_test
        )

        y_true.append(
            y_test[0]
        )

        y_pred.append(
            pred[0]
        )


    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )


    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    })

    prediction_dict[
        model_name
    ] = y_pred


    print("\n==============================")
    print(model_name)
    print("==============================")

    print(
        "Accuracy:",
        accuracy
    )

    print(
        "Macro F1:",
        macro_f1
    )

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


# =========================================================
# 5. 성능 결과 CSV 저장
# =========================================================

result_df = pd.DataFrame(
    results
)

result_df.to_csv(
    OUTPUT_DIR / "model_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[모델 성능 비교]")
print(result_df)


# =========================================================
# 6. 성능 비교 그래프
# =========================================================

result_df.set_index(
    "Model"
)[
    ["Accuracy", "Macro F1"]
].plot(
    kind="bar",
    figsize=(8, 5),
    ylim=(0, 1.1)
)

plt.title(
    "Model Performance Comparison"
)

plt.ylabel(
    "Score"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=150
)

plt.show()


# =========================================================
# 7. 실제값 vs 예측값 저장
# =========================================================

prediction_df = pd.DataFrame({
    "File": filenames,
    "Actual": y
})

for model_name in models:

    prediction_df[
        model_name
    ] = prediction_dict[
        model_name
    ]


prediction_df.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n[실제값 vs 예측값]")
print(prediction_df)


# =========================================================
# 8. 가장 높은 Macro F1 모델 선택
# =========================================================

best_model_name = result_df.loc[
    result_df["Macro F1"].idxmax(),
    "Model"
]

print(
    "\n최종 선택 모델:",
    best_model_name
)


# =========================================================
# 9. 전체 데이터로 최종 학습
# =========================================================

best_model = models[
    best_model_name
]

best_model.fit(
    X,
    y
)


# =========================================================
# 10. 웹에서 사용할 모델 패키지 저장
# =========================================================

model_package = {
    "model": best_model,
    "classes": CLASS_NAMES,
    "feature_method": "HOG",
    "image_size": (128, 128)
}


model_path = (
    MODEL_DIR /
    "traffic_classifier.pkl"
)

joblib.dump(
    model_package,
    model_path
)


print("\n최종 모델 저장 완료")
print("저장 위치:", model_path)
print(
    "파일 존재:",
    model_path.exists()
)