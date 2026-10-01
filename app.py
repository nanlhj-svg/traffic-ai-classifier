from pathlib import Path
import cv2
import joblib
import numpy as np

from flask import Flask, request, render_template_string
from skimage.feature import hog


# =========================================================
# 1. 경로 및 모델 불러오기
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "traffic_classifier.pkl"

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
classes = model_package["classes"]

print("모델 불러오기 성공")
print("분류 클래스:", classes)


# =========================================================
# 2. Flask
# =========================================================

app = Flask(__name__)


# =========================================================
# 3. 학습할 때와 동일한 이미지 전처리
# =========================================================

def extract_feature(image):

    image = cv2.resize(image, (128, 128))

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return feature


# =========================================================
# 4. 화면
# =========================================================

HTML = """
<!DOCTYPE html>
<html lang="ko">

<head>
    <meta charset="UTF-8">
    <title>교통안전 위반 유형 분류 AI</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 700px;
            margin: 60px auto;
            padding: 20px;
        }

        h1 {
            margin-bottom: 10px;
        }

        .box {
            border: 1px solid #cccccc;
            padding: 25px;
            margin-top: 25px;
        }

        button {
            margin-top: 15px;
            padding: 10px 20px;
            cursor: pointer;
        }

        .result {
            margin-top: 25px;
            padding: 20px;
            background: #f2f2f2;
        }
    </style>
</head>

<body>

<h1>CCTV 교통안전 위반 유형 분류</h1>

<p>
CCTV 이미지를 입력하면 학습된 모델을 이용하여
교통안전 위반 유형을 예측합니다.
</p>

<div class="box">

    <form method="POST"
          enctype="multipart/form-data">

        <input type="file"
               name="image"
               accept="image/*"
               required>

        <br>

        <button type="submit">
            예측하기
        </button>

    </form>

</div>


{% if prediction %}

<div class="result">

    <h2>예측 결과</h2>

    <p>
        {{ prediction }}
    </p>

</div>

{% endif %}


</body>
</html>
"""


# =========================================================
# 5. 이미지 입력 → 예측
# =========================================================

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["image"]

        # 업로드된 파일을 메모리에서 OpenCV 이미지로 변환
        file_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            prediction = "이미지를 읽을 수 없습니다."

        else:

            # 학습 때와 동일한 특징 추출
            feature = extract_feature(image)

            # 모델은 여러 샘플 형태를 요구하므로
            # (특징 개수,) → (1, 특징 개수)
            feature = feature.reshape(1, -1)

            result = model.predict(feature)[0]

            label_korean = {
                "motorcycle": "오토바이 인도 주행",
                "bicycle": "자전거 인도 주행",
                "kickboard": "전동킥보드 인도 주행"
            }

            prediction = label_korean.get(
                result,
                result
            )

    return render_template_string(
        HTML,
        prediction=prediction
    )


# =========================================================
# 6. 서버 실행
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )