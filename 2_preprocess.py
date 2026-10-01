from pathlib import Path

import cv2
import numpy as np
from skimage.feature import hog


# =========================================================
# 1. 경로 설정
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_DIR / "dataset"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 클래스 설정
# =========================================================

CLASS_NAMES = [
    "motorcycle",
    "bicycle",
    "kickboard"
]


# =========================================================
# 3. 이미지 → HOG 특징 추출 함수
# =========================================================

def extract_feature(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"이미지를 읽을 수 없습니다: {image_path}"
        )

    # 이미지 크기 통일
    image = cv2.resize(
        image,
        (128, 128)
    )

    # grayscale 변환
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # HOG 특징 추출
    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return feature


# =========================================================
# 4. 전체 이미지 전처리
# =========================================================

X = []
y = []
filenames = []

for class_name in CLASS_NAMES:

    class_dir = DATASET_DIR / class_name

    print(f"\n[{class_name}]")

    if not class_dir.exists():
        print("폴더 없음:", class_dir)
        continue

    image_files = sorted(
        class_dir.glob("*.jpg")
    )

    print("이미지 수:", len(image_files))

    for image_path in image_files:

        feature = extract_feature(
            image_path
        )

        X.append(feature)
        y.append(class_name)
        filenames.append(image_path.name)


# =========================================================
# 5. numpy 배열 변환
# =========================================================

X = np.array(X)
y = np.array(y)
filenames = np.array(filenames)


print("\n==============================")
print("전처리 결과")
print("==============================")

print("전체 이미지 수:", len(X))

if len(X) > 0:
    print("이미지당 HOG 특징 수:", X.shape[1])

for class_name in CLASS_NAMES:
    print(
        class_name,
        ":",
        np.sum(y == class_name)
    )


# =========================================================
# 6. 전처리 결과 저장
# =========================================================

save_path = OUTPUT_DIR / "preprocessed_data.npz"

np.savez(
    save_path,
    X=X,
    y=y,
    filenames=filenames
)

print("\n전처리 데이터 저장 완료")
print("저장 위치:", save_path)