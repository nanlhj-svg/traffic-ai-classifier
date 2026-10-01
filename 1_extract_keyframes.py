import cv2
import json
from pathlib import Path


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

VIDEO_DIR = PROJECT_DIR / "data" / "videos"
LABEL_DIR = PROJECT_DIR / "data" / "labels"
OUTPUT_DIR = PROJECT_DIR / "output" / "keyframes"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("현재 실행 파일 :", Path(__file__).resolve())
print("프로젝트 폴더 :", PROJECT_DIR)
print("영상 폴더     :", VIDEO_DIR)
print("라벨 폴더     :", LABEL_DIR)
print("출력 폴더     :", OUTPUT_DIR)


# ============================================================
# 2. 모든 JSON 파일 검색
# ============================================================

json_files = sorted(LABEL_DIR.glob("*.json"))

print("\n찾은 JSON 개수:", len(json_files))

for json_file in json_files:
    print(" -", json_file.name)

if not json_files:
    raise FileNotFoundError(
        f"JSON 파일을 찾을 수 없습니다: {LABEL_DIR}"
    )


# ============================================================
# 3. JSON 하나씩 처리
# ============================================================

total_saved = 0

for json_path in json_files:

    print("\n" + "=" * 60)
    print("JSON 처리:", json_path.name)
    print("=" * 60)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data["annotations"]

    # 사건 유형
    event_class = annotations["event_class"]

    print("이벤트 종류:", event_class)

    # evidence 정보
    evidence = annotations["evidence"]


    # ========================================================
    # 4. videos 리스트 순회
    # ========================================================

    for video_info in data["videos"]:

        video_name = video_info["filename"]
        view = video_info["view"]

        print(f"\n[{view} 처리 시작]")
        print("영상 파일:", video_name)

        video_path = VIDEO_DIR / video_name

        print("영상 경로:", video_path)
        print("영상 존재:", video_path.exists())

        if not video_path.exists():
            print("영상 파일이 없어서 건너뜁니다.")
            continue


        # ====================================================
        # 5. 해당 카메라의 근거 프레임 가져오기
        # ====================================================

        if view not in evidence:
            print(f"{view}에 해당하는 evidence가 없습니다.")
            continue

        frame_ids = evidence[view]["frame_id"]
        bboxes = evidence[view].get("obj_bbox", [])

        print("근거 프레임:", frame_ids)


        # ====================================================
        # 6. 영상 열기
        # ====================================================

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("영상 열기 실패:", video_path)
            continue

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        print("전체 프레임 수:", total_frames)


        # ====================================================
        # 7. 키프레임 추출
        # ====================================================

        for i, frame_id in enumerate(frame_ids):

            if frame_id >= total_frames:
                print(
                    f"프레임 {frame_id}은 영상 범위를 벗어났습니다."
                )
                continue

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_id
            )

            success, frame = cap.read()

            if not success:
                print("프레임 읽기 실패:", frame_id)
                continue


            # ================================================
            # 8. Bounding Box 표시
            # ================================================

            if i < len(bboxes):

                bbox = bboxes[i]

                x1, y1, x2, y2 = map(int, bbox)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    4
                )


            # ================================================
            # 9. 이미지 저장
            # ================================================

            video_stem = Path(video_name).stem

            output_name = (
                f"{video_stem}_frame_{frame_id}.jpg"
            )

            output_path = OUTPUT_DIR / output_name

            success_write = cv2.imwrite(
                str(output_path),
                frame
            )

            if success_write:
                total_saved += 1
                print("저장 완료:", output_path)
            else:
                print("저장 실패:", output_path)


        cap.release()


# ============================================================
# 10. 완료
# ============================================================

print("\n" + "=" * 60)
print("키프레임 추출 작업 완료!")
print("총 저장 이미지:", total_saved)
print("저장 위치:", OUTPUT_DIR)
print("=" * 60)