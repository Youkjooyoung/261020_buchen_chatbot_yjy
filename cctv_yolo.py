import os                # 환경변수(API 키)를 읽는 도구
import time
import json
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

# 저화질 CCTV 용 추적 설정 파일 (신뢰도 기준을 낮춤)
TRACKER = str(Path(__file__).parent / "bytetrack_cctv.yaml")

# 도로 객체만 탐지 (COCO 번호) : 0 사람, 1 자전거, 2 승용차, 3 오토바이, 5 버스, 7 트럭
#   -> 기차·신호등 같은 엉뚱한 결과 제외
ROAD_CLASSES = [0, 1, 2, 3, 5, 7]


# 1. ITS CCTV 목록 가져오기 (v06_03_openapi_yolo.py 의 its_cctv() 를 목록용으로 변형)
#    -> v06_03 은 import 하는 순간 영상 루프가 돌기 때문에 import 하지 않고 코드를 가져옴
#    road_type : its = 국도(일반도로) , ex = 고속도로
def get_cctv_list(road_type="its"):
    key = os.getenv("ITS_API_KEY")
    if not key:
        raise ValueError(".env 에 ITS_API_KEY 가 없습니다.")

    getType = "json"
    # 전국 범위 (경도 124~132, 위도 33~38.9) -> 부산·대구·강원·경기 북부까지 포함
    minX, maxX, minY, maxY = 124.0, 132.0, 33.0, 38.9

    url_cctv = (
        f"https://openapi.its.go.kr:9443/cctvInfo"
        f"?apiKey={key}&type={road_type}&cctvType=1"
        f"&minX={minX}&maxX={maxX}"
        f"&minY={minY}&maxY={maxY}&getType={getType}"
    )

    response = urllib.request.urlopen(url_cctv, timeout=10)
    json_str = response.read().decode("utf-8")      # bytes -> str
    json_object = json.loads(json_str)              # str -> dict

    cctv_play = pd.json_normalize(json_object["response"]["data"], sep='')
    cctv_play = cctv_play[["cctvname", "cctvurl"]].copy()

    # 같은 이름의 CCTV 는 두 번째부터 뒤에 (2), (3) 을 붙여 구분
    #   -> Streamlit selectbox 는 화면에 보이는 글자로 선택값을 구분하기 때문에 이름이 겹치면 안 됨
    dup_no = cctv_play.groupby("cctvname").cumcount()
    cctv_play["cctvname"] += dup_no.map(lambda n: f" ({n + 1})" if n else "")
    return cctv_play


# 2. 영상(CCTV URL 또는 영상 파일)을 정해진 시간 동안 YOLO 로 추적하며 탐지 결과 모으기
#    progress : 진행률(0~1)을 받는 함수 -> 화면에 진행 막대 표시용
def analyze_stream(source, seconds, model, progress=None):
    # 2-0. 응답 없는 CCTV 에서 오래 멈추지 않도록 열기·읽기 제한시간 10초
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
    ])
    if not cap.isOpened():
        raise ConnectionError("CCTV 영상을 열 수 없습니다. 다른 CCTV 를 골라보세요.")

    start_time = datetime.now()
    start = time.time()
    end = start + seconds

    unique_ids = {}        # 종류별 고유 track ID 모음 -> 지나간 수
    max_count = {}         # 종류별 한 화면 최대 수
    total_count = {}       # 종류별 누적 수 -> 평균 계산용
    frames = 0
    last_frame = None

    try:
        while time.time() < end:
            success, frame = cap.read()
            if not success:
                break

            # 2-1. 추적(track) : 같은 객체에 같은 ID 를 붙여줌 -> 중복 없이 셀 수 있음
            results = model.track(frame, persist=True, verbose=False,
                                  tracker=TRACKER, classes=ROAD_CLASSES)
            boxes = results[0].boxes
            names = results[0].names
            frames += 1

            # 2-2. 이번 프레임의 종류별 개수
            frame_count = {}
            # ID 가 없는 박스 = 추적 실패한 신뢰도 낮은 탐지 -> 세지 않음
            ids = boxes.id.int().tolist() if boxes.id is not None else []
            for cls, track_id in zip(boxes.cls.int().tolist(), ids):
                name = names[cls]
                frame_count[name] = frame_count.get(name, 0) + 1
                unique_ids.setdefault(name, set()).add(track_id)

            for name, cnt in frame_count.items():
                max_count[name] = max(max_count.get(name, 0), cnt)
                total_count[name] = total_count.get(name, 0) + cnt

            last_frame = results[0].plot()
            if progress:
                progress(min((time.time() - start) / seconds, 1.0))
        # 영상이 중간에 끊길 수 있으므로 실제로 분석한 시간을 기록
        elapsed = round(time.time() - start, 1)

    # 3. 자원해제 + 다음 분석을 위해 추적기 초기화 (중간에 오류가 나도 항상 실행)
    finally:
        cap.release()
        if model.predictor is not None and hasattr(model.predictor, "trackers"):
            del model.predictor.trackers

    if frames == 0:
        raise ConnectionError("CCTV 화면을 받아오지 못했습니다. 다른 CCTV 를 골라보세요.")

    stats = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": elapsed,
        "frames": frames,
        "classes": {
            name: {
                "unique": len(unique_ids.get(name, set())),
                "max": max_count[name],
                "avg": round(total_count[name] / frames, 1),
            }
            for name in sorted(max_count, key=lambda n: -max_count[n])
        },
    }
    image = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)     # BGR -> RGB (화면 표시용)
    return stats, image


# 4. 탐지 결과를 챗봇에게 넘길 글로 바꾸기
def make_summary_text(stats, cctv_name):
    lines = [
        f"- CCTV 이름 : {cctv_name}",
        f"- 분석 시작 시각 : {stats['start_time']}",
        f"- 분석 시간 : {stats['seconds']}초 ({stats['frames']} 프레임)",
        "- 탐지 결과 (YOLO11n, COCO 영어 클래스명) :",
    ]
    if not stats["classes"]:
        lines.append("  탐지된 객체 없음")
    for name, c in stats["classes"].items():
        lines.append(
            f"  {name} : 지나간 수(고유 ID) {c['unique']}, "
            f"한 화면 최대 {c['max']}, 평균 {c['avg']}"
        )
    return "\n".join(lines)
