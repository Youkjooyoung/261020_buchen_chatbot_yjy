import os                # 환경변수(API 키)를 읽는 도구
import time
import json
import urllib.request
from datetime import datetime

import cv2
import pandas as pd


# 1. ITS CCTV 목록 가져오기 (v06_03_openapi_yolo.py 의 its_cctv() 를 목록용으로 변형)
#    -> v06_03 은 import 하는 순간 영상 루프가 돌기 때문에 import 하지 않고 코드를 가져옴
def get_cctv_list():
    key = os.getenv("ITS_API_KEY")
    if not key:
        raise ValueError(".env 에 ITS_API_KEY 가 없습니다.")

    road_type, getType = "its", "json"          # its = 일반도로 , ex = 고속도로
    minX, maxX, minY, maxY = 120.95, 127.02, 30.55, 37.69

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
    return cctv_play[["cctvname", "cctvurl"]]


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

    while time.time() < end:
        success, frame = cap.read()
        if not success:
            break

        # 2-1. 추적(track) : 같은 객체에 같은 ID 를 붙여줌 -> 중복 없이 셀 수 있음
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes
        names = results[0].names
        frames += 1

        # 2-2. 이번 프레임의 종류별 개수
        frame_count = {}
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        for cls, track_id in zip(boxes.cls.int().tolist(), ids):
            name = names[cls]
            frame_count[name] = frame_count.get(name, 0) + 1
            if track_id is not None:
                unique_ids.setdefault(name, set()).add(track_id)

        for name, cnt in frame_count.items():
            max_count[name] = max(max_count.get(name, 0), cnt)
            total_count[name] = total_count.get(name, 0) + cnt

        last_frame = results[0].plot()
        if progress:
            progress(min((time.time() - start) / seconds, 1.0))

    # 3. 자원해제 + 다음 분석을 위해 추적기 초기화
    cap.release()
    if model.predictor is not None and hasattr(model.predictor, "trackers"):
        del model.predictor.trackers

    if frames == 0:
        raise ConnectionError("CCTV 화면을 받아오지 못했습니다. 다른 CCTV 를 골라보세요.")

    stats = {
        "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": seconds,
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
