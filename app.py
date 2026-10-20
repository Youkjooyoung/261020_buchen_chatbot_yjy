import streamlit as st
from dotenv import load_dotenv, find_dotenv
from ultralytics import YOLO

import llm
import cctv_yolo

# 1. .env 불러오기 (이 폴더부터 위로 올라가며 찾음 -> 없으면 Wori 루트의 .env 사용)
load_dotenv(find_dotenv())

st.set_page_config(page_title="Buchen chatbot", page_icon=":material/smart_toy:")
st.title(":material/smart_toy: CCTV 챗봇")
st.caption("Groq / Claude 챗봇 + YOLO11n CCTV 탐지 결과 질문하기")


# 2. 무거운 것은 한 번만 불러오기 (캐시)
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")       # 없으면 ultralytics 가 자동 다운로드


ROAD_TYPES = {"국도": "its", "고속도로": "ex"}


@st.cache_data(ttl=600, show_spinner="CCTV 목록 불러오는 중...")
def load_cctv_list(road_type):
    return cctv_yolo.get_cctv_list(road_type)


@st.cache_data(ttl=3600)
def load_groq_models():
    return llm.groq_models()


# 3. session_state 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "detect" not in st.session_state:
    st.session_state.detect = None      # {"summary": 글, "image": 결과 이미지}


# 4. 사이드바 - LLM 선택
with st.sidebar:
    st.header("AI 선택")
    provider = st.segmented_control("제공사", ["Groq", "Claude"], default="Groq", required=True)
    if provider == "Groq":
        model = st.selectbox("모델", load_groq_models())
    else:
        model = st.selectbox("모델", llm.CLAUDE_MODELS)
        st.caption("Claude 는 유료입니다 (ANTHROPIC_API_KEY 필요)")

    # 5. 사이드바 - CCTV 분석
    st.header("CCTV 분석")
    road = st.segmented_control("도로 종류", list(ROAD_TYPES), default="국도", required=True)
    try:
        cctv_df = load_cctv_list(ROAD_TYPES[road])
    except Exception as e:
        cctv_df = None
        st.warning(f"CCTV 목록을 불러오지 못했습니다 : {e}", title="CCTV 사용 불가")

    if cctv_df is not None and cctv_df.empty:
        cctv_df = None
        st.warning("CCTV 목록이 비어 있습니다. 잠시 후 다시 시도하세요.", title="CCTV 사용 불가")

    if cctv_df is not None:
        cctv_name = st.selectbox(
            "CCTV 선택", cctv_df["cctvname"].tolist(),
            help="목록을 연 상태에서 글자를 입력하면 검색됩니다 (예: 경부)",
        )
        seconds = st.slider("분석 시간(초)", 5, 30, 10)
        if st.button("분석 시작", icon=":material/videocam:", width="stretch"):
            # 같은 이름은 get_cctv_list 에서 (2), (3) 을 붙여 구분해 둠 -> 이름으로 찾아도 정확함
            cctv_url = cctv_df.loc[cctv_df["cctvname"] == cctv_name, "cctvurl"].iloc[0]
            bar = st.progress(0.0, text=f"{seconds}초 동안 YOLO 탐지 중...")
            try:
                stats, image = cctv_yolo.analyze_stream(
                    cctv_url, seconds, load_model(),
                    progress=lambda p: bar.progress(p, text=f"YOLO 탐지 중... {int(p * 100)}%"),
                )
                st.session_state.detect = {
                    "summary": cctv_yolo.make_summary_text(stats, cctv_name),
                    "image": image,
                    "name": cctv_name,
                }
                st.session_state.messages = []      # 새 분석 -> 이전 CCTV 대화 초기화
            except ConnectionError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다 : {e}")
            finally:
                bar.empty()                         # 오류가 나도 진행 막대는 지움

    st.divider()
    if st.button("대화 지우기", icon=":material/delete:", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# 6. 탐지 결과 보여주기
detect = st.session_state.detect
if detect:
    with st.expander(f"탐지 결과 : {detect['name']}", icon=":material/visibility:", expanded=True):
        st.image(detect["image"], alt=f"{detect['name']} YOLO 탐지 결과 화면")
        st.code(detect["summary"], language=None)
else:
    st.info("사이드바에서 CCTV 를 분석하면, 탐지 결과에 대해 질문할 수 있습니다.")


# 7. 시스템 프롬프트 만들기 (탐지 결과가 있으면 같이 넣음)
#    영어로 작성 -> 한글보다 토큰이 적음 (답변은 한국어로 하도록 지시)
system_prompt = "You are a helpful assistant. Always reply concisely in Korean only; never use Hanja or other languages."
if detect:
    system_prompt += (
        "\n\nBelow are YOLO11n detection results from a road CCTV video. "
        "Answer CCTV questions only from this data; if it is not in the data, say you don't know. "
        "Say class names in Korean (person=사람, car=승용차, etc.).\n"
        "'지나간 수' counts tracker IDs and can be too high on low-quality CCTV. "
        "When asked how many vehicles, give '한 화면 최대 / 평균' first and mention '지나간 수' only as a reference.\n\n"
        + detect["summary"]
    )


# 8. 대화 기록 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# 9. 새 질문 받기 -> 스트리밍 답변
if prompt := st.chat_input("질문을 입력하세요 (예: 지금 차가 몇 대 지나갔어?)", submit_mode="disable"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        answer = st.write_stream(
            llm.chat_stream(provider, model, system_prompt, st.session_state.messages)
        )
    # 빈 답변·오류 안내는 화면에만 남기고, AI 에게 보내는 기록에서는 질문과 함께 뺌 (skip)
    failed = not answer or llm.ERROR_MARK in answer
    st.session_state.messages[-1]["skip"] = failed
    st.session_state.messages.append({"role": "assistant", "content": answer, "skip": failed})
