# 🤖 CCTV 챗봇 (261020_buchen_chatbot_yjy)

> GroqAPI, CLAUDE API와 YOLO를 곁들인 Chatbot 구현

Groq / Claude API로 만든 Streamlit 챗봇입니다.
**국가교통정보센터(ITS) CCTV 영상을 YOLO11n으로 탐지**하고, 그 결과를 챗봇에게 물어볼 수 있습니다.

> 예) "지금 차가 몇 대 지나갔어?", "트럭이 제일 많았던 순간은 몇 대야?"

## ✨ 기능

| 기능 | 설명 |
|---|---|
| AI 선택 | 사이드바에서 **Groq**(무료)와 **Claude**(유료) 중 선택, 모델도 고를 수 있음 |
| 실시간 답변 | 답변이 글자 단위로 바로 나옴 (스트리밍) |
| CCTV 분석 | CCTV를 고르고 5~30초 동안 YOLO11n으로 **추적(track)** → 종류별 지나간 수·최대·평균 계산 |
| 탐지 결과로 질문 | 분석 결과를 글로 정리해서 챗봇에게 전달 → 챗봇이 데이터를 근거로 답함 |

### YOLO 결과를 어떻게 챗봇에 연결하나요?
LLM은 영상을 직접 보지 않습니다. YOLO가 찾은 결과를 **글로 바꿔서 시스템 프롬프트에 넣는** 방식입니다.

```
CCTV 영상 ──▶ YOLO11n 추적 ──▶ 탐지 요약 글 ──▶ 시스템 프롬프트 ──▶ Groq / Claude ──▶ 답변
            (cctv_yolo.py)     (make_summary_text)        (app.py)            (llm.py)
```

## 📁 폴더 구조

```
261020_buchen_chatbot_yjy/
├── app.py            # Streamlit 화면 (채팅 + 사이드바)
├── llm.py            # Groq / Claude 스트리밍 답변 함수
├── cctv_yolo.py      # ITS CCTV 목록 조회 + YOLO11n 탐지·요약
├── requirements.txt  # 필요한 라이브러리 목록
├── .env.example      # API 키 이름 예시 (값 없음)
├── .gitignore        # .env, 키 파일, *.pt 등 업로드 제외
└── README.md
```

## ⚙️ 설치

```bash
conda activate py311
cd 261020_buchen_chatbot_yjy
pip install -r requirements.txt
```

## 🔑 API 키 설정 (.env)

1. `.env.example`을 복사해서 이름을 `.env`로 바꿉니다.
2. `=` 뒤에 본인 키를 넣습니다.

```
GROQ_API_KEY=여기에_Groq_키
ANTHROPIC_API_KEY=여기에_Claude_키    # Claude를 쓸 때만
ITS_API_KEY=여기에_ITS_키             # CCTV 분석을 쓸 때만
```

| 키 | 발급처 | 비용 |
|---|---|---|
| `GROQ_API_KEY` | https://console.groq.com/keys | 무료 (요청 수 제한 있음) |
| `ANTHROPIC_API_KEY` | https://platform.claude.com/settings/keys | 유료 (크레딧 필요) |
| `ITS_API_KEY` | https://www.its.go.kr/opendata/ | 무료 |

> 이 폴더에 `.env`가 없으면 상위 폴더(`Wori/.env`)를 찾아서 씁니다.
> ⚠️ `.env`는 `.gitignore`로 막혀 있어 GitHub에 올라가지 않습니다. **키를 코드에 직접 쓰지 마세요.**

## ▶️ 실행

```bash
streamlit run app.py
```

브라우저에서 http://localhost:8501 이 열립니다.

## 💬 사용법

1. **AI 선택** : 사이드바에서 `Groq` 또는 `Claude`를 고르고 모델을 선택합니다.
   - Groq 기본 모델 : `openai/gpt-oss-120b`
   - Claude : `claude-opus-5-5`(기본) / `claude-sonnet-5-5` / `claude-haiku-5-5`
2. **그냥 대화** : 아래 입력창에 질문하면 일반 챗봇처럼 답합니다.
3. **CCTV 분석** : 사이드바에서 CCTV를 고르고 분석 시간을 정한 뒤 **[분석 시작]**을 누릅니다.
   - 본문 위에 탐지 결과 이미지와 요약이 나옵니다.
4. **탐지 결과 질문** : 이제 CCTV에 대해 물어보면 탐지 데이터를 근거로 답합니다.
5. **[대화 지우기]** 로 대화를 새로 시작합니다.

### 예시 질문
- "차가 몇 대 지나갔어?"
- "사람도 있었어?"
- "가장 많이 탐지된 건 뭐야?"
- "교통이 막히는 편이야?"

## 📝 참고

- `yolo11n.pt`가 없으면 처음 실행할 때 ultralytics가 자동으로 받습니다.
- "지나간 수"는 추적 ID 기준이라 가려지거나 다시 나타난 객체 때문에 실제와 조금 다를 수 있습니다.
- CCTV 스트림은 서버 상태에 따라 열리지 않을 수 있습니다. 그럴 땐 다른 CCTV를 골라보세요.
- Claude Opus/Sonnet은 거절(refusal) 시 서버가 다른 모델로 이어서 답하는 기능(`fallbacks="default"`)을 켜 두었습니다.
- CCTV 목록 조회 코드는 `261019_6_openapi/v06_03_openapi_yolo.py`의 `its_cctv()`를 바탕으로 만들었습니다.
