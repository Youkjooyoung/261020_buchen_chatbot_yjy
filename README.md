# 🤖 CCTV 챗봇 (261020_buchen_chatbot_yjy)

> GroqAPI, CLAUDE API와 YOLO를 곁들인 Chatbot 구현

Groq / Claude API로 만든 **Streamlit 챗봇**입니다.
국가교통정보센터(ITS)의 **국도·고속도로 CCTV 영상을 YOLO11n으로 탐지**하고, 그 결과를 챗봇에게 물어볼 수 있습니다.

> 예) "지금 차가 몇 대 지나갔어?", "트럭도 있었어?"

---

## 📑 목차

1. [기능](#-기능)
2. [폴더 구조](#-폴더-구조)
3. [처음 설치하기 (따라하기)](#-처음-설치하기-따라하기)
4. [API 키 발급받기](#-api-키-발급받기)
5. [.env 파일 만들기](#-env-파일-만들기)
6. [실행하기](#-실행하기)
7. [사용법](#-사용법)
8. [토큰 사용량 / 요금](#-토큰-사용량--요금)
9. [문제 해결](#-문제-해결)
10. [처음 요구사항 (구현 프롬프트)](#-처음-요구사항-구현-프롬프트)

---

## ✨ 기능

| 기능 | 설명 |
|---|---|
| AI 선택 | 사이드바에서 **Groq**(무료)와 **Claude**(유료) 중 선택, 모델도 고를 수 있음 |
| 실시간 답변 | 답변이 글자 단위로 바로 나옴 (스트리밍) |
| 도로 선택 | 전국 **국도**(약 6500개) / **고속도로**(약 4700개) CCTV 선택 |
| CCTV 분석 | 5~30초 동안 YOLO11n으로 **추적(track)** → 종류별 지나간 수·최대·평균 계산, 진행 막대 표시 |
| 탐지 결과로 질문 | 분석 결과를 글로 정리해서 챗봇에게 전달 → 챗봇이 데이터를 근거로 답함 |
| 토큰 절약 | 영어 시스템 프롬프트, 최근 대화 10개만 전송, 추론 강도 low ([자세히](#-토큰-사용량--요금)) |

### YOLO 결과를 어떻게 챗봇에 연결하나요?
LLM은 영상을 직접 보지 않습니다. YOLO가 찾은 결과를 **글로 바꿔서 시스템 프롬프트에 넣는** 방식입니다.

```
① CCTV 영상
     │  cctv_yolo.py : analyze_stream()      YOLO11n 으로 5~30초 추적
     ▼
② 탐지 요약 글
     │  cctv_yolo.py : make_summary_text()   "car : 지나간 수 6, 한 화면 최대 21, 평균 9.1 ..."
     ▼
③ 시스템 프롬프트
     │  app.py                               답변 규칙(영어) + 탐지 요약 글
     ▼
④ Groq / Claude
     │  llm.py : chat_stream()               최근 대화 10개와 함께 전송
     ▼
⑤ 답변 (한국어, 스트리밍)
```

---

## 📁 폴더 구조

```
261020_buchen_chatbot_yjy/
├── app.py               # Streamlit 화면 (채팅 + 사이드바) + 시스템 프롬프트
├── llm.py               # Groq / Claude 스트리밍 답변 함수
├── cctv_yolo.py         # ITS CCTV 목록 조회 + YOLO11n 탐지·요약
├── bytetrack_cctv.yaml  # 저화질 CCTV용 추적 설정 (신뢰도 기준 낮춤)
├── requirements.txt     # 필요한 라이브러리 목록
├── .gitignore           # .env, 키 파일, *.pt 등 업로드 제외
├── README.md
├── .env                 # ⚠️ 직접 만드는 파일 (API 키) - GitHub에 올라가지 않음
└── yolo11n.pt           # 처음 분석할 때 자동 다운로드 - GitHub에 올라가지 않음
```

---

## 🛠 처음 설치하기 (따라하기)

> 처음 한 번만 하면 됩니다. 다음부터는 [6. 실행하기](#-실행하기)만 하면 됩니다.

### 0단계. 필요한 프로그램 설치

| 프로그램 | 다운로드 | 비고 |
|---|---|---|
| **Anaconda** | https://www.anaconda.com/download | 파이썬 + 가상환경 관리 |
| **VS Code** | https://code.visualstudio.com | 코드 편집기 |
| **Git** | https://git-scm.com/download/win | 코드 내려받기 (ZIP으로 받으면 없어도 됨) |

VS Code를 연 뒤 왼쪽 **확장(Extensions, `Ctrl + Shift + X`)** 에서 **Python** 을 검색해 설치합니다.

### 1단계. 코드 내려받기

**방법 A - Git 사용**

원하는 위치(예: 바탕화면)에서 cmd를 열고 입력합니다.
```cmd
git clone https://github.com/Youkjooyoung/261020_buchen_chatbot_yjy.git
```

**방법 B - ZIP 다운로드**

GitHub 페이지에서 초록색 **[<> Code]** 버튼 → **Download ZIP** → 압축 풀기

### 2단계. VS Code로 폴더 열기

1. VS Code 실행
2. 상단 메뉴 **파일(File) → 폴더 열기(Open Folder...)**
3. `261020_buchen_chatbot_yjy` 폴더 선택 → **폴더 선택**
4. "이 폴더의 작성자를 신뢰하십니까?" 가 나오면 **예, 작성자를 신뢰합니다**

### 3단계. 터미널을 **cmd(명령 프롬프트)** 로 열기

> ⚠️ **PowerShell이 아니라 cmd를 사용하세요.**
> PowerShell에서는 `conda activate`가 아무 말 없이 실패해서 `'streamlit' 용어는 ... 인식되지 않습니다` 오류가 날 수 있습니다.

1. 상단 메뉴 **터미널(Terminal) → 새 터미널(New Terminal)** (단축키 `` Ctrl + ` ``)
2. 아래쪽에 터미널이 열리면, 터미널 오른쪽 위의 **`+` 옆 `∨` (화살표)** 클릭
3. **Command Prompt** 선택
4. 터미널 앞부분이 아래처럼 `PS` 없이 나오면 cmd 입니다.
   ```
   C:\...\261020_buchen_chatbot_yjy>
   ```

> 💡 매번 cmd로 열리게 하려면: `Ctrl + Shift + P` → **Terminal: Select Default Profile** → **Command Prompt**

### 4단계. 가상환경 만들기 + 켜기

**처음 한 번만** 가상환경을 만듭니다. (이미 `py311` 환경이 있으면 건너뛰기)
```cmd
conda create -n py311 python=3.11 -y
```

가상환경을 켭니다.
```cmd
conda activate py311
```

앞에 **`(py311)`** 이 붙으면 성공입니다.
```
(py311) C:\...\261020_buchen_chatbot_yjy>
```

### 5단계. 라이브러리 설치

```cmd
pip install -r requirements.txt
```

설치되는 라이브러리 : `streamlit`, `groq`, `anthropic`, `python-dotenv`, `ultralytics`, `opencv-python`, `pandas`, `lap`
(몇 분 걸릴 수 있습니다. 마지막에 `Successfully installed ...` 가 나오면 완료)

### 6단계. VS Code 파이썬 선택 (선택)

`Ctrl + Shift + P` → **Python: Select Interpreter** → **`py311`** 선택
→ 코드에 빨간 밑줄(라이브러리 못 찾음)이 사라집니다.

---

## 🔑 API 키 발급받기

| 키 이름 | 어디에 쓰나 | 필수 | 비용 |
|---|---|---|---|
| `GROQ_API_KEY` | Groq 챗봇 | ✅ | 무료 |
| `ITS_API_KEY` | CCTV 목록 | CCTV 분석할 때 | 무료 |
| `ANTHROPIC_API_KEY` | Claude 챗봇 | Claude 쓸 때만 | **유료** |

### Groq (무료)
1. https://console.groq.com 접속 → 구글 계정 등으로 로그인
2. 왼쪽 메뉴 **API Keys** → **Create API Key**
3. 이름 입력 후 생성 → `gsk_...` 로 시작하는 키 **복사** (한 번만 보여줌)

### ITS 국가교통정보센터 (무료)
1. https://www.its.go.kr/opendata/ 접속 → 회원가입 / 로그인
2. 오픈API 메뉴에서 **CCTV 정보** 활용 신청
3. 마이페이지에서 발급된 **인증키** 복사

### Claude (유료)
1. https://platform.claude.com 접속 → 로그인
2. **Settings → Billing** 에서 크레딧 충전 (claude.ai 구독과는 별도 결제)
3. **Settings → API Keys** (https://platform.claude.com/settings/keys) → **Create Key**
   - "ID 페더레이션" 안내가 나오면 → **API 키로 계속하기**
   - 이름 : 예) `buchen-chatbot` / 만료 : 30일 / 범위 : **기본 워크스페이스**
4. `sk-ant-...` 로 시작하는 키 **복사** (한 번만 보여줌)
5. (추천) **Settings → Limits** 에서 월 사용 한도를 작게 설정

---

## 📄 .env 파일 만들기

1. VS Code 왼쪽 **탐색기(Explorer)** 에서 `261020_buchen_chatbot_yjy` 폴더에 마우스를 올림
2. **새 파일(📄+)** 아이콘 클릭 → 파일 이름을 **`.env`** 로 입력 (앞에 점 `.` 꼭 포함, 확장자 없음)
3. 아래 내용을 붙여넣고 `=` 뒤에 본인 키를 넣은 뒤 **저장(`Ctrl + S`)**

```
GROQ_API_KEY=여기에_Groq_키
ITS_API_KEY=여기에_ITS_키
ANTHROPIC_API_KEY=여기에_Claude_키
```

> ⚠️ `=` 앞뒤에 띄어쓰기, 따옴표(`"`)를 넣지 마세요.
> ⚠️ `.env`는 `.gitignore`로 막혀 있어 GitHub에 올라가지 않습니다. **키를 코드(.py)에 직접 쓰지 마세요.**

---

## ▶️ 실행하기

> ⚠️ VS Code 오른쪽 위 **▶ 실행 버튼(또는 F5)으로 실행하면 안 됩니다.**
> Streamlit은 반드시 터미널에서 `streamlit run` 으로 실행해야 브라우저 화면이 뜹니다.

### 매번 실행하는 순서

1. VS Code에서 `261020_buchen_chatbot_yjy` 폴더 열기
2. **터미널 → 새 터미널** → **Command Prompt(cmd)** 인지 확인 ([3단계](#3단계-터미널을-cmd명령-프롬프트-로-열기) 참고)
3. 아래 두 줄 입력
   ```cmd
   conda activate py311
   streamlit run app.py
   ```
4. 브라우저에 **http://localhost:8501** 이 자동으로 열립니다.
   - 안 열리면 브라우저 주소창에 직접 `http://localhost:8501` 입력

> 💡 **처음 실행할 때** 터미널에 `Email:` 을 물어보면 그냥 **Enter** 를 누르면 됩니다.
> 💡 처음 CCTV 분석할 때 `yolo11n.pt` 모델(약 5MB)을 자동으로 내려받습니다.

### 종료하기

터미널을 클릭하고 **`Ctrl + C`** → 브라우저 탭 닫기

---

## 💬 사용법

1. **AI 선택** : 사이드바 맨 위에서 `Groq` 또는 `Claude` 선택 → 모델 선택
   - Groq 기본 모델 : `openai/gpt-oss-120b`
   - Claude : `claude-opus-5-5`(기본) / `claude-sonnet-5-5` / `claude-haiku-5-5`
     - 요금은 Opus > Sonnet > Haiku 순입니다. 탐지 결과 질문은 **Haiku로도 충분**합니다.
2. **그냥 대화** : 아래 입력창에 질문하면 일반 챗봇처럼 답합니다.
3. **CCTV 분석**
   1. **도로 종류** : `국도` / `고속도로` 선택
   2. **CCTV 선택** : 목록을 연 상태에서 글자를 입력하면 검색됩니다 (예: `경부`, `서울`)
      - 이름이 같은 CCTV가 여러 개면 두 번째부터 뒤에 `(2)`, `(3)` 이 붙습니다 (예: 같은 지점의 양방향 카메라)
   3. **분석 시간** : 5~30초
   4. **[분석 시작]** 클릭 → 진행 막대가 끝나면 본문 위에 탐지 이미지와 요약이 나옵니다.
      - 새로 분석하면 **이전 CCTV 대화는 자동으로 지워집니다.**
      - 분석은 **한 번에 하나만** 됩니다. 다른 브라우저 탭에서 분석 중이면 "다른 탭에서 분석 중입니다" 안내가 나옵니다. (YOLO 모델을 모든 탭이 같이 써서, 동시에 돌리면 추적 결과가 섞이기 때문)
4. **탐지 결과 질문** : CCTV에 대해 물어보면 탐지 데이터를 근거로 답합니다.
5. **[대화 지우기]** : 대화를 새로 시작합니다.

### 예시 질문
- "차가 몇 대 지나갔어?"
- "사람도 있었어?"
- "가장 많이 탐지된 건 뭐야?"
- "교통이 막히는 편이야?"

---

## 📊 토큰 사용량 / 요금

| AI | 요금 | 한도 |
|---|---|---|
| Groq `gpt-oss-120b` | 무료 | 하루 1000번, 1분 8000 토큰 (무료 키 기준) |
| Claude | 유료 (토큰당 과금) | 충전한 크레딧만큼 |

> Groq 한도를 넘으면 잠깐 막혔다가 시간이 지나면 풀립니다. 요금은 나오지 않습니다.

### 질문 1번에 드는 입력 토큰 (실측)

CCTV 탐지 결과가 있을 때 "지금 차가 몇 대 지나갔어?" 를 물어본 값입니다.

| 시스템 프롬프트 | Claude | Groq |
|---|---|---|
| 한글로 작성 (이전) | 574 | 426 |
| **영어로 작성 (지금)** | **468** (−18%) | **382** (−10%) |

- 여기에 답변(출력) 토큰이 더해집니다.
- 대화가 이어지면 이전 대화(최대 10개)도 같이 보내므로 그만큼 늘어납니다.

### 토큰을 줄이려고 적용한 것

| 방법 | 파일 | 효과 |
|---|---|---|
| 시스템 프롬프트를 **영어로** 작성 | `app.py` | 시스템 프롬프트는 질문할 때마다 매번 보내지므로 줄인 만큼 계속 절약. 답변은 한국어로 하도록 지시 |
| **최근 대화 10개만** 전송 | `llm.py` | 대화가 길어져도 토큰이 일정 수준 이상 늘지 않음 (화면에는 전체 대화가 남음) |
| 추론 강도 **low** | `llm.py` | Groq gpt-oss `reasoning_effort="low"`, Claude `effort: "low"` → 답하기 전 생각하는 토큰을 줄임 |

---

## 🧯 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `'streamlit' 용어는 ... 인식되지 않습니다` | PowerShell에서 가상환경이 안 켜짐 | 터미널을 **cmd** 로 바꾸고 `conda activate py311` 후 실행 |
| `'conda'은(는) ... 명령이 아닙니다` | Anaconda가 PATH에 없음 | 시작 메뉴의 **Anaconda Prompt** 로 실행하거나, Anaconda 재설치 시 "Add to PATH" 체크 |
| `missing ScriptRunContext` 경고가 잔뜩 나옴 | ▶ 버튼(F5)으로 실행함 | 터미널에서 `streamlit run app.py` 로 실행 |
| `ModuleNotFoundError: No module named ...` | 라이브러리 미설치 / 가상환경 안 켜짐 | `(py311)` 확인 후 `pip install -r requirements.txt` |
| "⚠️ .env 에 GROQ_API_KEY 가 없습니다" | `.env` 파일이 없거나 이름이 틀림 | [.env 파일 만들기](#-env-파일-만들기) 확인 (`.env.txt` 가 아닌지 확인) |
| "CCTV 목록을 불러오지 못했습니다" | `ITS_API_KEY` 없음 / 인터넷 | `.env` 에 `ITS_API_KEY` 확인 |
| "CCTV 영상을 열 수 없습니다" | 해당 CCTV 서버 문제 | 다른 CCTV 선택 (10초 안에 자동으로 멈춤) |
| "분석 중 오류가 발생했습니다" | 영상 프레임 오류 등 | 다시 분석하거나 다른 CCTV 선택. 영상이 중간에 끊기면 요약의 **분석 시간**이 고른 시간보다 짧게 나옵니다 |
| "Claude API 키가 올바르지 않거나 만료됐습니다" | 키 오류 / 30일 만료 | platform.claude.com 에서 새로 발급 → `.env` 교체 → 앱 재실행 |
| `Port 8501 is already in use` | 앱이 이미 켜져 있음 | 기존 터미널에서 `Ctrl + C` 또는 `streamlit run app.py --server.port 8502` |

---

## 📝 참고

### 탐지 숫자 읽는 법

| 값 | 뜻 | 믿을 만한 정도 |
|---|---|---|
| **한 화면 최대** | 한 프레임에 동시에 잡힌 가장 많은 수 | ✅ 가장 믿을 만함 |
| **평균** | 프레임마다 잡힌 수의 평균 | ✅ 가장 믿을 만함 |
| **지나간 수** | 추적 ID 개수 | ⚠️ 참고용. 저화질 CCTV에서는 같은 차의 ID가 바뀌거나 버스↔트럭으로 종류가 바뀌어 **실제보다 많게** 나올 수 있음 |

- **탐지 대상** : 도로 객체만 셉니다 (사람·자전거·승용차·오토바이·버스·트럭).
- **추적 설정** : CCTV 화질이 낮아(예: 720x480) 차가 낮은 신뢰도로 잡히기 때문에, `bytetrack_cctv.yaml` 에서 추적 기준을 ultralytics 기본값(0.25)보다 낮은 0.15 로 설정했습니다.

### 기타

- Claude Opus/Sonnet은 거절(refusal) 시 서버가 다른 모델로 이어서 답하는 기능(`fallbacks="default"`)을 켜 두었습니다.
- CCTV 목록 조회 코드는 `261019_6_openapi/v06_03_openapi_yolo.py`의 `its_cctv()`를 바탕으로 만들었습니다.

---

## 🧾 처음 요구사항 (구현 프롬프트)

이 챗봇을 처음 만들 때 Claude Code에 입력한 요구사항입니다. (작성한 그대로)

```
# CLAUDE API & CLAUDE CODE AI를 활용한 Chatbot 제작

# 파일명 : 261020_buchen_chatbot_yjy
# 작업폴더 경로 : wori/261020_7_groq_api/261020_buchen_chatbot_yjy

# 1. groq_api_key 활용하여 chatbot 제작.
# 2. 클로드 활용하여 로컬 형태 수준 혹시 자유롭게 개발
#   자유롭게 개발한다면 CLAUDE API를 사용하여 챗봇 제작을 할 것임.
# 3. .env 관리 + (.gitignore도 작성--> 이미 작성되어있다면 가져와서 쓸것)
# 4. requirment .txt 생성 필수(필요 라이브러리 목록 확인)
# 5. README.md(사용법 작성 필수)
# 6. YOLO11n.pt 모델도 같이 엮어서 사용 할 수 있는지?
#   ex) 영상안의 탐지된 내용을 챗봇으로 출력해준다.
```

> 작업 폴더는 이후 `Wori/261020_buchen_chatbot_yjy/` 로 옮겨서 별도 레포로 관리합니다.

### 요구사항별 구현 결과

| # | 요구사항 | 구현 |
|---|---|---|
| 1 | Groq API 챗봇 | `llm.py` 의 `groq_stream()`, 기본 모델 `openai/gpt-oss-120b` |
| 2 | Claude API 챗봇 | `llm.py` 의 `claude_stream()`, 사이드바에서 Groq / Claude 전환 |
| 3 | `.env` + `.gitignore` | 키는 `.env` 에만 저장, `.gitignore` 로 업로드 제외 |
| 4 | `requirements.txt` | 8개 라이브러리 ([5단계](#5단계-라이브러리-설치) 참고) |
| 5 | `README.md` 사용법 | 이 문서 |
| 6 | YOLO11n 연동 | ITS CCTV 영상을 YOLO11n 으로 추적 → 탐지 요약을 시스템 프롬프트에 넣어 챗봇이 답함 |
