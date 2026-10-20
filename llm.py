import os

import anthropic
from groq import Groq


# 1. 사용할 수 있는 모델
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
CLAUDE_MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]

# 채팅용이 아닌 모델 (음성 인식·음성 합성·안전 필터) 은 목록에서 제외
NOT_CHAT = ("whisper", "orpheus", "guard")


# 2. Groq 모델 목록 (서버에서 받아옴, 실패하면 기본 모델만)
def groq_models():
    try:
        models = Groq().models.list().data
        ids = sorted(m.id for m in models if not any(w in m.id for w in NOT_CHAT))
    except Exception:
        ids = []
    if GROQ_DEFAULT_MODEL in ids:
        ids.remove(GROQ_DEFAULT_MODEL)
    return [GROQ_DEFAULT_MODEL] + ids


# 3. Groq 스트리밍 답변 -> 글자를 조금씩 yield
def groq_stream(model, system_prompt, history):
    client = Groq()     # GROQ_API_KEY 환경변수를 자동으로 읽음
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + history,
        stream=True,
    )
    for chunk in stream:
        text = chunk.choices[0].delta.content
        if text:
            yield text


# 4. Claude 스트리밍 답변 -> 글자를 조금씩 yield
def claude_stream(model, system_prompt, history):
    client = anthropic.Anthropic()      # ANTHROPIC_API_KEY 환경변수를 자동으로 읽음

    # 4-1. Opus / Sonnet 은 거절(refusal) 시 서버가 다른 모델로 이어서 답하도록 설정
    #      (Haiku 5.5 는 이 기능을 지원하지 않음)
    extra = {}
    if "haiku" not in model:
        extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

    with client.beta.messages.stream(
        model=model,
        max_tokens=16000,
        system=system_prompt,
        messages=history,
        output_config={"effort": "low"},    # 채팅은 low 로도 충분 -> 빠르고 저렴
        **extra,
    ) as stream:
        for text in stream.text_stream:
            yield text


# 5. 제공사에 맞게 골라서 호출 + 오류를 한글 메시지로
def chat_stream(provider, model, system_prompt, history):
    key_name = "GROQ_API_KEY" if provider == "Groq" else "ANTHROPIC_API_KEY"
    if not os.getenv(key_name):
        yield f"⚠️ .env 에 {key_name} 가 없습니다. README 의 'API 키 설정' 을 참고해 넣어주세요."
        return

    try:
        if provider == "Groq":
            yield from groq_stream(model, system_prompt, history)
        else:
            yield from claude_stream(model, system_prompt, history)
    except anthropic.AuthenticationError as e:
        yield (
            f"⚠️ Claude API 키가 올바르지 않거나 만료됐습니다. ({e.message})\n\n"
            "https://platform.claude.com/settings/keys 에서 새로 발급받아 .env 에 넣어주세요."
        )
    except anthropic.RateLimitError:
        yield "⚠️ 요청이 너무 많습니다. 잠시 후 다시 시도하세요."
    except anthropic.APIConnectionError:
        yield "⚠️ 인터넷 연결을 확인하세요."
    except anthropic.APIStatusError as e:
        yield f"⚠️ Claude API 오류 ({e.status_code}) : {e.message}"
    except Exception as e:     # Groq 오류 등
        yield f"⚠️ 오류가 발생했습니다 : {e}"
