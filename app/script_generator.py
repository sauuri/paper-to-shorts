import base64
import json
import time
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from app.config import settings

# ─── 분위기별 설정 ──────────────────────────────────────────────────────────────
MOOD_CONFIG = {
    "dark": {
        "img_prefix": (
            "Cinematic noir, dark dramatic lighting, high contrast, "
            "photorealistic, no text, no faces, 9:16 vertical, moody atmosphere — "
        ),
        "tone": "긴장·공포·분노 — 사건/범죄 뉴스에 어울리는 무겁고 강렬한 톤",
        "emotion": "분노·놀람·공포·안타까움 중 하나를 선택해 일관되게 유지",
    },
    "bright": {
        "img_prefix": (
            "Bright cinematic, soft natural lighting, vibrant colors, "
            "photorealistic, no text, no faces, 9:16 vertical, uplifting atmosphere — "
        ),
        "tone": "희망·감동·흥미 — 긍정적 뉴스·기술·사회 이슈에 어울리는 밝고 경쾌한 톤",
        "emotion": "희망·감동·놀라움·흥미 중 하나를 선택해 일관되게 유지",
    },
}

# ─── JSON 스키마 ────────────────────────────────────────────────────────────────
_JSON_SCHEMA = """{
  "title": "2026년 최악의 사기 사건",
  "hook": "단 3일 만에 100억이 사라졌다",
  "points": [
    "언제·어디서: 2026년 3월, 서울 강남구에서 발생",
    "누가·무엇을: 전직 금융사 임원이 투자 사기를 주도",
    "어떻게: 300명 피해자, 평균 3천만 원씩 편취",
    "결과: 검거 당시 해외 도피 직전 상태였다",
    "현재: 피해금 회수율 5%, 수사 계속 중"
  ],
  "narration": "단 3일 만에 100억이 사라졌다. 2026년 3월... (300~340자 완성된 나레이션)",
  "outro_question": "이런 사기, 어떻게 막을 수 있을까?",
  "hashtags": ["#사기사건", "#실제사건", "#사건파일", "#충격사건", "#범죄"],
  "image_prompts": [
    "Cinematic noir ... — empty vault with scattered documents",
    "Cinematic noir ... — crowded city street at night",
    "Cinematic noir ... — abandoned office building interior",
    "Cinematic noir ... — surveillance camera footage aesthetic",
    "Cinematic noir ... — courtroom silhouette with harsh light",
    "Cinematic noir ... — dark foggy harbor at night"
  ]
}"""

_SCRIPT_TEMPLATE = """당신은 유튜브 숏츠 전문 스크립트 작가입니다.
아래 콘텐츠를 읽고 60초 분량의 한국어 숏츠 스크립트를 작성해주세요.

━━ 분위기 ━━
{tone}

━━ 스토리텔링 원칙 ━━
① 훅: 첫 1초 안에 손가락이 멈추는 강렬한 한 문장으로 시작
② 육하원칙 필수: 포인트는 언제·어디서·누가·무엇을·어떻게·왜를 나눠서 담을 것
③ 발단→전개→절정→결과→여운 구조로 서사 흐름 유지
④ 감정: {emotion}
⑤ 언어: 보고서 문체 금지. 친구에게 말하듯 생생하고 구체적으로
⑥ 마무리: 댓글 유도 질문으로 끝. 구독/좋아요 절대 금지

━━ 사실 정확성 원칙 (반드시 준수) ━━
- 입력 콘텐츠에 있는 사실만 사용. 없는 내용 창작 절대 금지
- 날짜·장소·인물·수치는 원문 그대로 사용
- 확인되지 않은 내용은 "~로 알려졌다", "~라고 전해진다" 표현으로 명시
- points 각 항목은 실제 사실 한 가지만. 지시문·설명 포함 금지

━━ image_prompts 작성 규칙 ━━
- 반드시 영어로 작성, 반드시 6개 (title + point1~5)
- 각 프롬프트는 반드시 이 prefix로 시작: "{img_prefix}"
- 그 뒤에 장면 내용에 맞는 구체적 묘사 추가

━━ 출력 형식 ━━
반드시 아래 JSON 형식으로만 응답. 다른 텍스트 절대 금지.

{json_schema}

━━ 입력 콘텐츠 ━━
{{content}}"""

_VISION_TEMPLATE = """당신은 유튜브 숏츠 전문 스크립트 작가입니다.
첨부된 이미지들을 분석하여 60초 분량의 한국어 숏츠 스크립트를 작성해주세요.

━━ 분위기 ━━
{tone}

━━ 스토리텔링 원칙 ━━
① 이미지에서 가장 인상적인 요소를 훅으로 사용
② 이미지 속 시각적 정보를 스토리로 연결
③ 육하원칙(언제·어디서·누가·무엇을·어떻게·왜)을 포인트에 분산해서 담을 것
④ 이미지에서 확인되는 사실만 사용. 없는 내용 창작 금지
⑤ 감정: {emotion}
⑥ 마무리: 댓글 유도 질문으로 끝. 구독/좋아요 절대 금지

━━ image_prompts 작성 규칙 ━━
- 반드시 영어로 작성, 반드시 6개 (title + point1~5)
- 각 프롬프트는 반드시 이 prefix로 시작: "{img_prefix}"

━━ 출력 형식 ━━
반드시 아래 JSON 형식으로만 응답.

{json_schema}"""


def _build_prompts(mood: str) -> tuple[str, str]:
    cfg = MOOD_CONFIG.get(mood, MOOD_CONFIG["dark"])
    escaped = _JSON_SCHEMA.replace("{", "{{").replace("}", "}}")
    script = _SCRIPT_TEMPLATE.format(
        tone=cfg["tone"], emotion=cfg["emotion"],
        img_prefix=cfg["img_prefix"], json_schema=escaped,
    )
    vision = _VISION_TEMPLATE.format(
        tone=cfg["tone"], emotion=cfg["emotion"],
        img_prefix=cfg["img_prefix"], json_schema=escaped,
    )
    return script, vision


def extract_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=15)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ", strip=True).split())[:3000]


def _extract_json(text: str) -> str:
    text = text.strip()
    for marker in ["```json", "```"]:
        if marker in text:
            text = text.split(marker)[1].split("```")[0].strip()
            break
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text


def _repair_script(data: dict, mood: str) -> dict:
    cfg = MOOD_CONFIG.get(mood, MOOD_CONFIG["dark"])
    title = data.get("title") or "사건파일"

    points = [p for p in data.get("points", []) if p]
    while len(points) < 5:
        points.append(f"핵심 포인트 {len(points) + 1}")
    data["points"] = points[:5]

    prompts = [p for p in data.get("image_prompts", []) if p]
    while len(prompts) < 6:
        prompts.append(f"{cfg['img_prefix']}{title}")
    data["image_prompts"] = prompts[:6]

    data.setdefault("outro_question", "어떻게 생각해?")
    data.setdefault("hashtags", [])
    data.setdefault("hook", data.get("narration", "")[:20])
    data.setdefault("narration", " ".join(data["points"]))
    return data


def _parse_and_repair(text: str, mood: str) -> dict:
    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw + "}")
        except Exception:
            raise ValueError(f"GPT 응답을 JSON으로 파싱할 수 없습니다:\n{text[:300]}")
    return _repair_script(data, mood)


def _call_with_retry(fn, retries: int = 2, delay: float = 2.0):
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay)
    raise last_exc


def generate_script(content: str, mood: str = "dark") -> dict:
    if not content.strip():
        raise ValueError("내용이 없습니다.")
    script_prompt, _ = _build_prompts(mood)

    def _call():
        llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.85,
        )
        chain = PromptTemplate(template=script_prompt, input_variables=["content"]) | llm
        result = chain.invoke({"content": content})
        return _parse_and_repair(result.content, mood)

    return _call_with_retry(_call)


def generate_script_from_images(image_paths: list[str], caption: str = "", mood: str = "dark") -> dict:
    _, vision_prompt = _build_prompts(mood)
    client = OpenAI(api_key=settings.openai_api_key)

    content_msgs = [{"type": "text", "text": vision_prompt}]
    if caption.strip():
        content_msgs.append({"type": "text", "text": f"\n추가 설명: {caption}"})

    for path in image_paths[:5]:
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content_msgs.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        })

    def _call():
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content_msgs}],
            temperature=0.85,
        )
        return _parse_and_repair(response.choices[0].message.content, mood)

    return _call_with_retry(_call)
