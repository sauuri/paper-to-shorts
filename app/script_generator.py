import base64
import json
import time
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from app.config import settings

# ─── JSON 스키마 ────────────────────────────────────────────────────────────────
_JSON_SCHEMA = """{
  "title": "제목 (15자 이내, 클릭하고 싶게)",
  "hook": "첫 문장 — 가장 충격적인 사실 또는 질문. 평범하면 실패 (20자 이내)",
  "points": [
    "포인트1: 배경/발단 — 상황을 설정하는 충격적 사실 (35자 이내)",
    "포인트2: 전개 — 긴장감 고조, '그런데...' 또는 '심지어...' 시작 (35자 이내)",
    "포인트3: 절정 — 가장 충격적이거나 반전되는 지점 (35자 이내)",
    "포인트4: 결과 — 사건의 결말 또는 영향 (35자 이내)",
    "포인트5: 여운 — 아직 풀리지 않은 의문 또는 교훈 (35자 이내)"
  ],
  "narration": "전체 나레이션 (300~340자). 첫 문장=hook. 문장 사이 리듬감 있게. 마지막 문장은 outro_question을 자연스럽게 녹여서 끝낼 것. 구독/좋아요 언급 절대 금지.",
  "outro_question": "시청자가 댓글 달고 싶어지는 짧고 강한 질문 (20자 이내, 물음표 필수)",
  "hashtags": ["#관련태그1", "#관련태그2", "#관련태그3", "#관련태그4", "#관련태그5"],
  "image_prompts": [
    "Title scene — English cinematic image prompt (detailed, 40 words max)",
    "Point 1 scene — English cinematic image prompt",
    "Point 2 scene — English cinematic image prompt",
    "Point 3 scene — English cinematic image prompt",
    "Point 4 scene — English cinematic image prompt",
    "Point 5 scene — English cinematic image prompt"
  ]
}"""

SCRIPT_PROMPT = """당신은 유튜브 숏츠 채널 "사건파일"의 전문 스크립트 작가입니다.
범죄·사회·과학·역사 속 충격적 사건을 다루며 시청자가 손가락을 멈추고 끝까지 볼 수밖에 없는 영상을 만듭니다.

━━ 스토리텔링 원칙 ━━
① 훅: 첫 1초 안에 "이게 뭐야?" 반응 유도 — 가장 충격적인 사실 또는 질문으로 시작
② 긴장 고조: 포인트마다 발단→전개→절정→결과→여운 구조로 서사 흐름 유지
③ 감정 자극: 분노·놀람·공포·안타까움 중 하나를 선택해 일관되게 유지
④ 언어: 딱딱한 보고서 문체 금지. 친구에게 말하듯 생생하고 구체적으로
⑤ 마무리: 댓글 달고 싶어지는 질문으로 끝. 구독/좋아요 절대 금지

━━ image_prompts 작성 규칙 ━━
- 반드시 영어로 작성, 반드시 6개 (title + point1~5)
- prefix: "Cinematic noir, dark dramatic lighting, high contrast, photorealistic, no text, no faces, 9:16 vertical, moody atmosphere — "
- 그 뒤에 장면별 구체적 묘사 추가

━━ 출력 형식 ━━
반드시 아래 JSON 형식으로만 응답. 다른 텍스트 절대 금지.

{json_schema}

━━ 입력 콘텐츠 ━━
{{content}}"""

VISION_PROMPT = """당신은 유튜브 숏츠 채널 "사건파일"의 전문 스크립트 작가입니다.
첨부된 이미지들을 분석하여 60초 분량의 한국어 숏츠 스크립트를 작성해주세요.

━━ 스토리텔링 원칙 ━━
① 이미지에서 가장 충격적이거나 흥미로운 요소를 훅으로 사용
② 이미지 속 시각적 정보를 스토리로 연결
③ 마무리: 댓글 달고 싶어지는 질문으로 끝. 구독/좋아요 절대 금지

━━ image_prompts 작성 규칙 ━━
- 반드시 영어로 작성, 반드시 6개 (title + point1~5)
- prefix: "Cinematic noir, dark dramatic lighting, high contrast, photorealistic, no text, no faces, 9:16 vertical, moody atmosphere — "

━━ 출력 형식 ━━
반드시 아래 JSON 형식으로만 응답.

{json_schema}"""

_escaped = _JSON_SCHEMA.replace("{", "{{").replace("}", "}}")
SCRIPT_PROMPT = SCRIPT_PROMPT.format(json_schema=_escaped)
VISION_PROMPT = VISION_PROMPT.format(json_schema=_escaped)

_NOIR_PREFIX = (
    "Cinematic noir, dark dramatic lighting, high contrast, "
    "photorealistic, no text, no faces, 9:16 vertical, moody atmosphere — "
)


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
    # 첫 { 부터 마지막 } 까지만 추출
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text


def _repair_script(data: dict) -> dict:
    """GPT 출력이 불완전할 때 기본값으로 보완"""
    title = data.get("title") or "사건파일"

    # points: 반드시 5개
    points = [p for p in data.get("points", []) if p]
    while len(points) < 5:
        points.append(f"핵심 포인트 {len(points) + 1}")
    data["points"] = points[:5]

    # image_prompts: 반드시 6개
    prompts = [p for p in data.get("image_prompts", []) if p]
    while len(prompts) < 6:
        prompts.append(f"{_NOIR_PREFIX}{title}")
    data["image_prompts"] = prompts[:6]

    data.setdefault("outro_question", "어떻게 생각해?")
    data.setdefault("hashtags", [])
    data.setdefault("hook", data.get("narration", "")[:20])
    data.setdefault("narration", " ".join(data["points"]))
    return data


def _parse_and_repair(text: str) -> dict:
    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 마지막 수단: 부분 JSON 복구 시도
        try:
            data = json.loads(raw + "}")
        except Exception:
            raise ValueError(f"GPT 응답을 JSON으로 파싱할 수 없습니다:\n{text[:300]}")
    return _repair_script(data)


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


def generate_script(content: str) -> dict:
    if not content.strip():
        raise ValueError("내용이 없습니다.")

    def _call():
        llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=0.85,
        )
        chain = PromptTemplate(template=SCRIPT_PROMPT, input_variables=["content"]) | llm
        result = chain.invoke({"content": content})
        return _parse_and_repair(result.content)

    return _call_with_retry(_call)


def generate_script_from_images(image_paths: list[str], caption: str = "") -> dict:
    client = OpenAI(api_key=settings.openai_api_key)

    content_msgs = [{"type": "text", "text": VISION_PROMPT}]
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
        return _parse_and_repair(response.choices[0].message.content)

    return _call_with_retry(_call)
