import base64
import json
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from openai import OpenAI
from app.config import settings

SCRIPT_PROMPT = """당신은 유튜브 숏츠 스크립트 작가입니다.
아래 콘텐츠를 읽고 60초 분량의 한국어 숏츠 스크립트를 작성해주세요.

규칙:
- 전체 나레이션은 280~320자 분량
- 흥미롭고 쉽게 설명
- 핵심 포인트 5개 추출
- 반드시 아래 JSON 형식으로만 응답

{{
  "title": "영상 제목 (20자 이내)",
  "hook": "첫 문장 (호기심 유발, 20자 이내)",
  "points": [
    "핵심 포인트 1 (30자 이내)",
    "핵심 포인트 2 (30자 이내)",
    "핵심 포인트 3 (30자 이내)",
    "핵심 포인트 4 (30자 이내)",
    "핵심 포인트 5 (30자 이내)"
  ],
  "narration": "전체 나레이션 텍스트 (280~320자)"
}}

문서 내용:
{content}"""

VISION_PROMPT = """당신은 유튜브 숏츠 스크립트 작가입니다.
첨부된 이미지들을 분석하여 60초 분량의 한국어 숏츠 스크립트를 작성해주세요.
이미지의 내용, 분위기, 핵심 메시지를 파악하여 자연스러운 영상 스크립트로 만들어주세요.

규칙:
- 전체 나레이션은 280~320자 분량
- 흥미롭고 쉽게 설명
- 이미지 내용 기반 핵심 포인트 5개 추출
- 반드시 아래 JSON 형식으로만 응답

{
  "title": "영상 제목 (20자 이내)",
  "hook": "첫 문장 (호기심 유발, 20자 이내)",
  "points": [
    "핵심 포인트 1 (30자 이내)",
    "핵심 포인트 2 (30자 이내)",
    "핵심 포인트 3 (30자 이내)",
    "핵심 포인트 4 (30자 이내)",
    "핵심 포인트 5 (30자 이내)"
  ],
  "narration": "전체 나레이션 텍스트 (280~320자)"
}"""


def extract_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ", strip=True).split())[:3000]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def generate_script(content: str) -> dict:
    if not content.strip():
        raise ValueError("내용이 없습니다.")

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.7,
    )
    chain = PromptTemplate(template=SCRIPT_PROMPT, input_variables=["content"]) | llm
    result = chain.invoke({"content": content})
    return _parse_json(result.content)


def generate_script_from_images(image_paths: list[str], caption: str = "") -> dict:
    """GPT-4o vision으로 이미지 분석 후 스크립트 생성"""
    client = OpenAI(api_key=settings.openai_api_key)

    content = [{"type": "text", "text": VISION_PROMPT}]

    if caption.strip():
        content.append({"type": "text", "text": f"\n추가 설명: {caption}"})

    for path in image_paths[:5]:
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        })

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        temperature=0.7,
    )
    return _parse_json(response.choices[0].message.content)
