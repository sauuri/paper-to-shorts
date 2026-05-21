from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from app.config import settings
import requests
from bs4 import BeautifulSoup
import json

PROMPT = """당신은 유튜브 숏츠 스크립트 작가입니다.
아래 콘텐츠(논문, 뉴스 기사, 블로그 등)를 읽고 60초 분량의 한국어 숏츠 스크립트를 작성해주세요.

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
  "narration": "전체 나레이션 텍스트 (280~320자)",
  "keywords": ["Pexels 검색용 영어 키워드1 (Pexels에 반드시 있을 일반적인 단어, 예: city, people, nature, sky, technology)", "키워드2", "키워드3", "키워드4", "키워드5"]
}}

문서 내용:
{content}"""


def extract_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return " ".join(text.split())[:3000]


def generate_script(content: str) -> dict:

    if not content.strip():
        raise ValueError("내용이 없습니다.")

    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.7,
    )

    prompt = PromptTemplate(template=PROMPT, input_variables=["content"])
    chain = prompt | llm

    result = chain.invoke({"content": content})
    text = result.content.strip()

    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    return json.loads(text)
