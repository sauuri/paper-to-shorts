# Paper to Shorts

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/GPT--4o--mini-LangChain-412991?style=flat-square&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenAI_TTS-nova-10A37F?style=flat-square&logo=openai&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pexels-royalty--free-05A081?style=flat-square"/>
  <img src="https://img.shields.io/badge/moviepy-2.0-FF6B6B?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

<p align="center">
논문 PDF 또는 뉴스 기사 URL을 입력하면<br/>
AI가 자동으로 유튜브 숏츠(1080×1920) 영상을 생성합니다.
</p>

---

## 파이프라인

```
PDF / URL 입력
      │
      ▼
┌─────────────────┐
│  GPT-4o-mini    │  스크립트 생성 (제목·후크·5포인트·나레이션·키워드)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI TTS     │  한국어 음성 합성 (nova voice, 1.1×)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Pexels API     │  키워드별 배경 영상 자동 검색 + 다운로드
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  moviepy + Pillow               │
│  ├── 제목 슬라이드 (BREAKING 태그) │
│  ├── 핵심 포인트 × 5 (진행바)    │
│  └── 나레이션 오디오 합성        │
└────────┬────────────────────────┘
         │
         ▼
   output.mp4 (1080×1920 · 60초)
```

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **PDF 업로드** | 논문, 보고서 PDF → 자동 요약 후 숏츠 생성 |
| **URL 입력** | 뉴스 기사, 블로그 URL → 크롤링 후 숏츠 생성 |
| **AI 스크립트** | GPT-4o-mini가 후크·5포인트·나레이션 구조화 생성 |
| **TTS 음성** | OpenAI TTS nova voice로 자연스러운 한국어 나레이션 |
| **배경 영상** | Pexels 로열티프리 영상 자동 매칭 + 폴백 처리 |
| **세로 영상** | 1080×1920 유튜브 숏츠 규격, 30fps |
| **웹 UI** | 드래그&드롭 업로드, 실시간 진행 표시, 영상 미리보기 |

---

## 빠른 시작

```bash
git clone https://github.com/sauuri/paper-to-shorts.git
cd paper-to-shorts

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env에 OPENAI_API_KEY, PEXELS_API_KEY 입력

uvicorn app.main:app --reload
```

브라우저에서 http://localhost:8000 접속

---

## 환경 변수

| 변수 | 설명 |
|------|------|
| `OPENAI_API_KEY` | OpenAI API 키 |
| `PEXELS_API_KEY` | [Pexels API](https://www.pexels.com/api/) 키 (무료) |
| `LLM_MODEL` | 기본값: `gpt-4o-mini` |
| `TTS_VOICE` | 기본값: `nova` |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| LLM | GPT-4o-mini, LangChain |
| TTS | OpenAI TTS (tts-1) |
| 영상 편집 | moviepy 2.0, Pillow |
| 배경 소스 | Pexels API (royalty-free) |
| 크롤링 | BeautifulSoup4, requests |
| PDF 파싱 | PyPDF (LangChain) |
| Backend | FastAPI |

---

## License

MIT
