from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.script_generator import generate_script, extract_from_url
from app.tts import generate_audio
from app.video_generator import create_video
from app.config import settings
from langchain_community.document_loaders import PyPDFLoader
import os
import uuid

app = FastAPI(title="Paper to Shorts", version="0.1.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

app.mount("/output", StaticFiles(directory=settings.output_dir), name="output")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


async def _run_pipeline(content: str, job_id: str) -> dict:
    audio_path = f"{settings.output_dir}/{job_id}.mp3"
    video_path = f"{settings.output_dir}/{job_id}.mp4"

    script = generate_script(content)
    generate_audio(script["narration"], audio_path)
    create_video(script, audio_path, video_path)

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return {"script": script, "video_url": f"/output/{job_id}.mp4"}


@app.post("/generate/pdf")
async def generate_from_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    job_id = str(uuid.uuid4())[:8]
    pdf_path = f"{settings.output_dir}/{job_id}.pdf"

    with open(pdf_path, "wb") as f:
        f.write(await file.read())

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        content = " ".join([d.page_content for d in docs[:10]])[:3000]
        return await _run_pipeline(content, job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


@app.post("/generate/url")
async def generate_from_url(url: str = Form(...)):
    try:
        content = extract_from_url(url)
        job_id = str(uuid.uuid4())[:8]
        return await _run_pipeline(content, job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
