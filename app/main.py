import asyncio
import os
import shutil
import uuid
from functools import partial
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.script_generator import generate_script, generate_script_from_images, extract_from_url
from app.tts import generate_audio
from app.video_generator import create_video
from app.config import settings
from langchain_community.document_loaders import PyPDFLoader

app = FastAPI(title="Paper to Shorts", version="0.3.0")

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(settings.output_dir, exist_ok=True)

app.mount("/output", StaticFiles(directory=settings.output_dir), name="output")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


def _save_to_dir(src: str, title: str, save_dir: str) -> str | None:
    try:
        expanded = os.path.expanduser(save_dir.strip())
        if not os.path.isdir(expanded):
            return None
        safe = "".join(c for c in title if c.isalnum() or c in " _-")[:28].strip() or "shorts"
        job_id = os.path.basename(src).split(".")[0]
        dest = os.path.join(expanded, f"{safe}_{job_id[:6]}.mp4")
        shutil.copy2(src, dest)
        return dest
    except Exception as e:
        print(f"[save] failed to copy to {save_dir}: {e}")
        return None


def _cleanup(*paths: str):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


async def _run_sync(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args, **kwargs))


async def _pipeline(content: str, save_dir: str = "", bg_images: list[str] | None = None, mood: str = "dark") -> dict:
    if not save_dir:
        save_dir = settings.default_save_dir

    job_id = str(uuid.uuid4())[:8]
    audio_path = f"{settings.output_dir}/{job_id}.mp3"
    video_path = f"{settings.output_dir}/{job_id}.mp4"

    try:
        if bg_images:
            script = await _run_sync(generate_script_from_images, bg_images, content, mood)
        else:
            script = await _run_sync(generate_script, content, mood)

        await _run_sync(generate_audio, script["narration"], audio_path)
        await _run_sync(create_video, script, audio_path, video_path, bg_images, mood)

    except Exception as e:
        _cleanup(audio_path, video_path)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup(audio_path)  # mp3는 항상 삭제

    saved_path = _save_to_dir(video_path, script["title"], save_dir) if save_dir else None

    return {
        "script": script,
        "video_url": f"/output/{job_id}.mp4",
        "saved_path": saved_path,
    }


@app.post("/generate/pdf")
async def generate_from_pdf(
    file: UploadFile = File(...),
    save_dir: str = Form(""),
    mood: str = Form("dark"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    job_id = str(uuid.uuid4())[:8]
    pdf_path = f"{settings.output_dir}/{job_id}.pdf"

    try:
        with open(pdf_path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")
        content = " ".join(d.page_content for d in docs[:10])[:3000]
        if not content.strip():
            raise HTTPException(status_code=400, detail="PDF 내용이 비어있습니다.")

        return await _pipeline(content, save_dir, mood=mood)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup(pdf_path)


@app.post("/generate/url")
async def generate_from_url(
    url: str = Form(...),
    save_dir: str = Form(""),
    mood: str = Form("dark"),
):
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="올바른 URL을 입력해주세요.")
    try:
        content = await _run_sync(extract_from_url, url)
        if not content.strip():
            raise HTTPException(status_code=400, detail="URL에서 내용을 추출할 수 없습니다.")
        return await _pipeline(content, save_dir, mood=mood)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/images")
async def generate_from_images(
    files: list[UploadFile] = File(...),
    caption: str = Form(""),
    save_dir: str = Form(""),
    mood: str = Form("dark"),
):
    if not files:
        raise HTTPException(status_code=400, detail="이미지를 업로드해주세요.")

    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    job_id = str(uuid.uuid4())[:8]
    saved_paths: list[str] = []

    try:
        for i, f in enumerate(files[:5]):
            ext = os.path.splitext(f.filename or "")[1].lower()
            if ext not in allowed:
                raise HTTPException(status_code=400, detail=f"지원하지 않는 형식: {f.filename}")
            path = f"{settings.output_dir}/upload_{job_id}_{i}{ext}"
            with open(path, "wb") as out:
                out.write(await f.read())
            saved_paths.append(path)

        return await _pipeline(caption, save_dir, bg_images=saved_paths, mood=mood)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _cleanup(*saved_paths)
