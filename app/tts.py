from openai import OpenAI
from app.config import settings
import os


def generate_audio(text: str, output_path: str) -> str:
    client = OpenAI(api_key=settings.openai_api_key)

    response = client.audio.speech.create(
        model=settings.tts_model,
        voice=settings.tts_voice,
        input=text,
        speed=1.05,
    )

    response.stream_to_file(output_path)
    return output_path
