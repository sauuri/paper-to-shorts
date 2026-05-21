from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    tts_model: str = "tts-1"
    tts_voice: str = "nova"
    output_dir: str = "output"
    pexels_api_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
