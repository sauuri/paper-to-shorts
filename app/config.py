from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    tts_model: str = "tts-1-hd"
    tts_voice: str = "onyx"
    output_dir: str = "output"
    pexels_api_key: str = ""
    default_save_dir: str = "/Users/jhlee/Documents/유튜브 숏츠"

    class Config:
        env_file = ".env"


settings = Settings()
