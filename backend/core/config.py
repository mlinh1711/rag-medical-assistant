from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str = ""
    data_path: str = "./data"
    vector_db_path: str = "./data/vectorstore"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    class Config:
        env_file = ".env"

settings = Settings()