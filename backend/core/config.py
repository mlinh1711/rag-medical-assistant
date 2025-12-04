from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # cấu hình LLM
    llm_provider: str = "deepseek"          # đọc từ LLM_PROVIDER trong .env
    deepseek_api_key: str = ""              # DEEPSEEK_API_KEY
    deepseek_model: str = "deepseek-chat"   # DEEPSEEK_MODEL
    voyage_api_key: str = ""                # VOYAGE_API_KEY

    # cấu hình dữ liệu
    data_path: str = "./data"
    vector_db_path: str = "./data/vectorstore"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # đọc biến môi trường từ file .env ở thư mục backend/
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
