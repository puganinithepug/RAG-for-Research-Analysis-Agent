from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    documents_path: str
    chroma_path: str
    collection_name: str
    checkpoint_db: str

    ollama_base_url: str
    llm_model: str
    embedding_model: str
    grader_model: str

    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_k: int
    max_attempts: int

    faithfulness_threshold: float
    evaluator_model: str





settings = Settings()

