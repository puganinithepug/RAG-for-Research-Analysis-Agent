from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    documents_path: str = "./documents"
    chroma_path: str = "./data/chroma"
    collection_name: str = "research"

    ollama_base_url: str = "http://ollama:11434"
    llm_model: str = "qwen3:8b"
    embedding_model: str = "nomic-embed-text"

    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    max_k: int = 12
    max_attempts: int = 2

    faithfulness_threshold: float = 0.75
    evaluator_model: str = "qwen3:8b"



settings = Settings()

