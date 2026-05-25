from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str = ""
    lore_dp_persist_dir: str = "./dp/world_lore"
    campaign_dp_persist_dir: str = "./dp/campaign_history"
    embed_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "gemini-3.5-flash"
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieve: int = 10
    top_k_rerank: int = 4

    class Config:
        env_file = ".env"


settings = Settings()
