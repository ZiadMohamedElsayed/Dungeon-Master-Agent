from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    lore_db_persist_dir: str = Field(default="./dp/world_lore", alias="LORE_DB_PERSIST_DIR")
    campaign_db_persist_dir: str = Field(default="./dp/campaign_history", alias="CAMPAIGN_DB_PERSIST_DIR")
    # Backwards-compat: old var names still accepted if set
    lore_dp_persist_dir: str = "./dp/world_lore"
    campaign_dp_persist_dir: str = "./dp/campaign_history"
    embed_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBED_MODEL")
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL")
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    top_k_retrieve: int = Field(default=10, alias="TOP_K_RETRIEVE")
    top_k_rerank: int = Field(default=4, alias="TOP_K_RERANK")
    short_term_turns: int = Field(default=5, alias="SHORT_TERM_TURNS")
    short_term_max_chars: int = Field(default=1500, alias="SHORT_TERM_MAX_CHARS")

    def resolved_lore_dir(self) -> str:
        # Prefer new var unless it is still default and old var was customized
        return self.lore_db_persist_dir

    def resolved_campaign_dir(self) -> str:
        return self.campaign_db_persist_dir


settings = Settings()
