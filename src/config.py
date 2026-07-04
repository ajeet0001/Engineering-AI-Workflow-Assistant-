"""
config.py
---------
Centralized configuration for the Engineering AI Workflow Assistant.

Loads all settings from environment variables / .env file using Pydantic Settings.
Import `settings` anywhere in the project to access configuration values.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or a .env file.

    All fields have sensible defaults except GOOGLE_API_KEY, which must
    be provided by the user.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Gemini API ──────────────────────────────────────────────────────────
    google_api_key: str = Field(
        default="",
        alias="GOOGLE_API_KEY",
        description="Google Gemini API key",
    )

    # ── Model Selection ─────────────────────────────────────────────────────
    llm_model: str = Field(
        default="gemini-2.0-flash",
        alias="LLM_MODEL",
        description="Gemini model name to use for LLM calls",
    )

    # ── Embedding Model ─────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
        description="Sentence Transformer model for RAG embeddings",
    )

    # ── RAG Settings ────────────────────────────────────────────────────────
    chunk_size: int = Field(
        default=500,
        alias="CHUNK_SIZE",
        description="Character size of each text chunk for RAG",
    )
    chunk_overlap: int = Field(
        default=50,
        alias="CHUNK_OVERLAP",
        description="Overlap between consecutive text chunks",
    )
    top_k_results: int = Field(
        default=5,
        alias="TOP_K_RESULTS",
        description="Number of top documents to retrieve from FAISS",
    )

    # ── Directory Paths ─────────────────────────────────────────────────────
    knowledge_base_dir: Path = Field(
        default=Path("data/knowledge_base"),
        alias="KNOWLEDGE_BASE_DIR",
        description="Directory containing engineering knowledge base documents",
    )
    vector_store_dir: Path = Field(
        default=Path("vector_store"),
        alias="VECTOR_STORE_DIR",
        description="Directory where the FAISS index is persisted",
    )
    reports_dir: Path = Field(
        default=Path("data/reports"),
        alias="REPORTS_DIR",
        description="Directory where input PDF reports are stored",
    )
    outputs_dir: Path = Field(
        default=Path("outputs"),
        alias="OUTPUTS_DIR",
        description="Directory where generated Markdown reports are saved",
    )

    def validate_api_key(self) -> None:
        """Raise a clear error if the Gemini API key is missing."""
        if not self.google_api_key or self.google_api_key == "your_gemini_api_key_here":
            raise ValueError(
                "\n\n[CONFIG ERROR] GOOGLE_API_KEY is not set.\n"
                "  1. Copy .env.example to .env\n"
                "  2. Open .env and set:  GOOGLE_API_KEY=your_actual_key\n"
                "  Get a free key at: https://aistudio.google.com/\n"
            )


# Singleton settings instance — import this everywhere
settings = Settings()
