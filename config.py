"""
Configuration module for the Supreme Court Judgments RAG System.
Handles GCP project settings, Vertex AI model identifiers, Cloud Storage bucket configuration,
Cloud SQL / PostgreSQL connections, and operational parameters for ingestion and online serving.
"""

import os
from typing import Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System-wide configuration settings with environment variable fallbacks."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # --------------------------------------------------------------------------
    # Google Cloud Platform & Vertex AI
    # --------------------------------------------------------------------------
    GCP_PROJECT: str = Field(
        default="sc-legal-rag-prod",
        description="Google Cloud Platform project ID"
    )
    GCP_REGION: str = Field(
        default="asia-south1",
        description="GCP region for Vertex AI endpoints and Cloud Run"
    )

    # Cloud Storage Bucket & Prefixes (v2 Architecture)
    BUCKET_NAME: str = Field(
        default="sc-legal-documents-raw",
        description="Google Cloud Storage bucket name for raw PDFs and parsed Parquet datasets"
    )
    GCS_BUCKET: str = Field(
        default="sc-legal-documents-raw",
        description="Alias for BUCKET_NAME"
    )
    GCS_RAW_PREFIX: str = Field(
        default="raw/",
        description="GCS blob prefix for raw unparsed Supreme Court judgment PDFs"
    )
    GCS_PARQUET_PREFIX: str = Field(
        default="parsed/",
        description="GCS blob prefix for segmented Parquet document tables"
    )
    GCS_CHUNKS_PREFIX: str = Field(
        default="chunks/",
        description="GCS blob prefix for fine-grained chunk embeddings tables"
    )
    GCS_EXPORTS_PREFIX: str = Field(
        default="exports/",
        description="GCS blob prefix for database analytical exports"
    )
    GCS_LOGS_PREFIX: str = Field(
        default="logs/",
        description="GCS blob prefix for batch ingestion execution audit logs"
    )

    # Vertex AI Model Specifications
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Vertex AI Gemini model for metadata extraction and citation verification"
    )
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-004",
        description="Vertex AI embedding model for legal document representation"
    )
    EMBEDDING_DIMENSION: int = Field(
        default=768,
        description="Vector output dimension of text-embedding-004"
    )

    # --------------------------------------------------------------------------
    # PostgreSQL & Cloud SQL Database Settings
    # --------------------------------------------------------------------------
    DB_USER: str = Field(default="postgres", description="Database username")
    DB_PASSWORD: str = Field(default="postgres", description="Database user password")
    DB_HOST: str = Field(default="127.0.0.1", description="Database host (IP or 127.0.0.1 for local/proxy)")
    DB_PORT: int = Field(default=5432, description="Database port")
    DB_NAME: str = Field(default="sc_rag_db", description="Database name")

    # Cloud SQL Instance Connection Name (e.g., 'sc-judgments-rag:asia-south1:sc-rag-db')
    CLOUD_SQL_CONNECTION_NAME: Optional[str] = Field(
        default=None,
        description="Cloud SQL instance connection name for Cloud Run Unix socket connectivity"
    )
    INSTANCE_CONNECTION_NAME: Optional[str] = Field(
        default=None,
        description="Alias for CLOUD_SQL_CONNECTION_NAME"
    )

    # Explicit DSN override (if provided, takes highest precedence)
    DB_DSN: str = Field(
        default="postgresql://postgres:postgres@127.0.0.1:5432/sc_rag_db",
        description="PostgreSQL Data Source Name (DSN) connection string"
    )
    DB_POOL_MIN_CONN: int = Field(default=2, description="Minimum pool connections")
    DB_POOL_MAX_CONN: int = Field(default=20, description="Maximum pool connections")

    @model_validator(mode="after")
    def resolve_cloud_and_db_settings(self) -> "Settings":
        """
        Dynamically resolves Cloud SQL connection strings, GCS bucket aliases,
        and local development PostgreSQL DSNs.
        """
        # Synchronize bucket aliases
        if self.BUCKET_NAME and not self.GCS_BUCKET:
            self.GCS_BUCKET = self.BUCKET_NAME
        elif self.GCS_BUCKET and not self.BUCKET_NAME:
            self.BUCKET_NAME = self.GCS_BUCKET

        # Check for explicit environment overrides
        env_dsn = os.getenv("DB_DSN") or os.getenv("DATABASE_URL")
        if env_dsn:
            self.DB_DSN = env_dsn
            return self

        # Check for Cloud SQL Instance connection (Cloud Run unix domain socket)
        cloud_sql_instance = (
            self.CLOUD_SQL_CONNECTION_NAME or
            self.INSTANCE_CONNECTION_NAME or
            os.getenv("CLOUD_SQL_CONNECTION_NAME") or
            os.getenv("INSTANCE_CONNECTION_NAME")
        )

        if cloud_sql_instance:
            self.CLOUD_SQL_CONNECTION_NAME = cloud_sql_instance
            self.INSTANCE_CONNECTION_NAME = cloud_sql_instance
            # Cloud Run mounts Cloud SQL instances under /cloudsql/<instance_connection_name>
            self.DB_DSN = (
                f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@/{self.DB_NAME}"
                f"?host=/cloudsql/{cloud_sql_instance}"
            )
            return self

        # Default TCP connection string for local development / Cloud SQL Auth Proxy
        self.DB_DSN = (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
        return self

    # --------------------------------------------------------------------------
    # Document Processing & Chunking Parameters
    # --------------------------------------------------------------------------
    PARENT_CHUNK_TARGET_WORDS: int = Field(
        default=1500,
        description="Target word count for parent section contexts"
    )
    PARENT_CHUNK_OVERLAP_WORDS: int = Field(
        default=100,
        description="Overlap word count between parent sections"
    )
    CHILD_CHUNK_TARGET_WORDS: int = Field(
        default=350,
        description="Target word count for child searchable chunks"
    )
    CHILD_CHUNK_OVERLAP_WORDS: int = Field(
        default=50,
        description="Overlap word count between child chunks"
    )

    # Margins for PyMuPDF Zone-Parsing (points)
    PDF_HEADER_MARGIN_PT: float = Field(
        default=50.0,
        description="Top margin boundary to strip running SCR headers"
    )
    PDF_FOOTER_MARGIN_PT: float = Field(
        default=50.0,
        description="Bottom margin boundary to strip page numbers & signatures"
    )
    PDF_OUTER_MARGIN_PT: float = Field(
        default=45.0,
        description="Outer margin boundary for margin note extraction"
    )

    # --------------------------------------------------------------------------
    # Search & Retrieval Configuration
    # --------------------------------------------------------------------------
    RRF_K_CONSTANT: int = Field(
        default=60,
        description="Smoothing constant for Reciprocal Rank Fusion"
    )
    VECTOR_CANDIDATE_LIMIT: int = Field(
        default=40,
        description="Initial retrieval depth for dense vector search"
    )
    FTS_CANDIDATE_LIMIT: int = Field(
        default=40,
        description="Initial retrieval depth for full-text search (tsvector)"
    )
    HYBRID_FINAL_TOP_K: int = Field(
        default=6,
        description="Final number of top ranked chunks to expand to parent context"
    )
    PARTY_SIMILARITY_THRESHOLD: float = Field(
        default=0.25,
        description="Trigram pg_trgm similarity threshold for party matching"
    )

    # --------------------------------------------------------------------------
    # Application & Serving Settings
    # --------------------------------------------------------------------------
    PORT: int = Field(
        default=8080,
        description="Server listening port (Cloud Run standard: 8080)"
    )
    HOST: str = Field(
        default="0.0.0.0",
        description="Server listening host"
    )
    DEFAULT_INGEST_LIMIT: int = Field(
        default=22,
        description="Default number of PDFs to process in batch ingestion (0 = process all)"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )


# Instantiate singleton config
settings = Settings()
