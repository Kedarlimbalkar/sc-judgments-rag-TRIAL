-- =============================================================================
-- Supreme Court Judgments RAG Database Schema
-- Target: PostgreSQL 15+ with pgvector, pg_trgm, uuid-ossp
-- =============================================================================

-- Ensure required PostgreSQL extensions are enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- -----------------------------------------------------------------------------
-- 1. Table: ingest_status
-- Tracks batch & stream ingestion state, file integrity hashes, page counts,
-- and error diagnostics for PDF judgments.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ingest_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_file VARCHAR(512) NOT NULL UNIQUE,
    gcs_uri VARCHAR(1024),
    file_hash VARCHAR(64),
    file_size_bytes BIGINT,
    total_pages INTEGER DEFAULT 0,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING', -- 'PENDING', 'PROCESSING', 'COMPLETED', 'FAILED', 'SKIPPED'
    error_message TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingest_status_status ON ingest_status (status);
CREATE INDEX IF NOT EXISTS idx_ingest_status_file_hash ON ingest_status (file_hash);

-- -----------------------------------------------------------------------------
-- 2. Table: cases
-- Master registry of Supreme Court of India judgments, docket numbers, dates,
-- bench composition, coram, and official legal citations.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_number VARCHAR(255) NOT NULL,
    case_title TEXT NOT NULL,
    judgment_date DATE,
    petitioner TEXT,
    respondent TEXT,
    bench TEXT[],                                  -- Array of presiding Justice names
    coram TEXT,                                    -- Full coram description
    citation VARCHAR(255),                         -- e.g., '2024 INSC 123', '(2024) 1 SCR 45'
    source_file VARCHAR(512) NOT NULL,
    gcs_uri VARCHAR(1024),
    total_pages INTEGER DEFAULT 0,
    headnote TEXT,
    tsv_metadata tsvector GENERATED ALWAYS AS (
        to_tsvector('english',
            coalesce(case_title, '') || ' ' ||
            coalesce(case_number, '') || ' ' ||
            coalesce(petitioner, '') || ' ' ||
            coalesce(respondent, '') || ' ' ||
            coalesce(citation, '') || ' ' ||
            coalesce(headnote, '')
        )
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_cases_source_file UNIQUE (source_file),
    CONSTRAINT uq_cases_number_date UNIQUE (case_number, judgment_date)
);

CREATE INDEX IF NOT EXISTS idx_cases_judgment_date ON cases (judgment_date DESC);
CREATE INDEX IF NOT EXISTS idx_cases_case_number ON cases (case_number);
CREATE INDEX IF NOT EXISTS idx_cases_tsv_metadata ON cases USING gin (tsv_metadata);
CREATE INDEX IF NOT EXISTS idx_cases_bench ON cases USING gin (bench);

-- -----------------------------------------------------------------------------
-- 3. Table: parties
-- Normalized registry of all litigation parties (Petitioners, Respondents,
-- Appellants, Interveners, Amici Curiae) for high-performance trigram fuzzy search.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parties (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    party_name TEXT NOT NULL,
    party_type VARCHAR(50) NOT NULL,               -- 'PETITIONER', 'RESPONDENT', 'APPELLANT', 'INTERVENER', 'AMICUS'
    normalized_name TEXT NOT NULL,                 -- Uppercase, stripped of punctuation
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_parties_case_id ON parties (case_id);
CREATE INDEX IF NOT EXISTS idx_parties_party_type ON parties (party_type);
CREATE INDEX IF NOT EXISTS idx_parties_name_trgm ON parties USING gin (party_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_parties_normalized_trgm ON parties USING gin (normalized_name gin_trgm_ops);

-- -----------------------------------------------------------------------------
-- 4. Table: article_mentions
-- Tracks constitutional article references (e.g. Article 14, 19(1)(a), 21, 32)
-- across judgments for article-level aggregation and constitutional jurisprudence queries.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS article_mentions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    article_raw VARCHAR(100) NOT NULL,             -- e.g., 'Article 19(1)(a)', 'Art. 21'
    article_clean VARCHAR(50) NOT NULL,            -- e.g., '19(1)(a)', '21', '32'
    mention_count INTEGER DEFAULT 1,
    context_snippet TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_article_mentions_clean ON article_mentions (article_clean);
CREATE INDEX IF NOT EXISTS idx_article_mentions_case_clean ON article_mentions (case_id, article_clean);

-- -----------------------------------------------------------------------------
-- 5. Table: case_cards
-- Synthesized executive summary, legal issues, ratio decidendi, acts referred,
-- and disposition extracted by Gemini 2.5 Flash for rapid judicial overview.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS case_cards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE UNIQUE,
    summary TEXT NOT NULL,
    ratio_decidendi TEXT,
    legal_issues TEXT[],
    disposition VARCHAR(150),                      -- e.g., 'Allowed', 'Dismissed', 'Partly Allowed', 'Referred to Larger Bench'
    acts_referred TEXT[],
    cases_cited TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_case_cards_case_id ON case_cards (case_id);
CREATE INDEX IF NOT EXISTS idx_case_cards_disposition ON case_cards (disposition);
CREATE INDEX IF NOT EXISTS idx_case_cards_metadata ON case_cards USING gin (metadata);

-- -----------------------------------------------------------------------------
-- 6. Table: parents
-- Coarse parent context sections (~1500 words) corresponding to judicial sections
-- or major topical partitions. Used for context expansion during synthesis.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS parents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    parent_index INTEGER NOT NULL,
    section_title VARCHAR(255),
    content TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_parents_case_idx UNIQUE (case_id, parent_index)
);

CREATE INDEX IF NOT EXISTS idx_parents_case_id ON parents (case_id);

-- -----------------------------------------------------------------------------
-- 7. Table: chunks
-- Fine-grained child chunks (~350 words) with 768-dimensional embeddings
-- (Vertex AI text-embedding-004) and pre-indexed full-text search vectors.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_id UUID NOT NULL REFERENCES parents(id) ON DELETE CASCADE,
    case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embedding vector(768) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_chunks_parent_idx UNIQUE (parent_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks (parent_id);
CREATE INDEX IF NOT EXISTS idx_chunks_case_id ON chunks (case_id);

-- HNSW Vector Index for sub-millisecond Cosine Similarity Search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- GIN Index for PostgreSQL Full-Text Search
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);

-- -----------------------------------------------------------------------------
-- Helper Triggers: Auto-update updated_at timestamps
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_ingest_status_updated_at ON ingest_status;
CREATE TRIGGER trg_ingest_status_updated_at
    BEFORE UPDATE ON ingest_status
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_cases_updated_at ON cases;
CREATE TRIGGER trg_cases_updated_at
    BEFORE UPDATE ON cases
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS trg_case_cards_updated_at ON case_cards;
CREATE TRIGGER trg_case_cards_updated_at
    BEFORE UPDATE ON case_cards
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

