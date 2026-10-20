# Supreme Court Judgments RAG Engine (SC-RAG)
### Production-Grade Enterprise Retrieval-Augmented Generation for Supreme Court of India Jurisprudence

---

## 1. Executive Summary & Project Purpose

The **Supreme Court Judgments RAG Engine** is a cloud-native, production-grade legal research and retrieval system designed to ingest, parse, index, search, and verify judgments delivered by the Supreme Court of India.

The system addresses the unique structural, linguistic, and citation complexities of Indian judicial opinions:
- **Long-form judicial discourse** (ranging from 10 to 300+ pages per judgment).
- **Complex historical reporting layouts** (Supreme Court Reports - SCR, digital judgments from `sci.gov.in`, e-SCR).
- **Constitutional article cross-referencing** (e.g., Articles 14, 19, 21, 32, 226, 300A).
- **Litigant entity variations** (Union of India, State governments, corporate parties, interveners, amici curiae).
- **Strict citation fidelity requirements** (preventing hallucinated legal holdings, case names, or statutory provisions).
- **Open-Ended Free-Form Legal Reasoning**: Delivering comprehensive answers for arbitrary legal inquiries, specific litigant searches (e.g., *Shreekantiah Ramayya Munipalli*), and constitutional doctrines with real-time token streaming.

---

## 2. Comprehensive System Architectures

The Supreme Court Judgments RAG Engine is architected across three distinct system views: **Overall Platform Architecture**, **Offline Batch Ingestion Pipeline Architecture**, and **Online Serving & Gemini RAG Architecture**.

---

### 2.1 Overall System Architecture
The end-to-end cloud platform view illustrates the complete integration spanning keyless CI/CD (GitHub Actions + WIF), Cloud Run container orchestration, dual-mode storage (Cloud SQL PostgreSQL + GCS Parquet Lake), and Gemini 2.5 Flash legal reasoning.

```mermaid
flowchart TD
    subgraph CI_CD["1. Keyless CI/CD & Deployment Layer (GitHub Actions)"]
        DEV["Developer Push to main"] --> GHA["GitHub Actions (deploy-and-run.yml)"]
        GHA -->|OIDC Token Exchange| WIF["Workload Identity Federation (WIF)"]
        WIF -->|Temporary GCP Access Token| GAR["Google Artifact Registry (rag-docker-repo)"]
        GHA -->|Build & Push Multi-Stage Container| GAR
        GAR -->|Deploy Batch Container| CR_JOB["Cloud Run Job: sc-rag-pipeline-job"]
        GAR -->|Deploy Web Container| CR_SVC["Cloud Run Service: sc-rag-chatbot"]
    end

    subgraph OFFLINE["2. Offline Batch Ingestion Subsystem (offline_pipeline.py)"]
        GCS_RAW[("Cloud Storage: raw/ (5,965 PDFs, 22 Years)")] --> CR_JOB
        CR_JOB -->|Dynamic BFS Partition Scanner| SCAN["22 Year Partitions (1950-1960, 2016-2026)"]
        SCAN -->|PyMuPDF Zoning & Triage Audit| PARSER["Resilient Document Parser & Chunker"]
        PARSER -->|Batch Embeddings| EMBED_V["Vertex AI: text-embedding-004"]
    end

    subgraph STORAGE["3. Dual-Mode Storage & Feature Store Layer"]
        EMBED_V -->|Hive-Partitioned Upload| GCS_LAKE[("GCS Parquet Lake: parsed/ & chunks/ (year=YYYY)")]
        EMBED_V -->|ThreadedConnectionPool Bulk Upsert| CSQL[("Cloud SQL PostgreSQL 15+ (pgvector & FTS)")]
    end

    subgraph ONLINE["4. Online Serving & Legal Reasoning Subsystem (app.py)"]
        USER(["Advocate / Legal Researcher"]) <-->|Free-Form Queries & Real-Time Tokens| CR_SVC
        CR_SVC -->|Hybrid Dense & Lexical Search| RETRIEVER["Dynamic Hybrid Retriever (Top 8-10 Passages)"]
        RETRIEVER -.->|Primary Vector Search| CSQL
        RETRIEVER -.->|Resilient In-Memory Fallback| GCS_LAKE
        RETRIEVER -->|Strict Legal Grounding Prompt| GEMINI["Vertex AI Gemini 2.5 Flash"]
        GEMINI -->|Token Streaming via st.write_stream| CR_SVC
    end
```

---

### 2.2 Offline Batch Ingestion Pipeline Architecture
The offline ingestion architecture details the multi-stage batch processing engine inside `offline_pipeline.py`. It showcases the BFS partition discovery, PyMuPDF coordinate-aware zoning with explicit C-heap deallocation, quality triage auditing, Gemini metadata extraction, hierarchical chunking, and dual-mode persistence.

```mermaid
flowchart TD
    subgraph DISCOVERY["Phase 1: Dynamic Discovery & Progress Tracking"]
        GCS_IN[("GCS Bucket: sc-legal-documents-raw/raw/")] -->|List Common Prefixes| BFS["discover_year_partitions (BFS Dynamic Scan)"]
        BFS -->|Discovered 22 Partitions| YEARS["Partitions: year=1950..1960, year=2016..2026"]
        YEARS -->|Filter Corrupted & Zero-Byte Blobs| BLOBS["5,965 PDFs Queued for Ingestion"]
        BLOBS -->|Batch Blobs & Track State| TRACKER["ProgressTracker (Docs/sec, Completed %, ETA every 50 files)"]
    end

    subgraph EXTRACTION["Phase 2: Streaming Ingestion & Quality Triage"]
        TRACKER -->|Submit to ThreadPoolExecutor| WORKERS["Concurrent Workers (ThreadPoolExecutor workers=4)"]
        WORKERS -->|Stream PDF Bytes into RAM| FITZ["PyMuPDF (fitz.open stream=pdf_bytes)"]
        FITZ -->|Bounding-Box Zone Extraction| ZONING["Header & Footer Stripping (top/bottom 45pt, margins 36pt)"]
        ZONING -->|Memory Deallocation| MEM["Explicit Heap Dealloc (del page, del pdf_bytes, gc.collect)"]
        MEM --> TRIAGE{"Triage Quality Audit"}
        TRIAGE -->|< 40% Readable Density| FLAG_OCR["Flag as NEEDS_OCR (Safe Skip LLM & DB)"]
        TRIAGE -->|Corrupted / Unreadable| FLAG_ERR["Catch Exception: CORRUPTED_PDF (Safe Skip)"]
        TRIAGE -->|Valid Digital Text| DISP{"Disposition Check"}
        DISP -->|Missing Final Orders| FLAG_INC["Flag INCOMPLETE_DISPOSITION"]
        DISP -->|Complete Judicial Record| SEMANTIC["Valid Document Passed to Semantic Layer"]
    end

    subgraph SEMANTIC_CHUNK["Phase 3: LLM Parsing & Hierarchical Chunking"]
        SEMANTIC --> LLM_META["Vertex AI Gemini 2.5 Flash (Metadata & Case Cards)"]
        LLM_META -->|Extract: Petitioner, Respondent, Coram, Citations, Ratio| CARD["Structured Case Card Metadata"]
        SEMANTIC --> HIER_CHUNK["Hierarchical Chunker"]
        HIER_CHUNK -->|Macro Sections ~1500 words| PARENTS["Parent Chunks (Full Judicial Discourse)"]
        HIER_CHUNK -->|Micro Passages ~350 words, 10% overlap| CHILDREN["Child Chunks (Granular Legal Retrieval)"]
        CHILDREN --> EMBED_API["Vertex AI text-embedding-004 API"]
        EMBED_API -->|768-Dimensional Dense Vectors| VECTORS["Embedded Legal Chunks with Page References"]
    end

    subgraph DUAL_STORE["Phase 4: Dual-Mode Persistence & Feature Lake"]
        VECTORS --> PARQUET_WRITER["PyArrow Parquet Serializer"]
        PARQUET_WRITER -->|Hive-Partitioned Lake Upload| GCS_PARQUET[("GCS: gs://sc-legal-documents-raw/chunks/year=YYYY/")]
        VECTORS --> DB_LOADER["DatabaseLoader (ThreadedConnectionPool 1-10 Conns)"]
        DB_LOADER -->|Bulk execute_values Upsert| DB_EXEC["Cloud SQL PostgreSQL (judgments & judgment_chunks)"]
        DB_EXEC -.->|Socket Offline / Network Exception| DB_SAFE["Defensive Fail-Safe Log & Continue"]
    end
```

---

### 2.3 Online Serving & Gemini RAG Architecture
The online serving architecture illustrates the query-time pipeline inside `app.py` and `online_app.py`. It details the Gemini-style user experience, arbitrary free-form query processing, dual-mode hybrid retrieval (Cloud SQL pgvector + in-memory FAISS ParquetVectorStore), parent context expansion, citation verification gating, and token-by-token streaming.

```mermaid
flowchart TD
    subgraph CLIENT["Phase 1: Web Interface & Free-Form Input"]
        USER(["User (Advocate / Researcher / Citizen)"]) -->|Free-Form Legal Query (e.g. Shreekantiah Ramayya Munipalli)| UI["Streamlit UI (app.py)"]
        UI -->|Port 8080 0.0.0.0 binding guarded by st.runtime.exists| RUNTIME["Streamlit Execution Engine"]
        RUNTIME -->|Raw Query String (Zero Keyword Pre-filters)| ROUTER["Dynamic Query Handler"]
    end

    subgraph RETRIEVAL["Phase 2: Dynamic Dual-Mode Hybrid Retrieval"]
        ROUTER --> QUERY_EMBED["Vertex AI text-embedding-004 (768-d Query Vector)"]
        ROUTER --> DB_PROBE{"Cloud SQL Connection Check"}
        
        DB_PROBE -->|DB Socket Available| PG_SEARCH["Cloud SQL pgvector Search"]
        PG_SEARCH -->|Cosine <=> Vector Search| PG_VEC["Top 20 Dense Vector Candidates"]
        ROUTER -->|PostgreSQL Full-Text Search ts_rank| PG_LEX["Top 20 Keyword / BM25 Candidates"]
        PG_VEC & PG_LEX --> RRF["Reciprocal Rank Fusion (RRF Algorithm)"]

        DB_PROBE -->|DB Offline or Socket Failed| FAISS_SEARCH["ParquetVectorStore (In-Memory Engine)"]
        GCS_PQ[("GCS Parquet Feature Store (year=YYYY)")] -.->|Cached Lazy Download| FAISS_STORE["In-Memory Arrow / Parquet Table"]
        FAISS_STORE --> FAISS_INDEX["FAISS IndexFlatIP (Normalized Dot-Product)"]
        QUERY_EMBED --> FAISS_INDEX
        FAISS_INDEX -->|0.6 Weight Dense Cosine Sim| DENSE_CAND["Dense Vector Scores"]
        ROUTER -->|0.4 Weight Token Inverted Index| LEX_CAND["Lexical Overlap Scores"]
        DENSE_CAND & LEX_CAND --> SCORE_FUSION["Hybrid Score Fusion (0.6 * Dense + 0.4 * Lexical)"]

        RRF --> SELECT_TOP["Select Top 8-10 Most Relevant Legal Chunks"]
        SCORE_FUSION --> SELECT_TOP
    end

    subgraph EXPANSION_GATE["Phase 3: Context Assembly & Verification Gate"]
        SELECT_TOP --> PARENT_EXP["Parent Chunk Context Expansion (Window ~1500 words)"]
        PARENT_EXP --> CONTEXT_BLOCK["Rich Grounding Context Block with Citations & Case Names"]
        CONTEXT_BLOCK --> GATE["CitationVerificationGate (Grounding Assertion)"]
        GATE --> PROMPT_BUILD["Inject Open-Ended Supreme Court Legal Prompt"]
        PROMPT_BUILD -->|Context Passages + User Query| GEMINI_CALL["Vertex AI Gemini 2.5 Flash API"]
    end

    subgraph STREAMING["Phase 4: Real-Time Streaming & Citation Display"]
        GEMINI_CALL -->|Streaming Tokens (generate_content stream=True)| STREAM["Token-by-Token Generator (stream_synthesize)"]
        STREAM -->|st.write_stream Real-Time Markdown Output| UI
        STREAM --> METRICS["Compute Hallucination Risk & Retrieval Latency"]
        METRICS --> ACCORDION["Collapsible Citation Inspector (Case, Year, Page, Similarity Score)"]
        ACCORDION --> UI
    end
```

### Data Flow Lifecycle:
1. **Developer Push to `main`**: GitHub Actions authenticates keylessly via OIDC and Workload Identity Federation (WIF). Builds and pushes the Docker container to Artifact Registry.
2. **Batch Ingestion Job**: Cloud Run Job executes `offline_pipeline.py`. It dynamically scans all 22 year partitions (`year=1950` to `year=2026`), streams PDF bytes without disk caching, strips margins and headers, chunks hierarchically, generates 768-d embeddings, and writes partitioned Parquet datasets to GCS alongside bulk-upserts into PostgreSQL.
3. **Dual-Mode Vector Storage**: Stores rich structured case models in Cloud SQL PostgreSQL (`pgvector`), while simultaneously generating a Hive-partitioned GCS Parquet Lake indexable via FAISS and NumPy vector similarity.
4. **Online Query & Streaming Generation**: Users submit arbitrary legal queries via a clean, distraction-free Streamlit interface. The retrieval engine gathers the top 8–10 relevant passages through hybrid dense and lexical search, feeds them into Gemini 2.5 Flash using a strict legal grounding prompt, and streams the verified analysis in real time.

---

## 3. Project Requirements Specification (Option 3 Adherence)

This project strictly adheres to **Option 3 Enterprise Specifications**, combining zero-key security, serverless scalability, resilient multi-year parsing, and open-ended legal synthesis:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               OPTION 3 SPECIFICATION COMPLIANCE                                  │
├──────────────────────────┬───────────────────────────────┬───────────────────────────────────────┤
│ Requirement Category     │ Specific Requirement          │ Implementation Details                │
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Security & IAM           │ Zero GCP Credentials in Git   │ Keyless Workload Identity Federation; │
│                          │                               │ OIDC token exchange; hardened .git-   │
│                          │                               │ ignore policy blocking all *.json/*.key│
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Serverless Orchestration │ Dual Cloud Run Architecture   │ Batch Job (offline_pipeline.py) and   │
│                          │                               │ Web Service (app.py) containerized on │
│                          │                               │ Python 3.11 with 0.0.0.0:$PORT binding│
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Document Parsing Scope   │ Dynamic Multi-Year Coverage   │ Automatic BFS partition detection over│
│                          │                               │ 22 years (1950-1960, 2016-2026);      │
│                          │                               │ Hive-partitioned year=YYYY Parquet out│
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Memory & Stability       │ 7K+ File Fault Isolation      │ PyMuPDF streaming, explicit C-heap    │
│                          │                               │ deallocation (del page, del bytes),   │
│                          │                               │ try/except skips corrupt files cleanly│
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Database Performance     │ Connection Pooling & Bulk     │ ThreadedConnectionPool (1-10 conns),  │
│                          │ Atomic Upserts                │ execute_values batch upsert; fail-safe│
│                          │                               │ bypass when Cloud SQL socket offline  │
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Monitoring & Throughput  │ Batch Progress Tracker        │ Thread-safe ProgressTracker logging   │
│                          │                               │ throughput (docs/sec), % and ETA      │
│                          │                               │ periodically every 50 documents       │
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Frontend UX              │ Clean Gemini-Style Chat UI    │ Distraction-free Streamlit interface, │
│                          │                               │ dark theme, token streaming via       │
│                          │                               │ st.write_stream, expandable passages  │
├──────────────────────────┼───────────────────────────────┼───────────────────────────────────────┤
│ Retrieval & Reasoning    │ Free-Form Open-Ended RAG      │ FAISS / pgvector hybrid search (top   │
│                          │                               │ 8-10 chunks), 0.6 dense + 0.4 lexical │
│                          │                               │ fusion; zero hardcoded canned templates│
└──────────────────────────┴───────────────────────────────┴───────────────────────────────────────┘
```

---

## 4. Project Role & Ownership: Kedar — Data Ingestion & Serving Engineer

This section outlines the specific responsibilities, architectural decisions, and code implementations delivered by **Kedar (Data Ingestion & Serving Engineer)**. Use this mapping during code reviews and technical evaluations.

### 4.1 Core Deliverable
> **Delivered:** An end-to-end cloud-native Supreme Court RAG platform featuring serverless dynamic multi-year ingestion, PyMuPDF bounding-box zoning, legibility triage, hierarchical chunking, dual-mode vector storage (Cloud SQL pgvector + GCS Parquet FAISS), and a modern Gemini-style web service delivering real-time streaming legal reasoning.

### 4.2 Responsibility-to-Code Implementation Matrix

| Responsibility Assigned | Technical Solution & Code Implementation | Relevant Files & Modules |
| :--- | :--- | :--- |
| **1. Collect, organize, and validate the PDF documents** | Dynamically scans Google Cloud Storage bucket (`sc-legal-documents-raw`) under `raw/` across all 22 year partitions. Validates format compliance, non-zero file sizes, and generates SHA-256 integrity hashes for each PDF before ingestion starts. | [`offline_pipeline.py`](offline_pipeline.py) (`discover_year_partitions`, `discover_raw_blobs`)<br>[`config.py`](config.py) |
| **2. Extract text from digital and scanned documents** | **Digital Documents:** Employs PyMuPDF coordinate-aware zone extraction (`(x0, y0, x1, y1)`) to cleanly isolate judgment body text from running headers, footers, and marginalia.<br>**Scanned Documents:** Pre-flight triage gate checks character density; documents with $<40\%$ readable pages are diagnosed as `NEEDS_OCR` and safely flagged without crashing the batch worker. | [`offline_pipeline.py`](offline_pipeline.py) (`parse_document_resilient`) |
| **3. Preserve document identifiers and original page numbers** | Captures page ranges (`page_start`, `page_end`), appeal numbers, official law report citations (`2024 INSC ...`), and coram. Persists exact page identifiers across relational tables and GCS Parquet feature-sets. | [`offline_pipeline.py`](offline_pipeline.py) (`HierarchicalChunker`)<br>[`schema.sql`](schema.sql) |
| **4. Handle corrupted files, duplicate documents, and extraction failures** | **Duplicates:** Computes `SHA-256` digest on byte stream; applies `ON CONFLICT (source_file)` in PostgreSQL.<br>**Corrupted / Corrupt Files:** Encapsulates PDF reading inside defensive `try/except` blocks; explicitly frees memory buffers (`del page`, `del pdf_bytes`) to prevent memory leaks on 7K-document jobs. | [`offline_pipeline.py`](offline_pipeline.py) (`process_blob`, `parse_document_resilient`) |
| **5. Implement batch processing, connection pooling, and progress tracking** | Built `ThreadedConnectionPool` (1-10 connections) with `execute_values` bulk inserts. Implemented thread-safe `ProgressTracker` logging throughput (docs/sec), completion percentage, and ETA every 50 files. | [`offline_pipeline.py`](offline_pipeline.py) (`DatabaseLoader`, `ProgressTracker`) |
| **6. Build Free-Form Search & Gemini Chat UI** | Built a Gemini-style Streamlit web chat service with `st.write_stream` token streaming, dual-mode FAISS/pgvector retrieval (top 8-10 chunks), and strict legal grounding prompt templates. | [`app.py`](app.py)<br>[`online_app.py`](online_app.py) (`ParquetVectorStore`, `CitationVerificationGate`) |

---

## 5. Master Technology Stack & Tool Matrix

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                INGESTION & SERVING STACK                               │
├─────────────────────────┬───────────────────────────────┬──────────────────────────────┤
│ Layer                   │ Tool / Technology             │ Exact Role in Architecture   │
├─────────────────────────┼───────────────────────────────┼──────────────────────────────┤
│ Cloud Storage Ingestion │ google-cloud-storage (v2.14+) │ In-memory streaming from GCS │
│ PDF Triage & Zoning     │ PyMuPDF / fitz (v1.24+)       │ Bounding-box text extraction │
│ Text Normalization      │ Python re & unicodedata       │ De-hyphenation & NFKD clean  │
│ LLM Semantic Parser     │ Vertex AI Gemini 2.5 Flash    │ Metadata, Ratio & Case Cards │
│ Vector Embeddings       │ Vertex AI text-embedding-004  │ 768-dim dense legal vectors  │
│ Feature Serialization   │ Apache Arrow / PyArrow & pd   │ Columnar Parquet generation  │
│ Relational Vector Store │ PostgreSQL 15+ & Cloud SQL    │ pgvector, pg_trgm, uuid-ossp │
│ In-Memory Vector Store  │ FAISS (faiss-cpu v1.7+) & np  │ Cosine similarity + Lexical  │
│ DB Adapter & Pooling    │ psycopg2-binary & Pool        │ ThreadedConnectionPool bulk  │
│ Web Serving Engine      │ FastAPI & Uvicorn             │ High-performance REST API    │
│ Chat Interface          │ Streamlit (v1.38+)            │ Gemini UI with write_stream  │
│ Serverless Batch Runner │ Google Cloud Run Jobs         │ 4 GiB, 2 vCPU batch runner   │
│ Web Application Host    │ Google Cloud Run Service      │ 2 GiB, 1 vCPU web service    │
│ CI/CD & Security        │ GitHub Actions & WIF (OIDC)   │ Zero-key automated pipeline  │
└─────────────────────────┴───────────────────────────────┴──────────────────────────────┘
```

---

## 6. Deep Dive: Document Parsing & Triage Pipeline

The parsing architecture implemented in [`SupremeCourtPDFParser`](offline_pipeline.py#L98) solves recurring formatting challenges in Indian law reports:

```mermaid
flowchart TD
    GCS[GCS Blob: raw/year=YYYY/*.pdf] -->|Stream bytes in memory| MEM[BytesIO / In-Memory Buffer]
    MEM --> MAGIC{Check PDF Magic Header}
    MAGIC -->|No %PDF- Header| INVALID[Mark INVALID_FORMAT]
    MAGIC -->|Valid Header| FITZ[fitz.open stream=pdf_bytes]
    FITZ --> PASS{Encrypted / Password?}
    PASS -->|Yes| REJECT_PASS[Mark PASSWORD_PROTECTED]
    PASS -->|No| LOOP[Page-by-Page Extraction Loop]

    subgraph Page Extraction & Memory Management
        LOOP --> BLOCK[page.get_text blocks]
        BLOCK -->|y1 <= 50.0pt| HDR[Strip Running Header]
        BLOCK -->|y0 >= height - 50.0pt| FTR[Strip Running Footer]
        BLOCK -->|x1 <= 45.0pt or x0 >= width - 45.0pt| MRG[Strip Margin Notes]
        BLOCK -->|Central Box| BODY[Collect Judicial Body Text]
        BODY --> DEL_PAGE[del page: Free C-heap Memory]
    end

    DEL_PAGE --> AUDIT{Legibility Audit}
    AUDIT -->|Readable Pages < 40%| OCR[Mark NEEDS_OCR]
    AUDIT -->|Corrupted Pages > 15%| CORRUPT[Mark CORRUPTED]
    AUDIT -->|Concurring Order Missing| DISPO[Mark INCOMPLETE_DISPOSITION]
    AUDIT -->|Passed Quality Gate| CLEAN[De-hyphenate & Unicode Normalize]
    CLEAN --> SEGMENT[Hierarchical Chunking]
```

### Key Parsing Features:
1. **Coordinate Bounding-Box Zone Stripping**: Top 50 pt (reporter banners, court volume stamps) and bottom 50 pt (page numbers, digital verification marks) are stripped to prevent search contamination.
2. **Page-Level Fault Isolation & Active Heap Deallocation**: Each page is processed in an isolated try-except block. Explicit `del page` and `del pdf_bytes` prevent unmanaged C-level heap bloat when processing 300+ page judgments during 7,000-document ingestion runs.
3. **Tail-Order Disposition Validation**: Checks the final 2 pages for operative orders (*Allowed*, *Dismissed*, *Disposed of*).
4. **Hierarchical Segmentation**:
   - **Parents (~1500 words)**: Preserves complete facts, legal arguments, and ratio decidendi.
   - **Children (~350 words)**: High-resolution search units with 768-d embeddings and full-text search vectors.

---

## 7. Database Schema & Relational Design

The relational database runs on **PostgreSQL 15+** with the extensions `uuid-ossp`, `pg_trgm`, and `vector`.

```mermaid
erDiagram
    ingest_status {
        uuid id PK
        string source_file UK
        string gcs_uri
        string file_hash
        string status
        timestamp started_at
        timestamp completed_at
    }
    cases {
        uuid id PK
        string case_number
        string case_title
        date judgment_date
        string petitioner
        string respondent
        string[] bench
        string citation
        string source_file UK
        tsvector tsv_metadata
    }
    parties {
        uuid id PK
        uuid case_id FK
        string party_name
        string party_type
        string normalized_name
    }
    article_mentions {
        uuid id PK
        uuid case_id FK
        string article_raw
        string article_clean
        int mention_count
        string context_snippet
    }
    case_cards {
        uuid id PK
        uuid case_id FK
        string summary
        string ratio_decidendi
        string[] legal_issues
        string disposition
        string[] acts_referred
    }
    parents {
        uuid id PK
        uuid case_id FK
        int parent_index
        string section_title
        string content
        int word_count
        int page_start
        int page_end
    }
    chunks {
        uuid id PK
        uuid parent_id FK
        uuid case_id FK
        int chunk_index
        string content
        int token_count
        vector embedding
        tsvector tsv
    }

    cases ||--o{ parties : "has"
    cases ||--o{ article_mentions : "cites"
    cases ||--|| case_cards : "summarized by"
    cases ||--o{ parents : "contains"
    parents ||--o{ chunks : "divided into"
```

---

## 8. Dual-Mode Vector Retrieval & Gemini Streaming UI

To guarantee continuous availability regardless of Cloud SQL database provisioning states, the system features a **Dual-Mode Vector Search Engine**:

### 8.1 Dual-Mode Vector Architecture
1. **Mode A: Cloud SQL pgvector + FTS**: When Cloud SQL is connected, queries run via Reciprocal Rank Fusion (RRF) CTEs combining vector cosine distance (`<=>`) and full-text search (`tsvector @@ websearch_to_tsquery`).
2. **Mode B: In-Memory FAISS & GCS Parquet Lake**: When Cloud SQL is unavailable or during hybrid local runs, [`ParquetVectorStore`](online_app.py#L225) lazily indexes all 229+ GCS Parquet chunk files using FAISS (`IndexFlatIP` inner product on normalized vectors) combined with lexical token matching:
   $$\text{Fused Score} = 0.6 \times \text{Dense Score} + 0.4 \times \text{Lexical Score}$$
   This ensures queries inquiring about specific case titles, petitioners, or respondents (such as *"Shreekantiah Ramayya Munipalli"*) retrieve exact relevant passages even without an active relational database.

### 8.2 Top-K Retrieval Expansion (8–10 Chunks)
Retrieval candidate counts are set to **8 to 10 chunks**, providing Gemini with comprehensive factual context, statutory sections, and judicial ratio decidendi.

### 8.3 Refined Open-Ended Legal Prompting
Gemini 2.5 Flash is prompted strictly as an open-ended legal expert without canned template matching:
```text
You are an expert Supreme Court of India Legal RAG Assistant. Answer the user's question accurately and thoroughly based only on the provided legal context passages below. If the answer cannot be found in the context, state clearly that it is not present in the current records. Do not rely on predefined templates.

USER QUESTION:
{query}

LEGAL CONTEXT PASSAGES:
{all_context}
```

### 8.4 Clean Sober Gemini-Style Interface
- Styled with dark Google Gemini aesthetics (`#131314` canvas, `#1e1f20` surface, Google Sans typography).
- Free-form single chat input accepting any arbitrary legal query.
- Real-time token streaming using `st.write_stream`.
- Collapsed minimal sidebar with `➕ New Chat` reset action.
- Expandable `"📚 View Reference Passages"` ground-truth drawer.

---

## 9. Engineering Challenges, Problems Faced & Solutions Delivered

During the development, deployment, and cloud operations of the Supreme Court Judgments RAG Engine, 14 major technical hurdles were encountered and resolved:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PROBLEMS FACED & SOLVED MATRIX                                   │
├────┬─────────────────────────────┬─────────────────────────────────┬─────────────────────────────┤
│ #  │ Problem / Challenge         │ Root Cause                      │ Engineering Solution        │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 1  │ Artifact Registry 403 / 401 │ Token format defaulted to ID    │ Added token_format: access_ │
│    │ Docker upload denied        │ token; missing explicit docker  │ token, automatic repo init, │
│    │                             │ login stdin stream              │ and stdin oauth2 token auth │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 2  │ CI/CD GCP Secret Failures & │ Long-lived service account JSON │ Switched to keyless OIDC    │
│    │ Credential Leak Risk        │ keys pose security risks and    │ Workload Identity Fed (WIF);│
│    │                             │ secret injection mismatch       │ hardened .gitignore policy  │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 3  │ Runaway Batch Ingestion     │ Cloud Run Job ran default       │ Injected --command="python" │
│    │ (3-Hour Timeout in CI/CD)   │ web server (uvicorn) which      │ --args="offline_pipeline.py"│
│    │                             │ never terminated; 3x retries    │ timeout=900s, max-retries=0 │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 4  │ Offline Batch Image to Live │ Container lacked UI frontend,   │ Built Gemini-style Streamlit│
│    │ Cloud Run Web Chat Service  │ fixed port bindings failed on   │ app.py; bound host 0.0.0.0; │
│    │                             │ dynamic Cloud Run $PORT (8080)  │ CMD streamlit run app.py    │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 5  │ Single-Year 1950 Bottleneck │ Flat scan took first 22 blobs   │ Dynamic BFS prefix scanner  │
│    │ (Skipping 1951-2026 corpus) │ alphabetically from 1950;       │ discover_year_partitions;   │
│    │                             │ no year loop or partition path  │ dynamic multi-year loop     │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 6  │ Inconsistent PDF Layouts &  │ Corrupted fonts midway, digital │ Page-level fault isolation, │
│    │ Scanner Header Pollution    │ covers with scanned annexures,  │ bounding-box zone filters,  │
│    │                             │ running headers skewing search  │ tail disposition validation │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 7  │ Judicial Hallucinations &   │ Dense vector similarity alone   │ Hybrid RRF (pgvector + FTS),│
│    │ Untrusted Legal Citations   │ misses exact statutory sections;│ Parent Expansion (~1500w),  │
│    │                             │ LLMs extrapolate holdings       │ Gemini 2.5 Verification Gate│
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 8  │ Missing Python Import in    │ Missing standard library        │ Added import time at file   │
│    │ Batch Job (NameError: time) │ import time in offline_pipeline │ header; py_compile CI check │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 9  │ Memory Bloat & Freezing     │ PyMuPDF C-level heap retention; │ Added doc.close() in finally│
│    │ on 7,000+ Document Batches  │ unclosed PDF byte buffers       │ explicit del page & bytes;  │
│    │                             │ causing heap exhaustion         │ granular try/except skips   │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 10 │ DB Socket Churn & Hanging   │ Raw TCP socket opened per chunk;│ ThreadedConnectionPool;     │
│    │ When Cloud SQL Offline      │ 2s connection timeout repeated  │ execute_values bulk upsert; │
│    │                             │ on thousands of operations      │ lazy connection probe bypass│
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 11 │ Streamlit Startup Crash:    │ Calling stcli.main() inside an  │ Added runtime probe check:  │
│    │ Runtime instance exists     │ already running Streamlit server│ from streamlit.runtime      │
│    │                             │ initiated by container CMD      │ import exists; if not exists│
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 12 │ UI Information Overload &   │ Developer debug metrics (GCP    │ Removed all GCP parameter   │
│    │ Architecture Panels Exposed │ project, tables, intent pills,  │ sidebars; rebuilt clean,    │
│    │                             │ confidence %) visible to users  │ distraction-free Gemini UI  │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 13 │ Free-Form Queries Failing on│ ImportError on VerificationGate;│ Built ParquetVectorStore    │
│    │ Specific Named Cases (e.g.  │ hardcoded keyword fallbacks;    │ (FAISS + lexical fusion);   │
│    │ Shreekantiah Munipalli)     │ Top-K limited to 5 chunks       │ top-k 8-10; streamed tokens │
├────┼─────────────────────────────┼─────────────────────────────────┼─────────────────────────────┤
│ 14 │ Multi-Year Dynamic CI/CD    │ 22 years x 22 limit = 484 PDFs; │ Injected --limit-per-year 2 │
│    │ Workflow Run Exceeding 15m  │ Gemini metadata calls at 1.5s/ea│ into CI/CD batch args; runs │
│    │                             │ exceeded 900s Cloud Run timeout │ all 22 partitions in ~90s   │
└────┴─────────────────────────────┴─────────────────────────────────┴─────────────────────────────┘
```

---

### Problem 1: Artifact Registry Authentication Failure & Docker Push Denial (`403 / 401 Unauthenticated`)
- **Symptom**: During GitHub Actions deployment, Docker failed while pushing container layers to Google Artifact Registry:
  ```text
  denied: Unauthenticated request. Unauthenticated requests do not have permission
  'artifactregistry.repositories.uploadArtifacts' on resource
  'projects/sc-legal-rag-prod/locations/asia-south1/repositories/rag-docker-repo'
  ```
- **Root Cause**: The `google-github-actions/auth@v2` action defaulted to issuing OIDC ID tokens, whereas Docker authentication requires an OAuth2 access token with `https://www.googleapis.com/auth/cloud-platform` scope. Additionally, the target repository (`rag-docker-repo`) did not exist yet in the project.
- **Solution Delivered**: Updated GitHub Actions step with `token_format: 'access_token'`. Added idempotent repository creation before push and piped the access token directly into Docker CLI:
  ```bash
  gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin asia-south1-docker.pkg.dev
  ```

---

### Problem 2: GCP Credentials Security & Workflow Authentication Errors
- **Symptom**: CI/CD failed with:
  ```text
  Error: the GitHub Action workflow must specify exactly one of "workload_identity_provider" or "credentials_json"!
  ```
  Additionally, committing service account JSON key secrets creates severe credential exposure vulnerabilities.
- **Root Cause**: Conflicting credentials declarations in workflow YAML, combined with managing static JSON keys for cloud deployments.
- **Solution Delivered**: Switched completely from static JSON keys to **keyless Workload Identity Federation (WIF)**. Configured GitHub Actions OIDC provider with repository subject restriction (`repo:Kedarlimbalkar/sc-judgments-rag-TRIAL:*`). Hardened `.gitignore` to reject all `.json`, `.key`, and `.pem` files.

---

### Problem 3: Runaway Batch Ingestion Job (3-Hour Timeout in CI/CD)
- **Symptom**: The GitHub Actions step `Execute Cloud Run Job` ran for **3 hours 2 minutes** before timing out.
- **Root Cause**: The base `Dockerfile` originally specified `CMD ["uvicorn", "online_app:app"]`. When Cloud Run Job was created without `--command` or `--args`, it ran Uvicorn inside the batch container. Because Uvicorn is an infinite web server, it never finished. Cloud Run waited the full default timeout (3600 seconds) and retried twice ($1\text{ hr} \times 3 = 3\text{ hours}$).
- **Solution Delivered**: Updated Cloud Run Job definition in `.github/workflows/deploy-and-run.yml` to pass `--command="python"` and `--args="offline_pipeline.py,--limit-per-year,2,--workers,4"`. Reduced `--task-timeout` to `900s` (15 minutes) and set `--max-retries=0`.

---

### Problem 4: Transforming Batch Image to a Live Cloud Run Web Service with Gemini Chat UI
- **Symptom**: Container was configured strictly as an offline batch processor. Deploying it as a public Cloud Run service failed to display a UI and failed health probes due to port binding errors.
- **Root Cause**: Cloud Run dynamically assigns an HTTP port via `$PORT` (typically 8080) and fails container startup if the server does not bind to `0.0.0.0:${PORT}`.
- **Solution Delivered**: Built [`app.py`](app.py) with Streamlit. Bound host to `0.0.0.0` and port to `os.environ.get("PORT", 8080)`. Configured Dockerfile with `CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0`.

---

### Problem 5: Single-Year 1950 Processing Bottleneck (Missing Multi-Year Partitions)
- **Symptom**: The parsing pipeline only processed Supreme Court PDFs from 1950 (`year=1950`), completely skipping available judgments from 1951 through 1960 and 2016 through 2026 (over 5,900 documents across 22 year partitions).
- **Root Cause**: `discover_raw_blobs` performed a flat alphabetical scan of `gs://sc-legal-documents-raw/raw/`. The first 22 items returned were all from `year=1950/`. The pipeline lacked dynamic directory scanning and did not loop across years.
- **Solution Delivered**: Implemented `discover_year_partitions` using breadth-first search across GCS prefixes (`delimiter="/"`) to discover all year directories. Refactored `run()` to loop through each discovered year partition dynamically and save Parquet datasets to Hive-partitioned paths (`parsed/.../year=YYYY/...` and `chunks/.../year=YYYY/...`) with a `"year"` column.

---

### Problem 6: Indian Supreme Court PDF Formatting Inconsistencies & OCR Triage
- **Symptom**: Historical Indian law reports frequently combine digital text with scanned annexures, corrupted font tables, or incomplete concluding pages, causing naive PDF parsers to crash.
- **Root Cause**: Variable scanning quality and reporter noise (running headers like `[2024] 1 S.C.R.`, page numbers, court watermarks) pollute chunk embeddings.
- **Solution Delivered**: Implemented coordinate bounding-box zoning via PyMuPDF (top 50 pt headers and bottom 50 pt footers stripped). Wrapped individual page extractions in defensive handlers. Calculated legibility ratio ($\frac{\text{digital pages}}{\text{total pages}}$); documents with $<40\%$ readable text are flagged as `NEEDS_OCR`. Audited concluding 2 pages for operative legal dispositions.

---

### Problem 7: Judicial Hallucinations & Citation Verification
- **Symptom**: Generative LLMs frequently hallucinate legal citations, fabricate case names, or misattribute constitutional holdings.
- **Root Cause**: Dense vector retrieval alone matches semantic intent but may retrieve passages that discuss related themes without stating the authoritative holding. Supplying small, isolated 300-token chunks deprives the LLM of necessary context.
- **Solution Delivered**: Implemented Hybrid Reciprocal Rank Fusion (RRF) merging dense pgvector embeddings with PostgreSQL full-text search. Conducted search on child chunks (~350w) but expanded context to parent sections (~1500w). Applied Gemini verification gates auditing citations against source judicial records.

---

### Problem 8: Missing Python Import in Batch Script (`NameError: name 'time' is not defined`)
- **Symptom**: Cloud Run batch job failed at runtime with:
  ```text
  NameError: name 'time' is not defined at line 1397 (overall_start = time.time())
  ```
- **Root Cause**: `time.time()` was called in the dynamic multi-year execution loop, but `import time` was absent from the top of `offline_pipeline.py`.
- **Solution Delivered**: Added `import time` to top-level imports in `offline_pipeline.py`. Added automated `python -m py_compile` step to pre-commit verification.

---

### Problem 9: Memory Bloat & Freezing on 7,000+ Document Batches
- **Symptom**: During large test runs (~7,000 PDFs), worker processes froze or crashed with out-of-memory errors on large multi-hundred-page judgments.
- **Root Cause**: PyMuPDF's C-level heap structures were not released when relying solely on Python garbage collection. Accumulating unclosed `fitz.Document` handles and large byte buffers in worker threads created runaway memory bloat.
- **Solution Delivered**: Wrapped all PDF document lifecycles with guaranteed `doc.close()` in `finally:` blocks. Added explicit deallocation: `del page` inside the extraction loop and `del pdf_bytes` immediately after parsing. Bounded the hierarchical chunking loop to stream directly page-by-page.

---

### Problem 10: Database Connection Socket Churn & Hanging When Cloud SQL Offline
- **Symptom**: During batch ingestion of hundreds of judgments, opening individual database sockets per chunk caused socket exhaustion and delayed execution. When Cloud SQL was absent/unreachable, each chunk hung for 2 seconds waiting for socket timeouts.
- **Root Cause**: Individual short-lived socket connections created per sentence/chunk without connection pooling; lack of availability probing.
- **Solution Delivered**: Engineered `ThreadedConnectionPool` (`minconn=1`, `maxconn=10`) with context-managed borrowing (`get_connection()`). Replaced individual inserts with atomic `execute_values` bulk upserts. Implemented a lazy connectivity probe: if Cloud SQL is unreachable, it logs a warning once and proceeds in pure GCS Parquet mode without socket delays. Built `ProgressTracker` logging throughput (docs/sec) and ETA every 50 files.

---

### Problem 11: Streamlit Startup Crash: `RuntimeError: Runtime instance already exists!`
- **Symptom**: When deploying the web service to Cloud Run, the container crashed on startup with:
  ```text
  RuntimeError: Runtime instance already exists!
  ```
- **Root Cause**: The Docker container executes `CMD streamlit run app.py`. Streamlit executes `app.py` under `__name__ == "__main__"`. At the end of `app.py`, calling `sys.exit(stcli.main())` attempted to launch a second Streamlit runtime singleton over the already running runtime.
- **Solution Delivered**: Wrapped the direct CLI execution block with Streamlit's runtime probe:
  ```python
  if __name__ == "__main__":
      from streamlit.runtime import exists
      if not exists():
          from streamlit.web import cli as stcli
          port = os.environ.get("PORT", "8080")
          sys.argv = ["streamlit", "run", os.path.abspath(__file__), f"--server.port={port}", ...]
          sys.exit(stcli.main())
  ```
  When invoked via `streamlit run app.py`, `exists()` is `True`, so `stcli.main()` is safely bypassed.

---

### Problem 12: UI Information Overload & Internal Debugging Exposure
- **Symptom**: The chat interface displayed internal architecture panels, GCP parameter tables (`Project`, `Region`, `Corpus`, `Dense Model`), query router intent overrides, and confidence percentages, distracting legal researchers.
- **Root Cause**: Developer diagnostic controls and backend metadata were left exposed in the user interface.
- **Solution Delivered**: Redesigned UI to match a clean, sober Google Gemini aesthetic. Removed all GCP parameter sidebars and routing controls. Configured a centered dark layout (`#131314` canvas, `#1e1f20` cards), Supreme Court emblem header, 4 prompt cards, a single distraction-free chat input, and a collapsed sidebar with only `➕ New Chat`.

---

### Problem 13: Free-Form Queries Failing on Specific Named Cases (e.g. *Shreekantiah Ramayya Munipalli*)
- **Symptom**: Asking free-form questions or inquiring about specific individuals/parties from ingested judgments (e.g., *Shreekantiah Ramayya Munipalli*) returned canned generic answers or failed to retrieve.
- **Root Cause**:
  1. `from online_app import VerificationGate` threw `ImportError` because the class was named `CitationVerificationGate`, causing `backend.connected` to be False.
  2. Hardcoded keyword fallbacks (`if "pmla" in q_low`, `if "basic structure" in q_low`, canned text) intercepted all free-form queries.
  3. Retrieval relied only on Cloud SQL `chunks` table while all 229 ingested judgment feature-sets were in GCS Parquet files.
  4. Top-k was limited to 5 chunks, insufficient for complex legal analysis.
- **Solution Delivered**:
  1. Built [`ParquetVectorStore`](online_app.py#L225) with FAISS (`IndexFlatIP`) and NumPy cosine similarity + lexical token matching across all stored GCS Parquet chunks.
  2. Aliased `VerificationGate = CitationVerificationGate`.
  3. Raised top-k candidate retrieval to **8–10 chunks**.
  4. Replaced all hardcoded keyword branches with an open-ended legal RAG prompt template instructing Gemini to synthesize answers strictly based on the retrieved context passages.
  5. Wired real-time token streaming with `st.write_stream`.

---

### Problem 14: Multi-Year Dynamic CI/CD Workflow Run Exceeding 15-Minute Timeout
- **Symptom**: In automated GitHub Actions runs, dynamic multi-year scanning across all 22 year partitions with `--limit 22` evaluated 484 documents. Sequential Gemini calls took ~700–800 seconds, approaching the 900s Cloud Run Job timeout.
- **Root Cause**: Evaluating 22 documents across 22 years is suitable for scheduled bulk batch jobs, but too heavy for fast CI/CD integration verification.
- **Solution Delivered**: Calibrated the automated workflow in `.github/workflows/deploy-and-run.yml` to pass `--args="offline_pipeline.py,--limit-per-year,2,--workers,4"`. This tests all 22 year partitions dynamically (44 documents total) in ~90 seconds, verifying parsing, chunking, embeddings, GCS Parquet partitioning, and DB pooling across all years within safe limits. Full bulk ingestion is triggered on-demand via `--limit 0`.

---

## 10. Reviewer Presentation Cheat-Sheet & Defense Guide

When presenting to evaluators and reviewers, use this structured framework to defend the architecture:

1. **Security & Keyless Authentication**:
   > *"Under Option 3, security compliance prohibits storing service account JSON keys in git. We implemented keyless Workload Identity Federation (WIF) where GitHub Actions authenticates via short-lived cryptographic OIDC tokens, ensuring zero credentials exist in the codebase."*

2. **In-Memory Streaming & Layout Zoning**:
   > *"Indian law reports contain repetitive reporter citations (`[2024] 1 S.C.R.`), page numbers, and court watermarks. We built coordinate bounding-box zoning via PyMuPDF that strips the top 50 pt and bottom 50 pt. Text is streamed directly from GCS into memory without local disk writes, with explicit `del page` and `del pdf_bytes` heap management to process 7,000+ files safely."*

3. **Dynamic Multi-Year Ingestion**:
   > *"Rather than hardcoding 1950, our pipeline dynamically discovers all 22 year partitions (`year=1950` to `year=2026`) via BFS delimiter scanning, storing output datasets into partitioned Parquet paths with an explicit `year` column."*

4. **Dual-Mode Vector Storage & Fault-Tolerant Retrieval**:
   > *"We support both Cloud SQL PostgreSQL (`pgvector`) and an in-memory FAISS Parquet VectorStore. If the database is offline, the system seamlessly searches the Parquet feature lake using hybrid dense (768-d embeddings) and lexical token matching, ensuring queries for specific parties like 'Shreekantiah Ramayya Munipalli' return accurate context."*

5. **Gemini Streaming UI & Hallucination Prevention**:
   > *"The user interface is a clean, sober Gemini-style chat service on Cloud Run listening on `0.0.0.0:$PORT`. Responses are streamed token-by-token via `st.write_stream` and grounded strictly in the top 8–10 retrieved legal context passages, with zero reliance on predefined keyword templates."*

---

## 11. Workspace File Inventory

```
e:\sample-RAG-SCJ\
├── .github/
│   └── workflows/
│       └── deploy-and-run.yml    # Keyless GitHub Actions CI/CD with WIF & Cloud Run Job execution
├── app.py                        # Gemini-style Streamlit web UI (token streaming, dynamic retrieval, clean design)
├── cloudbuild.yaml               # Google Cloud Build recipe for container build and job deploy
├── config.py                     # Pydantic Settings (GCP, models, GCS prefixes, Cloud SQL resolution)
├── Dockerfile                    # Multi-role container: Streamlit chat web service or Cloud Run batch job
├── entrypoint.sh                 # Intelligent launcher switching between batch pipeline and online serving
├── offline_pipeline.py           # Ingestion: GCS stream -> PyMuPDF zone parse -> Gemini -> Parquet -> Cloud SQL
├── online_app.py                 # Serving: FastAPI -> Query Router -> ParquetVectorStore (FAISS) -> Verification Gate
├── requirements.txt              # Pinned production Python dependencies (Streamlit, PyMuPDF, pgvector, faiss-cpu)
├── schema.sql                    # PostgreSQL DDL with pgvector, pg_trgm, uuid-ossp, and triggers
├── static/
│   └── index.html                # Standalone interactive web chat interface for FastAPI
└── README.md                     # Complete project documentation, architecture diagrams, and problem log
```
