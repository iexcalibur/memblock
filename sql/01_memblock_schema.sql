-- =====================================================================
-- MemBlock — Canonical PostgreSQL Schema (DDL, externally managed)
-- =====================================================================
-- Purpose: provision the complete memblock schema OUT OF BAND, so the
-- memblock Python package never needs CREATE/ALTER/TRIGGER privileges at
-- runtime. A DBA (or a migration tool: Flyway / Liquibase / Alembic /
-- sqitch) reviews and applies this file. The application then runs with a
-- DML-only role (see 02_memblock_grants.sql).
--
-- This is the FULL current-state schema (== SCHEMA_VERSION 7). It is the
-- exact layout a fresh `AsyncPostgreSQLAdapter.initialize()` produces,
-- with every sync-path migration (v2-v7 in src/memblock/migrations.py)
-- folded inline as final-state columns/indexes — NOT as ALTER steps.
--
-- ---------------------------------------------------------------------
-- SCHEMA PLACEHOLDER
-- ---------------------------------------------------------------------
-- Every object below is qualified with the schema `memblock`. The library
-- DEFAULT schema is "public" (AsyncPostgreSQLAdapter(schema="public")).
--   * Default deployment: replace `memblock.` with `public.` (or drop the
--     qualifier) and skip the CREATE SCHEMA line.
--   * Custom / per-tenant schema: keep a name like `tenant_xyz` and the
--     CREATE SCHEMA line applies.
-- Keep the chosen name consistent with the `schema=` you pass the adapter.
--
-- Migrations folded inline (do NOT emit these as ALTER for a fresh DB):
--   v2  memblock_blocks.content_hash          (migrations.py:73-78)
--   v3  idx_mb_blocks_content_hash            (migrations.py:86-89)
--   v4  memblock_metadata.session_id + index  (migrations.py:106-114)
--   v5  memblock_metadata.org_id/project_id/agent_id/custom_metadata + idx
--                                             (migrations.py:137-162)
--   v6  org Q&A analytics tables + indexes    (migrations.py:209-253)
--   v7  memblock_metadata.happened_at/_end/temporal_precision + idx
--                                             (migrations.py:276-291)
-- =====================================================================

-- ── Schema ───────────────────────────────────────────────────────────
-- No-op when targeting "public". Required for custom/tenant schemas.
CREATE SCHEMA IF NOT EXISTS memblock;

-- ── pgvector extension (OPTIONAL) ────────────────────────────────────
-- Required ONLY for the memblock_embeddings_vec table and its
-- hnsw/ivfflat index below. The adapter degrades to BYTEA-only
-- embeddings (memblock_embeddings) when this extension is absent.
-- Uncomment if you want server-side cosine similarity search:
-- CREATE EXTENSION IF NOT EXISTS vector;


-- =====================================================================
-- CORE TABLES  (FK order: blocks first, then dependents)
-- =====================================================================

-- ── memblock_blocks (root table) ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS memblock.memblock_blocks (
    id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    encryption_level TEXT NOT NULL DEFAULT 'none',
    encrypted BOOLEAN NOT NULL DEFAULT FALSE,
    parent_id TEXT,
    children_ids JSONB NOT NULL DEFAULT '[]',
    version INTEGER NOT NULL DEFAULT 1,
    op_hash TEXT NOT NULL DEFAULT '',
    tags JSONB NOT NULL DEFAULT '[]',
    deleted BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    content_tsv TSVECTOR,
    content_hash TEXT NOT NULL DEFAULT '',            -- migration v2
    PRIMARY KEY (id, user_id)
);

-- ── memblock_metadata (FK -> memblock_blocks) ────────────────────────
CREATE TABLE IF NOT EXISTS memblock.memblock_metadata (
    block_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'explicit',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by TEXT NOT NULL DEFAULT 'agent',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TIMESTAMPTZ,
    decay_rate REAL NOT NULL DEFAULT 0.01,
    ttl INTEGER,
    session_id TEXT,                                  -- migration v4
    org_id TEXT,                                      -- migration v5
    project_id TEXT,                                  -- migration v5
    agent_id TEXT,                                    -- migration v5
    custom_metadata JSONB,                            -- migration v5
    happened_at TIMESTAMPTZ,                          -- migration v7
    happened_at_end TIMESTAMPTZ,                      -- migration v7
    temporal_precision TEXT NOT NULL DEFAULT 'exact', -- migration v7
    PRIMARY KEY (block_id, user_id),
    FOREIGN KEY (block_id, user_id)
        REFERENCES memblock.memblock_blocks(id, user_id) ON DELETE CASCADE
);

-- ── memblock_edges ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memblock.memblock_edges (
    id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, user_id)
);

-- ── memblock_operations ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memblock.memblock_operations (
    id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    block_id TEXT NOT NULL DEFAULT '',
    author TEXT NOT NULL DEFAULT 'agent',
    clock INTEGER NOT NULL DEFAULT 0,
    action TEXT NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    hash TEXT NOT NULL DEFAULT '',
    prev_hash TEXT NOT NULL DEFAULT '',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, user_id)
);

-- ── memblock_embeddings (BYTEA — always present) ─────────────────────
CREATE TABLE IF NOT EXISTS memblock.memblock_embeddings (
    block_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    embedding BYTEA NOT NULL,
    PRIMARY KEY (block_id, user_id),
    FOREIGN KEY (block_id, user_id)
        REFERENCES memblock.memblock_blocks(id, user_id) ON DELETE CASCADE
);

-- ── memblock_embeddings_vec (OPTIONAL — requires pgvector) ───────────
-- Created only when the `vector` extension is installed. Uncomment the
-- CREATE EXTENSION line above to enable. Used for server-side cosine
-- similarity; otherwise the adapter falls back to Python cosine over the
-- BYTEA table above. IMPORTANT: the adapter creates the matching index
-- LAZILY on the first embedding write (see runtime DDL note in README) —
-- pre-create the index here too so the runtime role needs no CREATE.
-- CREATE TABLE IF NOT EXISTS memblock.memblock_embeddings_vec (
--     block_id TEXT NOT NULL,
--     user_id TEXT NOT NULL,
--     embedding vector NOT NULL,
--     PRIMARY KEY (block_id, user_id),
--     FOREIGN KEY (block_id, user_id)
--         REFERENCES memblock.memblock_blocks(id, user_id) ON DELETE CASCADE
-- );


-- =====================================================================
-- CORE INDEXES
-- =====================================================================
CREATE INDEX IF NOT EXISTS idx_mb_blocks_user_type
    ON memblock.memblock_blocks(user_id, type);
CREATE INDEX IF NOT EXISTS idx_mb_blocks_user_deleted
    ON memblock.memblock_blocks(user_id, deleted);
CREATE INDEX IF NOT EXISTS idx_mb_blocks_parent
    ON memblock.memblock_blocks(user_id, parent_id);
CREATE INDEX IF NOT EXISTS idx_mb_blocks_tsv
    ON memblock.memblock_blocks USING GIN(content_tsv);
CREATE INDEX IF NOT EXISTS idx_mb_blocks_content_hash          -- migration v3
    ON memblock.memblock_blocks(user_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_mb_edges_source
    ON memblock.memblock_edges(user_id, source_id);
CREATE INDEX IF NOT EXISTS idx_mb_edges_target
    ON memblock.memblock_edges(user_id, target_id);
CREATE INDEX IF NOT EXISTS idx_mb_ops_block
    ON memblock.memblock_operations(user_id, block_id);
CREATE INDEX IF NOT EXISTS idx_mb_ops_clock
    ON memblock.memblock_operations(user_id, clock);
CREATE INDEX IF NOT EXISTS idx_mb_meta_confidence
    ON memblock.memblock_metadata(user_id, confidence);

-- ── Metadata indexes added by migrations v4 / v5 / v7 ────────────────
CREATE INDEX IF NOT EXISTS idx_mb_metadata_session             -- migration v4
    ON memblock.memblock_metadata(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_mb_metadata_org                 -- migration v5
    ON memblock.memblock_metadata(user_id, org_id);
CREATE INDEX IF NOT EXISTS idx_mb_metadata_project             -- migration v5
    ON memblock.memblock_metadata(user_id, project_id);
CREATE INDEX IF NOT EXISTS idx_mb_metadata_agent               -- migration v5
    ON memblock.memblock_metadata(user_id, agent_id);
CREATE INDEX IF NOT EXISTS idx_mb_metadata_happened_at         -- migration v7
    ON memblock.memblock_metadata(user_id, happened_at);

-- ── pgvector index (OPTIONAL — requires pgvector) ────────────────────
-- The adapter creates this lazily on the FIRST save_embedding(), choosing
-- HNSW for embedding dims <= 2000 and IVFFlat for dims > 2000 (HNSW's hard
-- ceiling). Pre-create the one matching your embedding model so the
-- runtime path is a no-op and the app role needs no CREATE privilege:
-- CREATE INDEX IF NOT EXISTS idx_mb_embeddings_vec_hnsw
--     ON memblock.memblock_embeddings_vec
--     USING hnsw (embedding vector_cosine_ops);
-- -- OR, for > 2000-dim embeddings (e.g. gemini-embedding-001 @ 3072):
-- CREATE INDEX IF NOT EXISTS idx_mb_embeddings_vec_ivfflat
--     ON memblock.memblock_embeddings_vec
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);


-- =====================================================================
-- FULL-TEXT SEARCH: tsvector maintenance function + trigger
-- =====================================================================
-- NOTE: this FUNCTION + TRIGGER is the part the package can never run
-- under a DML-only role — `CREATE OR REPLACE FUNCTION` runs
-- unconditionally inside initialize() and requires table ownership. That
-- is the core reason initialize() must be SKIPPED in a locked-down
-- deployment (see README: manage_schema=False), not merely tolerated.
CREATE OR REPLACE FUNCTION memblock.memblock_update_tsv()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger names are unique PER TABLE, so the existence check is scoped to
-- this schema's memblock_blocks via tgrelid. Without the tgrelid scope a
-- trigger of the same name on public.memblock_blocks would short-circuit
-- creation here and leave content_tsv permanently NULL in custom schemas.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
         WHERE tgname = 'trg_memblock_tsv'
           AND tgrelid = 'memblock.memblock_blocks'::regclass
    ) THEN
        CREATE TRIGGER trg_memblock_tsv
        BEFORE INSERT OR UPDATE OF content
        ON memblock.memblock_blocks
        FOR EACH ROW
        EXECUTE FUNCTION memblock.memblock_update_tsv();
    END IF;
END;
$$;


-- =====================================================================
-- SCHEMA VERSION TRACKING  (seeded to 7 — no pending migrations)
-- =====================================================================
-- The package reads this at startup to decide whether migrations are
-- pending. Seeding it to the current SCHEMA_VERSION (7) tells memblock the
-- schema is fully provisioned, so its migration runner is a no-op.
CREATE TABLE IF NOT EXISTS memblock.memblock_schema_version (
    version INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- Seed the current version only when the table is empty (idempotent).
INSERT INTO memblock.memblock_schema_version (version)
SELECT 7
WHERE NOT EXISTS (SELECT 1 FROM memblock.memblock_schema_version);


-- =====================================================================
-- ANALYTICS TABLES  (org-level Q&A tracking — migration v6)
-- FK order: org_questions first, then its dependents.
-- These back enable_analytics / log_question(); harmless if unused.
-- =====================================================================
CREATE TABLE IF NOT EXISTS memblock.memblock_org_questions (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    frequency_count INTEGER NOT NULL DEFAULT 1,
    unique_user_count INTEGER NOT NULL DEFAULT 1,
    first_asked TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_asked TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, normalized_text)
);

CREATE TABLE IF NOT EXISTS memblock.memblock_org_question_users (
    org_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    PRIMARY KEY (org_id, question_id, user_id),
    FOREIGN KEY (question_id)
        REFERENCES memblock.memblock_org_questions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memblock.memblock_org_question_events (
    id BIGSERIAL PRIMARY KEY,
    org_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    asked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (question_id)
        REFERENCES memblock.memblock_org_questions(id) ON DELETE CASCADE
);

-- ── Analytics indexes ────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_mb_org_questions_org
    ON memblock.memblock_org_questions(org_id);
CREATE INDEX IF NOT EXISTS idx_mb_org_questions_freq
    ON memblock.memblock_org_questions(org_id, frequency_count DESC);
CREATE INDEX IF NOT EXISTS idx_mb_org_question_events_time
    ON memblock.memblock_org_question_events(org_id, asked_at);

-- =====================================================================
-- END OF SCHEMA
-- =====================================================================
