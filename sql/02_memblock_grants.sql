-- =====================================================================
-- MemBlock — Least-privilege runtime GRANTS (DML only, NO DDL)
-- =====================================================================
-- Apply AFTER 01_memblock_schema.sql. This grants the application role
-- exactly the data-plane verbs memblock needs at runtime and nothing more:
-- no CREATE, no ALTER, no DROP, no TRUNCATE. With these grants the package
-- physically cannot run DDL on the database, which is what the security
-- review is asking for.
--
-- Pair this with `manage_schema=False` (see sql/README.md) so the package
-- skips initialize()/migrations entirely. The schema is owned by a
-- separate admin/migration role that runs 01_memblock_schema.sql.
--
-- Replace `memblock_app` (role) and `memblock` (schema) with your names.
-- The verbs per table were derived from the package's actual queries in
-- src/memblock/storage/postgresql.py and async_postgresql.py — they are
-- intentionally narrow (no blanket GRANT ALL / ON ALL TABLES).
-- =====================================================================

-- 1. Schema usage (required to reference any object inside the schema)
GRANT USAGE ON SCHEMA memblock TO memblock_app;

-- 2. Table-level DML (exact verbs per table)
--    blocks/edges/embeddings get DELETE (direct deletes happen);
--    metadata is delete-by-cascade only (no direct DELETE privilege needed);
--    operations / question_events are append-only; schema_version is read-only.
GRANT SELECT, INSERT, UPDATE, DELETE ON memblock.memblock_blocks             TO memblock_app;
GRANT SELECT, INSERT, UPDATE         ON memblock.memblock_metadata           TO memblock_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON memblock.memblock_edges              TO memblock_app;
GRANT SELECT, INSERT                 ON memblock.memblock_operations         TO memblock_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON memblock.memblock_embeddings         TO memblock_app;
GRANT SELECT                         ON memblock.memblock_schema_version     TO memblock_app;
GRANT SELECT, INSERT, UPDATE         ON memblock.memblock_org_questions      TO memblock_app;
GRANT SELECT, INSERT                 ON memblock.memblock_org_question_users TO memblock_app;
GRANT SELECT, INSERT                 ON memblock.memblock_org_question_events TO memblock_app;

-- 3. Sequence grant — `id BIGSERIAL PRIMARY KEY` on memblock_org_question_events
--    auto-creates this sequence; INSERT consumes nextval(), which needs USAGE.
--    SELECT supports currval()/lastval(); UPDATE (setval) is deliberately withheld.
--    This is the ONLY sequence in the schema (all other PKs are TEXT/composite).
GRANT USAGE, SELECT ON SEQUENCE memblock.memblock_org_question_events_id_seq TO memblock_app;

-- 4. OPTIONAL — pgvector dual-write table (only if `vector` extension installed
--    and memblock_embeddings_vec was created). In BYTEA-only deployments the
--    table does not exist and this line will error — drop it if so.
GRANT SELECT, INSERT, UPDATE, DELETE ON memblock.memblock_embeddings_vec     TO memblock_app;

-- =====================================================================
-- Notes
-- =====================================================================
-- * No grant on the trigger function memblock_update_tsv() is needed: it
--   fires as part of INSERT/UPDATE on memblock_blocks and executes during
--   DML the role is already allowed to perform.
-- * Deletes from memblock_metadata / memblock_embeddings* /
--   memblock_org_question_* via ON DELETE CASCADE run with the table
--   owner's rights through the FK, NOT the issuing role's — so no extra
--   DELETE grant is required for cascade targets.
-- * memblock_schema_version is read-only at runtime (version check only).
--   The seed INSERT lives in the schema-provisioning step (file 01), run
--   by the admin/migration role — keep that separate from this app role.
-- * to_tsvector / to_tsquery and the pgvector `<=>` operator need no grants.
-- =====================================================================
