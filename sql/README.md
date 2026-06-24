# MemBlock — Externally-Managed Schema (separating DDL from the package)

This directory lets you provision the memblock PostgreSQL schema **out of
band** so the memblock Python package never runs `CREATE TABLE`,
`CREATE INDEX`, `CREATE FUNCTION`, or `CREATE TRIGGER` against your database
at runtime. A DBA / migration tool owns the schema; the app runs with a
DML-only role.

| File | What it does | Who runs it |
|------|--------------|-------------|
| [`01_memblock_schema.sql`](01_memblock_schema.sql) | Full current-state schema (tables, indexes, FTS function + trigger, version row seeded to 7). | Admin / migration role, once per environment (or per upgrade). |
| [`02_memblock_grants.sql`](02_memblock_grants.sql) | Least-privilege `GRANT`s for the runtime app role — DML only, no DDL. | Admin role, after `01`. |

---

## 1. The concept: DDL vs. DML (this resolves the disagreement)

The security review and "memory creation is the core principle" are about
two *different* kinds of SQL, and separating them satisfies both:

| | What it is | memblock's use | Needs schema-change rights? |
|---|---|---|---|
| **DDL** — `CREATE TABLE/INDEX/FUNCTION/TRIGGER`, `ALTER`, `CREATE SCHEMA` | Defines the **shape** of the DB | One-time provisioning in `initialize()` + version migrations | **Yes** — the review's concern |
| **DML** — `INSERT/UPDATE/DELETE/SELECT` | Reads/writes **rows** in existing tables | Every memory operation | **No** |

**"A memory is created / a new block is added" is DML, not DDL.** Creating
a memory is `INSERT INTO memblock_blocks (...)` — a *row* into a table that
already exists (`save_block` at `src/memblock/storage/async_postgresql.py`).
It does **not** require `CREATE` privileges. The `CREATE TABLE/INDEX/TRIGGER`
the review flagged only build the *container* the rows live in, and that is
a one-time setup step you can lift out of the package.

So the core principle is intact: you separate **one-time schema
provisioning** (DDL → reviewed and run by a DBA) from **runtime memory
writes** (DML → what the package does on every call, under a role with no
schema rights).

---

## 2. How to deploy

```bash
# 1. Admin/owner role provisions the schema (review this file first):
psql "$ADMIN_DSN" -f sql/01_memblock_schema.sql

# 2. Admin grants the app role DML-only access:
psql "$ADMIN_DSN" -f sql/02_memblock_grants.sql

# 3. App connects with the restricted role and SKIPS auto-DDL (see §4):
#    AsyncMemBlock(storage="postgresql+asyncpg://memblock_app:...@host/db",
#                  manage_schema=False)
```

Before running, in both files replace the placeholders:
- schema `memblock` → your schema (the library default is **`public`**),
- role `memblock_app` → your runtime role.

Keep the schema name identical to the `schema=` you pass the adapter.

---

## 3. Runtime DDL the package *also* emits (must pre-provision these)

The review said "DDL in `async_postgresql.py` and `migrations.py`" — but
DDL is **not** limited to the initial `initialize()`. We audited every
adapter path. There are five DDL surfaces; if you lock down privileges
without pre-provisioning all of them, the app will error (or silently
retry) at runtime:

1. **Lazy pgvector index** — `_ensure_pgvector_index()` issues a
   `CREATE INDEX ... USING hnsw/ivfflat` on `memblock_embeddings_vec` on the
   **first embedding write** of each process (not in `initialize()`). On
   failure it swallows the error and **retries on every subsequent embedding
   write**. → Pre-create the `vector` extension, the `memblock_embeddings_vec`
   table, **and** the index name matching your model (`hnsw` for ≤2000-dim,
   `ivfflat` for >2000-dim). The commented blocks in `01_…schema.sql` cover this.

2. **Migration runner** — sync `initialize()` calls `MigrationRunner.run()`,
   which emits `CREATE TABLE memblock_schema_version`, `ALTER TABLE ADD
   COLUMN`, and `CREATE INDEX` whenever the recorded version < 7. → Seeding
   `memblock_schema_version` to **7** (file `01` does this) makes the pending-
   migration loop empty, so no ALTER/CREATE fires.

3. **Analytics tables (sync)** — `initialize_analytics_tables()` runs
   `CREATE TABLE/INDEX` on demand when analytics is enabled. → File `01`
   pre-creates the three `memblock_org_*` tables, so it's a no-op.

4. **`CREATE SCHEMA`** — async `initialize()` runs `CREATE SCHEMA IF NOT
   EXISTS` for any non-`public` schema (needs the broad database-level
   `CREATE`). → Pre-create your tenant schema; file `01` includes it.

5. **`CREATE OR REPLACE FUNCTION` + `CREATE TRIGGER`** — runs in
   `initialize()` and is the reason §4 is **mandatory** (next section).

---

## 4. Why you must skip `initialize()` (not just rely on `IF NOT EXISTS`)

Most of the DDL uses `IF NOT EXISTS`, which is a no-op once provisioned.
**But the FTS function is not idempotent in the privilege sense:**
`initialize()` runs `CREATE OR REPLACE FUNCTION memblock.memblock_update_tsv()`
**unconditionally**, and `CREATE OR REPLACE` requires **ownership** of the
function — which a DML-only role does not have. So even against a fully
provisioned schema, a locked-down role would **error on the first
`initialize()` call**.

→ The app must **not call `initialize()`** at all. Today the package always
calls it (sync: `memblock.py:195`; async: `async_memblock.py:_ensure_initialized`),
and there is **no existing flag to skip it**. The clean fix is a one-line
constructor flag:

```python
# proposed: MemBlock(..., manage_schema: bool = True)
#           AsyncMemBlock(..., manage_schema: bool = True)
# When False, the adapter sets _initialized = True WITHOUT emitting any DDL.
```

- `manage_schema=True` (default) → current behavior; dev/single-tenant
  setups keep working unchanged.
- `manage_schema=False` → package never emits DDL; schema is your
  responsibility via the files here.

I can implement this flag (constructors + both adapters + a test) as a
follow-up — it's a small, backward-compatible change. Until then, an
interim workaround is to grant the app role temporary ownership during a
maintenance window, but the flag is the correct long-term answer.

---

## 5. Role separation summary

- **Admin / migration role** (owns the schema): runs `01` + `02`, and any
  future versioned migration files. Has DDL.
- **Runtime app role** (`memblock_app`): only the DML grants in `02`. Runs
  the application with `manage_schema=False`. Cannot alter the schema.

This is exactly the boundary the security review asked for: an external
package with **zero** DDL capability against your tables.

---

## 6. Upgrading an existing memblock DB

`01_memblock_schema.sql` is the **fresh-provisioning** (full current state,
v7). If you already have a memblock database at an older version and need
the incremental `ALTER` steps (v2→v7) as separate reviewable migration
files for Flyway/Liquibase, those live in `src/memblock/migrations.py` and
can be extracted the same way — ask and I'll generate the versioned set.

> Generated from an audited extraction of `async_postgresql.py`,
> `postgresql.py`, and `migrations.py`; the schema was adversarially
> verified complete against every column/index/constraint the package
> reads or writes.
