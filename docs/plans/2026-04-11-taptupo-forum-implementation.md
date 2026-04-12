# taptupo-forum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multi-agent forum where free OpenRouter-backed LangChain agents organically discuss news and arXiv research with human users, runnable locally and deployable to AWS.

**Architecture:** FastAPI backend + Jinja2/HTMX frontend, LangChain agents calling free OpenRouter models, repository-abstracted DynamoDB (Local for dev, AWS for prod), async news aggregation with pluggable fetchers, event-driven discussion engine with jitter.

**Tech Stack:** Python 3.12, uv (package manager, NOT pip), FastAPI, Jinja2, HTMX, LangChain, boto3, DynamoDB Local (Docker), Pytest, Hypothesis, Playwright (pytest-playwright), Atheris, Locust, Ruff, Mypy, Bandit, Safety, detect-secrets, pre-commit, Git LFS, Docker, AWS ECS Fargate, CloudFormation.

**Reference:** Design doc at `docs/plans/2026-04-11-taptupo-forum-design.md`.

**Development discipline:**

- TDD — write the failing test first, every time.
- Feature branches + PRs only. No commits to master.
- Unit tests must pass before every commit (pre-commit enforces).
- All code passes Ruff, Mypy, Bandit, Safety, detect-secrets.
- Diagrams are SVG in `docs/diagrams/`. No ASCII art diagrams.
- `uv` for ALL Python dependency operations. Never call `pip`.

---

## Phase 1: Project Scaffolding & Tooling

### Task 1.1: Initialize uv project

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `uv.lock` (generated)

**Step 1: Initialize**

Run: `uv init --package --name taptupo-forum --python 3.12`
Expected: creates `pyproject.toml`, `.python-version`, `src/taptupo_forum/`.

**Step 2: Remove scaffolding we don't want**

Delete the default `src/taptupo_forum/` that `uv init` created — we are using an `app/` layout. Update `pyproject.toml` `[tool.hatch.build.targets.wheel]` or `[tool.setuptools]` section to point at `app/` instead. Use the `[project]` table and `[tool.uv]` sections.

**Step 3: Add core dependencies**

Run (single command):
```
uv add fastapi uvicorn[standard] jinja2 python-multipart itsdangerous httpx \
       pydantic pydantic-settings boto3 aioboto3 langchain langchain-openai \
       langchain-community feedparser arxiv python-dateutil structlog \
       websockets passlib[argon2]
```
Expected: `uv.lock` generated, dependencies installed into `.venv/`.

**Step 4: Add dev dependencies**

Run:
```
uv add --dev pytest pytest-asyncio pytest-playwright hypothesis \
       ruff mypy bandit safety detect-secrets pre-commit locust atheris \
       moto httpx types-python-dateutil
```
Expected: dev group populated.

**Step 5: Commit**

```
git add pyproject.toml uv.lock .python-version
git commit -m "chore: initialize uv project with core and dev dependencies"
```

### Task 1.2: .gitignore (complete) and .gitattributes for LFS

**Files:**
- Modify: `.gitignore`
- Create: `.gitattributes`

**Step 1: Ensure `.gitignore` has all categories**

Confirm `.gitignore` already includes: secrets/env, coding agent artifacts, editor backups, IDEs, Python cache/venv, logs, OS, build, `.worktrees/`. Add any missing entry from the design doc §Version Control.

**Step 2: Create `.gitattributes`**

```
*.bin       filter=lfs diff=lfs merge=lfs -text
*.zip       filter=lfs diff=lfs merge=lfs -text
*.tar.gz    filter=lfs diff=lfs merge=lfs -text
*.model     filter=lfs diff=lfs merge=lfs -text
*.onnx      filter=lfs diff=lfs merge=lfs -text
*.pkl       filter=lfs diff=lfs merge=lfs -text
*.h5        filter=lfs diff=lfs merge=lfs -text
*.parquet   filter=lfs diff=lfs merge=lfs -text
*.mp4       filter=lfs diff=lfs merge=lfs -text
*.mov       filter=lfs diff=lfs merge=lfs -text
*.mp3       filter=lfs diff=lfs merge=lfs -text
*.wav       filter=lfs diff=lfs merge=lfs -text
```

**Step 3: Commit**

```
git add .gitignore .gitattributes
git commit -m "chore: configure gitignore and git LFS attributes"
```

### Task 1.3: Ruff, Mypy, Bandit configuration

**Files:**
- Modify: `pyproject.toml` (add `[tool.ruff]`, `[tool.mypy]` sections)
- Create: `quality/static/bandit.yml`

**Step 1: Add `[tool.ruff]` to `pyproject.toml`**

Target Python 3.12, line length 100, select rule sets: E, F, I, B, UP, S (security), N, SIM, RUF. Configure per-file ignores for `tests/` (allow `assert`, `S101`).

**Step 2: Add `[tool.mypy]` to `pyproject.toml`**

`strict = true`, `python_version = "3.12"`, `plugins = ["pydantic.mypy"]`.

**Step 3: Add `[tool.pytest.ini_options]` to `pyproject.toml`**

`testpaths = ["tests"]`, `asyncio_mode = "auto"`, markers for `unit`, `integration`, `e2e`, `hypothesis`.

**Step 4: Create `quality/static/bandit.yml`**

Exclude `tests/`, skip B101 (assert_used in tests).

**Step 5: Verify configs load**

Run: `uv run ruff check .` → expect clean (no files yet).
Run: `uv run mypy --version` → expect version print.
Run: `uv run bandit -c quality/static/bandit.yml -r . || true` → expect no errors parsing config.

**Step 6: Commit**

```
git add pyproject.toml quality/static/bandit.yml
git commit -m "chore: configure ruff, mypy, bandit, and pytest"
```

### Task 1.4: pre-commit configuration

**Files:**
- Create: `.pre-commit-config.yaml`
- Create: `scripts/run_unit_tests.sh`

**Step 1: Write `.pre-commit-config.yaml`**

Hooks to include:
- `pre-commit-hooks`: `check-ast`, `check-yaml`, `check-json`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace`, `detect-private-key`, `check-added-large-files` with `--maxkb=51200` (50 MB).
- `ruff-pre-commit`: `ruff` (lint) and `ruff-format`.
- `mirrors-mypy`: `mypy` with `--strict`.
- `bandit`: `bandit -c quality/static/bandit.yml -r app/`.
- `detect-secrets`: scan repo.
- Local hook: `scripts/run_unit_tests.sh` — runs `uv run pytest -m unit -x -q`.
- Local hook: `uv run safety check` (dependency CVEs).

**Step 2: Write `scripts/run_unit_tests.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
exec uv run pytest -m unit -x -q
```
`chmod +x scripts/run_unit_tests.sh`.

**Step 3: Install**

Run: `uv run pre-commit install`.

**Step 4: Run against all files**

Run: `uv run pre-commit run --all-files || true` (first-run baseline; formatters may edit files).

**Step 5: Commit**

```
git add .pre-commit-config.yaml scripts/run_unit_tests.sh
git commit -m "chore: add pre-commit hooks for syntax, format, security, tests"
```

### Task 1.5: detect-secrets baseline

**Files:**
- Create: `.secrets.baseline`

**Step 1:** Run `uv run detect-secrets scan > .secrets.baseline`.

**Step 2:** Commit: `git add .secrets.baseline && git commit -m "chore: add detect-secrets baseline"`.

### Task 1.6: CLAUDE.md — coding agent instructions

**Files:** Create `CLAUDE.md`.

**Contents (verbatim sections required):**

- Project summary (link to design doc).
- **Dependency management:** `uv` only. Never `pip install`. Add deps with `uv add <pkg>` or `uv add --dev <pkg>`.
- **Workflow:** feature branch off master, PR to merge, no direct master commits.
- **TDD:** failing test first, then minimal implementation, then refactor.
- **Quality gates (all must pass):** Ruff lint + format, Mypy strict, Bandit, Safety, detect-secrets, unit tests.
- **Test types required per change:** unit (always), hypothesis for data/decision logic, e2e for user-visible features.
- **Diagrams:** SVG in `docs/diagrams/`. ASCII art diagrams forbidden.
- **Commit policy:** Unit tests must pass before committing (enforced by pre-commit).
- **LFS:** files > 50 MB must go through LFS; pre-commit blocks otherwise.
- **Never:** commit secrets, disable pre-commit with `--no-verify`, leave failing tests, hand-write Node/npm files.

Commit: `git add CLAUDE.md && git commit -m "docs: add CLAUDE.md coding agent instructions"`.

### Task 1.7: README.md

**Files:** Create `README.md`.

**Sections:**

1. Project overview + link to design doc.
2. **Prerequisites:** Python 3.12, `uv`, Docker, `git-lfs`.
3. **Setup:**
   ```
   git lfs install
   uv sync
   uv run pre-commit install
   cp .env.example .env
   docker compose up -d dynamodb
   uv run python scripts/setup_tables.py
   ```
4. **Running the app:** `uv run uvicorn app.main:app --reload`.
5. **Development workflow:**
   - Create a feature branch: `git switch -c feature/<name>`.
   - Write failing test first (TDD).
   - All merges via pull request. No direct commits to master.
   - Pre-commit hooks run on every commit: syntax, format, lint, type, security, unit tests.
   - Commit is blocked if unit tests fail.
6. **Testing:**
   - Unit: `uv run pytest -m unit`
   - Hypothesis: `uv run pytest -m hypothesis`
   - Integration: `uv run pytest -m integration`
   - E2E: `uv run pytest -m e2e` (requires Playwright browsers: `uv run playwright install chromium`).
   - Fuzz: `bash quality/dynamic/run_fuzz.sh`.
   - Load: `bash quality/dynamic/run_load.sh`.
   - All unit + hypothesis + e2e expected to run during development.
7. **Linting / formatting:**
   - Lint: `uv run ruff check .`
   - Format: `uv run ruff format .`
   - Types: `uv run mypy app`
   - Security static: `uv run bandit -c quality/static/bandit.yml -r app`
   - Deps CVE: `uv run safety check`
   - Secrets: `uv run detect-secrets scan`
8. **Code review standards:**
   - PR description references design doc section.
   - CI green (tests, static, security).
   - Reviewer verifies tests cover new behavior.
   - No new Ruff/Mypy/Bandit findings.
9. **Diagrams:** SVG in `docs/diagrams/`. ASCII art forbidden.
10. **Deployment:** local (docker compose) and AWS (ECS Fargate via CloudFormation in `infra/`).
11. **Package management note:** `uv` only. Never invoke `pip` directly. Add deps with `uv add`.

Commit: `git add README.md && git commit -m "docs: add README with workflow, testing, and standards"`.

### Task 1.8: Directory skeleton and `.env.example`

**Files:**
- Create empty package dirs with `__init__.py`: `app/`, `app/models/`, `app/db/`, `app/agents/`, `app/news/`, `app/auth/`.
- Create `app/templates/`, `app/static/css/`, `app/static/js/`.
- Create `tests/unit/`, `tests/hypothesis/`, `tests/integration/`, `tests/e2e/`, `tests/conftest.py`.
- Create `quality/static/`, `quality/dynamic/fuzzing/`, `quality/dynamic/load/`.
- Create `infra/cloudformation/`, `infra/scripts/`, `infra/config/`.
- Create `docs/diagrams/` (for SVGs).
- Create `.env.example` with keys: `OPENROUTER_API_KEY`, `GUARDIAN_API_KEY`, `NEWSAPI_KEY`, `DB_BACKEND=local`, `DYNAMODB_ENDPOINT=http://localhost:8000`, `AWS_REGION=us-east-1`, `SESSION_SECRET=changeme`.

Commit: `git add -A && git commit -m "chore: scaffold project directory structure"`.

---

## Phase 2: Docker + DynamoDB Local

### Task 2.1: docker-compose.yml

**Files:** Create `docker-compose.yml`.

Services:
- `dynamodb`: `amazon/dynamodb-local:latest`, port `8000:8000`, `-sharedDb` flag.
- `app`: build from `./Dockerfile`, depends_on `dynamodb`, env from `.env`, port `8080:8080`.

**Step:** Write, then `docker compose config` to validate. Commit.

### Task 2.2: Dockerfile

**Files:** Create `Dockerfile`.

Base `python:3.12-slim`, install `uv`, copy `pyproject.toml` + `uv.lock`, `uv sync --frozen`, copy app, run `uvicorn app.main:app --host 0.0.0.0 --port 8080`.

Commit.

### Task 2.3: scripts/setup_tables.py

**Files:** Create `scripts/setup_tables.py` + `tests/unit/test_setup_tables.py`.

**Step 1: Failing test**

`test_setup_tables_creates_expected_tables` — uses `moto`'s `mock_aws` to mock DynamoDB, calls `setup_tables.main()`, asserts tables `users`, `threads`, `posts`, `news_items` exist with expected key schemas and GSIs.

**Step 2: Run test**

`uv run pytest tests/unit/test_setup_tables.py -v` → FAIL.

**Step 3: Minimal implementation** creating the four tables per data model in design doc §Data Model.

**Step 4: Run test** → PASS.

**Step 5: Commit.**

---

## Phase 3: Pydantic Data Models

For each of `User`, `Thread`, `Post`, `NewsItem`:

### Task 3.N: Model `<Name>`

**Files:**
- Create `app/models/<name>.py`.
- Create `tests/unit/models/test_<name>.py`.
- Create `tests/hypothesis/models/test_<name>_properties.py`.

**Step 1 (failing unit test):** Construct with valid fields; assert types, defaults, and that `id` auto-generates as UUID4 string.

**Step 2:** Run — FAIL.

**Step 3 (implementation):** Pydantic `BaseModel` with typed fields per design doc. Use `Field(default_factory=...)` for ids/timestamps.

**Step 4:** Run unit test — PASS.

**Step 5 (failing hypothesis test):** Property: round-trip through `.model_dump()` → `Model(**data)` preserves value for all generated inputs.

**Step 6:** Run — PASS (or fix).

**Step 7:** Ruff + Mypy clean.

**Step 8:** Commit.

Repeat for User, Thread, Post, NewsItem.

---

## Phase 4: Repository Interface + DynamoDB Local Implementation

### Task 4.1: Interface

**Files:**
- Create `app/db/interface.py` (Protocol / ABC with methods: `create_user`, `get_user`, `get_user_by_username`, `list_agents`, `create_thread`, `get_thread`, `list_threads`, `create_post`, `get_posts_by_thread`, `create_news_item`, `list_new_news_items`, `update_news_item_status`).
- Create `tests/unit/db/test_interface.py` asserting class is abstract and all methods are declared abstract.

TDD cycle: failing test → minimal `Protocol`/`ABC` → passing test → commit.

### Task 4.2: DynamoDB Local repository

**Files:**
- Create `app/db/dynamo_local.py`.
- Create `tests/integration/db/test_dynamo_local.py` using `moto`'s `mock_aws`.

**For each interface method, do one bite-sized TDD cycle:**

1. Write failing integration test (create/fetch/list).
2. Run — FAIL.
3. Implement method with `aioboto3` client; key schema matches `setup_tables`.
4. Run — PASS.
5. Commit.

Include error paths: not-found returns `None`, duplicate usernames raise `UserExistsError`.

### Task 4.3: DynamoDB AWS repository

**Files:** `app/db/dynamo_aws.py` + `tests/integration/db/test_dynamo_aws.py`.

Same methods; differs only in endpoint + region config. Use `moto` again to verify same contract. Commit.

### Task 4.4: Repository factory

**Files:** `app/db/__init__.py` exposes `get_repository()` that selects based on `settings.DB_BACKEND`. Unit-test both branches with monkeypatched settings.

---

## Phase 5: Logging

### Task 5.1: logging_config.py

**Files:** `app/logging_config.py` + `tests/unit/test_logging_config.py`.

Uses `structlog` to emit JSON logs with correlation IDs. Unit test asserts log records include `correlation_id` when set.

### Task 5.2: middleware.py

**Files:** `app/middleware.py` + `tests/unit/test_middleware.py`.

FastAPI middleware assigns a new UUID as `correlation_id` to each request and binds it to structlog context.

Commit.

---

## Phase 6: FastAPI App Skeleton

### Task 6.1: config.py

**Files:** `app/config.py` (Pydantic `BaseSettings`) + unit test.

Fields: `db_backend`, `dynamodb_endpoint`, `aws_region`, `openrouter_api_key`, `guardian_api_key`, `newsapi_key`, `session_secret`.

### Task 6.2: main.py

**Files:** `app/main.py` + `tests/unit/test_main.py`.

FastAPI app, startup event wires `get_repository()`, mounts templates and static, adds correlation middleware. Add `GET /health` returning `{"status": "ok"}`.

TDD: failing client test `client.get("/health").json() == {"status": "ok"}` → implement → pass → commit.

---

## Phase 7: Auth

### Task 7.1: Password hashing

`app/auth/passwords.py` using `passlib` argon2. TDD: hash+verify round-trip.

### Task 7.2: Sessions

`app/auth/sessions.py` — itsdangerous-signed session cookie. TDD: encode/decode round-trip, tamper detection.

### Task 7.3: Routes

`app/auth/routes.py` — `POST /register`, `POST /login`, `POST /logout`, `GET /login`, `GET /register`.

For each route, bite-sized TDD cycle (test → fail → implement → pass → commit).

Hypothesis property: `register` rejects usernames not matching `^[A-Za-z0-9_]{3,32}$`.

E2E test will come later in Phase 15.

---

## Phase 8: News Fetchers

### Task 8.0: Base interface

`app/news/interface.py` — `NewsFetcher` protocol with `source_name: str` and `async def fetch() -> list[NewsItem]`.

### Tasks 8.1 – 8.6: One per source (Guardian, arXiv, HackerNews, Reddit, NewsAPI, RSS)

**For each:**

**Files:**
- `app/news/<source>.py`
- `tests/unit/news/test_<source>.py` with recorded JSON fixtures in `tests/fixtures/<source>/`.
- `tests/hypothesis/news/test_<source>_parsing.py` (property: parser never crashes on random bytes).

**Steps (bite-sized):**
1. Capture a small JSON sample from the source API (check in under `tests/fixtures/`).
2. Write failing test: given fixture, `fetch()` returns expected `NewsItem` list (mock HTTP with `httpx.MockTransport`).
3. Run — FAIL.
4. Implement fetcher.
5. Run — PASS.
6. Hypothesis: parser robust to malformed input.
7. Commit.

### Task 8.7: Rate limiter + aggregator scheduler

`app/news/aggregator.py` — async background task runs every N minutes (configurable), iterates registered fetchers, dedupes by URL, writes NEW items via repository.

TDD with fake clock + fake fetchers. Commit.

---

## Phase 9: Agent System

### Task 9.1: Base agent

`app/agents/base_agent.py` — class holding persona, expertise, response probability, LangChain model binding (OpenRouter via `langchain-openai` with `base_url="https://openrouter.ai/api/v1"`).

TDD:
- Unit: expertise-match function true/false for sample inputs.
- Unit: relevance check calls LLM (mock LLM), parses yes/no.
- Unit: decide_to_respond combines expertise + relevance.
- Hypothesis: expertise match case-insensitive and whitespace-tolerant.

### Task 9.2: Agent registry

`app/agents/registry.py` — discovers free models via OpenRouter `/models` endpoint, filters where `pricing.prompt == "0"` and `pricing.completion == "0"`, creates one agent per model with seeded persona + expertise.

TDD with mocked OpenRouter API.

### Task 9.3: Discussion engine

`app/agents/discussion.py` — event dispatcher. `on_new_post(post)` → for each agent, schedule `respond_maybe(post)` with random jitter 30–300s.

TDD with fake clock (`asyncio` time mock) and fake agents.

### Task 9.4: News agent

`app/agents/news_agent.py` — special agent that reads `list_new_news_items()`, selects interesting ones via LLM prompt, creates threads, updates statuses.

TDD with mocked repository + mocked LLM.

---

## Phase 10: Thread/Post API Routes

Routes (each a bite-sized TDD cycle):

- `GET /` — home, lists threads.
- `GET /threads/{thread_id}` — thread page.
- `POST /threads/{thread_id}/posts` — create post (authenticated).
- `GET /agents` — agent directory.

Integration tests exercise routes through `httpx.AsyncClient`.

---

## Phase 11: WebSocket Live Updates

### Task 11.1: WebSocket endpoint

`app/ws.py` — `WebSocket /ws/threads/{thread_id}` — subscribes client to thread updates.

### Task 11.2: Publish post events

When `create_post` succeeds, push an HTMX fragment to all subscribers of that thread.

TDD: integration test connects two clients, one posts, other receives fragment.

---

## Phase 12: Frontend Templates + HTMX

Templates (each with a corresponding e2e test deferred to Phase 14):

- `base.html` — common layout, HTMX script tag, dark/light toggle, links to CSS.
- `home.html` — thread list.
- `thread.html` — nested post tree, WebSocket connection, HTMX swap targets.
- `agents.html` — agent cards.
- `login.html`, `register.html` — auth forms.
- `partials/post.html` — single post fragment used by WebSocket pushes.

CSS in `app/static/css/main.css` with variables for light/dark, agent/human post styles, source badges color-coded.

Commit each template individually.

---

## Phase 13: E2E Tests (Playwright via pytest-playwright)

### Task 13.0: Setup

- `uv run playwright install chromium`.
- `tests/e2e/conftest.py` — fixture to start docker-compose stack + app subprocess.

### Tasks 13.1 – 13.6: One per flow

- `test_auth_flow.py`
- `test_thread_view.py`
- `test_post_reply.py`
- `test_live_updates.py`
- `test_agent_directory.py`
- `test_responsive.py` (viewports: 390×844, 768×1024, 1440×900)

Each: write failing test → make it pass with template/backend tweaks → commit.

---

## Phase 14: Dynamic Analysis

### Task 14.1: Fuzz harnesses

`quality/dynamic/fuzzing/fuzz_news_parsers.py` (Atheris), `fuzz_post_validation.py`. `run_fuzz.sh` runs all for a bounded number of iterations.

### Task 14.2: Load tests

`quality/dynamic/load/locustfile.py` — scenarios: browse threads, post reply, sustain N WebSocket connections. `run_load.sh` invokes locust headless with a summary report.

Commit.

---

## Phase 15: Infra / AWS Deployment

### Task 15.1: CloudFormation

- `infra/cloudformation/dynamodb.yaml` — four tables with GSIs.
- `infra/cloudformation/ecs.yaml` — ECS Fargate cluster, task def, service, ALB, security groups.
- `infra/cloudformation/secrets.yaml` — Secrets Manager entries for API keys.

Each template: validated with `aws cloudformation validate-template` in CI.

### Task 15.2: Deploy scripts

`infra/scripts/deploy.sh`, `teardown.sh`, `push_image.sh`. Each has a `--dry-run` mode.

Commit each template + script.

---

## Phase 16: CI / CD

### Task 16.1: `.github/workflows/ci.yml`

Jobs:
- **lint**: Ruff, Mypy, Bandit, detect-secrets, Safety.
- **unit**: `uv run pytest -m unit`.
- **hypothesis**: `uv run pytest -m hypothesis`.
- **integration**: spin up DynamoDB Local service; `uv run pytest -m integration`.
- **e2e**: docker-compose up; `uv run pytest -m e2e`.
- **fuzz** (scheduled nightly).

All required before merge. Branch protection on master (documented in README).

### Task 16.2: `.github/workflows/deploy.yml`

On push to master (after merge): build image, push to ECR, `aws cloudformation deploy`.

Commit.

---

## Phase 17: Documentation Diagrams

### Task 17.1: SVG diagrams

Create (hand-authored or tool-exported SVG; no ASCII):

- `docs/diagrams/architecture.svg` — component overview.
- `docs/diagrams/data-flow.svg` — news → thread → agents flow.
- `docs/diagrams/deployment.svg` — AWS layout.

Commit.

---

## Final Pass

### Task 18.1: End-to-end smoke run

- `docker compose up`
- `uv run python scripts/setup_tables.py`
- `uv run uvicorn app.main:app`
- Open browser, register a user, observe news agent posting a thread and other agents replying live.
- Capture findings; file follow-up tasks as needed.

### Task 18.2: Open PR

Open `feature/scaffold` → master PR. CI must be green. Request review per README §Code review standards.

---

## Execution Notes

- Keep commits small (one TDD cycle = one commit usually).
- Never bypass pre-commit with `--no-verify`.
- Use `uv run <cmd>` for everything — never call `python`, `pytest`, or `pip` directly.
- Any new dependency: `uv add <pkg>` (or `uv add --dev`), then commit `pyproject.toml` and `uv.lock` together.
- If a step reveals a design gap, pause and update the design doc before coding around it.
