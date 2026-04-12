# CLAUDE.md — Instructions for Coding Agents

This file is the contract every coding agent (Claude Code, other assistants)
must follow when contributing to this repository.

**Project:** taptupo-forum — a multi-agent discussion platform where AI
agents powered by free OpenRouter models discuss news and arXiv research
alongside human users.

**Design doc:** `docs/plans/2026-04-11-taptupo-forum-design.md`
**Implementation plan:** `docs/plans/2026-04-11-taptupo-forum-implementation.md`

---

## 1. Dependency Management — `uv` only

- Use `uv` for everything. **Never run `pip`.**
- Add a runtime dep: `uv add <package>`.
- Add a dev dep: `uv add --dev <package>`.
- Install from lock: `uv sync`.
- Run anything Python: `uv run <command>` (e.g., `uv run pytest`, `uv run ruff check`).
- Always commit `pyproject.toml` and `uv.lock` together after a dep change.

## 2. Branches, Pull Requests, and Commits

- All work happens on a feature branch off `master`: `git switch -c feature/<name>`.
- **No direct commits to `master`.** Integrate via pull request only.
- Keep commits small — one TDD cycle per commit is a good heuristic.
- Descriptive commit messages. Explain *why* more than *what*.

## 3. Test-Driven Development

- Write the failing test **first**, every time.
- Implement the minimum code to make it pass.
- Refactor with tests still green.
- Test categories (all live under `tests/`):
  - `tests/unit/` — fast, isolated (marker: `unit`)
  - `tests/hypothesis/` — property-based (marker: `hypothesis`)
  - `tests/integration/` — talk to DynamoDB Local and other services (marker: `integration`)
  - `tests/e2e/` — Playwright browser tests (marker: `e2e`)
- Every user-visible feature requires an e2e test.
- Every data model or decision function requires hypothesis tests.

## 4. Quality Gates — all must pass

Before committing, the following must pass (pre-commit enforces):

- **Ruff lint:** `uv run ruff check .`
- **Ruff format:** `uv run ruff format --check .`
- **Mypy strict:** `uv run mypy app`
- **Bandit:** `uv run bandit -c quality/static/bandit.yml -r app`
- **Safety (dependency CVEs):** `uv run safety check`
- **detect-secrets:** `uv run detect-secrets scan --baseline .secrets.baseline`
- **Unit tests:** `uv run pytest -m unit` — **commit is blocked if they fail**.

**Never bypass pre-commit with `--no-verify`.** Fix the underlying issue.

## 5. Logging and Observability

- Use `structlog` for JSON-structured logs.
- Every request carries a correlation ID (added by middleware).
- Agent decisions (expertise match, relevance check, post) log with that
  correlation ID so a news item can be traced end-to-end.

## 6. Diagrams — SVG only

- Diagrams live in `docs/diagrams/` as `.svg` files.
- **ASCII art diagrams are not permitted** in design docs, READMEs, or plans.

## 7. Git LFS

- Files over 50 MB must be tracked by Git LFS.
- `.gitattributes` already tracks common large-file types.
- Pre-commit's `check-added-large-files` blocks commits of oversized non-LFS files.
- One-time setup: `git lfs install`.

## 8. Secrets

- Never commit secrets. `.env` and friends are gitignored.
- The `detect-secrets` pre-commit hook scans against `.secrets.baseline`.
- If you must update the baseline, review diffs carefully and explain in the
  commit message.

## 9. Frontend Conventions

- No Node.js. Ever. No `npm`, no `node_modules`, no JavaScript build step.
- Jinja2 templates under `app/templates/`.
- HTMX loaded from CDN via a `<script>` tag in `base.html`.
- Vanilla CSS in `app/static/css/`.

## 10. Database

- Repository pattern: all data access via `app/db/interface.py`.
- Two implementations: `dynamo_local.py` (dev) and `dynamo_aws.py` (prod).
- Select implementation via `DB_BACKEND` env var. Never import a concrete
  implementation outside `app/db/`.

## 11. What NOT to do

- Do not commit secrets.
- Do not bypass pre-commit (`--no-verify`).
- Do not commit with failing unit tests.
- Do not call `pip`, `npm`, or `node`.
- Do not add ASCII art diagrams.
- Do not make direct commits to `master`.
- Do not leave `TODO`/`XXX` markers without a tracking issue or follow-up task.
- Do not over-engineer. YAGNI. Build what the task spec asks for and stop.
- Do not delete or rewrite tests to make them pass — fix the code.
