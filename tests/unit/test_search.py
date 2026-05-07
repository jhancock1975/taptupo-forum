from __future__ import annotations

from datetime import UTC, date, datetime

from app.models.schemas import AgentConfig, Post, Thread, User
from app import search as search_mod
from app.search import SearchFilters, parse_date_input, search_threads


def test_parse_date_input_returns_none_for_blank_values() -> None:
    assert parse_date_input("") is None


def test_search_threads_matches_text_and_builds_snippet() -> None:
    thread = Thread(thread_id="t1", title="Rust performance", created_by="human")
    post = Post(
        thread_id="t1",
        author_id="agent",
        content="These benchmarks show Rust getting faster every release.",
        created_at=datetime(2026, 5, 6, 12, tzinfo=UTC),
    )
    users = {
        "human": User(user_id="human", username="alice"),
        "agent": User(
            user_id="agent",
            username="Delta",
            is_agent=True,
            agent_config=AgentConfig(
                model_id="openrouter/meta-llama",
                model_label="Meta Llama",
                persona_name="Delta",
            ),
        ),
    }

    hits = search_threads(
        [thread],
        {"t1": [post]},
        users,
        SearchFilters(query="benchmarks"),
    )

    assert [hit.thread.thread_id for hit in hits] == ["t1"]
    assert hits[0].snippet.startswith("These benchmarks")
    assert hits[0].matched_agents == ("Delta",)
    assert hits[0].matched_models == ("Meta Llama",)


def test_search_threads_combines_agent_model_and_date_filters() -> None:
    matching_thread = Thread(
        thread_id="match",
        title="Model notes",
        created_by="human",
        created_at=datetime(2026, 5, 5, tzinfo=UTC),
    )
    older_thread = Thread(
        thread_id="old",
        title="Older notes",
        created_by="human",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    users = {
        "human": User(user_id="human", username="alice"),
        "agent": User(
            user_id="agent",
            username="Delta",
            is_agent=True,
            agent_config=AgentConfig(
                model_id="openrouter/meta-llama",
                model_label="Meta Llama",
                persona_name="Delta",
            ),
        ),
        "other-agent": User(
            user_id="other-agent",
            username="Nova",
            is_agent=True,
            agent_config=AgentConfig(
                model_id="openrouter/other",
                model_label="Other Model",
                persona_name="Nova",
            ),
        ),
    }
    posts_by_thread = {
        "match": [
            Post(
                thread_id="match",
                author_id="agent",
                content="Daily summary",
                created_at=datetime(2026, 5, 6, 9, tzinfo=UTC),
            )
        ],
        "old": [
            Post(
                thread_id="old",
                author_id="other-agent",
                content="Daily summary",
                created_at=datetime(2026, 5, 2, 9, tzinfo=UTC),
            )
        ],
    }

    hits = search_threads(
        [matching_thread, older_thread],
        posts_by_thread,
        users,
        SearchFilters(
            agent_username="Delta",
            model="openrouter/meta-llama",
            start_date=date(2026, 5, 6),
            end_date=date(2026, 5, 6),
        ),
    )

    assert [hit.thread.thread_id for hit in hits] == ["match"]


def test_search_threads_returns_no_hits_without_filters_or_matches() -> None:
    thread = Thread(thread_id="t1", title="General chat", created_by="human")
    users = {"human": User(user_id="human", username="alice")}

    assert search_threads([thread], {"t1": []}, users, SearchFilters()) == []
    assert (
        search_threads([thread], {"t1": []}, users, SearchFilters(query="missing")) == []
    )


def test_search_threads_handles_thread_only_and_date_only_matches() -> None:
    thread = Thread(
        thread_id="t1",
        title="Rust benchmarks",
        created_by="human",
        created_at=datetime(2026, 5, 6, 8, tzinfo=UTC),
    )
    late_post = Post(
        thread_id="t1",
        author_id="human",
        content="Later follow-up without the keyword.",
        created_at=datetime(2026, 5, 8, 12, tzinfo=UTC),
    )
    users = {"human": User(user_id="human", username="alice")}

    title_hits = search_threads(
        [thread],
        {"t1": []},
        users,
        SearchFilters(query="benchmarks"),
    )
    assert title_hits[0].last_match_at == thread.created_at

    date_hits = search_threads(
        [thread],
        {"t1": [late_post]},
        users,
        SearchFilters(start_date=date(2026, 5, 8), end_date=date(2026, 5, 8)),
    )
    assert date_hits[0].matched_post_count == 1
    assert date_hits[0].snippet.startswith("Later follow-up")

    assert (
        search_threads(
            [thread],
            {"t1": [late_post]},
            users,
            SearchFilters(start_date=date(2026, 5, 9), end_date=date(2026, 5, 9)),
        )
        == []
    )


def test_search_helper_edges_are_covered() -> None:
    assert search_mod._match_score("anything", "", []) == 0
    assert search_mod._match_score("rust parser", "rust code", ["rust", "code"]) == 4
    assert search_mod._build_snippet("", "rust") == ""
    assert search_mod._build_snippet("Short text", "") == "Short text"
    assert search_mod._build_snippet("Short text", "missing") == "Short text"
    centered = search_mod._build_snippet(
        "prefix " + ("x" * 120) + " rust " + ("y" * 120) + " suffix",
        "rust",
    )
    assert centered.startswith("...")
    assert centered.endswith("...")


def test_in_date_range_excludes_values_after_end_date() -> None:
    filters = SearchFilters(end_date=date(2026, 5, 6))

    assert search_mod._in_date_range(datetime(2026, 5, 6, 23, 59, tzinfo=UTC), filters)
    assert not search_mod._in_date_range(
        datetime(2026, 5, 7, 0, 0, tzinfo=UTC),
        filters,
    )
