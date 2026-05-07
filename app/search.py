from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.schemas import Post, Thread, User

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_SNIPPET_RADIUS = 90


@dataclass(frozen=True)
class SearchFilters:
    query: str = ""
    agent_username: str = ""
    model: str = ""
    start_date: date | None = None
    end_date: date | None = None

    @property
    def has_active_filters(self) -> bool:
        return any(
            (
                self.query.strip(),
                self.agent_username.strip(),
                self.model.strip(),
                self.start_date,
                self.end_date,
            )
        )


@dataclass(frozen=True)
class SearchHit:
    thread: Thread
    snippet: str
    matched_post_count: int
    participants: tuple[str, ...]
    matched_agents: tuple[str, ...]
    matched_models: tuple[str, ...]
    last_match_at: datetime
    score: int


def search_threads(
    threads: list[Thread],
    posts_by_thread: dict[str, list[Post]],
    users_by_id: dict[str, User],
    filters: SearchFilters,
    *,
    limit: int = 50,
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    normalized_query = _normalize_text(filters.query)
    tokens = _query_tokens(normalized_query)

    for thread in threads:
        posts = posts_by_thread.get(thread.thread_id, [])
        hit = _search_thread(
            thread,
            posts,
            users_by_id,
            filters,
            normalized_query,
            tokens,
        )
        if hit is not None:
            hits.append(hit)

    hits.sort(key=lambda hit: (hit.score, hit.last_match_at), reverse=True)
    return hits[:limit]


def parse_date_input(value: str) -> date | None:
    text = value.strip()
    if not text:
        return None
    return date.fromisoformat(text)


def _search_thread(
    thread: Thread,
    posts: list[Post],
    users_by_id: dict[str, User],
    filters: SearchFilters,
    normalized_query: str,
    tokens: list[str],
) -> SearchHit | None:
    participants = _participants(thread, posts, users_by_id)
    agent_participants = _agent_participants(thread, posts, users_by_id)
    matched_agents = _filter_agents(agent_participants, filters)
    if filters.agent_username or filters.model:
        if not matched_agents:
            return None
    elif not filters.has_active_filters:
        return None

    eligible_posts = [
        post for post in posts if _in_date_range(post.created_at, filters)
    ]
    thread_in_range = _in_date_range(thread.created_at, filters)
    if (
        (filters.start_date or filters.end_date)
        and not eligible_posts
        and not thread_in_range
    ):
        return None

    thread_match_score = 0
    thread_match_time = thread.created_at
    snippet_source = thread.summary or thread.title
    matching_posts: list[tuple[Post, int]] = []
    if normalized_query:
        thread_text = _searchable_thread_text(thread)
        thread_match_score = _match_score(thread_text, normalized_query, tokens)
        matching_posts = [
            (
                post,
                _match_score(_normalize_text(post.content), normalized_query, tokens),
            )
            for post in posts
        ]
        matching_posts = [(post, score) for post, score in matching_posts if score > 0]
        if not matching_posts and thread_match_score == 0:
            return None
        if matching_posts:
            top_post, top_score = max(matching_posts, key=lambda item: item[1])
            snippet_source = top_post.content
            thread_match_time = top_post.created_at
            thread_match_score += top_score
        elif thread_match_score > 0:
            thread_match_time = thread.created_at
    else:
        if eligible_posts:
            latest_post = max(eligible_posts, key=lambda post: post.created_at)
            snippet_source = latest_post.content
            thread_match_time = latest_post.created_at
        elif posts:
            latest_post = max(posts, key=lambda post: post.created_at)
            snippet_source = latest_post.content
            thread_match_time = latest_post.created_at

    filtered_posts = (
        eligible_posts if (filters.start_date or filters.end_date) else posts
    )
    if normalized_query:
        matched_post_count = len(
            [
                post
                for post, score in matching_posts
                if score > 0 and (post in filtered_posts or not filtered_posts)
            ]
        )
    else:
        matched_post_count = len(filtered_posts)

    final_agents = (
        tuple(sorted(agent.username for agent in matched_agents))
        if matched_agents
        else tuple(sorted(agent.username for agent in agent_participants))
    )
    final_models = _agent_models(matched_agents or agent_participants)

    score = thread_match_score
    if filters.agent_username:
        score += 25
    if filters.model:
        score += 25
    if filters.start_date or filters.end_date:
        score += 10
    if not normalized_query:
        score += max(1, matched_post_count)

    snippet = _build_snippet(snippet_source, normalized_query)
    return SearchHit(
        thread=thread,
        snippet=snippet,
        matched_post_count=matched_post_count,
        participants=participants,
        matched_agents=final_agents,
        matched_models=final_models,
        last_match_at=thread_match_time,
        score=score,
    )


def _participants(
    thread: Thread, posts: list[Post], users_by_id: dict[str, User]
) -> tuple[str, ...]:
    names: set[str] = set()
    for user_id in [thread.created_by, *(post.author_id for post in posts)]:
        user = users_by_id.get(user_id)
        names.add(user.username if user else "Unknown")
    return tuple(sorted(names))


def _agent_participants(
    thread: Thread, posts: list[Post], users_by_id: dict[str, User]
) -> list[User]:
    seen: set[str] = set()
    agents: list[User] = []
    for user_id in [thread.created_by, *(post.author_id for post in posts)]:
        if user_id in seen:
            continue
        seen.add(user_id)
        user = users_by_id.get(user_id)
        if user and user.is_agent and user.agent_config:
            agents.append(user)
    return agents


def _filter_agents(agents: list[User], filters: SearchFilters) -> list[User]:
    filtered = agents
    if filters.agent_username:
        wanted = filters.agent_username.casefold()
        filtered = [agent for agent in filtered if agent.username.casefold() == wanted]
    if filters.model:
        wanted_model = filters.model.casefold()
        filtered = [
            agent
            for agent in filtered
            if agent.agent_config
            and (
                agent.agent_config.model_id.casefold() == wanted_model
                or agent.agent_config.model_label.casefold() == wanted_model
            )
        ]
    return filtered


def _agent_models(agents: list[User]) -> tuple[str, ...]:
    values = {
        agent.agent_config.model_label or agent.agent_config.model_id
        for agent in agents
        if agent.agent_config
    }
    return tuple(sorted(value for value in values if value))


def _searchable_thread_text(thread: Thread) -> str:
    parts = [
        thread.title,
        thread.summary or "",
        " ".join(thread.categories),
        thread.source_url or "",
        thread.source_type or "",
    ]
    return _normalize_text(" ".join(parts))


def _match_score(text: str, normalized_query: str, tokens: list[str]) -> int:
    if not normalized_query:
        return 0
    score = 0
    if normalized_query in text:
        score += 80
    if tokens:
        token_hits = sum(1 for token in tokens if token in text)
        if token_hits == len(tokens):
            score += 30 + token_hits * 5
        elif token_hits > 0:
            score += token_hits * 4
    return score


def _build_snippet(source: str, normalized_query: str) -> str:
    text = _WHITESPACE_RE.sub(" ", source).strip()
    if not text:
        return ""
    if not normalized_query:
        return text[:220] + ("..." if len(text) > 220 else "")

    index = text.casefold().find(normalized_query)
    if index < 0:
        return text[:220] + ("..." if len(text) > 220 else "")

    start = max(0, index - _SNIPPET_RADIUS)
    end = min(len(text), index + len(normalized_query) + _SNIPPET_RADIUS)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"...{snippet}"
    if end < len(text):
        snippet = f"{snippet}..."
    return snippet


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _query_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _in_date_range(value: datetime, filters: SearchFilters) -> bool:
    if filters.start_date is not None:
        start_at = datetime.combine(filters.start_date, time.min, tzinfo=UTC)
        if value < start_at:
            return False
    if filters.end_date is not None:
        end_exclusive = datetime.combine(
            filters.end_date + timedelta(days=1),
            time.min,
            tzinfo=UTC,
        )
        if value >= end_exclusive:
            return False
    return True
