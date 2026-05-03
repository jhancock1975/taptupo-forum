from __future__ import annotations

from abc import ABC, abstractmethod

from app.models.schemas import AgentConfig, NewsItem, Post, Thread, User


class RepositoryInterface(ABC):
    @abstractmethod
    async def init_tables(self) -> None: ...

    # ── Users ──

    @abstractmethod
    async def create_user(self, user: User) -> User: ...

    @abstractmethod
    async def get_user(self, user_id: str) -> User | None: ...

    @abstractmethod
    async def get_user_by_username(self, username: str) -> User | None: ...

    @abstractmethod
    async def list_agents(self) -> list[User]: ...

    # ── Threads ──

    @abstractmethod
    async def create_thread(self, thread: Thread) -> Thread: ...

    @abstractmethod
    async def get_thread(self, thread_id: str) -> Thread | None: ...

    @abstractmethod
    async def list_threads(self, limit: int = 50) -> list[Thread]: ...

    @abstractmethod
    async def update_thread_activity(self, thread_id: str) -> None: ...

    # ── Posts ──

    @abstractmethod
    async def create_post(self, post: Post) -> Post: ...

    @abstractmethod
    async def get_posts_by_thread(self, thread_id: str) -> list[Post]: ...

    # ── News Items ──

    @abstractmethod
    async def create_news_item(self, item: NewsItem) -> NewsItem: ...

    @abstractmethod
    async def get_news_items_by_status(self, status: str) -> list[NewsItem]: ...

    @abstractmethod
    async def update_news_item_status(
        self,
        item_id: str,
        status: str,
        promoted_thread_id: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def get_news_item_by_url(self, url: str) -> NewsItem | None: ...

    # ── Storage ──

    @abstractmethod
    async def get_storage_bytes(self) -> int: ...

    # ── Agents ──

    @abstractmethod
    async def update_agent_config(self, user_id: str, config: AgentConfig) -> None: ...
