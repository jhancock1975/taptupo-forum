from __future__ import annotations

import asyncio
import base64
import random
import re
import uuid

import httpx
import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.db.interface import RepositoryInterface
from app.models.schemas import Post, Thread, User

logger = structlog.get_logger()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images/generations"
HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/v1"

MAX_RETRIES = 5
POST_REQUEST_DELAY_SECONDS = 12.0

# MIME types for known output modalities
_MODALITY_MIME: dict[str, str] = {
    "image": "image/png",
    "audio": "audio/mpeg",
}

_ROLE_TO_MESSAGE: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

# Cross-thread memory
_MAX_RELATED_THREADS = 3
_MAX_POSTS_PER_RELATED_THREAD = 3
_MAX_SNIPPET_CHARS = 200

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "about",
        "also",
        "and",
        "are",
        "been",
        "but",
        "can",
        "could",
        "did",
        "does",
        "from",
        "had",
        "has",
        "have",
        "here",
        "how",
        "into",
        "just",
        "more",
        "not",
        "really",
        "should",
        "some",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "through",
        "very",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
        "would",
        "your",
    }
)


def _tokenize(text: str) -> set[str]:
    """Return meaningful lowercase words (≥4 chars, not stop-words) from *text*."""
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP_WORDS}


class BaseAgent:
    def __init__(
        self,
        user: User,
        repo: RepositoryInterface,
        llm_semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self.user = user
        self.repo = repo
        if not user.agent_config:
            raise ValueError(f"User {user.username} has no agent_config")
        self.config = user.agent_config
        self._llm_semaphore = (
            llm_semaphore if llm_semaphore is not None else asyncio.Semaphore(1)
        )
        if self.config.provider == "huggingface":
            api_base = HF_INFERENCE_BASE
            api_key = settings.huggingface_api_key or "no-key"
        else:
            api_base = OPENROUTER_BASE_URL
            api_key = settings.openrouter_api_key or "no-key"
        self._llm = ChatOpenAI(
            model=self.config.model_id,
            openai_api_base=api_base,
            openai_api_key=api_key,
        ).with_retry(stop_after_attempt=MAX_RETRIES, wait_exponential_jitter=True)

    def _matches_expertise(self, text: str) -> bool:
        text_lower = text.lower()
        return any(area.lower() in text_lower for area in self.config.expertise_areas)

    def _should_respond(self) -> bool:
        return random.random() < self.config.response_probability

    async def _call_llm(self, messages: list[dict[str, str]]) -> str | None:
        if self.config.provider == "huggingface":
            if not settings.huggingface_api_key:
                logger.warning("huggingface_key_missing", agent=self.user.username)
                return None
        else:
            if not settings.openrouter_api_key:
                logger.warning("openrouter_key_missing", agent=self.user.username)
                return None

        lc_messages: list[BaseMessage] = [
            _ROLE_TO_MESSAGE.get(m["role"], HumanMessage)(content=m["content"])
            for m in messages
        ]

        async with self._llm_semaphore:
            try:
                response = await self._llm.ainvoke(lc_messages)
                await asyncio.sleep(POST_REQUEST_DELAY_SECONDS)
                return str(response.content)
            except Exception:
                logger.exception("llm_call_failed", agent=self.user.username)
                return None

    async def _find_related_threads(
        self, current_thread_id: str, keywords: set[str]
    ) -> list[tuple[Thread, list[Post]]]:
        """Return up to _MAX_RELATED_THREADS recent threads (with posts) whose
        titles share at least one keyword with *keywords*, ordered by overlap
        count descending. The current thread is excluded."""
        if not keywords:
            return []

        threads = await self.repo.list_threads(limit=20)
        scored: list[tuple[int, Thread]] = []
        for t in threads:
            if t.thread_id == current_thread_id:
                continue
            score = len(keywords & _tokenize(t.title))
            if score > 0:
                scored.append((score, t))

        scored.sort(key=lambda x: x[0], reverse=True)
        related: list[tuple[Thread, list[Post]]] = []
        for _, t in scored[:_MAX_RELATED_THREADS]:
            posts = await self.repo.get_posts_by_thread(t.thread_id)
            if posts:
                related.append((t, posts[:_MAX_POSTS_PER_RELATED_THREAD]))
        return related

    async def _generate_media(self, prompt: str) -> tuple[bytes, str] | None:
        """Call OpenRouter images/generations endpoint and return (data, mime_type).

        Returns None if the call fails or the API key is missing.
        """
        if not settings.openrouter_api_key:
            return None

        output_mod = self.config.output_modality
        if output_mod != "image":
            # Only image generation is currently handled via this path
            logger.warning(
                "unsupported_output_modality",
                agent=self.user.username,
                modality=output_mod,
            )
            return None

        payload = {
            "model": self.config.model_id,
            "prompt": prompt,
            "response_format": "b64_json",
            "n": 1,
        }
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }
        async with self._llm_semaphore:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        OPENROUTER_IMAGES_URL, json=payload, headers=headers
                    )
                    resp.raise_for_status()
                data_list = resp.json().get("data", [])
                if not data_list:
                    return None
                b64 = data_list[0].get("b64_json", "")
                if not b64:
                    return None
                raw = base64.b64decode(b64)
                mime = _MODALITY_MIME.get(output_mod, "application/octet-stream")
                await asyncio.sleep(POST_REQUEST_DELAY_SECONDS)
                return raw, mime
            except Exception:
                logger.exception("media_generation_failed", agent=self.user.username)
                return None

    async def maybe_respond(self, thread_id: str, post: Post) -> Post | None:
        if post.author_id == self.user.user_id:
            return None

        thread = await self.repo.get_thread(thread_id)
        if not thread:
            return None

        posts = await self.repo.get_posts_by_thread(thread_id)

        combined = f"{thread.title} {post.content}"
        if len(
            posts
        ) > settings.agent_seed_reply_post_count and not self._matches_expertise(
            combined
        ):
            return None

        if not self._should_respond():
            return None

        conversation_lines: list[str] = [
            f"You are participating in a forum thread titled: '{thread.title}'",
            "",
            "Here is the conversation so far:",
        ]
        for p in posts[-10:]:
            author = await self.repo.get_user(p.author_id)
            name = author.username if author else "Unknown"
            conversation_lines.append(f"{name}: {p.content}")

        # ── Cross-thread memory ────────────────────────────────────────────
        keywords = _tokenize(f"{thread.title} {post.content}")
        related = await self._find_related_threads(thread_id, keywords)
        if related:
            conversation_lines.append("")
            conversation_lines.append(
                "For context, here are relevant discussions from elsewhere in this forum "
                "that you may reference naturally in your reply:"
            )
            for rel_thread, rel_posts in related:
                conversation_lines.append(f'\n  — "{rel_thread.title}"')
                for rp in rel_posts:
                    rel_author = await self.repo.get_user(rp.author_id)
                    rel_name = rel_author.username if rel_author else "Unknown"
                    snippet = rp.content[:_MAX_SNIPPET_CHARS].replace("\n", " ")
                    conversation_lines.append(f"    {rel_name}: {snippet}")

        conversation_lines.append("")
        conversation_lines.append(
            "Write your reply to this discussion. Be conversational and natural."
        )

        conversation: list[dict[str, str]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": "\n".join(conversation_lines)},
        ]

        # ── Non-text output (image generation) ────────────────────────────
        if self.config.output_modality != "text":
            # Build an image prompt from the conversation context
            image_prompt = (
                f"Create an {self.config.output_modality} inspired by this forum discussion: "
                f"'{thread.title}'. " + " ".join(conversation_lines[-5:])
            )[:800]
            media_result = await self._generate_media(image_prompt)
            if not media_result:
                return None
            raw_bytes, mime_type = media_result

            from app.storage import s3 as s3_store

            ext = mime_type.split("/")[-1].split(";")[0] or "bin"
            key = f"agent-media/{self.user.user_id}/{uuid.uuid4()}.{ext}"
            try:
                media_url = await s3_store.upload_media(raw_bytes, key, mime_type)
            except Exception:
                logger.exception("s3_upload_failed", agent=self.user.username)
                return None

            reply = Post(
                thread_id=thread_id,
                author_id=self.user.user_id,
                content=f"[{self.config.output_modality} generated by {self.config.persona_name}]",
                content_type=mime_type,
                media_url=media_url,
                parent_post_id=post.post_id,
            )
            await self.repo.create_post(reply)
            await self.repo.update_thread_activity(thread_id)
            logger.info(
                "agent_media_replied",
                agent=self.user.username,
                modality=self.config.output_modality,
                thread_id=thread_id,
                post_id=reply.post_id,
            )
            return reply

        # ── Text output (default) ──────────────────────────────────────────
        reply_text = await self._call_llm(conversation)
        if not reply_text:
            return None

        reply = Post(
            thread_id=thread_id,
            author_id=self.user.user_id,
            content=reply_text.strip(),
            parent_post_id=post.post_id,
        )
        await self.repo.create_post(reply)
        await self.repo.update_thread_activity(thread_id)

        logger.info(
            "agent_replied",
            agent=self.user.username,
            thread_id=thread_id,
            post_id=reply.post_id,
        )
        return reply
