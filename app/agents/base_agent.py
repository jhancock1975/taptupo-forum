from __future__ import annotations

import asyncio
import base64
import json
import random
import re
import uuid
from typing import Any

import httpx
import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import settings
from app.db.interface import RepositoryInterface
from app.mcp.catalog import MCPToolCatalog
from app.models.schemas import Post, Thread, User

logger = structlog.get_logger()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images/generations"
HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/v1"
HF_INFERENCE_MODELS = "https://api-inference.huggingface.co/models"

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

_TOOL_REQUEST_RE = re.compile(
    r"^TOOL_REQUEST:\s*([a-zA-Z0-9_.-]+)\s*\|\s*(\{.*\})\s*$"
)

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
        tool_catalog: MCPToolCatalog | None = None,
    ) -> None:
        self.user = user
        self.repo = repo
        if not user.agent_config:
            raise ValueError(f"User {user.username} has no agent_config")
        self.config = user.agent_config
        self._llm_semaphore = (
            llm_semaphore if llm_semaphore is not None else asyncio.Semaphore(1)
        )
        self._tool_catalog = tool_catalog
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

    def _parse_tool_requests(
        self, text: str, max_tools: int = 2
    ) -> list[tuple[str, dict[str, Any]]]:
        results: list[tuple[str, dict[str, Any]]] = []
        for line in text.splitlines():
            line = line.strip()
            match = _TOOL_REQUEST_RE.match(line)
            if not match:
                continue
            tool_name = match.group(1)
            raw_args = match.group(2)
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning(
                    "tool_request_json_invalid",
                    agent=self.user.username,
                    tool=tool_name,
                )
                parsed = {}
            args: dict[str, Any] = {}
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    args[str(key)] = value
            results.append((tool_name, args))
            if len(results) >= max_tools:
                break
        return results

    @staticmethod
    def _strip_tool_request_lines(text: str) -> str:
        cleaned = [
            line
            for line in text.splitlines()
            if not line.strip().startswith("TOOL_REQUEST:")
        ]
        return "\n".join(cleaned).strip()

    async def _generate_media(self, prompt: str) -> tuple[bytes, str] | None:
        """Generate media (image or audio) and return (raw_bytes, mime_type).

        Routes to the correct provider/endpoint based on agent config.
        Returns None if the call fails or the API key is missing.
        """
        output_mod = self.config.output_modality
        if output_mod not in _MODALITY_MIME:
            logger.warning(
                "unsupported_output_modality",
                agent=self.user.username,
                modality=output_mod,
            )
            return None

        if self.config.provider == "huggingface":
            return await self._generate_media_huggingface(prompt, output_mod)
        return await self._generate_media_openrouter(prompt, output_mod)

    async def _generate_media_openrouter(
        self, prompt: str, output_mod: str
    ) -> tuple[bytes, str] | None:
        if not settings.openrouter_api_key:
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

    async def _generate_media_huggingface(
        self, prompt: str, output_mod: str
    ) -> tuple[bytes, str] | None:
        if not settings.huggingface_api_key:
            return None
        url = f"{HF_INFERENCE_MODELS}/{self.config.model_id}"
        headers = {
            "Authorization": f"Bearer {settings.huggingface_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": prompt}
        async with self._llm_semaphore:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                mime = _MODALITY_MIME.get(output_mod, "application/octet-stream")
                await asyncio.sleep(POST_REQUEST_DELAY_SECONDS)
                return resp.content, mime
            except Exception:
                logger.exception(
                    "hf_media_generation_failed", agent=self.user.username
                )
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

        tool_profile = self.config.tool_profile
        if self._tool_catalog is not None and tool_profile.affinity != "none":
            tool_list_response = await self._tool_catalog.invoke("meta.list_tools", {})
            tool_result = tool_list_response.get("result", {})
            raw_tools = tool_result.get("tools", []) if isinstance(tool_result, dict) else []
            if isinstance(raw_tools, list) and raw_tools:
                tool_descriptions: dict[str, str] = {}
                tool_context = "\n".join(
                    [thread.title, post.content, *(p.content for p in posts[-3:])]
                )
                recent_content = [p.content for p in posts[-5:]]
                suggested_tools = self._tool_catalog.suggest_tools(
                    tool_context,
                    preferred_tools=tool_profile.preferred_tools,
                    recent_posts=recent_content,
                )
                conversation_lines.append("")

                max_tools = tool_profile.max_tools_per_turn
                if tool_profile.tool_nudge == "always":
                    conversation_lines.append(
                        "You should use MCP tools to verify factual, current, "
                        "or external claims before replying."
                    )
                elif tool_profile.tool_nudge == "rarely":
                    conversation_lines.append(
                        "You may use an MCP tool only if you really need "
                        "external data to answer accurately."
                    )
                else:
                    conversation_lines.append(
                        "Use a relevant MCP tool when it helps verify factual, current, "
                        "or external claims before replying."
                    )

                if suggested_tools:
                    conversation_lines.append(
                        "Strong tool candidates for this thread:"
                    )
                    for tool_name in suggested_tools:
                        conversation_lines.append(f"- {tool_name}")

                conversation_lines.append(
                    f"You may request up to {max_tools} tools. "
                    "Put each on its own line at the end:"
                )
                conversation_lines.append(
                    'TOOL_REQUEST: <tool_name> | {"arg": "value"}'
                )
                conversation_lines.append(
                    "Available tools from the meta.list_tools service:"
                )
                for tool in raw_tools:
                    if not isinstance(tool, dict):
                        continue
                    name = str(tool.get("name", "")).strip()
                    description = str(tool.get("description", "")).strip()
                    if not name:
                        continue
                    tool_descriptions[name] = description
                    if description:
                        conversation_lines.append(f"- {name}: {description}")
                    else:
                        conversation_lines.append(f"- {name}")
                if suggested_tools:
                    conversation_lines.append("")
                    conversation_lines.append(
                        "If one of the suggested tools fits, prefer using it instead "
                        "of guessing."
                    )
                    for tool_name in suggested_tools:
                        description = tool_descriptions.get(tool_name, "")
                        if description:
                            conversation_lines.append(
                                f"- Suggested: {tool_name}: {description}"
                            )

        conversation_lines.append("")
        conversation_lines.append(
            "Write your reply to this discussion. Use a relevant tool call "
            "if helpful, but if no tool is useful, reply normally."
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

        if self._tool_catalog is not None:
            max_tools = self.config.tool_profile.max_tools_per_turn
            tool_requests = self._parse_tool_requests(reply_text, max_tools=max_tools)

            if tool_requests:
                tool_coros = [
                    self._tool_catalog.invoke(name, args)
                    for name, args in tool_requests
                ]
                tool_responses = await asyncio.gather(
                    *tool_coros, return_exceptions=True
                )

                results_parts: list[str] = []
                for (name, _), response in zip(tool_requests, tool_responses):
                    if isinstance(response, Exception):
                        results_parts.append(
                            f"Tool '{name}' failed: {type(response).__name__}: {response}"
                        )
                    else:
                        results_parts.append(
                            f"Tool '{name}' response:\n"
                            f"{json.dumps(response, ensure_ascii=True)}"
                        )

                follow_up_prompt = (
                    "You requested tools. Here are the results:\n"
                    + "\n\n".join(results_parts)
                    + "\nWrite the final forum reply using these results when helpful. "
                    "Do not mention using a tool, checking a tool, internal reasoning, "
                    "or any draft/planning language in the final reply."
                )
                follow_up_messages = [
                    *conversation,
                    {"role": "user", "content": follow_up_prompt},
                ]
                follow_up_reply = await self._call_llm(follow_up_messages)

                if follow_up_reply is None:
                    logger.warning(
                        "tool_follow_up_failed",
                        agent=self.user.username,
                        tools=[name for name, _ in tool_requests],
                    )
                    return None

                chained_requests = self._parse_tool_requests(
                    follow_up_reply, max_tools=1
                )
                if chained_requests:
                    chain_name, chain_args = chained_requests[0]
                    chain_response = await self._tool_catalog.invoke(
                        chain_name, chain_args
                    )
                    chain_prompt = (
                        f"Chained tool '{chain_name}' response:\n"
                        f"{json.dumps(chain_response, ensure_ascii=True)}\n"
                        "Write the final forum reply. "
                        "Do not mention tools or internal process."
                    )
                    chain_messages = [
                        *follow_up_messages,
                        {"role": "user", "content": chain_prompt},
                    ]
                    final_reply = await self._call_llm(chain_messages)
                    if final_reply:
                        reply_text = final_reply
                    else:
                        reply_text = self._strip_tool_request_lines(follow_up_reply)
                else:
                    reply_text = follow_up_reply

        cleaned_reply = self._strip_tool_request_lines(reply_text)
        if not cleaned_reply:
            return None

        reply = Post(
            thread_id=thread_id,
            author_id=self.user.user_id,
            content=cleaned_reply,
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
