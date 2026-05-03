"""Discover free LLMs from OpenRouter and generate per-model skills for each persona."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.db.interface import RepositoryInterface
from app.models.schemas import AgentConfig, User

logger = structlog.get_logger()

OPENROUTER_API_MODELS = "https://openrouter.ai/api/v1/models"
SKILL_GENERATOR_MODEL = "openai/gpt-oss-20b:free"

HF_API_MODELS = "https://huggingface.co/api/models"
HF_INFERENCE_BASE = "https://router.huggingface.co/hf-inference/v1"
# Query params: warm inference-ready text-generation models, by downloads, max 30
HF_QUERY_PARAMS = {
    "pipeline_tag": "text-generation",
    "sort": "downloads",
    "direction": "-1",
    "limit": "30",
    "inference": "warm",
}

# Model IDs / prefixes confirmed broken: Venice-backed (429), expiring, or problematic
_EXCLUDED_IDS: frozenset[str] = frozenset(
    {
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen3-8b:free",
        "mistralai/mistral-7b-instruct:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-4-31b-it:free",
        "inclusionai/ling-2.6-flash:free",
        "inclusionai/ling-2.6-1t:free",
    }
)
_EXCLUDED_PREFIXES: tuple[str, ...] = ("google/gemma",)

_HF_EXCLUDED_PREFIXES: tuple[str, ...] = ("google/",)

_PROVIDER_DOMAINS: dict[str, str] = {
    "openai": "openai.com",
    "nvidia": "nvidia.com",
    "liquid": "liquid.ai",
    "nousresearch": "nousresearch.com",
    "minimax": "minimaxi.com",
    "z-ai": "zhipuai.cn",
    "anthropic": "anthropic.com",
    "mistralai": "mistral.ai",
    "meta-llama": "meta.com",
    "qwen": "qwenlm.github.io",
    "deepseek": "deepseek.com",
    "cohere": "cohere.com",
    "perplexity": "perplexity.ai",
    "microsoft": "microsoft.com",
    "amazon": "aws.amazon.com",
    "featherless": "featherless.ai",
    "huggingfaceh4": "huggingface.co",
}

# Simple injection-attempt pattern: "ignore/disregard/forget … above/previous/instructions"
_INJECTION_RE = re.compile(
    r"(?i)(ignore|disregard|forget)\s.{0,40}(above|previous|instruction)",
)

# ── Discovery log ────────────────────────────────────────────────────────────

# Each entry: {"ts": str (ISO-8601 UTC), "event": str, "data": dict[str, Any]}
DiscoveryLog = list[dict[str, Any]]
_MAX_LOG_ENTRIES = 200


def _log_event(log: DiscoveryLog | None, event: str, **data: Any) -> None:
    """Append a timestamped event to the discovery log (no-op if log is None)."""
    if log is None:
        return
    log.append(
        {
            "ts": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "event": event,
            "data": data,
        }
    )
    # Trim to keep only the most recent entries
    if len(log) > _MAX_LOG_ENTRIES:
        del log[: len(log) - _MAX_LOG_ENTRIES]


# ── Pure helpers (importable without side-effects) ───────────────────────────


def _is_free(model: dict[str, Any]) -> bool:
    p = model.get("pricing", {})
    return str(p.get("prompt", "1")) == "0" and str(p.get("completion", "1")) == "0"


def _is_excluded(model_id: str) -> bool:
    if model_id in _EXCLUDED_IDS:
        return True
    return any(model_id.startswith(pfx) for pfx in _EXCLUDED_PREFIXES)


def _input_modality(model: dict[str, Any]) -> str:
    """Return the input portion of 'input->output' modality string."""
    return model.get("architecture", {}).get("modality", "text->text").split("->")[0]


def _output_modality(model: dict[str, Any]) -> str:
    """Return the primary output type: 'text', 'image', 'audio', etc."""
    modality = model.get("architecture", {}).get("modality", "text->text")
    # e.g. "text->image", "text->text,image" → "image"; "text->text" → "text"
    output_part = modality.split("->")[1] if "->" in modality else "text"
    # Take first listed output type
    primary = output_part.split(",")[0].strip()
    return primary


def _provider_of(model_id: str) -> str:
    return model_id.split("/")[0] if "/" in model_id else model_id


def _icon_url(provider: str) -> str:
    domain = _PROVIDER_DOMAINS.get(provider, f"{provider}.com")
    return f"https://www.google.com/s2/favicons?domain={domain}&sz=32"


def _hf_is_accessible(model: dict[str, Any]) -> bool:
    """Return True for openly accessible (non-gated, non-excluded) HF models."""
    gated = model.get("gated", False)
    if gated is True or gated == "manual":
        return False
    model_id: str = model.get("id", "")
    return not any(model_id.startswith(pfx) for pfx in _HF_EXCLUDED_PREFIXES)


def _hf_model_to_api_format(model: dict[str, Any]) -> dict[str, Any]:
    """Normalise a HuggingFace model dict to the shape expected by build_agent_config."""
    hf_id: str = model.get("id", "")
    author = hf_id.split("/")[0] if "/" in hf_id else hf_id
    display_name = (hf_id.split("/")[-1] if "/" in hf_id else hf_id).replace("-", " ")
    return {
        "id": hf_id,
        "name": display_name,
        "description": "",
        "context_length": 4096,  # HF API does not expose context length; use safe default
        "pricing": {"prompt": "0", "completion": "0"},
        "architecture": {"modality": "text->text"},
        # Internal keys used by discovery logic — not stored in AgentConfig
        "_provider_type": "huggingface",
        "_diversity_provider": f"hf-{author}",
    }


def _context_label(ctx: int) -> str:
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M tokens"
    if ctx >= 1024:
        return f"{ctx // 1024}K tokens"
    return f"{ctx} tokens"


# ── Service ──────────────────────────────────────────────────────────────────


class ModelDiscoveryService:
    """Fetch free models from OpenRouter and assign one per agent persona."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def fetch_free_models(self) -> list[dict[str, Any]]:
        """Return all free models from OpenRouter whose input is text, sorted best-first.

        Includes text→text, text→image, text→audio, etc. — any model that takes
        text as input and produces any form of output.
        """
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                OPENROUTER_API_MODELS,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            resp.raise_for_status()
            raw: list[dict[str, Any]] = resp.json().get("data", [])

        free = [
            m
            for m in raw
            if _is_free(m)
            and not _is_excluded(m.get("id", ""))
            and _input_modality(m).startswith("text")  # text input required
        ]
        # Prefer larger context windows as a proxy for model capability
        free.sort(key=lambda m: m.get("context_length", 0), reverse=True)
        return free

    async def fetch_free_hf_models(self, hf_api_key: str) -> list[dict[str, Any]]:
        """Return normalised, accessible text-generation models from HuggingFace Hub.

        Requires a HuggingFace API key for authenticated access (higher rate limits).
        Only models with open/auto-gating that are warm (inference-ready) are returned.
        """
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                HF_API_MODELS,
                params=HF_QUERY_PARAMS,
                headers={"Authorization": f"Bearer {hf_api_key}"},
            )
            resp.raise_for_status()
            raw: list[dict[str, Any]] = resp.json()

        return [_hf_model_to_api_format(m) for m in raw if _hf_is_accessible(m)]

    @staticmethod
    def select_diverse_models(
        models: list[dict[str, Any]], count: int
    ) -> list[dict[str, Any]]:
        """Pick `count` models with one per provider (best context window first).

        Respects the ``_diversity_provider`` key when present (used for HuggingFace
        models so they are treated as distinct from same-author OpenRouter models).
        """
        seen: set[str] = set()
        selected: list[dict[str, Any]] = []
        for m in models:
            provider = m.get("_diversity_provider") or _provider_of(m.get("id", ""))
            if provider not in seen:
                seen.add(provider)
                selected.append(m)
            if len(selected) >= count:
                break
        return selected

    async def generate_model_skill(
        self,
        model: dict[str, Any],
        persona_name: str,
        persona_traits: list[str],
    ) -> str:
        """Call GPT-OSS to write a 2-3 sentence skill addon for this model + persona.

        For non-text-output models, generates a prompt describing what kind of
        visual/audio content they should produce and how.
        """
        if not self._api_key:
            return ""

        out_mod = _output_modality(model)

        llm = ChatOpenAI(
            model=SKILL_GENERATOR_MODEL,
            openai_api_key=self._api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=120,
            temperature=0.4,
        ).with_retry(stop_after_attempt=2)

        model_name = model.get("name") or model.get("id", "unknown")
        description = (model.get("description") or "")[:500]
        ctx = model.get("context_length", 0)

        if out_mod == "text":
            prompt = (
                f"Write 2-3 sentences (max 80 words) telling an AI forum participant "
                f"named {persona_name!r} (traits: {', '.join(persona_traits)}) how to "
                f"leverage the specific strengths of the LLM powering it in forum discussions.\n\n"
                f"Model: {model_name}\n"
                f"Description: {description or 'No description available.'}\n"
                f"Context window: {_context_label(ctx)}\n\n"
                f"Be concrete and actionable. Output only the sentences, no preamble or labels."
            )
        else:
            prompt = (
                f"Write 2-3 sentences (max 80 words) instructing an AI forum participant "
                f"named {persona_name!r} (traits: {', '.join(persona_traits)}) on how to "
                f"use this {out_mod}-generating model to contribute to forum discussions. "
                f"Describe what kind of {out_mod} content it should create and when.\n\n"
                f"Model: {model_name}\n"
                f"Description: {description or 'No description available.'}\n\n"
                f"Be concrete and actionable. Output only the sentences, no preamble or labels."
            )

        try:
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            skill = str(result.content).strip()
            # Sanitize: remove obvious prompt-injection attempts from model descriptions
            skill = _INJECTION_RE.sub("", skill)
            return skill[:400]
        except Exception:
            logger.exception("skill_generation_failed", model=model.get("id"))
            return ""

    async def curate_candidate_models(
        self,
        models: list[dict[str, Any]],
        count: int,
    ) -> list[dict[str, Any]]:
        """Ask a free LLM to rank the candidate model pool for forum discussion fitness.

        Sends a compact catalogue of model IDs, names, context lengths, and providers
        to the skill-generator model and asks it to return the best ``count`` models
        as a JSON array of IDs.  Falls back silently to the original order if the LLM
        call fails, returns unparseable output, or no API key is configured.
        """
        if not self._api_key or not models:
            return models

        id_to_model: dict[str, dict[str, Any]] = {}
        catalogue_lines: list[str] = []
        for m in models[:50]:  # cap catalogue size
            mid = m.get("id", "")
            name = m.get("name") or mid
            ctx = m.get("context_length", 0)
            provider = m.get("_provider_type", "openrouter")
            desc = (m.get("description") or "")[:100]
            catalogue_lines.append(
                f"- {mid} | {name} | ctx:{ctx} | {provider} | {desc}"
            )
            id_to_model[mid] = m

        catalogue = "\n".join(catalogue_lines)

        prompt = (
            f"You are selecting AI models to power forum discussion agents. "
            f"From the catalogue below, choose the best {count} models for "
            f"diverse, engaging forum conversations. "
            f"IMPORTANT: you MUST include at least one model with provider=huggingface "
            f"if any appear in the catalogue. Prefer variety across providers "
            f"(openrouter and huggingface) and a mix of context-window sizes. "
            f"Return ONLY a JSON array of model IDs, best first. "
            f'Example: ["id1", "id2"]\n\n'
            f"Catalogue:\n{catalogue}"
        )

        llm = ChatOpenAI(
            model=SKILL_GENERATOR_MODEL,
            openai_api_key=self._api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=300,
            temperature=0.2,
        ).with_retry(stop_after_attempt=2)

        try:
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = _INJECTION_RE.sub("", str(result.content).strip())
            # Extract first JSON array from the response (may have surrounding prose)
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if not match:
                logger.warning("curator_no_json_array", response=raw[:200])
                return models
            ranked_ids: list[str] = json.loads(match.group())
            ranked = [id_to_model[mid] for mid in ranked_ids if mid in id_to_model]
            mentioned = {m.get("id") for m in ranked}
            remaining = [m for m in models if m.get("id") not in mentioned]
            logger.info(
                "model_curation_complete", ranked=len(ranked), remaining=len(remaining)
            )
            return ranked + remaining
        except Exception:
            logger.exception("model_curation_failed")
            return models

    async def build_agent_config(
        self,
        model: dict[str, Any],
        persona: dict[str, Any],
    ) -> AgentConfig:
        """Build a full AgentConfig from model metadata + LLM-generated skill."""
        model_id = model.get("id", "")
        provider = _provider_of(model_id)
        model_name = model.get("name") or model_id
        ctx = model.get("context_length", 0)
        out_mod = _output_modality(model)

        skill = await self.generate_model_skill(
            model,
            persona["persona_name"],
            persona.get("personality_traits", []),
        )

        base_prompt: str = persona["base_system_prompt"]
        if out_mod != "text":
            # For image/audio models replace text instructions with generation prompt
            system_prompt = (
                f"{base_prompt}\n\nYou generate {out_mod} content. " f"{skill}"
                if skill
                else base_prompt
            )
        else:
            system_prompt = (
                f"{base_prompt}\n\nLeverage your model's strengths: {skill}"
                if skill
                else base_prompt
            )

        provider_type: str = model.get("_provider_type", "openrouter")
        if provider_type == "huggingface":
            label = f"HuggingFace · {model_name}"
            icon = _icon_url("huggingfaceh4")
        else:
            label = f"{provider.title()} · {model_name}"
            icon = _icon_url(provider)

        return AgentConfig(
            model_id=model_id,
            persona_name=persona["persona_name"],
            expertise_areas=persona["expertise_areas"],
            personality_traits=persona["personality_traits"],
            response_probability=persona["response_probability"],
            system_prompt=system_prompt,
            model_label=label,
            model_icon_url=icon,
            model_description=(model.get("description") or "")[:300],
            model_specializations=[],
            model_benchmarks=[],
            model_context_length=_context_label(ctx),
            model_params="",
            output_modality=out_mod,
            provider=provider_type,
        )

    async def refresh_agent_models(
        self,
        agents: list[User],
        repo: RepositoryInterface,
        personas: list[dict[str, Any]],
        log: DiscoveryLog | None = None,
        hf_api_key: str = "",
    ) -> int:
        """
        Fetch free models from OpenRouter and (optionally) HuggingFace, pick one
        per provider, assign each to an agent persona.
        Returns the number of agents whose config was successfully updated.
        """
        _log_event(log, "job_started")

        # ── OpenRouter free models ────────────────────────────────────────
        or_models: list[dict[str, Any]] = []
        try:
            or_models = await self.fetch_free_models()
        except Exception:
            logger.exception("model_discovery_fetch_failed")
            _log_event(log, "fetch_failed")

        # ── HuggingFace free models (optional) ───────────────────────────
        hf_models: list[dict[str, Any]] = []
        if hf_api_key:
            try:
                hf_models = await self.fetch_free_hf_models(hf_api_key)
                logger.info("hf_models_fetched", count=len(hf_models))
            except Exception:
                logger.exception("hf_model_discovery_fetch_failed")

        free_models = or_models + hf_models

        if not free_models:
            # Both sources returned nothing (or both failed)
            if not or_models and not hf_models:
                _log_event(log, "fetch_failed")
            else:
                logger.warning("no_free_models_found")
                _log_event(log, "no_free_models")
            return 0

        _log_event(
            log,
            "models_fetched",
            count=len(free_models),
            model_ids=[m.get("id") for m in free_models[:20]],
        )

        persona_map = {p["username"]: p for p in personas}
        pairs = [
            (u, persona_map[u.username])
            for u in agents
            if u.is_agent and u.agent_config and u.username in persona_map
        ]
        if not pairs:
            _log_event(log, "no_agent_pairs")
            return 0

        # ── LLM curation: reorder the pool by fitness for forum discussion ──
        # Ask for 3x the needed count so diversity selection has HF models to pick from
        curate_count = min(len(free_models), len(pairs) * 3)
        free_models = await self.curate_candidate_models(free_models, curate_count)
        _log_event(log, "curation_complete", pool_size=len(free_models))

        selected = self.select_diverse_models(free_models, len(pairs))
        if not selected:
            _log_event(log, "no_selected_models")
            return 0

        updated = 0
        for i, (user, persona) in enumerate(pairs):
            model = selected[i % len(selected)]
            try:
                new_config = await self.build_agent_config(model, persona)
                await repo.update_agent_config(user.user_id, new_config)
                logger.info(
                    "agent_model_assigned",
                    agent=user.username,
                    model=model.get("id"),
                )
                _log_event(
                    log,
                    "agent_assigned",
                    agent=user.username,
                    model_id=model.get("id"),
                    model_label=new_config.model_label,
                    output_modality=new_config.output_modality,
                )
                updated += 1
            except Exception:
                logger.exception("agent_update_failed", agent=user.username)
                _log_event(
                    log, "agent_failed", agent=user.username, model_id=model.get("id")
                )

        _log_event(log, "job_complete", updated=updated)
        return updated
