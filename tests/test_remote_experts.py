"""Hosted GLM-5.2 / DeepSeek-V4-Pro roles in the expert pool."""

from __future__ import annotations

import pytest

from legal_research.config import Settings
from legal_research.llm.pool import DEEPSEEK, GLM, WRITER, build_expert_pool


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"llm_mode": "openai"}
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def test_hosted_roles_absent_by_default() -> None:
    pool = build_expert_pool(_settings())
    assert GLM not in pool.roles
    assert DEEPSEEK not in pool.roles


def test_enabled_without_key_stays_absent() -> None:
    pool = build_expert_pool(_settings(glm_enable=True, deepseek_enable=True))
    assert GLM not in pool.roles
    assert DEEPSEEK not in pool.roles


def test_enabled_providers_join_the_pool() -> None:
    pool = build_expert_pool(
        _settings(
            glm_enable=True,
            glm_api_key="glm-key",
            deepseek_enable=True,
            deepseek_api_key="ds-key",
        )
    )
    assert pool.get(GLM).name == GLM
    assert pool.get(DEEPSEEK).name == DEEPSEEK


def test_role_map_reassigns_a_local_role() -> None:
    pool = build_expert_pool(
        _settings(glm_enable=True, glm_api_key="glm-key", remote_role_map="writer=glm")
    )
    assert pool.get(WRITER) is pool.get(GLM)


def test_role_map_ignores_inactive_provider() -> None:
    """The mapping can stay in .env while the provider is toggled back off."""

    pool = build_expert_pool(_settings(remote_role_map="writer=glm"))
    assert pool.get(WRITER).name == WRITER


def test_unprefixed_env_names_are_shared_with_patent_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLM_ENABLE", "true")
    monkeypatch.setenv("GLM_API_KEY", "shared-key")
    settings = Settings(_env_file=None, llm_mode="openai")  # type: ignore[call-arg]
    assert settings.glm_enable is True
    assert settings.glm_api_key == "shared-key"
