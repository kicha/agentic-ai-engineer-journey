from __future__ import annotations

from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import QueryingConfig


def create_query_llm(config: QueryingConfig) -> BaseChatModel:
    """Create the LLM used by rewrite, multi-query and HyDE generation."""

    if not config.groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is required for query rewriting, multi-query and HyDE."
        )

    return ChatGroq(
        model=config.llm_model,
        api_key=config.groq_api_key,
        temperature=0.0,
    )
