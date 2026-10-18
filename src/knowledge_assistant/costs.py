"""Estimated API cost of indexing and answering.

Prices are the Gemini API paid-tier list prices in US dollars per million
tokens, checked on the date below on https://ai.google.dev/gemini-api/docs/pricing.
They change over time and must be checked again before relying on them. The
free tier costs nothing but has rate limits, and Google may use free-tier
content to improve its products.
"""

from __future__ import annotations

from dataclasses import dataclass

from .llm import TokenUsage

PRICES_CHECKED_ON = "2026-10-18"
PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float = 0.0  # thinking tokens are billed as output


PRICES: dict[str, ModelPrice] = {
    "gemini-3.5-flash-lite": ModelPrice(0.30, 2.50),
    # Introductory price until 31 December 2026 ($1.50 / $7.50 afterwards).
    "gemini-3.8-flash": ModelPrice(0.75, 3.75),
    "gemini-embedding-001": ModelPrice(0.15),
}


def estimate_tokens(text: str) -> int:
    """Rough token count for English text (about 4 characters per token),
    used when the API does not report the number of tokens (embeddings)."""
    return max(1, round(len(text) / 4))


def _price(model: str) -> ModelPrice | None:
    # The API may report a dated version, for example "gemini-3.5-flash-lite-001".
    for name in sorted(PRICES, key=len, reverse=True):
        if model == name or model.startswith(f"{name}-"):
            return PRICES[name]
    return None


def generation_cost(model: str, usage: TokenUsage) -> float | None:
    """Cost of one model call in dollars, or None for an unknown model."""
    price = _price(model)
    if price is None:
        return None
    output = usage.output_tokens + usage.thinking_tokens
    return (usage.prompt_tokens * price.input_per_million + output * price.output_per_million) / 1e6


def embedding_cost(model: str, tokens: int) -> float | None:
    price = _price(model)
    if price is None:
        return None
    return tokens * price.input_per_million / 1e6
