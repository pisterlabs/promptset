"""Configuration for Hephaestus."""

from pathlib import Path

import langchain.globals
from langchain.cache import SQLiteCache


def configure_langchain_cache(cache_dir: Path) -> None:
    """Configure the LLM cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not langchain.globals.get_llm_cache():
        langchain.globals.set_llm_cache(
            SQLiteCache(database_path=str(cache_dir / ".langchain.db"))
        )
