import numpy as np
import json
from anthropic_bedrock import AnthropicBedrock
import anthropic_bedrock
import os
from functools import lru_cache
from ..models import LLM_Cache
from .. import db, app

client = AnthropicBedrock(
    aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
)

from sqlalchemy.exc import IntegrityError


def llm_wrapper(text: str) -> str:
    """Wrapper for the LLM API with caching to database."""
    with app.app_context():
        # Attempt to fetch or create a new cache entry
        try:
            cached_result = LLM_Cache.query.filter_by(text_prompt=text).first()
            if not cached_result:
                llm_response = client.completions.create(
                    model="anthropic.claude-v2:1",
                    max_tokens_to_sample=1024,
                    prompt=text,
                    temperature=0.2,
                ).completion

                cached_result = LLM_Cache(text_prompt=text, llm_response=llm_response)
            return cached_result.llm_response
        except IntegrityError:
            db.session.rollback()
            # Handle the race condition here by re-querying the database
            return LLM_Cache.query.filter_by(text_prompt=text).first().llm_response

def save_llm_result(text_prompt: str, llm_response: str) -> None:
    try:
        with app.app_context():
            db.session.add(LLM_Cache(text_prompt=text_prompt, llm_response=llm_response))
            db.session.commit()
    except IntegrityError:
        None