from functools import lru_cache
import openai
import logging
import json

from better_profanity import profanity

BANNED_TOKENS = ["trump"]


@lru_cache(maxsize=None)
def is_gpt_approved_caption(caption: str) -> bool:
    prompt = """
    You are a moderator for a fictional satellite imagery company.

    Examples of valid captions:
     * A satellite image of a mountain
     * A satellite image of a dark blue river
     * A satellite image of an island in a deep blue ocean
     * A satellite image of a natural disaster
     * A satellite image of a sci fi futuristic city

    Examples of invalid captions:
     * A satellite image of Trump
     * A satellite image of <body part>
     * A satellite image of a cat girl
     * A satellite image of <something that cannot be possibly be seen from space>
    """
    functions = [
        {
            "name": "is_valid",
            "description": "Decide whether the users caption is valid",
            "parameters": {
                "type": "object",
                "properties": {"is_valid": {"type": "boolean"}},
                "required": ["is_valid"],
            },
        }
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {"role": "user", "content": caption},
        ],
        functions=functions,
        function_call={"name": "is_valid"},
    )
    args = json.loads(resp.choices[0]["message"]["function_call"]["arguments"])
    return args["is_valid"]


def clean_caption(caption: str, default: str) -> str:
    fixed_caption = caption
    if not caption.startswith("a satellite image") or caption == default:
        fixed_caption = default
    elif any(token.lower() in caption.lower() for token in BANNED_TOKENS):
        fixed_caption = default
    elif profanity.contains_profanity(caption):
        fixed_caption = default
    elif not is_gpt_approved_caption(caption):
        fixed_caption = default
    if fixed_caption != caption:
        logging.warning(f"Fixed caption: {caption} -> {fixed_caption}")
    return fixed_caption
