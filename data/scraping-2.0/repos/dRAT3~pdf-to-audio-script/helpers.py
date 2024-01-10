import asyncio
import re
from openai import AsyncOpenAI
from pathlib import Path

import settings

### CLEANING HELPERS
def remove_extra_whitespaces(text):
    # Replace multiple whitespaces (except newlines) with a single space
    cleaned_text = re.sub(r'[ \t]+', ' ', text)
    return cleaned_text

async def clean_up_text_with_3_5_turbo(text):
    client = AsyncOpenAI()
    prompt = settings.CLEANER_PROMPT + text
    response = await client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": settings.SYSTEM_PROMPT},
            {"role": "user", "content": settings.EXAMPLE_PROMPT_TO_FIX},
            {"role": "assistant", "content": settings.EXAMPLE_RESPONSE_FIXED},
            {"role": "user", "content": prompt},
        ]
    )

    response_text_stripped = response.choices[0].message.content.strip()

    # Check if the response is smaller than five lines
    if response_text_stripped.count('\n') < 5:
        return text
    else:
        return response.choices[0].message.content
