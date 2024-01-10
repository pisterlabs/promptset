from dataclasses import dataclass
import os
import yaml
from typing import Optional
import openai
from aiolimiter import AsyncLimiter
from datetime import date

MODEL_OLD = "ft:gpt-3.5-turbo-0613:personal::7wldgc8B"
MODEL = "ft:gpt-3.5-turbo-0613:personal::8Ese4HRZ"

SYSTEM_PROMPT_TEMPLATE = """You are a Wikidata administrator in {}. You will be shown a Wikidata item and an edit to that item.
You should decide whether the edit should be reverted and then output a rationale for your decision and your decision in YAML format.
"""


def make_system_prompt():
    # get current month and year
    month = date.today().strftime("%B")
    year = date.today().strftime("%Y")
    today = f"{month} {year}"
    return SYSTEM_PROMPT_TEMPLATE.format(today)


@dataclass
class ClassificationResult:
    revert: Optional[bool]
    rationale: Optional[str]
    doc: str


class Classifier:
    def __init__(self):
        key_file = os.path.expanduser("~/openai.key")
        with open(key_file, 'r') as f:
            openai.api_key = f.read().strip()
        # maximum rate of 3/minute
        self._limiter = AsyncLimiter(3)

    async def classify(self, doc) -> Optional[ClassificationResult]:
        async with self._limiter:
            try:
                completion = await openai.ChatCompletion.acreate(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": make_system_prompt()},
                        {"role": "user", "content": doc},
                    ]
                )
                if completion:
                    doc = completion.get("choices")[0].get(
                        "message").get("content")
                    try:
                        response = yaml.safe_load(doc)
                        revert = response.get("revert")
                        rationale = response.get("rationale")
                        return ClassificationResult(revert, rationale, doc)
                    except:
                        return ClassificationResult(None, None, doc)
            except:
                return None
