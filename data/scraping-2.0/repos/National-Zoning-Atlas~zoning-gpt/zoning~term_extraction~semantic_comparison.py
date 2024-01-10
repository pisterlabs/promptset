import json

import diskcache as dc
import openai
from tenacity import retry, retry_if_exception_type, wait_random_exponential

from ..utils import get_jinja_environment, get_project_root, cached

template  = get_jinja_environment().get_template("semantic_comparison.pmpt.tpl")
cache = dc.Cache(get_project_root() / ".diskcache")


@cached(cache, lambda *args: json.dumps(args))
@retry(
    retry=retry_if_exception_type(
        (
            openai.error.APIError,
            openai.error.RateLimitError,
            openai.error.APIConnectionError,
            openai.error.ServiceUnavailableError,
            openai.error.Timeout,
            openai.error.TryAgain,
        )
    ),
    wait=wait_random_exponential(multiplier=1, max=60),
)
def semantic_comparison(expected: str, actual: str) -> bool:
    # TODO: Is there a way to share this implementation with our generic prompt
    # function?
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.0, # We want these responses to be deterministic
        max_tokens=1,
        messages=[
            {
                "role": "user",
                "content": template.render(
                    actual=actual,
                    expected=expected,
                ),
            },
        ],
    )
    top_choice = resp.choices[0]
    text = top_choice.message.content
    return text == "Y"
