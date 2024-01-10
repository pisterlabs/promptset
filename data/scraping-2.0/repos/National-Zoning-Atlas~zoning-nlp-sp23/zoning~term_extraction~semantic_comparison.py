import openai
from joblib import Memory
from retry import retry

from ..utils import get_project_root, get_jinja_environment


template  = get_jinja_environment().get_template("semantic_comparison.pmpt.tpl")
memory = Memory(get_project_root() / ".joblib_cache", verbose=0)


@memory.cache
@retry(exceptions=openai.error.RateLimitError, tries=-1, delay=10, backoff=1.25, jitter=(1, 10))  # type: ignore
def semantic_comparison(expected: str, actual: str) -> bool:
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
