import os

from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic

MY_PROMPT = """
"""

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

COMP = anthropic.completions.create(
    max_tokens_to_sample=1000,
    model="claude-2",
    prompt=f"{HUMAN_PROMPT} {MY_PROMPT} {AI_PROMPT}",
)

print(COMP.completion)
