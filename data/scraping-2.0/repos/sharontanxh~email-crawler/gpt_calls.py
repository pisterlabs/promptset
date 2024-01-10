import openai
import asyncio

_DUMMY_INPUT = "Mary had a little lamb, its fleece was white as snow."
_DUMMY_PROMPT = "You are a poet. Make a haiku from the input."

def dummy_query_gpt4(text: str = _DUMMY_INPUT, max_tokens: int = 200):
    # MUST REMOVE
    openai.api_key = "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"
    response = openai.ChatCompletion.create(
        temperature=0,
        max_tokens = max_tokens,
        model = 'gpt-4',
        messages = [
            {"role": "system", "content": _DUMMY_PROMPT},
            {"role": "user", "content": text}
        ]
    )
    res = response["choices"][0]["message"]["content"]
    input_tokens = response["usage"]["prompt_tokens"]
    output_tokens = response["usage"]["completion_tokens"]
    return res, input_tokens, output_tokens

async def async_query_gpt(queue: asyncio.Queue):
    while True:
        text = await queue.get()
        print(f"Sending query to GPT-4...")
        res, input_tokens, output_tokens = dummy_query_gpt4(text)
        print(f"Received response from GPT-4.")
        queue.task_done()
        yield res, input_tokens, output_tokens