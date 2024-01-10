from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT


def search_from_claudia(text):
    anthropic = Anthropic(api_key='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')
    stream = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT}{text}{AI_PROMPT}",
        stream=True
    )
    #print(completion.response)
    # for complation in stream:
    #     print(complation.completion, end='', flush=True)
    return stream

#search_from_claudia()

