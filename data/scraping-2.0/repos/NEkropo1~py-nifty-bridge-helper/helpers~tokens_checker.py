import openai


def check_tokens_size(query: str) -> None:
    max_tokens = 4096
    tokens_used = openai.Completion.create(
        model="gpt-3.5-turbo",
        prompt=query,
        max_tokens=max_tokens,
        temperature=0,
    ).usage["total_tokens"]

    if tokens_used > max_tokens:
        message = (
            f"Error: The number of tokens used ({tokens_used}) exceeds the maximum limit "
            f"of {max_tokens}. Please reduce the length of your text input."
        )
        raise ValueError(message)
