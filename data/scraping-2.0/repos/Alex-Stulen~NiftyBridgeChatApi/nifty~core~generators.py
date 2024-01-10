import openai

from langchain.vectorstores.redis import Redis

from conf import settings

from .search import similarity_search


def generate_response(message: str, redis: Redis) -> str:
    """ Returns generated answer message from AI model """
    if not message:
        return "How can I help you?"

    text_chunks = similarity_search(message, redis)

    if not text_chunks:
        return (
            "I'm sorry, I don't have an answer to that. "
            "Please contact support at support@nifty-bridge.com."
        )

    chunks_set = set(text_chunks)

    response_message = (
            f"Answer on this message: `{message}`\n\n"
            "using only this info:" + "`" + "\n\n".join(chunks_set) + "`"
    )

    if len(response_message) > settings.OPENAI_MAX_TOKENS:
        return f"Your message with similar pieces is too long: {len(response_message)} symbols."

    response = openai.ChatCompletion.create(
        model=settings.OPENAI_USE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant which "
                           "talks only about info related in user content."
            },
            {"role": "user", "content": response_message}
        ],
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    if not response.choices:
        return 'Failed to generate an answer.'

    return response.choices[0].message.content.strip()
