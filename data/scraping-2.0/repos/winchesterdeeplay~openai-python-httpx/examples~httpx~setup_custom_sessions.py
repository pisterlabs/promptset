import openai
from httpx import Limits

if __name__ == "__main__":
    openai.api_key = ...

    openai.setup_custom_sync_session(
        limits=Limits(max_keepalive_connections=1, max_connections=1)
    )

    # Now you can use openai with custom params for session

    completion = openai.Completion.create(
        engine="davinci",
        prompt="Translate the following English text to French: 'Hello, how are you?'",
        max_tokens=50
    )

    print(completion)
