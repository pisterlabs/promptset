from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,  # noqa: F401
    ChatCompletionAssistantMessageParam,  # noqa: F401
    ChatCompletionUserMessageParam,  # noqa: F401
)
from app.settings import Settings

settings = Settings()


class LLMModel:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def completion_stream(self, history: list[ChatCompletionMessageParam]):
        stream = await self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[messsage for messsage in history],
            temperature=0.7,
            max_tokens=512,
            n=1,
            stream=True,
        )

        # Stream responses
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
