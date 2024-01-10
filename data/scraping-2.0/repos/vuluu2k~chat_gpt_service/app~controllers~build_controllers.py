from fastapi.encoders import jsonable_encoder
# import tiktoken
from app.controllers.key_controllers import get_openai_key_by_user_id
# import openai

# openai.api_key = "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"
# openai.api_base = "http://localhost:1337"


import g4f
import asyncio

_providers = [
    g4f.Provider.Aichat,
    g4f.Provider.ChatBase,
    g4f.Provider.Bing,
    g4f.Provider.GptGo,
    g4f.Provider.You,
    g4f.Provider.Yqcloud,
    g4f.Provider.Liaobots,
]

_providers_sync = [
    g4f.Provider.Raycast,
    g4f.Provider.GeekGpt,
    g4f.Provider.Theb,
    g4f.Provider.Vercel,
]


async def run_provider(provider: g4f.Provider.BaseProvider, message: str = "Hello"):
    try:
        response = await g4f.ChatCompletion.create_async(
            model=g4f.models.default,
            messages=[{"role": "user", "content": message}],
            provider=provider,
        )

        return {f"{provider.__name__}": response}
    except Exception as e:
        return {f"{provider.__name__}": str(e)}


async def run_provider_sync(provider: g4f.Provider.BaseProvider, message: str = "Hello"):
    pass


async def run_all(message: str = "Hello"):
    calls = [
        run_provider(provider, message) for provider in _providers
    ]
    return await asyncio.gather(*calls)


async def build_learning_path(request, body):
    responses = await run_all(body.message)

    return {"responses": responses}


async def conversation(request, body):
    provider = g4f.Provider.BaseProvider

    match body.provider:
        case "bing":
            provider = g4f.Provider.Bing
        case "chatbase":
            provider = g4f.Provider.ChatBase
        case "theb":
            provider = g4f.Provider.Theb
        case "gpt":
            provider = g4f.Provider.GptGo

    responses = await g4f.ChatCompletion.create(
        model=g4f.models.default,
        messages=[{"role": "user", "content": body.message}],
        provider=provider
    )
    print(responses)

    return {"responses": responses}
