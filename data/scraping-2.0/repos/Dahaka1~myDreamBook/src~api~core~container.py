from openai import AsyncOpenAI

from infra.requesters import GPTRequester
from config import get_settings

settings = get_settings()

__gpt_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
gpt_requester = GPTRequester(__gpt_client)
