import asyncio
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from dotenv import load_dotenv
from langchain.cache import InMemoryCache
import os

set_llm_cache(InMemoryCache())
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# llm = OpenAI(temperature=0)
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.6, cache=False)


async def main():
    llm = OpenAI(temperature=0, max_tokens=5)
    with get_openai_callback() as cb:
        llm("What is the square root of 4?")

    total_tokens = cb.total_tokens
    print(cb)
    print("Total tokens: ", total_tokens)

if __name__ == "__main__":
    asyncio.run(main())
