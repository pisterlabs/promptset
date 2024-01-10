from dotenv import load_dotenv
from langchain.utilities import GoogleSerperAPIWrapper
import asyncio

load_dotenv()

google = GoogleSerperAPIWrapper()


async def two_plus_one(a, b):
    return a + b


num_list = [(1, 2), (2, 3), (4, 5)]
tasks = []

for i in num_list:
    tasks.append(two_plus_one(i[0], i[1]))


async def main():
    results = await asyncio.gather(*tasks)
    print(results)


loop = asyncio.get_event_loop()
loop.run_until_complete(main())
