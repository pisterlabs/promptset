import asyncio
import bopenai as openai
import openai_secret

openai.organization = openai_secret.openai_org
openai.api_key = openai_secret.openai_secret

if __name__ == "__main__":
    async def test():
        async def task():
            s = await openai.Completion.create(prompt="Write Hello World: ", max_tokens=10, model="text-ada-001", temperature=1.2)
            t = ""
            async for i in s: t += i.text
            return t
        
        openai.AsyncConfiguration.set_maximum_collection_period(0.1)
        openai.AsyncConfiguration.set_batch_size(2)

        tasks = await asyncio.gather(*(task() for i in range(4)))
        for t in tasks:
            print("t", [t])

    asyncio.run(test())