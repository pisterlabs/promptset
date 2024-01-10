import openai
import asyncio
import time

async def get_values_async(text, keywords, limiter):
    # await limiter.acquire()
    async with limiter:
        prompt = f"""Extract {', '.join(keywords)} from following in ["{'", "'.join(keywords)}"] (python list) format. Fill with X if not exists.\nInput: {text}\nOutput:"""
        response = await openai.Completion.acreate(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        try:
            return eval(response.choices[0].text)
        except:
            return ['X' for _ in range(len(keywords))]



