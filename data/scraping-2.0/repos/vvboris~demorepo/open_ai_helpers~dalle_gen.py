import openai_async
import aiofiles


async def gen(prompt):
    async with aiofiles.open('tokens.txt', "r") as file:
        lines = await file.readlines()
    for line_number, line in enumerate(lines, start=1):
        gpt_token = line
        response = await openai_async.generate_img(
            gpt_token.strip(),
            timeout=60,
            payload={
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024"
            },
        )
        try:
            if 'safety system' in response.json()['error']['message']:
                return response.json()['error']['message']
        except Exception as e:
            print(e)
        if 'error' not in response.json():
            print(f"    Token number: {line_number}, Content: {gpt_token.strip()}")
            return response.json()["data"][0]["url"]
