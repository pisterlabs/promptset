import aiofiles
import openai_async
from data.prompts import prompts_list
from colorama import Fore, Back, Style


async def gen(user_contexts, prompt_mode):

    try:
        prompt = prompts_list[prompt_mode]
    except:
        prompt = prompts_list["sys_prompt"]

    print(f'PROMPT TEXT IS: {prompt}')
        
    async with aiofiles.open('tokens.txt', "r") as file:
        lines = await file.readlines()
    for line_number, line in enumerate(lines, start=1):
        gpt_token = line.strip()
        response = await openai_async.chat_complete(
            gpt_token,
            timeout=160,
            payload={
                # https://platform.openai.com/docs/models/overview
                'model': "gpt-3.5-turbo",
                # 'model': "gpt-4",
                "messages": [{"role": "system", "content": f'{prompt}'}] + user_contexts
            }
        )
        print(response.json())
        if 'error' not in response.json():
            print(f"    Token number: {line_number}, Content: {gpt_token}")
            return response.json()['choices'][0]['message']['content']
