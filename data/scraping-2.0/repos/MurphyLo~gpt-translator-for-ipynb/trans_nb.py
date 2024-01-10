import asyncio
import nbformat
import time
from openai import AsyncOpenAI
from openlimit import ChatRateLimiter
import tiktoken

rate_limiter = ChatRateLimiter(request_limit=100, token_limit=4e5)
encoding = tiktoken.get_encoding("cl100k_base")
PROMPTS = {
    "markdown": "Translate this English markdown formated text wrapped by triple `\"` to Chinese and output only the translated content (so you shouldn't output triple `\"`), don't include any explainations:\n\"\"\"\n{}\n\"\"\"\nThe text is from a markdown cell of a kaggle notebook, please provide a natural and understandable translation in terms of the context.",
    "code": "Translate only the comments lines begin with a `#` in the following Python code wrapped by triple `\"` to Chinese and output the entire content (but you shouldn't output triple `\"`), don't include any explaination, don't translate any code:\n\"\"\"\n{}\n\"\"\"\nThe code is from a code cell of a kaggle notebook regarding Titanic survival predictions, please provide a natural and understandable translation in terms of the context.",
    # "raw": "RAW"
}


async def translate_cell(api_key, content, prompt):
    # if prompt == "RAW":
    #     return content
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    token_num = 0
    token_num += len(encoding.encode(prompt.format(content)))
    tout = token_num * 3 / 10  # assume an avg token rate of 10tks/sec and relative 3tks/char(CN/EN)
    print('Prompt tokens:', token_num)
    m = [{
            "role": "user",
            # "content": f"Translate this English markdown formated text wrapped by triple backticks to Chinese and output only the translated content, don't include any explainations:\n```\n{text}\n```\nThe text is from a markdown cell of a kaggle notebook, please provide a natural and understandable translation in terms of the context."
            "content": prompt.format(content)
    }]
    max_retries = 5
    for try_time in range(max_retries):
        try:
            async with rate_limiter.limit(model='gpt-4-0314', messages=m, timeout=tout):
                response = await client.chat.completions.create(
                    model="gpt-4-0314",
                    messages=m,
                    temperature=0.7,
                    timeout=tout
                )
            token_num += len(encoding.encode(response.choices[0].message.content))
            print('Context tokens:', token_num, 'RESP:', response.choices[0].message.content[:25].replace('\n', ' '), end='\t\n')
            return response.choices[0].message.content.strip().strip('```').strip().strip('"""').strip()
        except Exception as e:
            print(f'RETRY {try_time}:', e.args)
    print('FATAL')
    return '______NONE______'


async def process_notebook(api_key, file_path):
    notebook = nbformat.read(file_path, as_version=4)
    tasks = []

    for cell in notebook.cells:
        task = translate_cell(api_key, cell.source, PROMPTS[cell.cell_type])
        tasks.append(task)

    translated_texts = await asyncio.gather(*tasks)

    for cell, translated_text in zip(notebook.cells, translated_texts):
        if cell.cell_type in ['markdown', 'code']:
            cell.source = translated_text

    new_file_path = file_path.replace('.ipynb', '_translated.ipynb')
    nbformat.write(notebook, new_file_path)

async def main():
    api_key = "sk-xxxxxxxxxxxxxxxxxxxxx"
    file_path = "/path/to/example.ipynb"
    await process_notebook(api_key, file_path)

asyncio.run(main())

