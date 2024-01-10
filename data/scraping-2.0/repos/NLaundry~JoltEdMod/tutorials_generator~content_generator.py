import os
import openai
import asyncio
import time
import aiohttp
import aiofiles
import nbformat as nbf
import markdown2
from markdownify import markdownify as md
import json
import uuid
from alive_progress import alive_bar
from colorama import Fore, Style
from tutorials_generator.block import Block
from tutorials_generator.cell_type import CellType
from tutorials_generator.block_factory import BlockFactory


class ContentGenerator:
    def __init__(self, model="gpt-3.5-turbo", system_block=None, max_tokens=1024, n=1, stop=None, temperature=0.7, blocks=None):
        self.model = model
        self.system_block = system_block
        self.max_tokens = max_tokens
        self.n = n
        self.stop = stop
        self.temperature = temperature
        self._set_api_key()
        self.blocks = blocks

    def _set_api_key(self):
        """Set OpenAI API key from the environment variable."""
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("Environment variable OPENAI_API_KEY not set")

    def create_notebook(self, config_file, output_file):
        """Create a Jupyter Notebook file from the given config file."""
        asyncio.run(self._create_notebook_async(config_file, output_file))

    async def _create_notebook_async(self, config_file, output_file):
        nb = nbf.v4.new_notebook()
        blocks = await self._create_content_async(config_file)
        self._generate_notebook_blocks(blocks, nb)
        nbf.write(nb, output_file)
        return nb

    def _generate_notebook_blocks(self, blocks, nb):
        """Generate notebook cells from given blocks."""
        for block in blocks:
            if block.cell_type == CellType.CODE:
                new_cell = nbf.v4.new_code_cell(block.content)
                new_cell['id'] = str(uuid.uuid4())  # Generate and set cell id
                nb.cells.append(new_cell)
            elif block.cell_type == CellType.MARKDOWN:
                nb.cells.append(nbf.v4.new_markdown_cell(block.content))
        return nb

    def create_wiki(self, config_file, output_file):
        """Create a wiki file from the given config file."""
        asyncio.run(self._create_wiki_async(config_file, output_file))

    async def _create_wiki_async(self, config_file, output_file):
        blocks = await self._create_content_async(config_file)
        await self._create_markdown_file_async(blocks, output_file)

    async def _create_markdown_file_async(self, blocks, file_path):
        """Create a markdown file from the given blocks."""
        markdown_text = ""
        async with aiofiles.open(file_path, "w") as f:
            for block in blocks:
                markdown_text += f"{markdown2.markdown(block.content)}"
            await f.write(md(markdown_text))

    def create_content(self, config_file):
        """Create content for blocks from the given config file."""
        return asyncio.run(self._create_content_async(config_file))

    async def _create_content_async(self, config_file):
        async with aiohttp.ClientSession() as self._session:
            blocks = self._parse_config_file(config_file)
            self._update_context(config_file, blocks)

            if blocks[0].type == 'SeedBlock':
                self.system_block = blocks[0]
                blocks.pop(0)
            else:
                self.system_block = None

            await self._generate_all_block_content_async(blocks)
            return blocks

    @staticmethod
    def _update_context(config_file, blocks):
        """Update block context using the given config file."""
        with open(config_file) as f:
            config = json.load(f)
            for block, block_config in zip(blocks, config['blocks']):
                if 'context' in block_config and block_config['context'] is not None:
                    block.set_context(blocks[block_config['context']])

    @staticmethod
    def _parse_config_file(config_file):
        """Parse the given config file and create blocks."""
        with open(config_file) as f:
            config = json.load(f)
            blocks = []
            for block_config in config['blocks']:
                block = BlockFactory.create_block(block_config)
                blocks.append(block)
            return blocks

    async def _generate_all_block_content_async(self, blocks):
        """Generate content for all blocks asynchronously."""
        dependent_blocks, independent_blocks = self._split_blocks(blocks)

        title_color = f"{Fore.CYAN}{Style.BRIGHT}Generating block content{Style.RESET_ALL}"
        spinner_style = 'waves'

        with alive_bar(len(blocks), title=title_color, spinner=spinner_style, bar="smooth") as pbar:
            for block in dependent_blocks:
                self.generate_block_content(block)
                pbar()
            await self._generate_all_block_content_async_helper(independent_blocks)
            pbar(len(independent_blocks))

    @staticmethod
    def _split_blocks(blocks):
        """Split blocks into dependent and independent blocks."""
        dependent_blocks = [
            block for block in blocks if block.context is not None]
        independent_blocks = [
            block for block in blocks if block.context is None]

        return dependent_blocks, independent_blocks

    async def _generate_all_block_content_async_helper(self, independent_blocks):
        """Asynchronously generate content for independent blocks."""
        tasks = [
            asyncio.create_task(self.generate_block_content_async(block))
            for block in independent_blocks
        ]
        await asyncio.gather(*tasks)

    async def generate_block_content_async(self, block):
        while True:
            try:
                response = await self.chat_completion_create_async(
                    session=self._session,
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_block.generate_prompt(),
                        },
                        {"role": "user", "content": block.generate_prompt()},
                    ],
                    max_tokens=self.max_tokens,
                    n=self.n,
                    stop=self.stop,
                    temperature=self.temperature,
                )
                block.set_content(response["choices"][0]["message"]["content"])
                break  # If the response was successful, break out of the loop

            except Exception as e:
                if "429" in str(e):  # Check if the error code is 429
                    wait_time = 60  # You can set this to the desired waiting time in seconds
                    print(
                        f"Error 429 encountered. Retrying in {wait_time} seconds...")
                    # Use asyncio.sleep instead of time.sleep for async code
                    await asyncio.sleep(wait_time)
                else:
                    raise  # If it's another exception, raise it as usual

    def generate_block_content(self, block):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.system_block.generate_prompt(),
                        },
                        {"role": "user", "content": block.generate_prompt()},
                    ],
                    max_tokens=self.max_tokens,
                    n=self.n,
                    stop=self.stop,
                    temperature=self.temperature,
                )
                block.set_content(response["choices"][0]["message"]["content"])
                break  # If successful, break out of the loop

            except Exception as e:
                if "429" in str(e):  # Check if the error code is 429
                    wait_time = 60  # You can set this to the desired waiting time in seconds
                    print(
                        f"Error 429 encountered. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise  # If it's another exception, raise it as usual

    @staticmethod
    async def chat_completion_create_async(session, model, messages, max_tokens, n, stop, temperature):
        url = f"https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
        }

        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "n": n,
            "stop": stop,
            "temperature": temperature,
        }

        while True:
            try:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status != 200:
                        raise Exception(f"Error: {response.status}, {await response.text()}")
                    return await response.json()

            except Exception as e:
                if "429" in str(e):  # Check if the error code is 429
                    wait_time = 60  # You can set this to the desired waiting time in seconds
                    print(
                        f"Error 429 encountered. Retrying in {wait_time} seconds...")
                    # Use asyncio.sleep instead of time.sleep for async code
                    await asyncio.sleep(wait_time)
                else:
                    raise  # If it's another exception, raise it as usual
