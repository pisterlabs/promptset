import datetime
import pytz

from langchain.agents import Tool
from langchain.utilities import PythonREPL

from ...config import Config
from ...db import query_memory, add_memory, delete_memory


def get_memory_tools(tokenizer):
    def memorize_run(text):
        add_memory(text)
        return 'Memorized.'

    async def memorize_arun(text):
        return memorize_run(text)

    memorize_tool = Tool(
        name="memorize",
        description="If the user tell you to remember something, you MUST use this tool in order to remember it. Do not use this tool unless the user explicitly tells you to remember something. The content that you want to remember should be the input. The input MUST contain both the value and a description of the value.",
        func=memorize_run,
        coroutine=memorize_arun,
    )

    def check_memory_run(text):
        if not text:
            return 'Error: You must provide a input while using this tool.'
        if not isinstance(text, str):
            return 'Error: The input of this tool must be a string.'
        text = get_memories_text(text, tokenizer=tokenizer)
        if text:
            text += '\nNote that newer memories should override older ones if they have conflicts. It is possible that the above memories does not provide sufficient information about the topic you specified, in such case, you need to change your input or use other tools to find information.'
        else:
            text = 'No related memories found.'
        return text

        return text

    async def check_memory_arun(text):
        return check_memory_run(text)

    check_memory_tool = Tool(
        name="check_memory",
        description="Use this tool to check if you have memories related to the specified topic. This should be the first priority to find information. You do not need to check your memory against the user's current message, as it's already done automatically.",
        func=check_memory_run,
        coroutine=check_memory_arun,
    )

    return [memorize_tool, check_memory_tool]


def get_memories_text(query, tokenizer):
    memories = query_memory(query, n_results=Config.agent.memory_top_n)
    memories = [m for m in memories if m['distance'] <= Config.agent.memory_max_distance]
    memories_text = []
    tz = pytz.timezone(Config.timezone)
    for m in memories:
        text = m['document']
        text = text.strip()
        text = text.replace('\n', ' ')
        tokenized_text = tokenizer.encode(text)
        if len(tokenized_text) > Config.agent.memory_max_token_limit:
            text = tokenizer.decode(tokenized_text[:Config.agent.memory_max_token_limit]) + \
                ' ... (truncated)'
        created_at = m['metadata'].get('created_at', None)
        if created_at:
            dt = datetime.datetime.fromtimestamp(created_at)
            dt = dt.astimezone(tz)
            formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
            text = f'[Memorized at {formatted_dt}]: {text}'
        else:
            text = f'[Memorized at unknown date]: {text}'

        memories_text.append(text)

    memories_text = '\n'.join(memories_text)

    return memories_text
