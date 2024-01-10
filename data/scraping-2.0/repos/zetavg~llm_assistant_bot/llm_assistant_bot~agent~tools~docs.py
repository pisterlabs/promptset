import datetime
import pytz

from langchain.agents import Tool

from ...config import Config
from ...db import query_docs


def get_docs_tools(tokenizer):
    def find_docs_run(text):
        if not text:
            return 'Error: You must provide a input while using this tool.'
        if not isinstance(text, str):
            return 'Error: The input of this tool must be a string.'
        text = get_docs_text(text, tokenizer=tokenizer)
        if text:
            text += '\nNote that newer docs should override older ones if they have conflicts. It is possible that the above docs does not provide sufficient information about the topic you specified, in such case, you need to change your input or use other tools to find information.'
        else:
            text = 'No related documents found.'
        return text

    async def find_docs_arun(text):
        return find_docs_run(text)

    find_docs_tool = Tool(
        name="find_docs",
        description="Use this tool find related documents about a specified topic.",
        func=find_docs_run,
        coroutine=find_docs_arun,
    )

    return [find_docs_tool]


def get_docs_text(query, tokenizer):
    docs = query_docs(query, n_results=Config.agent.docs_top_n)
    docs = [m for m in docs if m['distance'] <= Config.agent.docs_max_distance]
    docs_text = []
    tz = pytz.timezone(Config.timezone)
    for m in docs:
        text = m['document']
        text = text.strip()
        text = text.replace('\n', ' ')
        tokenized_text = tokenizer.encode(text)
        if len(tokenized_text) > Config.agent.docs_max_token_limit:
            text = tokenizer.decode(tokenized_text[:Config.agent.docs_max_token_limit]) + \
                ' ... (truncated)'
        created_at = m['metadata'].get('saved_at', None)
        if created_at:
            dt = datetime.datetime.fromtimestamp(created_at)
            dt = dt.astimezone(tz)
            formatted_dt = dt.strftime('%Y-%m-%d %H:%M:%S')
            text = f'[Saved at {formatted_dt}]: {text}'
        else:
            text = f'[Saved at at unknown date]: {text}'

        docs_text.append(text)

    docs_text = '\n'.join(docs_text)

    return docs_text
