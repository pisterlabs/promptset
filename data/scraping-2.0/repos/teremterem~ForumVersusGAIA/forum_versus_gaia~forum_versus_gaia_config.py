# pylint: disable=wrong-import-position
"""
Various settings and utility functions for the forum_versus_gaia project.
"""
from functools import partial

from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum
from dotenv import load_dotenv

load_dotenv()

import promptlayer

async_openai_client = promptlayer.openai.AsyncOpenAI()

# FAST_GPT = "gpt-3.5-turbo-1106"
# FAST_GPT = "gpt-4-1106-preview"
SLOW_GPT = "gpt-4-1106-preview"
# SLOW_GPT = "gpt-3.5-turbo-1106"

REMOVE_GAIA_LINKS = True

forum = Forum()

zero_temperature_completion = partial(
    openai_chat_completion,
    async_openai_client=async_openai_client,
    temperature=0,
)

# fast_gpt_completion = partial(
#     zero_temperature_completion,
#     model=FAST_GPT,
# )

slow_gpt_completion = partial(
    zero_temperature_completion,
    model=SLOW_GPT,
)
