"""Module Containing Agents used in AI Swarm"""
import logging
from typing import Literal, Optional, List
import os
import openai

from langchain.tools import BaseTool
from langchain.tools.render import format_tool_to_openai_function
from langchain.chat_models import ChatOpenAI
from langchain.agents import OpenAIMultiFunctionsAgent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

from SparksAI import config
from SparksAI import databases
from SparksAI.memory import AIMemory
from SparksAI.swarm import Swarm

logger = logging.getLogger(__name__)

openai_client = openai.Client()

memory = AIMemory(databases.FireBaseStrategy(os.getenv("FIREBASE_TABLE_ID")))

swarm = Swarm()


async def image_agent(prompt: str, style: Optional[Literal["vivid", "natural"]]) -> str:
    """Generate Image Agent

    Args:
        prompt (str): Prompt used to generate image
        style (str): The style of the generated images. Must be one of vivid or natural. Defaults to vivid.
                        Vivid causes the model to lean towards generating hyper-real and dramatic images
                        Natural causes the model to produce more natural, less hyper-real looking images.

    Returns:
        str: url to image generated
    """
    if len(prompt) > config.DALL_E_MAX_PROMPT_SIZE:
        return f"ValueError: Prompt size too large. Please try again with a prompt size less than {config.DALL_E_MAX_PROMPT_SIZE} characters."

    if not style:
        style = "vivid"

    if style not in ["vivid", "natural"]:
        return f"ValueError: Invalid value '{style}' for style. Please use either 'vivid' or 'natural' instead."

    logger.info("Generating %s image with prompt: %s", style, prompt)

    try:
        api_response = openai_client.images.generate(
            model=config.DALL_E_MODEL_NAME,
            prompt=prompt,
            style=style,
            size=config.DALL_E_SIZE,
            quality=config.DALL_E_QUALITY,
        )

        response = api_response.data[0].url
    except openai.OpenAIError as e:
        response = f"There was an error with image generation. Error Details:\n{e}"

    logger.info("Generated image: %s", response)

    return response


async def research_agent(prompt: str, username: str) -> dict:
    """Research Agent, will provide detailed info regarding a topic

    Args:
        prompt (str): Topics to research
        username (str): Username of questionor

    Returns:
        dict: returns two outputs. The first is analysis of previous interactions. The second is detailed review from an analyst.
    """
    convo_memory = memory.get_convo_mem(username=username).messages
    logger.info("Getting message summary")
    message_summary = await swarm.get_archivist(username).ainvoke(
        {"input_message": prompt, "memory": convo_memory}
    )

    logger.info("Getting Analyst Comments")
    analyst_review = await swarm.get_analyst_agent().ainvoke(
        {"content": f"Context: {message_summary}\n\nUser message: {prompt}"}
    )

    return {
        "prior_messages_analysis": message_summary.content,
        "analyst_review": analyst_review["output"],
    }
