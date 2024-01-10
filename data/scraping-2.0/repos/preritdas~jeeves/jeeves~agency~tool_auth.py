"""Load tools depending on authorization."""
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.callbacks.base import BaseCallbackHandler

from keys import KEYS
from jeeves.permissions import User

from jeeves.agency import retrieval
from jeeves.agency import news
from jeeves.agency import send_texts
from jeeves.agency import make_calls
from jeeves.agency.user_memory import create_user_memory_tools
from jeeves.agency.serper_wrapper import GoogleSerperAPIWrapperURL


ANSWERER_JSON_STRING_INPUT_INSTRUCTIONS = (
    'Input must be a JSON string with the keys "source" and "query".'
)

NO_AUTH_TOOLS: list[BaseTool] = [
    Tool(
        name="Google Search",
        func=GoogleSerperAPIWrapperURL(serper_api_key=KEYS.GoogleSerper.api_key).run,
        description=(
            "Useful for when you need to search Google. Provides links to search results "
            "that you can use Website Answerer to answer for more information."
        ),
    ),
    Tool(
        name="Website Answerer",
        func=retrieval.WebsiteAnswerer.answer_json_string,
        description=(
            "Useful for when you need to answer a question about the content on a website. "
            "You can use this to answer questions about links found in Google Search results. "
            f'{ANSWERER_JSON_STRING_INPUT_INSTRUCTIONS} "source" is the URL of the website. '
            "Do not make up websites to search - you can use Google Search to find relevant urls."
        ),
    ),
    Tool(
        name="YouTube Answerer",
        func=retrieval.YouTubeAnswerer.answer_json_string,
        description=(
            "Useful for when you need to answer a question about the contents of a YouTube video. "
            f'{ANSWERER_JSON_STRING_INPUT_INSTRUCTIONS} "source" is the URL of the YouTube video. '
            "Do not make up YouTube videos - you can use Google Search to find relevant videos, or "
            "accept them directly from the user."
        ),
    ),
    Tool(
        name="Headline News",
        func=news.manual_headline_news,
        description=(
            "Useful for when you need to get the top headlines from a specific category. "
            "Input must be a string with the category name. Category must be one of "
            f"{news.MANUAL_AVAILABLE_CATEGORIES}."
        ),
    ),
    Tool(
        name="Wolfram Alpha",
        func=WolframAlphaAPIWrapper(wolfram_alpha_appid=KEYS.WolframAlpha.app_id).run,
        description=(
            "Useful for when you need to do math or anything quantitative/computational. "
            'Input should ideally be math expressions, ex. "8^3", but can also be '
            "natural language if a math expression is not possible."
        ),
    ),
    make_calls.CallTool()
]


def build_tools(
    user: User, callback_handlers: list[BaseCallbackHandler]
) -> list[BaseTool]:
    """Build all authenticated tools given a phone number."""
    added_tools: list[BaseTool] = []

    # Zapier
    if user.zapier_access_token:
        zapier_wrapper = ZapierNLAWrapper(
            zapier_nla_oauth_access_token=user.zapier_access_token
        )
        zapier_toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier_wrapper)
        added_tools.extend(zapier_toolkit.get_tools())

    # Text messages
    TextToolClass = send_texts.create_text_message_tool(user.phone)
    added_tools.append(TextToolClass())

    # User longterm memory
    added_tools.extend(create_user_memory_tools(user.phone))

    # Add all tools together
    tools = NO_AUTH_TOOLS + added_tools

    # Check for proper tool types
    if not all(isinstance(tool, BaseTool) for tool in tools):
        raise TypeError("All tools must be of type BaseTool (or subclass thereof).")

    # Add callback manager to all tools
    for tool in tools:
        tool.callbacks = callback_handlers

    return tools
