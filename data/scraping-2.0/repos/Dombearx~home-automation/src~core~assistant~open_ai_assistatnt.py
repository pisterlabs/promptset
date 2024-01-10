from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.chat_models import ChatOpenAI
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

from src.core.assistant.assistant_template import ChatBotTemplate
from src.core.tools.tools import TOOLS


class OpenAIChatBot(ChatBotTemplate):
    def __init__(self, model_name: str, temperature: float = 0.7):
        main_llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        super().__init__(
            main_llm,
            tools=TOOLS,
            format_function=format_to_openai_functions,
            tool_format_function=format_tool_to_openai_function,
            output_parser=OpenAIFunctionsAgentOutputParser,
        )
