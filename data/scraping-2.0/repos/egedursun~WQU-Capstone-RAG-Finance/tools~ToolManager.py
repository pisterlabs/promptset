import dotenv
from langchain.agents import StructuredChatAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool

from config import TOOL_DESC_FUNDAMENTAL_DATA, TOOL_DESC_TECHNICAL_DATA, TOOL_DESC_TECHNICAL_INDICATORS, \
    TOOL_DESC_TICKER_NEWS, TOOL_DESC_DIVIDENDS
from functions.container.code_interpreter import run_code
from tools.container.dividends import get_dividends
from tools.container.fundamental_data import get_fundamental_data
from tools.container.technical_data import get_technical_data
from tools.container.technical_indicators import get_technical_indicators
from tools.container.ticker_news import get_ticker_news


dotenv.load_dotenv()
config = dotenv.dotenv_values()


class ToolManager:

    def __init__(self):
        self.tools = [

            ####################################################################################################
            # API TOOLS
            ####################################################################################################
            Tool("get_fundamental_data", get_fundamental_data, TOOL_DESC_FUNDAMENTAL_DATA),
            Tool("get_technical_data", get_technical_data, TOOL_DESC_TECHNICAL_DATA),
            Tool("get_technical_indicators", get_technical_indicators, TOOL_DESC_TECHNICAL_INDICATORS),
            Tool("get_ticker_news", get_ticker_news, TOOL_DESC_TICKER_NEWS),
            Tool("get_dividends", get_dividends, TOOL_DESC_DIVIDENDS),
            ####################################################################################################
            ####################################################################################################
            ####################################################################################################
            # CODE INTERPRETING TOOLS
            ####################################################################################################
            Tool("run_code", run_code, TOOL_DESC_FUNDAMENTAL_DATA),
            ####################################################################################################

        ]
        self.llm = ChatOpenAI(openai_api_key=config["OPENAI_API_KEY"],
                              streaming=True,
                              model_name="gpt-4-1106-preview",
                              temperature="0.5")
        # self.vectorstore = vectorstore
        self.structured_agent = StructuredChatAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools)
        self.chat_agent = AgentExecutor.from_agent_and_tools(
            agent=self.structured_agent,
            tools=self.tools
        )
