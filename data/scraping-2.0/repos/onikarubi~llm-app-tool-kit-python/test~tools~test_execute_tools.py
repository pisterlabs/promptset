from src.agent.function_calling.base_function_agent import BaseFunctionAgent
from langchain.chat_models import ChatOpenAI
from src.services.crypts.crypts import CryptocurrencyPriceTool
from src.services.yfinance.yfinance import YahooFinanceTool
import pytest

@pytest.mark.skip(reason="This test is too slow")
def test_execute_tools():
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-1106",
        streaming=True,
    )

    tools = [YahooFinanceTool(), CryptocurrencyPriceTool()]

    agent = BaseFunctionAgent(
        llm=llm,
        tools=[tool for tool in tools],
    )

    executor = agent.get_executor(debug=True)

    result = executor.run("現在のマイクロソフトの株価はNvidiaと比較してどうですか？")
    print(result)