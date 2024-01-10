# Local imports
from experts import (AaveContextProvider,  # LangchainContextProvider,
                     AirstackContextProvider, GnosisContextProvider,
                     OneInchContextProvider, UniswapContextProvider)
from langchain import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
# 
from langchain.llms import OpenAI
from tools.code import PythonCodeWriter
from tools.etherscan import (EtherScanGetContractABI, EtherScanGetContractCode,
                             EtherScanGetTXStatus)
from tools.ethsend import EthSend, GetEthBalance


class AIHelper:
    def __init__(self, callback_handler=None):
        self.llm = OpenAI(temperature=0.0)
        self.search = SerpAPIWrapper()
        self.callback_handler = callback_handler

        self.tools = [
            EtherScanGetContractABI(),
            # Tools
            EtherScanGetContractCode(),
            EtherScanGetTXStatus(),
            PythonCodeWriter(),
            EthSend(),
            GetEthBalance(),
            # LangchainContextProvider(),
            # Experts
            AirstackContextProvider(),
            AaveContextProvider(),
            OneInchContextProvider(),
            GnosisContextProvider(),
            UniswapContextProvider(),
            # Web3PyContextProvider(),
        ]

        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            # callback_handler=callback_handler,
        )

    def run_query(self, query):
        self.agent.run(
            {
                "input": query,
                "chat_history": [],
            },
            callbacks=[self.callback_handler]
        )


if __name__ == "__main__":
    helper = AIHelper()
    query = "What is the balance of spink.eth?"
    helper.run_query(query)
