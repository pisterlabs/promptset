import os
import requests
from tools.util_tools import SyncTool
from langchain.chat_models import ChatOpenAI
from web3_config import w3


class ExecuteReadTool(SyncTool):
    name = "EXECUTE_CONTRACT_READ"
    description = (
        "This tool executes a read on the blockchain. It should be used once we know the contract address, and "
        "we after we have used the etherscan_abi_query tool to get its abi. Also, this should be called only on "
        "read-only functions (like contract balance, ownership and etc).\n"
        "We will call this tool as: <contract>;<function signature>;<arguments>. If we have multiple <arguments> "
        "they are comma separated.\n"
        "Example #1: 0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0;name() -> string; \nOutput: Matic Token\n"
        "Example #2: 0x6B175474E89094C44Da98b954EedeAC495271d0F;balanceOf(address owner) -> uint256;"
        "0xcC227A599c10A39265Fda98beC977aee99adA5d1\n Output: 2343242398032\n"
        "the address we are interested in is 0xcC227A599c10A39265Fda98beC977aee99adA5d1. We use this tool to find the "
        "abi of 0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0 in a natural language format.\n"
        "Example #3: We call this tool on a SINGLE contract address. Otherwise we fail."
    )

    def _run(self, query: str) -> str:
        try:
            non_fixed_args = query.split(';')
            if len(non_fixed_args) != 3:
                chat = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
                query = chat.call_as_llm(
                    f"Fit the {query} into the format <contract address>;<function signature>;<arguments>. "
                    "If we have multiple <arguments> they are comma separated."
                )
                non_fixed_args = query.split(';')

            if len(non_fixed_args) != 3:
                return (
                    f"Error in parsing {query}: we expected a <contract>;<function signature>;<arguments> "
                    f"format for the query. We got {len(non_fixed_args)} blocks instead of 3."
                )

            contract, signature_str, arguments = non_fixed_args
        except Exception as e:
            return f"Query failed with exception: {e}!"

        try:
            etherscan_key = os.environ["ETHERSCAN_API_KEY"]
            response = requests.get(
                f"https://api.etherscan.io/api?module=contract&action=getabi&address={contract}"
                f"&apikey={etherscan_key}"
            )

            abi = eval(response.text)['result']
            contract_web3_obj = w3.eth.contract(address=contract, abi=abi)
        except Exception as e:
            return f"Query failed with exception: {e}!"

        try:
            func_name = signature_str.split('(')[0]
            contract_func = contract_web3_obj.functions[func_name]
        except Exception as e:
            return (
                f"Couldn't get the function {signature_str}: {e}! "
            )

        try:
            arguments_list = arguments.split(',')
            result = contract_func(*arguments_list).call()
            return (
                f"We called {func_name} from {contract} with {arguments_list} and received: {result}.\n"
                f"We reached this point, so this means that we successfully executed a query on the chain and the "
                f"answer {result} here is valid."
            )
        except Exception as e:
            return (
                f"Couldn't execute the function: {e}! "
            )