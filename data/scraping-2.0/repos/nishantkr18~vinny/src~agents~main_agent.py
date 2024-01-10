from src.bot_state import BotState, AgentMemory
from src.tools.product_search_tool import ProductSearchTool, get_filtered_products
import logging
import json
import textwrap
from typing import Any
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, FunctionMessage, AIMessage
from langchain.callbacks import get_openai_callback
from langchain.tools import format_tool_to_openai_function

logging.basicConfig(level=logging.INFO,
                    format='::::::%(levelname)s:::::: %(message)s')

MAX_LOOP_LIMIT = 10

class MainAgent():
    name = 'chief_complaint_agent'
    def __init__(self, state: BotState):
        self.state = state
        if self.name not in self.state.conv_hist:
            self.state.conv_hist[self.name] = AgentMemory()
        self.conv_hist = self.state.conv_hist[self.name]

        if len(self.conv_hist) == 0:
            system_prompt = textwrap.dedent(
                f"""
            You are Vinny, the helpful AI salesbot for Winecentral.
            Your task is to answer questions about wine products and engage with the user in a friendly non condescending manner.
            You are an expert at wine and alcohol products.
            You can help both novices and experts alike, discover new wines.
            Where necessary you will ask questions one at a time to understand the customers taste preference before recommending wines. 
            Do not hallucinate about products.
            Use the product search tool to look for products in stock.
            Only recommend products that are in stock.
            Keep your answers short and concise. Answer in points.
            Try to reply in 50 words or less.
            """)

            self.conv_hist.append(SystemMessage(content=system_prompt))

        self.llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo-0613')
        self.tools = [
            ProductSearchTool()
        ]
        self.tool_dict = {tool.name: tool for tool in self.tools}

    def ask(self) -> str:
        
        # human input appended in history
        self.conv_hist.append(HumanMessage(content=self.state.last_human_input))
        for _ in range(MAX_LOOP_LIMIT):
            logging.info('Waiting for the main agent...')
            response = self.llm(self.conv_hist, functions=[
                                format_tool_to_openai_function(tool) for tool in self.tools])
            self.conv_hist.append(response)

            self.state.save_to_file()

            # Check if a function call is present in the reposnse
            function_call = response.additional_kwargs.get("function_call")
            if function_call is not None:
                # Execute the tool and get the response. Continue the loop!

                # Find the tool
                tool_to_run = self.tool_dict.get(function_call.get('name'))
                logging.info(f'Running tool: {tool_to_run.name}')

                tool_result = tool_to_run.run(
                    json.loads(function_call.get('arguments'))
                )
                logging.info(f'Tool results: {tool_result}')

                # Converting list of products to json string for the conversation memory. Also appending to the state.
                if tool_to_run.name == 'product_search_tool':
                    self.state.products_list = tool_result
                    tool_result = get_filtered_products(tool_result)

                # Add the repsonse to the conversation memory
                self.conv_hist.append(FunctionMessage(
                    name=function_call.get('name'), content=tool_result))
                pass
            else:
                # return self._language_check(input, response.content)
                return response.content

        raise Exception('Max loop limit reached')