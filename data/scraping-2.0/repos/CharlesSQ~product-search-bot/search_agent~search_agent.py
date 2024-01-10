from typing import List, Optional

from langchain.agents import Tool
from langchain.tools import StructuredTool
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, Tool
from langchain.schema.messages import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent

from .tools.product_search_tool import ProductSearchTool
from .tools.product_info_tool import ProductInfoTool
from .tools.review_search_tool import ReviewSearchTool
from .tools.metadata_filter_tool import MetadataFilterTool

import json


class SearchAgent():
    def __init__(self):
        # Set OpenAI LLM and embeddings
        llm_chat = self._set_llm()

        tools = self._set_tools()

        system_message = SystemMessage(
            content=(
                "Do your best to answer the questions. "
                "Feel free to use any tools available to look up relevant information, only if necessary. "
                "Do not invent information."
            )
        )

        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message
        )
        agent = OpenAIFunctionsAgent(llm=llm_chat, tools=tools, prompt=prompt)
        self.search_agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            max_iterations=3,
            verbose=True,
        )

        # self.search_agent = create_conversational_retrieval_agent(
        #     llm_chat, tools, system_message=system_message, remember_intermediate_steps=False, max_iterations=3, verbose=True)

    def run_agent(self, input: str):
        agent_input = ''

        user_input = input['query']

        if 'product_ids' in input and len(input['product_ids']) > 0:
            agent_input = f'Query: {user_input}, product ids: {input["product_ids"]}'
        else:
            agent_input = f'{user_input}'

        print('\n\nagent_input:', agent_input)
        return self.search_agent.run(agent_input)

    def use_as_tool(self):
        return Tool(
            name="search_product_review",
            func=self.run_agent,
            description="""If the user is asking to search for a product or review. Input: {"query": A well formulated question, "product_ids": A list of prducts ids related to the query}""",
        )

    def _set_tools(self):
        return [ProductSearchTool().get_tool(), ReviewSearchTool().get_tool(), MetadataFilterTool().get_tool(), ProductInfoTool().get_tool()]

    def _set_llm(self):
        return ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo-0613', client='')


def main():
    query = "Tell me more about Mario Badescu and Frank Body serums, product ids: ['63b260653d66c49aca71d738', '64439cf4dad6ccfb902d7337']"
    # user_prompt = input("Usuario: ")
    agent = SearchAgent().search_agent
    response = agent(query)
    print(f'Assistant: ${response["output"]}')


if __name__ == '__main__':
    main()
