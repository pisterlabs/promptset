from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import Tool

from typing import List
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from pydantic import BaseModel, Field

from langchain.agents.agent_toolkits.conversational_retrieval.tool import (
    create_retriever_tool,
)

import json
from langchain.schema.agent import AgentActionMessageLog, AgentFinish
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function

class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[str] = Field(
        description="List of sources that contain answer to the question. Only include a source if it contains relevant information"
    )

class ChatAgent:
    """
    A class to encapsulate the functionality for a chat agent with
    various tools including a conversational model, memory, and
    retrieval QA chain.
    """

    def __init__(self, openai_api_key, vectorstore):
        # Initialize the language model
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("system", "{chat_history}"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name='gpt-3.5-turbo-1106',  # gpt-3.5-turbo didn't provide sources
            temperature=0.0
        )

        self.retriever = create_retriever_tool(
            vectorstore.as_retriever(search_kwargs={"k": 5}),
            "ibm-generative-ai-sdk-retriever",
            "Query a retriever to get information about IBM Generative AI Python SDK",
        )

        self.llm_with_tools = self.llm.bind(
            functions=[
                # The retriever tool
                format_tool_to_openai_function(self.retriever),
                # Response schema
                convert_pydantic_to_openai_function(Response),
            ]
        )

        # Set up conversational memory
        self.conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            output_key='answer',
            k=5,
            return_messages=True
        )


        # Initialize the agent with tools
        self.agent = self.initialize_agent()

    def parse(self, output):
        # If no function was invoked, return to user
        if "function_call" not in output.additional_kwargs:
            return AgentFinish(return_values={"answer": output.content}, log=output.content)

        # Parse out the function call
        function_call = output.additional_kwargs["function_call"]
        name = function_call["name"]
        inputs = json.loads(function_call["arguments"])

        # If the Response function was invoked, return to the user with the function inputs
        if name == "Response":
            return AgentFinish(return_values=inputs, log=str(function_call))
        # Otherwise, return an agent action
        else:
            return AgentActionMessageLog(
                tool=name, tool_input=inputs, log="", message_log=[output]
            )


    def initialize_agent(self):
        """
        Initialize the agent with tools and settings.

        :return: The initialized agent.
        """

        custom_agent = (
                {
                    "input": lambda x: x["input"],
                    # Format agent scratchpad from intermediate steps
                    "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
                    "chat_history" : lambda x: x["chat_history"]
                }
                | self.prompt
                | self.llm_with_tools
                | self.parse
        )
        return AgentExecutor(tools=[self.retriever], agent=custom_agent,
                             verbose=True, memory=self.conversational_memory)

    def run(self, query):
        """
        Run the agent with a given query.

        :param query: The query to be processed.
        :return: The result of the agent processing.
        """
        return self.agent.invoke({"input": query}, return_only_outputs=True)