"""File with class to manage the LLM models for the OpenAI function.

Returns:
     _type_: _description_
"""
import os

from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryMemory,ChatMessageHistory
from langchain.agents import load_tools,initialize_agent,AgentType

from lib.packages_utility.logger import logging
from lib.packages_utility.db_manager import DBManagerMemory

class LLModel:
    """A language model that can be used to generate text.  This is a base class for specific implementations of LLMs, such as GPT."""

    def __init__(self,openai_key:str,language:str,gpt_version:str,max_tokens:int,prompt:str) -> None:  # noqa: PLR0913
        """Initialize the language model with openai key and other parameters for GPT-3 API call.

        Args:
            openai_key (str): The openai key
            language (str): The language for the conversation
            gpt_version (str): The version of GPT
            max_tokens (int): The quantity for tokens
            prompt (str): The initial prompt from the settings
        """
        os.environ['OPENAI_API_KEY'] = openai_key
        llm = OpenAI(temperature=0.7,model_name=gpt_version,max_tokens=int(max_tokens))
        self.lang = language
        self.history = ChatMessageHistory()
        self.prompt = prompt.replace("{language}",'english' if self.lang == 'en' else 'italian')
        self.db_memory = DBManagerMemory()
        self.db_memory.init()

        self.load_history()

        self.memory = ConversationSummaryMemory.from_messages(memory_key="chat_history",llm=llm,chat_memory=self.history,return_messages=True)
        tools = load_tools(["wikipedia"], llm=llm) #to see other tools but with base is OK

        self.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        memory=self.memory
        )
        self.agent.agent.llm_chain.prompt.template = self.prompt

    def load_history(self):
        """Load previous chat history from file into `ChatMessageHistory` object."""
        messages = self.db_memory.get_messages()
        for message in messages:
            self.history.add_user_message(message[1])
            self.history.add_ai_message(message[2])

    def gen_response(self,input):
        """Generate a response to an input message using the loaded language model.

        Args:
            input (_type_): _description_

        Returns:
            _type_: _description_
        """
        logging.info("Generating response")
        try:
            response = self.agent.run({'input':input,'chat_history':self.memory.chat_memory})
            self.db_memory.add_messages(input,response)
            return response
        except  Exception as e:
            logging.error(f"Error: {e}")
            return "Error in the request"
