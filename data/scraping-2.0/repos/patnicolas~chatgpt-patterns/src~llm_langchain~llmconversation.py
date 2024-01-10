__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2022, 23. All rights reserved."

from typing import AnyStr, Dict, NoReturn
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationTokenBufferMemory, ConversationBufferWindowMemory, \
    ConversationSummaryBufferMemory, ConversationBufferMemory, ConversationEntityMemory

"""
   Wraps LLM conversation with the appropriate memory model to build up context
            - token memory buffer
            - sliding window memory buffer
            - summary memory buffer
            - entity memory buffer
            - Generic conversation memory buffer
"""


class LLMConversation(object):
    token_memory_type = 'token_memory'
    window_memory_type = 'window_memory'
    summary_memory_type = "summary_memory"
    entity_memory_type = "entity_memory"
    conversation_memory_type = "conversation_memory"

    """
        Constructor for the conversational buffer
        
        :param _model Model name (i.e. gpt-3.5-turbo-0613)
        :param memory_buffer_type Type of memory buffer used (i.e. token_memory, window_memory,..)
        :param argument:  max_tokens_imit for summary and token buffer memory or number of requests in
                memory for window buffer
        :param _verbose Boolean flag
    """
    def __init__(self, _model: AnyStr, memory_buffer_type: AnyStr, argument: int, _verbose: bool = True):
        self.llm = ChatOpenAI(temperature=0.0, model=_model)
        if memory_buffer_type == LLMConversation.token_memory_type:
            self.memory = ConversationTokenBufferMemory(llm=self.llm, max_tokens_limit=argument)
        elif memory_buffer_type == LLMConversation.window_memory_type:
            self.memory = ConversationBufferWindowMemory(k=argument)
        elif memory_buffer_type == LLMConversation.summary_memory_type:
            self.memory = ConversationSummaryBufferMemory(llm=self.llm, max_token_limit=argument)
        elif memory_buffer_type == LLMConversation.entity_memory_type:
            self.memory = ConversationEntityMemory(llm=self.llm)
        else:
            self.memory = ConversationBufferMemory()
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory, verbose=_verbose)

    def __call__(self, prompt: AnyStr) -> AnyStr:
        """
            Apply ta given conversational memory model for request to LLM
            :param prompt Prompt or content of the request
            :return response as string
        """
        return self.conversation.predict(input=prompt)

    def save_context(self, key_values: Dict[AnyStr, AnyStr]) -> NoReturn:
        """
            Save the context (memory buffer) into local file
            :param key_values Dictionary of key, value as string
            :return None
        """
        for key, value in key_values.items():
            self.memory.save_context({'input' : key}, {'output' : value})

    def load_memory_variables(self) -> Dict[AnyStr, AnyStr]:
        """
            Load the memory variables
            :return Dictionary of memory variable key -> value
        """
        return self.memory.load_memory_variables({})


