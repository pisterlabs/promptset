import streamlit as st
from typing import Union

from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory, RedisChatMessageHistory

from src.packages.chains.api_prompted_chain import APIPromptedChain
from src.packages.chains.conversational_api_chain import ConversationalAPIChain
from src.packages.constants.api_documentation import APIDocumentation
from src.packages.constants.config_constants import ConfigConstants
from src.packages.constants.environ_constants import EnvironConstants
from src.packages.constants.messages_constants import MessagesConstants
from src.packages.llms.neural_hermes import NeuralHermes
from src.packages.prompts.api_response_prompts import APIResponsePromptWithHistory, APIResponsePromptWithoutHistory
from src.packages.prompts.api_url_prompts import APIURLPromptWithHistory, APIURLPromptWithoutHistory
from src.packages.prompts.classification_prompts import ClassificationPromptWithoutMemory, \
    ClassificationPromptWithMemory
from src.packages.prompts.conversational_prompts import ConversationalPromptWithoutMemory, \
    ConversationalPromptWithMemory
from src.packages.sessions.session_manager import SessionManager
from src.packages.utils.parameter_server import ParameterServer


class ChainController:
    def __init__(
        self,
        st_memory: Union[ConversationSummaryBufferMemory, None],
        st_first_question: Union[ConversationSummaryBufferMemory, None],
    ):
        self.parameter_server = ParameterServer()
        self.environ_constants = EnvironConstants()
        self.session_manager = SessionManager()

        # region LLMs Parameters

        self.memory_parameters = self.parameter_server.settings.memory_model
        self.classification_parameters = self.parameter_server.settings.classification_model
        self.api_parameters = self.parameter_server.settings.api_model

        # endregion

        st.session_state.memory = self.memory = st_memory if st_memory else self._construct_memory()
        st.session_state.is_first_question = self.is_first_question = st_first_question if st_first_question else False

        self.chain_with_memory = self._construct_chain_with_memory(memory=self.memory)
        self.chain_without_memory = self._construct_chain_without_memory(memory=self.memory)

    def _construct_memory(self) -> ConversationSummaryBufferMemory:
        memory = ConversationSummaryBufferMemory(
            llm=NeuralHermes.construct_llm(**self.memory_parameters),
            memory_key='chat_history',
            input_key='query',
            output_key='answer',
            return_messages=True,
            # Max tokens in buffer memory has to be larger than model's to accommodate for memory pruning
            max_token_limit=self.memory_parameters.max_tokens,
            chat_memory=RedisChatMessageHistory(
                session_id=self.session_manager.session_id,
                url=self.environ_constants.REDIS_URL,
            )
        )

        return memory

    def _construct_chain_without_memory(
            self,
            memory: ConversationSummaryBufferMemory,
    ) -> ConversationalAPIChain:
        conversational_api_chain_without_memory: ConversationalAPIChain = ConversationalAPIChain(
            prompt=ConversationalPromptWithoutMemory.PROMPT,
            llm=NeuralHermes.construct_llm(**self.memory_parameters),
            api_chain=APIPromptedChain.from_llm_and_api_docs(
                llm=NeuralHermes.construct_llm(**self.api_parameters),
                api_docs=APIDocumentation.RL_DOCS,
                api_url_prompt=APIURLPromptWithoutHistory.API_URL_PROMPT,
                api_response_prompt=APIResponsePromptWithoutHistory.API_URL_PROMPT,
            ),
            classification_chain=LLMChain(
                llm=NeuralHermes.construct_llm(**self.classification_parameters),
                prompt=ClassificationPromptWithoutMemory.QUERY_CLASSIFICATION_PROMPT
            ),
            memory=memory,
        )

        return conversational_api_chain_without_memory

    def _construct_chain_with_memory(
        self,
        memory: ConversationSummaryBufferMemory,
    ) -> ConversationalAPIChain:
        conversational_api_chain_with_memory: ConversationalAPIChain = ConversationalAPIChain(
            prompt=ConversationalPromptWithMemory.PROMPT,
            llm=NeuralHermes.construct_llm(**self.memory_parameters),
            api_chain=APIPromptedChain.from_llm_and_api_docs(
                llm=NeuralHermes.construct_llm(**self.api_parameters),
                api_docs=APIDocumentation.RL_DOCS,
                api_url_prompt=APIURLPromptWithHistory.API_URL_PROMPT,
                api_response_prompt=APIResponsePromptWithHistory.API_URL_PROMPT,
            ),
            classification_chain=LLMChain(
                llm=NeuralHermes.construct_llm(**self.classification_parameters),
                prompt=ClassificationPromptWithMemory.QUERY_CLASSIFICATION_PROMPT
            ),
            memory=memory,
        )

        return conversational_api_chain_with_memory

    async def __call__(self, query: str) -> dict[str, str]:
        chain: ConversationalAPIChain = \
            self.chain_without_memory if self.is_first_question else self.chain_without_memory

        response: dict[str, str] = await chain.acall({
            'query': query,
            'api_docs': APIDocumentation.RL_DOCS,
            'api_url': ConfigConstants.RL_API_URL,
            'not_relevant_output': MessagesConstants.NOT_RELEVANT_QUERY,
            'chat_history': self.memory.chat_memory,
        })

        return response

