from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from pydantic import Extra
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models.base import BaseChatModel

class Neo4jRAGChain(Chain):
    """
    RAG with Neo4j Database
    """
    retriever: VectorStoreRetriever
    system_prompt: SystemMessagePromptTemplate
    context_prompt: SystemMessagePromptTemplate
    user_prompt: HumanMessagePromptTemplate
    chat: BaseChatModel
    output_key: str = "content"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.user_prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "neo4j_vector_chain"

    def call_with_history(self,
        inputs: Dict[str, Any],
        history: List[BaseMessage],
        run_manager: Optional[CallbackManagerForChainRun] = None):

        return self._call(inputs, history, run_manager)

    def _call(
        self,
        inputs: Dict[str, Any],
        history: List[BaseMessage] = [],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # System Prompt
        system_prompt = self.system_prompt.format()

        # Retrieve Documents from Vector Store
        documents = self.retriever.get_relevant_documents(inputs["question"])

        # Context Prompt
        context_prompt = self.context_prompt.format(documents=documents)

        # Build message and append to history
        messages = [
            system_prompt,
            context_prompt,
        ] + history + [
            self.user_prompt.format(**inputs)
        ]

        # Get response from LLM
        response = self.chat(messages=messages)

        if run_manager:
            run_manager.on_text(response)

        return {self.output_key: response.content}

    def _acall(
        self,
        inputs: Dict[str, Any],
        history: List[BaseMessage] = [],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # System Prompt
        system_prompt = self.system_prompt.format()

        # Retrieve Documents from Vector Store
        documents = self.retriever.get_relevant_documents(inputs["question"])

        # Context Prompt
        context_prompt = self.context_prompt.format(documents=documents)

        # Build message and append to history
        messages = [
            system_prompt,
            context_prompt,
        ] + history + [
            self.user_prompt.format(**inputs)
        ]

        # Get response from LLM
        response = self.chat(messages=messages)

        if run_manager:
            run_manager.on_text(response)

        return {self.output_key: response.content}
