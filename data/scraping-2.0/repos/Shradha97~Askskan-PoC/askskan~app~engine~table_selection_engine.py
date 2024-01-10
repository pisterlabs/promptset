from typing import Optional
from langchain.callbacks.tracers import LangChainTracer
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseMemory
from langchain.base_language import BaseLanguageModel
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.callbacks.manager import Callbacks

from app.configurations.development.config_parser import args
from app.configurations.development.settings import (
    AZURE_API_BASE,
    AZURE_API_KEY,
    AZURE_API_TYPE,
    MODEL_DEPLOYMENT_NAME,
    EMBED_DEPLOYMENT_NAME,
    EMBED_API_VERSION,
    MODEL_API_VERSION,
    OPENAI_API_KEY,
)


class TableSelectionEngine:
    """Chain for chatting with an index."""

    # manager = AsyncCallbackManager([])
    # question_manager = AsyncCallbackManager([question_handler])
    # stream_manager = AsyncCallbackManager([stream_handler])

    """If set, restricts the docs to return from store based on tokens, enforced only
    for StuffDocumentChain"""

    # def _set_chain_tracer(self, question_handler, stream_handler):
    #     manager = AsyncCallbackManager([])
    #     question_manager = AsyncCallbackManager([question_handler])
    #     stream_manager = AsyncCallbackManager([stream_handler])

    #     tracer = LangChainTracer()
    #     tracer.load_default_session()
    #     manager.add_handler(tracer)
    #     question_manager.add_handler(tracer)
    #     stream_manager.add_handler(tracer)

    def get_table_selection_llm(
        self, temperature: Optional[float] = 0.0, verbose: Optional[bool] = False
    ):
        """llm for multiple table selection."""
        if args.personal_token:
            table_selection_llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=temperature,
                verbose=verbose,
            )
        else:
            table_selection_llm = AzureChatOpenAI(
                deployment_name=MODEL_DEPLOYMENT_NAME,
                openai_api_version=MODEL_API_VERSION,
                openai_api_base=AZURE_API_BASE,
                openai_api_key=AZURE_API_KEY,
                temperature=temperature,
                verbose=verbose,
            )
        return table_selection_llm

    def get_table_selection_chain(
        self,
        table_selection_llm: BaseLanguageModel,
        table_selection_prompt: BasePromptTemplate,
        verbose: Optional[bool] = False,
        callbacks: Callbacks = None,
    ) -> LLMChain:
        """Create an LLMChain for selecting relevant tables for querying."""
        table_selection_chain = LLMChain(
            llm=table_selection_llm,
            prompt=table_selection_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )
        return table_selection_chain
