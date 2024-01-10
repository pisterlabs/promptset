from typing import Optional, Dict
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import BaseRetriever
from langchain.schema import BaseMemory
from langchain.base_language import BaseLanguageModel
from langchain.chains import ConversationalRetrievalChain, ConversationChain
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


class QueryGenerationEngine:
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

    def get_chain_type_kwargs(self, instruction_prompt: BasePromptTemplate):
        chain_type_kwargs = {"prompt": instruction_prompt}
        return chain_type_kwargs

    def get_combine_doc_llm(
        self,
        temperature: Optional[float] = 0.0,
        streaming: Optional[bool] = False,
    ):
        """Streaming llm for combining documents. This happens after question generation
        FIXME: ALLOW TO TAKE MORE MODELS"""

        if args.personal_token:
            combine_docs_llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                streaming=streaming,
                temperature=temperature,
                # callback_manager=stream_manager,
            )
        else:
            # Initialize LangChain with Azure OpenAI
            combine_docs_llm = AzureChatOpenAI(
                deployment_name=MODEL_DEPLOYMENT_NAME,
                openai_api_version=MODEL_API_VERSION,
                openai_api_base=AZURE_API_BASE,
                openai_api_key=AZURE_API_KEY,
                temperature=temperature,
                streaming=streaming,
            )
        return combine_docs_llm

    def get_question_generation_llm(self, temperature: Optional[float] = 0.0):
        """Non streaming llm for question generation.
        FIXME: ALLOW TO TAKE MORE MODELS"""
        if args.personal_token:
            question_gen_llm = ChatOpenAI(
                openai_api_key=OPENAI_API_KEY,
                temperature=temperature,
                # callback_manager=question_manager,
            )
        else:
            # Initialize LangChain with Azure OpenAI
            question_gen_llm = AzureChatOpenAI(
                deployment_name=MODEL_DEPLOYMENT_NAME,
                openai_api_version=MODEL_API_VERSION,
                openai_api_base=AZURE_API_BASE,
                openai_api_key=AZURE_API_KEY,
                temperature=temperature,
            )
        return question_gen_llm

    def query_generation_chain_input(self, user_input, selected_tables_dict=None):
        query_generation_chain_input = {
            "question": user_input,
        }
        if selected_tables_dict:
            query_generation_chain_input["data_table_dictionary"] = selected_tables_dict

        return query_generation_chain_input

    @classmethod
    def get_query_generation_chain(
        cls,
        combine_doc_llm: BaseLanguageModel,
        retriever: BaseRetriever,
        memory: Optional[BaseMemory] = None,
        question_generation_llm: Optional[BaseLanguageModel] = None,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        verbose: Optional[bool] = False,
        combine_docs_chain_kwargs: Optional[Dict] = None,
        callbacks: Callbacks = None,
        tracing: Optional[bool] = False,
    ) -> ConversationalRetrievalChain:
        """Create a ConversationalRetrievalChain for question/answering."""

        # if tracing:
        #     cls._set_chain_tracer(question_handler, stream_handler)

        generated_query = ConversationalRetrievalChain.from_llm(
            llm=combine_doc_llm,
            retriever=retriever,
            memory=memory,
            condense_question_llm=question_generation_llm,
            condense_question_prompt=condense_question_prompt,
            chain_type=chain_type,
            verbose=verbose,
            combine_docs_chain_kwargs=combine_docs_chain_kwargs,
            get_chat_history=lambda h: h,
            callbacks=callbacks,
        )
        return generated_query
