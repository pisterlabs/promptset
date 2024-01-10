import logging
import os
import time
from functools import partial
from typing import Optional, Type, Union

from llama_index import VectorStoreIndex, ServiceContext, OpenAIEmbedding
from llama_index.agent import ReActAgent
from llama_index.agent.react.formatter import ReActChatFormatter
from llama_index.agent.react.output_parser import ReActOutputParser

from llama_index.callbacks import CallbackManager
from llama_index.chat_engine.types import BaseChatEngine
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.llm_predictor.base import BaseLLMPredictor
from llama_index.llms import OpenAI, HuggingFaceLLM, ChatMessage, MessageRole
from llama_index.memory import BaseMemory, ChatMemoryBuffer
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT
from llama_index.utils import print_text

from src.Llama_index_sandbox.config import MAX_CONTEXT_LENGTHS
from src.Llama_index_sandbox.constants import OPENAI_MODEL_NAME, LLM_TEMPERATURE, NUMBER_OF_CHUNKS_TO_RETRIEVE, OPENAI_INFERENCE_MODELS
from src.Llama_index_sandbox.custom_react_agent.logging_handler import JSONLoggingHandler
from src.Llama_index_sandbox.custom_react_agent.tools.default_prompt_selectors import DEFAULT_TEXT_QA_PROMPT_SEL, DEFAULT_REFINE_PROMPT_SEL, DEFAULT_TREE_SUMMARIZE_PROMPT_SEL
from src.Llama_index_sandbox.custom_react_agent.tools.reranker.custom_vector_store_index import CustomVectorStoreIndex
from src.Llama_index_sandbox.prompts import SYSTEM_MESSAGE, QUERY_TOOL_RESPONSE, QUERY_ENGINE_TOOL_DESCRIPTION
from src.Llama_index_sandbox.custom_react_agent.ReActAgent import CustomReActAgent
from src.Llama_index_sandbox.custom_react_agent.formatter import CustomReActChatFormatter
from src.Llama_index_sandbox.custom_react_agent.output_parser import CustomReActOutputParser
from src.Llama_index_sandbox.custom_react_agent.tools.fn_schema import ToolFnSchema

from src.Llama_index_sandbox.custom_react_agent.tools.query_engine import CustomQueryEngineTool
from src.Llama_index_sandbox.custom_react_agent.tools.tool_output import log_and_store
from src.Llama_index_sandbox.utils.store_response import store_response
from src.Llama_index_sandbox.utils.utils import timeit


def get_query_engine(index, service_context, verbose=True, similarity_top_k=5):
    """Get a response synthesizer."""
    text_qa_template = DEFAULT_TEXT_QA_PROMPT_SEL
    refine_template = DEFAULT_REFINE_PROMPT_SEL
    simple_template = DEFAULT_SIMPLE_INPUT_PROMPT
    summary_template = DEFAULT_TREE_SUMMARIZE_PROMPT_SEL
    return index.as_query_engine(similarity_top_k=similarity_top_k,
                                 service_context=service_context,
                                 verbose=verbose,
                                 text_qa_template=text_qa_template,
                                 refine_template=refine_template,
                                 simple_template=simple_template,
                                 summary_template=summary_template,
                                 )


def get_inference_llm(llm_model_name):
    if llm_model_name in OPENAI_INFERENCE_MODELS:
        return OpenAI(model=llm_model_name)
    else:
        return HuggingFaceLLM(model_name=llm_model_name)


def set_inference_llm_params(temperature,
                      service_context,
                      # llm,
                      stream=False,
                      callback_manager: Optional[CallbackManager] = None,
                      max_tokens: Optional[int] = None,
                      ):
    llm = service_context.llm
    if isinstance(llm, OpenAI):
        llm.temperature = temperature
        llm.callback_manager = callback_manager
        llm.max_tokens = max_tokens
        # llm.model = OPENAI_MODEL_NAME
        # llm.stream = stream TODO 2023-10-17: determine where to set stream bool
        # if callback_manager is not None:
        #     llm.callback_manager = callback_manager
    else:
        llm.temperature = temperature
        llm.max_new_tokens = max_tokens  # TODO 2023-10-29: TBD what to set here
        llm.context_window = MAX_CONTEXT_LENGTHS[llm.model_name]
    return llm


def get_chat_engine(index: CustomVectorStoreIndex,
                    service_context: ServiceContext,
                    query_engine_as_tool: bool,
                    stream: bool,
                    log_name: str,
                    chat_mode: str = "react",
                    verbose: bool = True,
                    similarity_top_k: int = 5,
                    max_iterations: int = 10,
                    memory: Optional[BaseMemory] = None,
                    memory_cls: Type[BaseMemory] = ChatMemoryBuffer,
                    temperature=LLM_TEMPERATURE):
    # NOTE 2023-09-29: creating a (react) chat engine from an index transforms that
    #  query as a tool and passes it to the agent under the hood. That query tool can receive a description.
    #  We need to determine (1) if we pass several query engines as tool or build a massive single one (cost TBD),
    #  and (2) if we pass a description to the query tool and what is the expected retrieval impact from having a description versus not.

    logging.info(f"Fetching query engine tool from index: {index}")
    query_engine = get_query_engine(index=index, service_context=service_context, verbose=verbose, similarity_top_k=similarity_top_k)
    logging.info(f"Successfully created the query engine!")
    # NOTE 2023-10-14: the description assigned to query_engine_tool should have extra scrutiny as it is passed as is to the agent
    #  and the agent formats it into the react_chat_formatter to determine whether to perform an action with the tool or respond as is.
    # NOTE 2023-10-15: It is unclear how GPT exactly interprets the fn_schema, it is difficult to have a consistent result. Usually GPT greatly
    #  simplifies the query sent to the query engine tool, and the query engine does very poorly. We force the input to the query engine to be the user question.
    query_engine_tool = CustomQueryEngineTool.from_defaults(query_engine=query_engine)
    query_engine_tool.metadata.description = QUERY_ENGINE_TOOL_DESCRIPTION
    query_engine_tool.metadata.fn_schema = ToolFnSchema
    react_chat_formatter: Optional[ReActChatFormatter] = CustomReActChatFormatter(tools=[query_engine_tool])

    output_parser: Optional[ReActOutputParser] = CustomReActOutputParser()
    # callback_manager: Optional[CallbackManager] = None  # NOTE 2023-10-06: to configure

    logging_event_ends_to_ignore = []
    logging_event_starts_to_ignore = []
    json_logging_handler = JSONLoggingHandler(event_ends_to_ignore=logging_event_ends_to_ignore, event_starts_to_ignore=logging_event_starts_to_ignore, log_name=log_name, similarity_top_k=similarity_top_k)
    # Instantiate the CallbackManager and add the handlers
    callback_manager = CallbackManager(handlers=[json_logging_handler])

    chat_history = []

    max_tokens: Optional[int] = None  # NOTE 2023-10-05: tune timeout and max_tokens
    # TODO 2023-10-15: determine where to set stream bool
    llm = set_inference_llm_params(temperature=temperature, stream=stream, callback_manager=callback_manager, max_tokens=max_tokens, service_context=service_context)
    # service_context.llm_predictor.llm = llm
    memory = memory or memory_cls.from_defaults(chat_history=chat_history, llm=llm)

    if query_engine_as_tool:
        return CustomReActAgent.from_tools(
            tools=[query_engine_tool],
            react_chat_formatter=react_chat_formatter,
            llm=llm,
            max_iterations=max_iterations,
            memory=memory,
            output_parser=output_parser,
            verbose=verbose,
        )
    else:  # without having query engine as tool (but external to agent)
        return CustomReActAgent.from_tools(
            tools=[],
            react_chat_formatter=react_chat_formatter,
            llm=llm,
            max_iterations=max_iterations,
            memory=memory,
            output_parser=output_parser,
            verbose=verbose,
        )


def ask_questions(input_queries, retrieval_engine, query_engine, store_response_partial, engine, query_engine_as_tool, reset_chat, chat_history, run_application=False):
    # TODO 2023-10-15: We need metadata filtering at database level else for the query to look over Documents metadata else it fails e.g. when asked to
    #  retrieve content from authors. It would search in paper content but not necessarily correctly fetch all documents, and might return documents that cited the author but which can be irrelevant.
    all_formatted_metadata = None
    for query_str in input_queries:
        # TODO 2023-10-08: add the metadata filters  # https://docs.pinecone.io/docs/metadata-filtering#querying-an-index-with-metadata-filters
        if isinstance(retrieval_engine, BaseChatEngine):
            if not query_engine_as_tool:
                response = query_engine.query(query_str)
                str_response, all_formatted_metadata = log_and_store(store_response_partial, query_str, response, chatbot=True)
                str_response = QUERY_TOOL_RESPONSE.format(question=query_str, response=str_response)
                logging.info(f"Message passed to chat engine:    \n\n[{str_response}]")
                logging.info(f"With input chat history: [{chat_history}]")
                response, all_formatted_metadata = retrieval_engine.chat(message=str_response, chat_history=chat_history)
            else:
                if os.environ.get('ENVIRONMENT') == 'LOCAL':
                    logging.info(f"The question asked is: [{query_str}]")
                    logging.info(f"With input chat history: [{chat_history}]")
                response, all_formatted_metadata = retrieval_engine.chat(message=query_str, chat_history=chat_history)
            if not run_application:
                logging.info(f"[End output shown to client for question [{query_str}]]:    \n```\n{response}\n```")
                if os.environ.get('ENVIRONMENT') == 'LOCAL':
                    print_text(f"[End output shown to client for question [{query_str}]]:    \n```\n{response}\n\n Fetched based on the following sources: \n{all_formatted_metadata}\n```\n", color='green')
            if reset_chat:
                logging.info(f"Resetting chat engine after question.")
                retrieval_engine.reset()  # NOTE 2023-10-27: comment out to reset the chat after each question and see the performance, i.e. correct response given memory versus hallucination.

            if len(input_queries) > 1:
                response = ChatMessage(
                    role=MessageRole.USER,
                    content=response.response,
                )
                chat_history.append(response)

        elif isinstance(retrieval_engine, BaseQueryEngine):
            logging.info(f"Querying index with query:    [{query_str}]")
            response = retrieval_engine.query(query_str)
            response, all_formatted_metadata = log_and_store(store_response_partial, query_str, response, chatbot=False)
        else:
            logging.error(f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}")
            assert False
    if (len(input_queries) == 1) or (all_formatted_metadata is not None):
        return response, all_formatted_metadata


@timeit
def get_engine_from_vector_store(embedding_model_name: str,
                                 embedding_model: Union[OpenAIEmbedding, HuggingFaceEmbedding],
                                 llm_model_name: str,
                                 service_context: ServiceContext,
                                 text_splitter_chunk_size: int,
                                 text_splitter_chunk_overlap_percentage: int,
                                 index: CustomVectorStoreIndex,
                                 query_engine_as_tool: bool,
                                 stream: bool,
                                 similarity_top_k: int,
                                 log_name: str,
                                 engine='chat',
                                 ):

    # TODO 2023-09-29: determine how we should structure our indexes per document type
    # create partial store_response with everything but the query_str and response
    store_response_partial = partial(store_response, embedding_model_name, llm_model_name, text_splitter_chunk_size, text_splitter_chunk_overlap_percentage)

    if engine == 'chat':
        retrieval_engine = get_chat_engine(index=index, stream=stream, service_context=service_context, chat_mode="react", verbose=True, similarity_top_k=similarity_top_k, query_engine_as_tool=query_engine_as_tool, log_name=log_name)
        query_engine = get_query_engine(index=index, service_context=service_context, verbose=True, similarity_top_k=similarity_top_k)
    elif engine == 'query':
        query_engine = None
        retrieval_engine = get_query_engine(index=index, service_context=service_context, verbose=True, similarity_top_k=similarity_top_k)
    else:
        assert False, f"Please specify a retrieval engine amongst ['chat', 'query'], current input: {engine}"

        # TODO 2023-10-05 [RETRIEVAL]: in particular for chunks from youtube videos, we might want
        #   to expand the window from which it retrieved the chunk
        # TODO 2023-10-05 [RETRIEVAL]: since many chunks can be retrieved from a single youtube video,
        #   what should be the returned timestamp to these references? should we return them all? return the one with highest score?
        # TODO 2023-10-05 [RETRIEVAL]: add weights such that responses from older sources have less importance in the answer
        # TODO 2023-10-05 [RETRIEVAL]: should we weight more a person which is an author and has a paper?
        # TODO 2023-10-07 [RETRIEVAL]: ADD metadata filtering e.g. "only video" or "only papers", or "from this author", or "from this channel", or "from 2022 and 2023" etc
        # TODO 2023-10-07 [RETRIEVAL]: in the chat format, is the rag system keeping in memory the previous retrieved chunks? e.g. if an answer is too short can it develop it further?
        # TODO 2023-10-07 [RETRIEVAL]: should we allow the external user to tune the top-k retrieved chunks?

        # TODO 2023-10-09 [RETRIEVAL]: use metadata tags for users to choose amongst LVR, Intents, MEV, etc such that it can increase the result speed (and likely accuracy)
        #  and this upfront work is likely a low hanging fruit relative to payoff.

    return retrieval_engine, query_engine, store_response_partial
