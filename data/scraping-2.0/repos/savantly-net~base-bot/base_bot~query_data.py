"""Create a ConversationalRetrievalChain for question/answering."""
import imp
import logging
import sys
from typing import Union

from langchain.callbacks.base import BaseCallbackManager, BaseCallbackHandler
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores.base import VectorStore

from base_bot.prompts import QA_PROMPT, REPHRASE_PROMPT

from . import config


# dynamic import
def dynamic_imp(name):
    # find_module() method is used
    # to find the module and return
    # its description and path
    try:
        fp, path, desc = imp.find_module(name, [".", "base_bot/llm"])

    except ImportError as e:
        logging.error("module not found: " + name + " " + str(e))

    try:
        # load_modules loads the module
        # dynamically and takes the filepath
        # module and description as parameter
        return imp.load_module(name, fp, path, desc)

    except Exception as e:
        logging.error("error loading module: " + name + " " + str(e))


def get_chain(
    vectorstore: Union[VectorStore, any], rephrase_handler: BaseCallbackHandler, stream_handler: BaseCallbackHandler, tracing: bool = False
) -> ConversationalRetrievalChain:
    _vectorstore = vectorstore() if callable(vectorstore) else vectorstore
    manager = BaseCallbackManager([])
    rephrase_manager = BaseCallbackManager([rephrase_handler])
    stream_manager = BaseCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        rephrase_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    llm_package = dynamic_imp(config.LLM_MODULE)
    rephrase_generator_llm = llm_package.getLLM(
        model=config.LLM_REPRHASING_MODEL,
        temperature=config.LLM_REPHRASING_TEMPERATURE,
        verbose=config.LLM_REPHRASING_VERBOSE,
        callback_manager=rephrase_manager,
    )
    streaming_llm = llm_package.getLLM(
        streaming=True,
        callback_manager=stream_manager,
        verbose=config.LLM_STREAMING_VERBOSE,
        temperature=config.LLM_STREAMING_TEMPERATURE,
        model=config.LLM_STREAMING_MODEL,
    )

    rephrase_generator = LLMChain(
        llm=rephrase_generator_llm, prompt=REPHRASE_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = ConversationalRetrievalChain(
        retriever=_vectorstore.as_retriever(),
        combine_docs_chain=doc_chain,
        question_generator=rephrase_generator,
        callback_manager=manager,
    )
    return qa
