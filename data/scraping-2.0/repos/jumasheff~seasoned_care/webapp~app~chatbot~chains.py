"""Create a ChatVectorDBChain for question/answering."""
from langchain.agents import AgentExecutor, LLMSingleActionAgent
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

from langchain.vectorstores.base import VectorStoreRetriever

from .utils import (
    AppointmentsOutputParser,
    get_appointment_chat_prompt,
    get_appointment_json_prompt,
    get_appointment_tools,
    get_general_chat_prompt,
    get_intent_prompt,
    get_symptoms_qa_prompt,
)


def get_symptoms_chain(
    retriever: VectorStoreRetriever,
    question_handler,
    stream_handler,
    tracing: bool = False,
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = OpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm,
        chain_type="stuff",
        prompt=get_symptoms_qa_prompt(),
        callback_manager=manager,
    )

    qa = ConversationalRetrievalChain(
        retriever=retriever,
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
    )
    return qa


def get_intents_chain():
    chat = ChatOpenAI(
        temperature=0,
        verbose=True,
    )
    chat_prompt = get_intent_prompt()
    return LLMChain(llm=chat, prompt=chat_prompt)


def get_general_chat_chain(stream_handler, memory: ConversationBufferMemory = None):
    if memory is None:
        memory = ConversationBufferMemory()
    chat_prompt = get_general_chat_prompt()
    stream_manager = AsyncCallbackManager([stream_handler])
    chat = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )
    return LLMChain(llm=chat, prompt=chat_prompt)


def get_appointment_agent_executor(
    stream_handler,
    memory: ConversationBufferMemory = None,
):
    stream_manager = AsyncCallbackManager([stream_handler])
    if memory is None:
        memory = ConversationBufferMemory()
    tools = get_appointment_tools(stream_manager)
    tool_names = [tool.name for tool in tools]
    chat_prompt = get_appointment_chat_prompt(tools=tools)
    chat = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        temperature=0,
    )
    llm_chain = LLMChain(llm=chat, prompt=chat_prompt)
    output_parser = AppointmentsOutputParser()
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True
    )
    return agent_executor


def get_appointment_chain(memory: ConversationBufferMemory = None):
    if memory is None:
        memory = ConversationBufferMemory()
    chat_prompt = get_appointment_json_prompt()
    chat = ChatOpenAI(
        streaming=False,
        verbose=True,
        temperature=0,
    )
    return ConversationChain(llm=chat, prompt=chat_prompt, memory=memory)
