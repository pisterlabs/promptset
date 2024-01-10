import asyncio
import json
import logging
from typing import Any, AsyncGenerator, List, Union

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever

from crc_api import dao

log = logging.getLogger(__name__)

DEFAULT_BOUNDARY_STRING = "------[RESPONSE BOUNDARY]------"


def populate_vectorstore(
    openai_api_key: str,
    persist_directory: str,
    documents: list,
    chunk_size_tokens=1200,
    chunk_overlap_tokens=100,
):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens, chunk_overlap=chunk_overlap_tokens
    )

    texts = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    return db


def create_retriever(
    openai_api_key: str,
    persist_directory: str,
    num_docs_to_retrieve=2,
) -> VectorStoreRetriever:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return db.as_retriever(
        search_type="similarity", search_kwargs={"k": num_docs_to_retrieve}
    )


def get_answer(
    openai_api_key: str,
    retriever,
    conversation_id: str,
    question: str,
    model: str = "gpt-3.5-turbo",
) -> AsyncGenerator[str, None]:
    conv = dao.get_conversation(conversation_id)
    if "memory" not in conv:
        conv["memory"] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    memory = conv["memory"]

    # callback_handler = AsyncIteratorCallbackHandler()

    from langchain.callbacks.base import BaseCallbackHandler

    class PrintStreamHandler(BaseCallbackHandler):
        # def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        #     self.container = container
        #     self.text = initial_text

        def on_llm_new_token(self, token: str, *args, **kwargs) -> None:
            # self.text += token
            # self.container.markdown(self.text)
            print(token, end="")

    class PrintRetrievalHandler(BaseCallbackHandler):
        # def __init__(self, container):
        #     self.container = container.expander("Context Retrieval")

        def on_retriever_start(self, query: str, *args, **kwargs):
            print(f"**Question:** {query}")

        def on_retriever_end(self, documents, **kwargs):
            # self.container.write(documents)
            for doc in documents:
                # source = os.path.basename(doc.metadata["source"])
                # self.container.write(f"**Document {idx} from {source}**")
                print(doc.page_content)

    stream_handler = PrintStreamHandler()
    retrieval_handler = PrintRetrievalHandler()

    condense_quuestion_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model)
    qa_prompt_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=model,
        streaming=True,
        # callbacks=[callback_handler],
        temperature=0,
    )

    question_generator = LLMChain(
        llm=condense_quuestion_llm, prompt=CONDENSE_QUESTION_PROMPT
    )
    combine_docs_chain = load_qa_chain(
        qa_prompt_llm, chain_type="stuff", prompt=QA_PROMPT
    )

    conv_retrieval_chain = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
    )

    resp = conv_retrieval_chain.run(
        question=question,
        callbacks=[retrieval_handler, stream_handler],
    )
    return resp


async def get_answer_async(
    openai_api_key: str,
    retriever,
    conversation_id: str,
    question: str,
    model: str = "gpt-3.5-turbo",
    condense_question_prompt: str = None,
    qa_prompt: str = None,
    boundary_string=DEFAULT_BOUNDARY_STRING,
) -> AsyncGenerator[str, None]:
    condense_question_prompt = condense_question_prompt or CONDENSE_QUESTION_PROMPT
    qa_prompt = qa_prompt or QA_PROMPT

    conv = dao.get_conversation(conversation_id)
    if "memory" not in conv:
        conv["memory"] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
    memory = conv["memory"]

    callback_handler = AsyncIteratorCallbackHandler()

    from langchain.callbacks.base import BaseCallbackHandler

    class RetrievedDocsHandler(BaseCallbackHandler):
        done: asyncio.Event = asyncio.Event()
        docs: List[Document] = []

        def on_llm_error(
            self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
        ) -> None:
            log.exception(str(error))
            self.done.set()

        def on_retriever_end(self, documents, *args, **kwargs):
            for doc in documents:
                self.docs.append(doc)
            self.done.set()

    retrieved_docs_handler = RetrievedDocsHandler()

    condense_quuestion_llm = ChatOpenAI(openai_api_key=openai_api_key, model=model)
    qa_prompt_llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=model,
        streaming=True,
        callbacks=[callback_handler],
        temperature=0,
    )

    question_generator = LLMChain(
        llm=condense_quuestion_llm, prompt=condense_question_prompt
    )
    combine_docs_chain = load_qa_chain(
        qa_prompt_llm, chain_type="stuff", prompt=qa_prompt
    )

    conv_retrieval_chain = ConversationalRetrievalChain(
        retriever=retriever,
        memory=memory,
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        # callbacks=[retrieved_docs_handler],
    )

    run = asyncio.create_task(
        conv_retrieval_chain.arun(question=question, callbacks=[retrieved_docs_handler])
    )

    await retrieved_docs_handler.done.wait()
    for doc in retrieved_docs_handler.docs:
        yield json.dumps({"content": doc.page_content}) + "\n"

    yield boundary_string + "\n"

    async for token in callback_handler.aiter():
        yield token

    await run
