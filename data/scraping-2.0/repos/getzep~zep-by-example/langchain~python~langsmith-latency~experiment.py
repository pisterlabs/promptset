from __future__ import annotations

import asyncio
import time
from uuid import uuid4

from dotenv import load_dotenv
from langchain.callbacks.manager import (
    atrace_as_chain_group,
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory, CombinedMemory
from langchain.memory import VectorStoreRetrieverMemory
from langchain.memory import ZepMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers import ZepRetriever
from langchain.schema import Document, AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from zep_python import Memory, Message, ZepClient

from chat_history import history, supplemental_human_messages

TEMPERATURE = 0.0
ZEP_API_URL = "http://localhost:8000"
PROMPT_TEMPLATE = """You are very knowledgeable about the world and enjoy sharing your 
knowledge with others. Please answer the human's question. Limit your answers to a 
maximum of three sentences.

Relevant pieces of previous conversation:
{retriever_results}

(You do not need to use these pieces of information if not relevant)

The history of this conversation:
{chat_history}

Current conversation:
Human: {input}
AI:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["input", "chat_history", "retriever_results"],
)


async def openai_chroma_test(
    run_tag_prefix: str,
    runs: int = 5,
):
    for i in range(runs):
        run_tag = f"{run_tag_prefix}_{i}"

        print(f"Running {run_tag}\n")

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            tags=[run_tag],
            temperature=TEMPERATURE,
        )

        chromadb = init_chroma(run_tag)

        retriever = chromadb.as_retriever(search_kwargs=dict(top_k=5), tags=[run_tag])
        retriever_memory = VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="retriever_results",
            input_key="input",
            exclude_input_keys=["chat_history"],
        )

        chat_memory = ConversationSummaryBufferMemory(
            max_token_limit=350,
            memory_key="chat_history",
            llm=llm,
            input_key="input",
        )

        chat_memory = init_summary_memory(chat_memory)

        combined_memory = CombinedMemory(
            memories=[retriever_memory, chat_memory], tags=[run_tag]
        )

        chain = LLMChain(
            llm=llm, memory=combined_memory, prompt=PROMPT, verbose=True, tags=[run_tag]
        )

        async with atrace_as_chain_group(run_tag) as async_group_manager:
            for msg in supplemental_human_messages:
                print(
                    await chain.arun(
                        {"input": msg["content"]},
                        callbacks=async_group_manager,
                        tags=[run_tag],
                    )
                )
                time.sleep(0.2)

        # Cleanup
        chromadb.delete_collection()
        time.sleep(0.2)


async def zep_test(
    run_tag_prefix: str,
    runs: int = 5,
):
    for i in range(runs):
        run_tag = f"{run_tag_prefix}_{i}"
        zep_session_id = str(uuid4())

        init_zep(zep_session_id)

        print(f"Running {run_tag}\n")

        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            # tags=[run_tag],
            temperature=TEMPERATURE,
        )

        retriever = ZepRetriever(
            url=ZEP_API_URL,
            session_id=zep_session_id,
            top_k=5,
            tags=[run_tag, "zep_retriever"],
        )

        chat_memory = ZepMemory(
            url=ZEP_API_URL,
            session_id=zep_session_id,
            memory_key="chat_history",
            input_key="input",
        )

        chain = LLMChain(
            llm=llm,
            memory=chat_memory,
            prompt=PROMPT,
            verbose=True,
        )

        async with atrace_as_chain_group(run_tag) as async_group_manager:
            for msg in supplemental_human_messages:
                print(
                    await chain.arun(
                        {
                            "input": msg["content"],
                            "retriever_results": docs_to_messages_str(
                                retriever.get_relevant_documents(
                                    msg.get("content"),
                                    tags=[run_tag],
                                )
                            ),
                        },
                        callbacks=async_group_manager,
                        tags=[run_tag],
                    )
                )
                time.sleep(0.2)


def docs_to_messages_str(docs: list[Document]) -> str:
    return "\n".join(
        [f"{doc.metadata.get('role')}: {doc.page_content}" for doc in docs]
    )


def get_history_messages(role: list[str] | None = None) -> list[str]:
    if role is None:
        return [f"{msg['role']}: {msg['content']}" for msg in history]

    return [msg["content"] for msg in history if msg["role"] in role]


def init_summary_memory(
    memory: ConversationSummaryBufferMemory,
) -> ConversationSummaryBufferMemory:
    for msg in history:
        if msg["role"] == "ai":
            memory.chat_memory.add_message(AIMessage(content=msg["content"]))
        else:
            memory.chat_memory.add_message(HumanMessage(content=msg["content"]))

    return memory


def init_zep(session_id: str):
    messages = [Message(role=m["role"], content=m["content"]) for m in history]

    memory = Memory(messages=messages)

    zep = ZepClient(base_url=ZEP_API_URL)
    zep.add_memory(session_id=session_id, memory_messages=memory)

    # Let the index operation on batch complete
    time.sleep(10)


def init_chroma(run_tag: str) -> Chroma:
    messages = [Document(page_content=s) for s in get_history_messages()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(messages)

    chromadb = Chroma.from_documents(
        collection_name=run_tag,
        documents=docs,
        embedding=OpenAIEmbeddings(),
    )

    # Let the index operation on batch complete
    time.sleep(10)

    return chromadb


async def test_suite():
    runs = 5

    await openai_chroma_test("openai_chroma", runs=runs)

    await zep_test("zep", runs=runs)


def main():
    load_dotenv()
    asyncio.run(test_suite())


if __name__ == "__main__":
    main()
