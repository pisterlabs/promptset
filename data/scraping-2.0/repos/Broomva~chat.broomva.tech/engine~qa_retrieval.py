from typing import Dict, Optional

import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch
from langchain.chains import (ConversationalRetrievalChain,
                              RetrievalQAWithSourcesChain)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()
vector_store = FAISS.load_local("../docs.faiss", embeddings)


@cl.on_chat_start
async def init():
    cl.AppUser(username="Broomva", role="ADMIN", provider="header")
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="OpenAI - Model",
                values=[
                    "gpt-3.5-turbo",
                    "gpt-3.5-turbo-1106",
                    "gpt-4",
                    "gpt-4-1106-preview",
                ],
                initial_index=0,
            ),
            Switch(id="streaming", label="OpenAI - Stream Tokens", initial=True),
            Slider(
                id="temperature",
                label="OpenAI - Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            Slider(
                id="k",
                label="RAG - Retrieved Documents",
                initial=3,
                min=1,
                max=20,
                step=1,
            ),
        ]
    ).send()

    chat_profile = cl.user_session.get("chat_profile")

    if chat_profile == "Turbo Agent":
        settings["model"] = "gpt-3.5-turbo"
    elif chat_profile == "GPT4 Agent":
        settings["model"] = "gpt-4-1106-preview"

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # # Create a chain that uses the Chroma vector store
    # chain = ConversationalRetrievalChain.from_llm(
    #     ChatOpenAI(
    #         temperature=settings["temperature"],
    #         streaming=settings["streaming"],
    #         model=settings["model"],
    #     ),
    #     chain_type="stuff",
    #     retriever=vector_store.as_retriever(search_kwargs={"k": int(settings["k"])}),
    #     memory=memory,
    #     return_source_documents=True,
    # )

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(
            temperature=settings["temperature"],
            streaming=settings["streaming"],
            model=settings["model"],
        ),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": int(settings["k"])}),
        memory=memory,
        return_source_documents=True,
    )

    cl.user_session.set("settings", settings)
    cl.user_session.set("chain", chain)


def format_url(input_string):
    # Remove the leading '../../../'
    modified_string = input_string[9:]

    # Replace '.md' with an empty string
    modified_string = modified_string.replace(".md", "")

    # Prepend the base URL
    formatted_url = f"https://book.broomva.tech/{modified_string}"

    return formatted_url


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    res = await chain.acall(message.content, callbacks=[cb])

    answer = res["answer"]

    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"Ref. {source_idx}"
            # Create the text element referenced in the message

            text_content = f"""{format_url(source_doc.metadata['source'])} \n
            {source_doc.page_content}
            """

            text_elements.append(cl.Text(content=text_content, name=source_name))
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
