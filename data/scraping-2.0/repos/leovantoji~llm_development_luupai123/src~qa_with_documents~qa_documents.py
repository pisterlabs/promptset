import tempfile

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

welcome_message = """
Welcome to QA with documents!
    1. Upload a PDF or TXT file.
    2. Ask a question about the file.
"""


def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    if file.type == "application/pdf":
        Loader = PyPDFLoader

    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(file.content)
        loader = Loader(temp_file.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents=documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # save data in the user session
    cl.user_session.set(key="docs", value=docs)

    # create a unique namespace for the file
    docsearch = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
    )
    return docsearch


@cl.on_chat_start
async def start():
    # sending an image with the local file path
    await cl.Message(content="You can now chat with your pdfs.").send()
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=120,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing '{file.name}'...")
    await msg.send()

    # no async implementation in the Pinecone client, fallback to sync
    docsearch = await cl.make_async(get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=ChatOpenAI(temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(max_tokens_limit=1000),
    )

    # let the user know that the system is ready
    msg.content = f"'{file.name}' is ready. Ask a question!"
    await msg.update()

    cl.user_session.set(key="chain", value=chain)


@cl.on_message
async def main(message: str):
    chain = cl.user_session.get(key="chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"],
    )
    cb.answer_reached = True
    result = await chain.acall(message, callbacks=[cb])

    answer = result["answer"]
    sources = result["sources"].strip()
    source_elements = []

    # get the documents from the user session
    docs = cl.user_session.get(key="docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [m["source"] for m in metadatas]

    if sources:
        found_sources = []

        # add the sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")

            # get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue

            text = docs[index].page_content
            found_sources.append(source_name)

            # create the text element referenced in the message
            source_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found."

    if cb.has_streamed_final_answer:
        cb.final_stream.elements = source_elements
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer, elements=source_elements).send()
