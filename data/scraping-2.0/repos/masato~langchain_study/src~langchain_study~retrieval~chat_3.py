from pathlib import Path

import chainlit as cl
from chainlit.message import Message
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.text_splitter import SpacyTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

llm = ChatOpenAI(model="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "文章を元に質問に答えてください。\n\n文章: {document}"),
        ("human", "質問: {query}"),
    ],
)

text_splitter = SpacyTextSplitter(chunk_size=1000, pipeline="ja_core_news_sm")

database = Chroma(persist_directory="./data", embedding_function=embeddings)


@cl.on_chat_start
async def on_chat_start() -> None:
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="PDF を選択してください。",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    file = files[0]

    if not Path("tmp").exists():
        Path("tmp").mkdir()
    Path(f"tmp/{file.name}").write_bytes(file.content)

    documemts = PyMuPDFLoader(f"tmp/{file.name}").load()

    splitted_documents = text_splitter.split_documents(documemts)

    database = Chroma(embedding_function=embeddings)
    database.add_documents(splitted_documents)

    cl.user_session.set("database", database)

    await cl.Message(
        content=f"`{file.name}` の読み込みが完了しました。質問を入力してください。",
    ).send()


@cl.on_message
async def on_message(input_message: Message) -> None:
    database = cl.user_session.get("database")  # type: Chroma

    documents = database.similarity_search(input_message.content)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ----------------------------
    {document.page_content}
    """

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {"document": documents_string, "query": input_message.content},
    )
    await cl.Message(content=str(result)).send()
