import os
import chainlit as cl
from langchain.chat_models import BedrockChat
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores import Chroma
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1"
)

chat = BedrockChat(
    model_id="anthropic.claude-v2"
)

prompt = PromptTemplate(
    template="""文章を元に質問に答えてください。

    文章：
    {document}

    質問： {query}
    """, input_variables=["document", "query"]
)

text_splitter = SpacyTextSplitter(chunk_size=300, pipeline="ja_core_news_sm")


@cl.on_chat_start
async def on_chat_start():
    files = None

    while files is None:
        files = await cl.AskFileMessage(
            max_size_mb=20,
            content="PDFを選択してください",
            accept=["application/pdf"],
            raise_on_timeout=False,
        ).send()
    file = files[0]

    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    with open(f"tmp/{file.name}", "wb") as f:
        f.write(file.content)

    documents = PyMuPDFLoader(f"tmp/{file.name}").load()
    splitted_documents = text_splitter.split_documents(documents)

    database = Chroma(
        embedding_function=embeddings,
    )

    database.add_documents(splitted_documents)

    cl.user_session.set(
        "database",
        database
    )

    await cl.Message(content=f"`{file.name}`の読み込みが完了しました。質問を入力してください。").send()


@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ： " + input_message.content)

    database = cl.user_session.get("database")

    documents = database.similarity_search(input_message.content)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ----------------------------
    {document.page_content}
    """

    result = chat([
        HumanMessage(content=prompt.format(
            document=documents_string, query=input_message.content))
    ])
    await cl.Message(content=result.content).send()
