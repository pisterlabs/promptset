import chainlit as cl
from langchain.chat_models import BedrockChat
from langchain.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
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
""", input_variables=["document", "query"])

database = Chroma(
    persist_directory="./.data",
    embedding_function=embeddings
)


@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="準備ができました！メッセージを入力してください！").send()


@cl.on_message
async def on_message(input_message):
    print("入力されたメッセージ： " + input_message.content)
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
