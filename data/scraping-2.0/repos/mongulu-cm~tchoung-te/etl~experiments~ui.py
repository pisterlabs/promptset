import os

import chainlit as cl
import sentry_sdk
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import FAISS


sentry_sdk.init(
    dsn="https://a38e91a66c70912c38406fef32d86809@o4504301629407232.ingest.sentry.io/4506436450844672",
    # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100% of sampled transactions.We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

system_template = """Vous êtes un assistant IA qui fournit des informations sur les associations camerounaises en France. Vous recevez une question et fournissez une réponse claire et structurée. Lorsque cela est pertinent, utilisez des points et des listes pour structurer vos réponses.

Utilisez les éléments de contexte suivants pour répondre à la question de l'utilisateur. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

Si vous souhaitez connaître le nombre d'associations, je vous recommande de visiter le site web "tchoung-te.mongulu.cm" pour obtenir des informations actualisées à ce sujet.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

embedding_pth = "embeddings"
embeddings = OpenAIEmbeddings()
if os.path.exists(embedding_pth):
    vectors = FAISS.load_local(embedding_pth, embeddings)
else:
    loader = CSVLoader(
        file_path="ref-rna-real-mars-2022-enriched-qualified.csv", encoding="utf-8"
    )
    data = loader.load()
    vectors = FAISS.from_documents(data, embeddings)
    vectors.save_local(embedding_pth)

llm = ChatOpenAI(max_tokens=500, temperature=0, model_name="gpt-3.5-turbo",streaming=True)
chain_type_kwargs = {"prompt": CHAT_PROMPT}


@cl.on_chat_start
async def main():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectors.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs=chain_type_kwargs,
        chain_type="stuff",
        memory=memory,
    )
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")

    res = await cl.make_async(chain)(message)
    # Send the response
    await cl.Message(content=res["answer"]).send()
