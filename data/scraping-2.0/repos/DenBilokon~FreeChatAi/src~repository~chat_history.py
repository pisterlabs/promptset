from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_
import ast

from src.database.models import User, ChatHistory
from src.schemas.chat_history_schemas import ChatHistoryBase
from src.conf.messages import ChatMessages
from src.repository.chats import get_chat_by_id

from fastapi import HTTPException, status
from typing import List
from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

load_dotenv()


async def get_context(chat_id: int, db: AsyncSession, user: User) -> List:
    chat = await get_chat_by_id(chat_id, db, user)
    context = chat.file_url
    if context is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=ChatMessages.NOT_FOUND)

    with open(context, "r", encoding="utf-8") as file:
        text = file.read()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_conversation_chain(user_question, context) -> str:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=context, embedding=embeddings)

    llm = ChatOpenAI(model='gpt-3.5-turbo')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    response = conversation_chain({'question': user_question})
    chat_history = response['chat_history']

    # for separate question/answer items
    response_dict = {'user_messages': [], 'bot_messages': []}

    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            response_dict['user_messages'].append(message.content)
        else:
            response_dict['bot_messages'].append(message.content)

    return str(response_dict)


async def create_message(chat_id: int, body: ChatHistoryBase, db: AsyncSession, user: User) -> ChatHistory:
    context = await get_context(chat_id, db, user)
    answer = get_conversation_chain(body.message, context)

    response_dict = ast.literal_eval(answer)
    user_question = 'Q: ' + response_dict['user_messages'][0]
    bot_response = 'A: ' + response_dict['bot_messages'][0]

    question = ChatHistory(message=user_question, user_id=user.id, chat_id=chat_id)
    response = ChatHistory(message=bot_response, user_id=user.id, chat_id=chat_id)
    db.add(question)
    db.add(response)

    await db.commit()
    await db.refresh(response)
    return response


async def get_history_by_chat(chat_id: int, db: AsyncSession) -> ChatHistory:
    return db.query(ChatHistory).filter(and_(ChatHistory.chat_id == chat_id)).order_by(ChatHistory.created_at).all()
