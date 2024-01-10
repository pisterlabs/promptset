from pydantic import Json
from sqlalchemy.orm import Session

from entity import crud, models
from entity.schemas import reqChat
from langchain_core.messages import AIMessage, HumanMessage
from fun import Knowledge
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from openAi import openAiUtil

vectorstoreMap = {}
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""


def loadVectorstore(db: Session, knowledge: models.knowledge):
    vectorstore = Knowledge.get_knowledge(knowledge, db)
    vectorstoreMap[knowledge.id] = vectorstore


def send_open_ai(db: Session, res: reqChat):
    vectorstore = vectorstoreMap[res.knowledge_id]
    if vectorstore is None:
        return {"err": "未加载数据"}
    # 获取历史聊天记录
    message: list = get_history(db, res)
    # 设置openAI key
    setting = crud.get_user_setting(db)
    llm = ChatOpenAI(model_name=openAiUtil.get_open_model(res.model), temperature=0,
                     openai_api_key=setting.openai_api_key,
                     openai_api_base=setting.openai_api_base)

    other_data = []

    def format_docs(docs):
        for doc in docs:
            other_data.append(doc.page_content)
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vectorstore.as_retriever()

    # 这里调用问题终结
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    from langchain_core.messages import AIMessage, HumanMessage

    contextualize_q_chain.invoke(
        {
            "chat_history": [
                HumanMessage(content="What does LLM stand for?"),
                AIMessage(content="Large language model"),
            ],
            "question": "What is meant by large",
        }
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    def contextualized_question(input: dict):
        if input.get("chat_history"):
            return contextualize_q_chain
        else:
            return input["question"]

    rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | llm
    )
    question = message.pop()
    # return rag_chain.stream({"question": question.content, "chat_history": message})
    content: str = ""
    for chunk in rag_chain.stream({"question": question.content, "chat_history": message}):
        print(chunk.content)
        print(type(chunk))
        if chunk.content is not None:
            yield chunk.content
            content = content + chunk.content
    chatHistDetails = models.chat_hist_details()
    chatHistDetails.chat_id = res.chat_id
    chatHistDetails.content = content
    chatHistDetails.other_data = json.dumps(other_data)
    chatHistDetails.role = "assistant"
    crud.save_chat_hist_details(db, chatHistDetails)


def get_history(db: Session, res: reqChat) -> list:
    chat_id = res.chat_id
    chatHistList = crud.get_chat_hist_details(db, res.chat_id)
    print(f"chatId:{chat_id}获取的历史记录:{chatHistList}")
    message = []
    for chat in chatHistList:
        if "user" == chat.role:
            message.append(HumanMessage(content=chat.content))
        elif "assistant" == chat.role:
            message.append(AIMessage(content=chat.content))
        elif "err" == chat.role:
            continue
        else:
            message.append(HumanMessage(content=chat.content))
    return message
