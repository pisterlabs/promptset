import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
import sys
from langchain.embeddings import FakeEmbeddings
from key import OPENAI_API_KEY

user_chat_history = {}  # 用户对文档聊天记录


def chat(question, user_id):
    global user_chat_history
    try:
        # 加载数据

        embeddings = FakeEmbeddings(size=1352)
        persist_directory = os.path.join(sys.path[0], 'db')
        memorydb = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings, collection_name=user_id)

        retriever = memorydb.as_retriever()
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

        qa = ConversationalRetrievalChain.from_llm(
            model, retriever=retriever, return_source_documents=True)

        result = qa(
            {"question": question, "chat_history": get_conversation(user_id)})

        retain_5_conversation(user_id)
        user_chat_history[user_id].append((question, result["answer"]))

        # arr_doc = []
        # for ref_doc in result['source_documents']:
        #     arr_doc.append(
        #         {'page_content': ref_doc.page_content, 'page': ref_doc.metadata['page'], 'source': os.path.basename(ref_doc.metadata['source'])})
        res_json = {
            'status': 200,
            'answer': result['answer'],
        }
        return res_json
    except Exception as e:
        print(e)
        res_json = {
            'status': 500,
            'answer': 'server error'
        }
        return res_json


def get_conversation(user_id):
    global user_chat_history
    if user_id in user_chat_history:
        return user_chat_history[user_id]
    else:
        user_chat_history[user_id] = []
        return []


# 保存最近五条聊天记录
def retain_5_conversation(user_id):
    global user_chat_history
    if len(user_chat_history[user_id]) >= 5:
        user_chat_history[user_id].pop(0)
