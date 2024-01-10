import os
import sys
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_KEY')

sys.path.append('../Credit_All_In_One/')
from my_configuration import _get_mongodb


mongo_db = _get_mongodb()
mongo_collection = mongo_db["official_website"] 
persist_directory = './chroma_db'
embedding = OpenAIEmbeddings() # default: “text-davinci-003”
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
  

def load_data(mongo_history):
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    template = """
    你是一個數家銀行信用卡介紹與優惠資訊的聊天機器人，請依據下方給定的內容與使用者本次問題進行回覆。
    請依使用者提問的語言回答他的問題，若有不清楚使用者語言或是簡體中文，一律使用繁體中文回答。
    You are given the following extracted parts of a long document and a question. Provide a conversational answer.
    If you don't know the answer, just say "抱歉，我目前沒有這個問題的相關資訊。您可以調整您的提問，或是詢問我其他問題。" Don't try to make up an answer.
    回答信用卡資訊時，也提供資料庫內該信用卡的link連結給使用者自行檢索。
    如果使用者的問題與信用卡、基本問候無關的話，請回答「抱歉，我目前沒有這個問題的相關資訊。」
    Question: {question}
    =========
    {context}
    =========
    Answer in Markdown:"""
    QA_PROMPT = PromptTemplate(template=template, input_variables=["question", "context"])

    # 4. Now we can load the persisted database from disk
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    
    # retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2}) 
    # retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5})
    retriever = _build_self_query_retriever(vectordb)

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=mongo_history, return_messages=True)
    conversation_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory,
        verbose=True,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        max_tokens_limit=300)
    return conversation_qa_chain


def _build_self_query_retriever(vectorstore): #cards_list
    metadata_field_info = [
        AttributeInfo(
            name="card_name",
            description=f"The name of the credit card", 
            type="string",
        )
    ]
    document_content_description = "The metadata of the credit card information"
    llm = OpenAI(temperature=0)
    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, verbose=True, enable_limit=True
    )
    return retriever



if __name__ == '__main__':
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    result = _build_self_query_retriever(vectordb).get_relevant_documents('可以幫我介紹一下聯邦銀行吉鶴卡嗎？')
    result = _build_self_query_retriever(vectordb).get_relevant_documents('海外消費有什麼現金回饋比較多的信用卡？')
    print(result)