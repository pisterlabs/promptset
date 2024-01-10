#https://qiita.com/shimajiroxyz/items/7c3983d658dea421052b
#https://python.langchain.com/docs/modules/data_connection/retrievers/web_research

from langchain.retrievers.web_research import WebResearchRetriever
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

def web_research_retriever(user_input,model_name,temperature):

    #空のFAISSを作成
    dummy_text, dummy_id = "1", 1
    vectorstore = FAISS.from_texts([dummy_text], OpenAIEmbeddings(), ids = [dummy_id])
    vectorstore.delete([dummy_id])

    # Vectorstore
    # vectorstore = Chroma(
    #     embedding_function=OpenAIEmbeddings(), 
    #     persist_directory="./chroma_db_oai"
    # )

    # LLM
    llm = ChatOpenAI(model= model_name, temperature=temperature)

    # Search
    os.environ["GOOGLE_CSE_ID"]
    os.environ["GOOGLE_API_KEY"]
    search = GoogleSearchAPIWrapper()
    
    # Initialize
    web_research_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore,
        llm=llm,
        search=search,
    )


    #user_input = "How do LLM Powered Autonomous Agents work?"
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm, retriever=web_research_retriever
    )
    result = qa_chain({"question": user_input})
    print(result['question'])
    print(result['answer'])
    print(result['sources'])
    return result
#{'question': '2023年で、日本の球団はどこが日本一になった？', 
# 'answer': '2023年で、日本一になった球団は阪神タイガースです。\n', 
# 'sources': 'https://npb.jp/'}

if __name__ == "__main__":
    user_input = "2023年で、日本の球団はどこが日本一になった？日本語で教えてください。"
    web_research_retriever(user_input,'gpt-4-1106-preview',1.0)