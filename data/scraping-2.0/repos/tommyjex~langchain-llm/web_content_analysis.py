from langchain.document_loaders import WebBaseLoader
import streamlit as st
from langchain import PromptTemplate
from langchain import LLMChain
from llm_wrapper import Baichuan
from langchain.embeddings import MiniMaxEmbeddings
from langchain.vectorstores import PGVector

llm = Baichuan() 
embeddings = MiniMaxEmbeddings()
template = """
    根据以下内容回答问题{doc}，不能自己编造内容。
    
    问题：{question}
"""

prompt = PromptTemplate(input_variables=['doc','question'],template=template)
chain = LLMChain(llm=llm,prompt=prompt)

def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("您好，我是AI助手，很高兴为您服务🥰")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
        
    else:
        st.session_state.messages = []

    return st.session_state.messages

st.header("向量数据库postgresql连接")
col1,col2 = st.columns([0.5,0.5],gap="medium")

with col1:
    
    # db_user = "xujianhua"
    # db_password = "AihymTs4X*7z*QGp"
    # db_connection_url = "postgres773bcb6637f1.rds-pg.ivolces.com"
    # db_name = "PGvector_db"
    db_connection_url = st.text_input("pgsql连接地址：")
    db_name = st.text_input("数据库名称")

with col2:
    db_user = st.text_input("数据库用户名：")
    db_password = st.text_input(label="密码：",type='password')

st.header("获取网页内容")
def init_db(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    st.success("网页内容抓取成功！")
 
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["。", "、", "\n\n", "\n", " ", "", ],
        chunk_size = 500,
        chunk_overlap = 20,
        length_function = len,
        add_start_index = True,
    )

    docs = text_splitter.split_documents(data)  
    # st.markdown(docs)

    COLLECTION_NAME = "pgvector"
    CONNECTION_STRING = f"postgresql+psycopg2://{db_user}:{db_password}@{db_connection_url}:5432/{db_name}"
    global db
    db = PGVector.from_documents(embedding=embeddings,documents=docs,collection_name=COLLECTION_NAME,connection_string=CONNECTION_STRING)

    return db

if url := st.text_input("网页url："):
    # if st.button("分析网页"):
        
        # loader = WebBaseLoader("https://zhuanlan.zhihu.com/p/597586623")     
    db = init_db(url=url)
    st.success("网页内容已转化为向量，并成功存入向量数据库postgresql")
st.header("利用大模型，针对网页内容问答")
def main():

    # url = st.text_input("网页url：")
    messages = init_chat_history()
    if question :=st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message('user',avatar='🧑‍💻'):
            st.markdown(question)
        messages.append({"role": "user", "content": question})
        doc = db.similarity_search(query=question)
        response = chain.run({'doc':doc,'question':question})
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown(response)
        messages.append({"role": "assistant", "content": response})
        st.button("清空对话", on_click=clear_chat_history)

if __name__ == "__main__":
    main()