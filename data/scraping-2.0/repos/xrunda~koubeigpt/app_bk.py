from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.schema import Document
from langchain.document_loaders.csv_loader import CSVLoader
# from langchain.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import json


from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from langchain.retrievers.multi_query import MultiQueryRetriever




load_dotenv()
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 1.矢量化数据
file_path = 'formatted_documents_split_columns_corrected_100.csv'
data = pd.read_csv(file_path)
docs=[]
for index, row in data.iterrows():
    page_content = row['page_content']
    metadata = row['metadata'].replace("'", '"')
    docs.append(Document(page_content=page_content,metadata=json.loads(metadata)))

# vectorstore = Chroma.from_documents(docs, embeddings)

# loader = CSVLoader("formatted_documents_no_brackets.csv")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db_100")


db3 = Chroma(persist_directory="./chroma_db_100", embedding_function=embedding_function)

# loader = CSVLoader(file_path="reshaped_car_data_1000.csv")
# documents = loader.load()
# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(documents,embeddings)
# 2.做相似性搜索
def retrieve_info(query):
    # similar_response = db.similarity_search(query,k=3)
    # similar_response = db3.similarity_search(query)

    # retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db3.as_retriever(), llm=llm)
    # unique_docs = retriever_from_llm.get_relevant_documents(query=query)
    # len(unique_docs)

    metadata_field_info = [
        AttributeInfo(
            name="brand",
            description="汽车品牌",
            type="string",
        ),
        AttributeInfo(
            name="model",
            description="汽车型号",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="上市年份",
            type="string",
        ),
        AttributeInfo(
            name="price", 
            description="售价", 
            type="string"
        ),
        AttributeInfo(
            name="rating", 
            description="车型特点", 
            type="string"
        ),
    ]
    document_content_description = "汽车评论"
    retriever = SelfQueryRetriever.from_llm(
        llm, db3, document_content_description, metadata_field_info, verbose=True
    )

    print(retriever)
    # page_contents_array = [doc.page_content for doc in retriever]

    return retriever

# custom_prompt = """
#     我想合作或定制服务，怎么联系？
# """
# results=retrieve_info(custom_prompt)
# print(results)

# 3.设置LLMChain和提示

import os
os.environ["DASHSCOPE_API_KEY"] = 'sk-38e455061c004036a70f661a768ba779'
DASHSCOPE_API_KEY='sk-38e455061c004036a70f661a768ba779'
from langchain.llms import Tongyi
from langchain import PromptTemplate, LLMChain

llm = Tongyi(model_kwargs={"api_key":DASHSCOPE_API_KEY},model_name= "qwen-7b-chat-v1")
# llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613')
template = """
    你是一名可以回答购车意向用户、准车主在用车买车时候的所有问题.
    我讲给你一个用户发来的问题，并且给你与这个问题相关的几个答案，这些答案来自于真实的车主反馈信息。
    请你根据这些真实的车主信息，给出你认为最佳的问题答案。
    从而帮助用户解决问题，提高用户的购车意向。解决用车方面的疑问。
    1/ 在篇幅、语气、逻辑论证和其他细节方面，答复应符合智能客服的要求和标准。
    2/ 如果与最佳实践无关，则应尽量模仿最佳实践的风格，以传达潜在客户的信息。
    以下是我从潜在客户那里收到的信息：
    {message}
    以下是此类用车问题，的真实车主反馈
    {best_practice}
    请给出你认为最佳的问题答案：
"""
prompt=PromptTemplate(
    input_variables=["message","best_practice"],
    template=template
)

# llm_chain = LLMChain(prompt=prompt, llm=llm)
chain=LLMChain(llm=llm,prompt=prompt)
# 4.检索生成结果
def generate_response(message):
    best_practice = retrieve_info(message)
    st.write("message:",message,best_practice)
    response = chain.run(message=message,best_practice=best_practice)
    return response


def main():
    st.set_page_config(
        page_title="用车口碑GPT",page_icon="🚗")

    st.header("用车口碑GPT 🚗")
    message = st.text_area("名爵MG72010款售价？")
    if message:
        info=st.write("正在生成回复内容，请稍后...")
        result = generate_response(message)
        st.info(result)


if __name__ == "__main__":
    main()

















# 
# save to disk
# db2 = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
# docs = db2.similarity_search(query)

# # load from disk
# db3 = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)
# docs = db3.similarity_search(query)
# print(docs[0].page_content)
# https://python.langchain.com/docs/integrations/vectorstores/chroma#basic-example-including-saving-to-disk