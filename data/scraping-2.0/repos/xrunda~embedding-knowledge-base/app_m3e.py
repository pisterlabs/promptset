from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
import streamlit as st
import os


os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

load_dotenv()
# 1.矢量化数据
loader = CSVLoader(file_path="updated_magical_book_m3e.csv")
documents = loader.load()


embeddings = HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(documents,embeddings)


# model = SentenceTransformer("moka-ai/m3e-base")
# embeddings = model.encode(documents)
# text_documents = documents["question"].tolist()


# 2.做相似性搜索
def retrieve_info(query):
    similar_response = db.similarity_search(query,k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    print(page_contents_array)
    return page_contents_array
# custom_prompt = """
#     我想合作或定制服务，怎么联系？
# """
# results=retrieve_info(custom_prompt)
# print(results)

# 3.设置LLMChain和提示
llm=ChatOpenAI(temperature=0, model='gpt-3.5-turbo-16k-0613')
template = """
    You are an excellent intelligent customer service agent.
    I'm going to share information about a potential customer with you, and you're going to give the best answer that
    Based on past best practices, I should send to this prospect,and follow all the rules below.
    1/ The response should be very similar, if not identical, to past best practices,in terms of length, tone of voice, logical arguments and other details.
    2/ If best practices are irrelevant, try to mimic the style of the best practice to get the prospect's message across.Here's the message I received from the prospect:
    {message}
    Here are best practices for how we typically respond to prospects in similar situations:
    {best_practice}
    Please write the best response I should give to this prospect:

    All replies are in Chinese
"""
prompt=PromptTemplate(
    input_variables=["message","best_practice"],
    template=template
)
chain=LLMChain(llm=llm,prompt=prompt)
# 4.检索生成结果
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message,best_practice=best_practice)
    return response

message = """
我想联系商务对接
"""
# response=generate_response(message)
# print(response)
# 5.创建一个应用使用streamlit框架
def main():
    st.set_page_config(
        page_title="Customer response generator",page_icon=":bird:")

    st.header("Customer response generator :bird:")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")

        result = generate_response(message)
        
        st.info(result)

if __name__ == "__main__":
    main()