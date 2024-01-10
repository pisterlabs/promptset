from langchain.prompts import PromptTemplate

app_name = "DENSO GPT Expert"

# BOILERPLATE


from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.vectorstores.chroma import Chroma
#from .prompts import prompt

from dotenv import load_dotenv,find_dotenv
import os


####################STAGE 0 LOAD CONFIG ############################
load_dotenv(find_dotenv(),override=True)
CHROMADB_HOST = os.environ.get("CHROMADB_HOST")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY")

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, openai_api_key=OPEN_AI_API_KEY)
import streamlit as st
import os
#model = HuggingFaceEmbeddings(model_name = "bkai-foundation-models/vietnamese-bi-encoder")
model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
database = Chroma(persist_directory="../chroma_db", embedding_function=model)
st.set_page_config(layout='centered', page_title=f'{app_name}')
ss = st.session_state
if 'debug' not in ss: ss['debug'] = {}

#from DENSO_GPT_Expert.src.Core.model import get_similar_chunks, get_response_from_query
import streamlit as st

st.title("💬 DENSO GPT Expert")
st.caption("🚀 A chatbot powered by SmartABI")

st.sidebar.title("🤖 DENSO GPT Expert")
st.sidebar.write("Welcome to the DENSO GPT Expert")


def get_similar_chunks(query, db=database, k=4):
    chunks = db.similarity_search_with_score(query=query, k=k)
    return chunks

def get_response_from_query(query, chunks):
    docs = " ".join([d[0].page_content for d in chunks])

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0, openai_api_key=OPEN_AI_API_KEY)

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            ###
            Bạn là một trợ lý quy trình, bạn có kiến thức về quy trình, hướng dẫn và tài liệu máy dựa trên tài liệu của nhà máy.
            Dựa trên tài liệu được cung cấp dưới đây, hãy cung cấp hướng dẫn cho câu hỏi dưới đây dựa trên tài liệu đã cung cấp.
            Hãy sử dụng ngôn ngữ hướng dẫn, kỹ thuật và một cách ngắn gọn.
            
            Tài liệu: {docs}
            Câu hỏi: {question}
            
            Hãy cung cấp tất cả các câu trả lời bằng tiếng Việt.
            ###
            """,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    test_prompt = prompt.format(question=query, docs=docs)
    st.write(test_prompt)
    output = chain.run({'question': query, 'docs': docs})
    return output


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("Say something")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    chunks = get_similar_chunks(query=prompt)
    response = get_response_from_query(query=prompt,chunks=chunks)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

