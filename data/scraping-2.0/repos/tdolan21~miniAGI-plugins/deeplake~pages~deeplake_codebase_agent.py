import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.vectorstores import DeepLake
load_dotenv()


root_dir = "../"

docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".py") and "*venv/" not in dirpath:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
print(f"{len(docs)}")



text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)
print(f"{len(texts)}")


embeddings = OpenAIEmbeddings()
embeddings

from langchain.vectorstores import DeepLake


username = os.getenv("ACTIVELOOP_USERNAME")


db = DeepLake(
    dataset_path=f"hub://tdolan21/miniAGI-code",
    read_only=True,
    embedding=embeddings,
)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"

## Agent Initialization
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

model = ChatOpenAI(model_name="gpt-3.5-turbo-0613")  # 'ada' 'gpt-3.5-turbo-0613' 'gpt-4',
qa = ConversationalRetrievalChain.from_llm(model, retriever=retriever)



question = ""
chat_history = []
qa_dict = {}

st.title("Deeplake Codebase Search :computer:")
st.info("This is a search engine for the miniAGI codebase. It can be configured for any codebase by using the configuration in the .env")


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    question = prompt  # Directly use the prompt as a string

with st.chat_message("assistant"):
    st_callback = StreamlitCallbackHandler(st.container())

    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    st.write(f"-> **Question**: {question} \n")
    st.write(f"**Answer**: {result['answer']} \n")


            
           