import streamlit as st
import base64
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.chains import RetrievalQAWithSourcesChain
import faiss
from langchain.vectorstores import FAISS 
from langchain.docstore import InMemoryDocstore 
from langchain.utilities import GoogleSearchAPIWrapper

# Uncomment below lines when OPENAI or compatible API is being used
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings

@st.cache_resource
def _llm_specifications():
    embeddings_model = LlamaCppEmbeddings(model_path="models/llama-7b.ggmlv3.q4_0.bin")
    # embeddings_model = OpenAIEmbeddings()  # enable if openai api is being used
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    # LLM local hosted
    llm = LlamaCpp(model_path="models/llama-7b.ggmlv3.q4_0.bin", n_ctx=2048, verbose=True)
    
    # Uncomment below lines when OPENAI or compatible API is being used
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    search = GoogleSearchAPIWrapper()   
    web_finder = WebResearchRetriever.from_llm(vectorstore=vectorstore_public, llm=llm, search=search, num_search_results=2)
    return web_finder, llm

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background_image(image_path):
    image_url = get_base64_of_bin_file(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{image_url});
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
local_image_path = "logo/internet.jpeg"
set_background_image(local_image_path)
st.info("`A bot that can find the answers over internet and summarize the web pages.`")
st.header("`Web Finder`")
class StreamOutputHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class ContextRetrievalVisualizer(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)

web_finder, llm = _llm_specifications()
question = st.text_input("`Ask a question:`")

if question:
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_finder)
    
    retrieval_streamer_cb = ContextRetrievalVisualizer(st.container())
    answer = st.empty()
    stream_handler = StreamOutputHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])