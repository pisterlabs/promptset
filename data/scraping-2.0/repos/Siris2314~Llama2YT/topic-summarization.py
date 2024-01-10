

from langchain.llms import LlamaCpp, CTransformers
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.memory import ConversationSummaryBufferMemory
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from tqdm import tqdm
from langchain.chains.summarize import load_summarize_chain
from utils import get_prompt, delete_folder_contents, wrap_text_preserve_newlines, process_llm_response
import chromadb
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from experimentyt import get_text
import os
import sys
from transcript_data import generate_topic_transcript





folder_to_clear = "db/chroma_db"
delete_folder_contents(folder_to_clear)


generate_topic_transcript(sys.argv[1])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm =  LlamaCpp(model_path="./llama-2-7b-chat.ggmlv3.q8_0.bin", temperature=0.01, n_ctx=4000, verbose=True)


instruction = "Given the context that has been provided. \n {context}, Answer the following question: \n{question}"

sys_prompt = """You are an expert in YouTube topic summarization.
You will be given context to answer from. Answer the questions with as much detail as possible.
In case you do not know the answer, you can say "I don't know" or "I don't understand".
"""

prompt_sys = get_prompt(instruction, sys_prompt)

 

template = PromptTemplate(template=prompt_sys, input_variables=['context', 'question'])

# Retrieval Augmented Generation Example Below

def data_loader():
    loader = DirectoryLoader('experimental_topic', glob="**/*.txt",loader_cls=TextLoader, show_progress=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                               chunk_overlap=20)
    texts = text_splitter.split_documents(documents)
    return texts


data = data_loader()

print(data)

full_response = {}

rel_docs = []

def build_query(query):

    texts = data_loader()


    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-en",
                                        model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})


    vectordb = Chroma.from_documents(texts,embeddings, persist_directory='db/chroma_db',collection_name='yt_summarization')

    vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    
    rel_docs.append(retriever.get_relevant_documents(query, k=5))

    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True,
                                    chain_type_kwargs={'prompt': template})
    
    


    query = query
    llm_response = qa_chain(query)
    full_response = llm_response
    return full_response


ans = build_query(' '.join(sys.argv[2:]))
print(rel_docs)
process_llm_response(ans)
