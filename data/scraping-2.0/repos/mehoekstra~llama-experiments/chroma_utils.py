import chromadb
import torch
import os

from langchain.vectorstores import Chroma
# use this to configure the Chroma database  
from chromadb.config import Settings
from dotenv import load_dotenv
from typing import List
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.document_loaders import PyPDFLoader, BSHTMLLoader, TextLoader
from langchain_core.documents import Document
from langchain.embeddings import SentenceTransformerEmbeddings

import transformers_utils

import logging

sentence_transformer_model="all-MiniLM-L6-v2"
_ = load_dotenv()

RAG_COLLECTION_NAME = "Transcripts_Store"
logger = logging.getLogger('transformers_streamlit.chroma_utils')







#config = AutoConfig.from_pretrained(original_model_id, torchscript=True)
# configure our database

def get_vector_store(transformers_config, rescan_dir=False, reset_db=False):
    vector_store = None
    persistent_client = chromadb.PersistentClient(transformers_config.rag_db_dir)

    #delete the collection if resetting the entire db
    if reset_db:
        try:
            persistent_client.delete_collection(RAG_COLLECTION_NAME)
        except ValueError:
            pass

    #Use a cosine distance measurement and sentence transformer embeddings
    collection = persistent_client.get_or_create_collection(RAG_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    #If not using GPU, set to CPU explicitly.  Otherwise let the framework pick.
    if not transformers_config.use_GPU:
        model_kwargs=dict(device=torch.device("cpu"))
        embeddings = SentenceTransformerEmbeddings(
            model_name=sentence_transformer_model,
            model_kwargs=model_kwargs
        )
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name=sentence_transformer_model,
        )      

    # Create a langchain vectorstore object
    client_settings = Settings(
        persist_directory=transformers_config.rag_db_dir, #location to store 
        anonymized_telemetry=False # optional but showing how to toggle telemetry
    )
    vector_store = Chroma(collection_name=RAG_COLLECTION_NAME, client_settings=client_settings, client=persistent_client,
                            embedding_function=embeddings, persist_directory=transformers_config.rag_db_dir)
    
    if rescan_dir:
        add_files_to_vector_store(vector_store, transformers_config.rag_file_dir, transformers_config)

    return vector_store

#Check to see if a particular file is found in the database already so we don't
#try to add it twice
def file_in_collection(vector_store, file_path):
    docs = vector_store.get(where={"source" : file_path}, limit=1)
    return (len(docs["ids"]) > 0)

#Add additional metadata to a list of documents
def add_metadata(docs : List[Document], data : dict):
    for doc in docs:
        for key, value in data.items():
            doc.metadata[key] = value


# check if the database exists already
# if not, create it, otherwise read from the database
def add_files_to_vector_store(vector_store, path, transformers_config):

    logger.debug("Creating database")
    headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = transformers_config.rag_doc_max_chars,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )

    markdown_splits = []
    #different split strategy based on doc type
    for filename in os.listdir(path):
        large_docs = None
        file_path = os.path.join(path, filename)
        #recurse if we hit another directory
        if os.path.isdir(file_path):
            add_files_to_vector_store(vector_store, str(file_path), transformers_config)
        elif not file_in_collection(vector_store, file_path):
            logger.debug(f"Adding file {file_path}")
            if filename.endswith('.md') or filename.endswith('.MD'):
                with open(file_path) as f:
                    large_docs = markdown_splitter.split_text(f.read())
                    add_metadata(large_docs, {"source" : file_path})
            elif filename.endswith('pdf') or filename.endswith('.PDF'):
                pdf_loader = PyPDFLoader(file_path)
                large_docs = pdf_loader.load_and_split()
            elif filename.endswith('.html') or filename.endswith('.HTML'):
                html_loader = BSHTMLLoader(file_path)
                large_docs = html_loader.load()
            elif filename.endswith('txt') or filename.endswith('.TXT'):
                text_loader = TextLoader(file_path)
                large_docs = text_loader.load()
            if large_docs is not None:
                text_splits = text_splitter.split_documents(large_docs)
                #file could have all whitespace
                if(len(text_splits) > 0):
                    text = [document.page_content for document in text_splits]
                    metadata = [document.metadata for document in text_splits]
                    vector_store.add_texts(text, metadata)
        else:
            logger.debug(f"File {file_path} already found")



#Main function if called directly
def main():


    config = transformers_utils.read_config()
    logger.setLevel(logging.DEBUG)

    vector_store = get_vector_store(config, config.rescan_RAG_files, config.reset_RAG_db)
    #Here higher is better
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold",
                                            search_kwargs={'k': 10, 'score_threshold' : config.rag_relevance_limit})
    docs = retriever.invoke("tell me about the security and privacy research group")
    for doc in docs:
        print(str(doc))

    model, tokenizer = transformers_utils.load_optimized_model(config)

    query_list = ["Hello my name is Anjo",
    "who are you?",
    "tell me about the Security and Privacy Research group",
    "who leads the group?",
    "tell me a joke"]

    transformers_utils.send_rag_queries(vector_store, model, tokenizer, query_list, config)

if __name__ == "__main__":
    main()
    exit(0)




# attempts to use langchain below here
rag_template = """Use the following pieces of context, delimited by 3 single quotes
to answer the question at the end.  If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.
'''{context}'''
Question: "{question}"
Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(rag_template)



"""

from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_experimental.chat_models import Llama2Chat


#Almost there with langchain, if only the Llama2Chat object could be instantiated
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
hf = HuggingFacePipeline(pipeline=pipe)

chat = Llama2Chat(llm=hf)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt_custom
    | chat
    | StrOutputParser()
)


for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)


#rag_chain.invoke("What is Task Decomposition?")
"""


