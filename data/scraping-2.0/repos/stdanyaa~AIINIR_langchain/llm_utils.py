from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

from langchain.schema.prompt_template import format_document
from langchain.prompts import PromptTemplate

from langchain.retrievers import RePhraseQueryRetriever

from prompt_templates import QA_PROMPT_TEMPLATE, QUERY_REPHRASE_PROMPT_TEMPLATE




"""
Documents loading and preprocessing
"""

# def process_docs(docs):
#     prompt = PromptTemplate.from_template("{page_content}\n")
#     return [format_document(doc, prompt) for doc in docs]


def load_documents(
        docs_path, text_splitter=None, 
        loaders={
        '.pdf': PyPDFLoader,
        '.csv': CSVLoader,},
        loader_kwargs=None
    ):
    def create_directory_loader(file_type, directory_path):
        return DirectoryLoader(
            path=directory_path,
            glob=f"**/*{file_type}",
            loader_cls=loaders[file_type],
            loader_kwargs=loader_kwargs
        )
    pdf_loader = create_directory_loader('.pdf', docs_path)
    csv_loader = create_directory_loader('.csv', docs_path)
    if text_splitter:
        pdf_documents = pdf_loader.load_and_split(text_splitter=text_splitter)
        csv_documents = csv_loader.load_and_split(text_splitter=text_splitter)
    else:
        pdf_documents = pdf_loader.load()
        csv_documents = csv_loader.load()
    return pdf_documents + csv_documents

def get_text_splitter(chunk_size=800, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
        separators=['\d+\.\s', '\d+\.\d+\.\s', '\d+(\.\d+){2}\.\s', '\n\n', '\n'],
        is_separator_regex=True
    )
    return text_splitter


"""
Vector database, embedder and retriever
"""

def get_db(chunks, embedder_name='cointegrated/LaBSE-en-ru'):
    embeddings_model = HuggingFaceEmbeddings(model_name=embedder_name)
    db = Chroma.from_documents(chunks, embeddings_model)
    return db

def get_query_rephraser(llm):
    query_prompt = PromptTemplate(
    input_variables=["question"],
    template=QUERY_REPHRASE_PROMPT_TEMPLATE
    )
    return LLMChain(llm=llm, prompt=query_prompt)


def get_retriever(vectorstore, search_kwargs={"k": 2}, rephraser=None):
    retriever=vectorstore.as_retriever(search_kwargs=search_kwargs)
    if rephraser:
        return RePhraseQueryRetriever(
            retriever=retriever, llm_chain=rephraser
        )
    else:
        return retriever



"""
LLM and QA-langchain
"""

def get_llm(model_path='models/llama-2-7b-chat.Q4_K_M.gguf', n_ctx=4096):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llm = LlamaCpp(
        model_path=model_path,
        temperature=0.0, #0.75,
        max_tokens=min(n_ctx, 4000),
        n_ctx=n_ctx,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
    )
    return llm


def get_qa_langchain(model, retriever):
    template = QA_PROMPT_TEMPLATE

    prompt = ChatPromptTemplate.from_template(template)
    chain = {
        "context": itemgetter("question") | retriever, 
        "question": itemgetter("question") 
    } | prompt | model | StrOutputParser()

    return chain

