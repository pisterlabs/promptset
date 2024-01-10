import os

import requests

from lighthouz import Lighthouz
from lighthouz.benchmark import Benchmark
from lighthouz.evaluation import Evaluation

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.llms import HuggingFaceHub, HuggingFaceEndpoint
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "sk-xx"

def hf_example_function(query: str) -> str:
    API_URL = (
        "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
    )
    API_KEY = "hf_xx"
    headers = {f"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": query, }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]["generated_text"]

def langchain_llm_model():
    print("Initializing LangChain RAG Agent")

    # DOCUMENTS_FOLDER = "FOLDER_NAME_OF_DOCUMENTS_TO_BUILD_RAG"
    DOCUMENTS_FOLDER = "DATA"
    chunk_size = 2000
    chunk_overlap = 150
    collection_name = "data-test_vect_embedding"
    local_directory = "data-test_vect_embedding"
    persist_directory = os.path.join(os.getcwd(), local_directory)
    if not os.path.exists(persist_directory) or not os.listdir(persist_directory):
        embeddings = OpenAIEmbeddings()
        documents = []
        for file in os.listdir(DOCUMENTS_FOLDER):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(DOCUMENTS_FOLDER, file)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())
        text_splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        splitdocument = text_splitter.split_documents(documents)
        vectDB = Chroma.from_documents(
            splitdocument,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory,
        )
        vectDB.persist()
    else:
        # Load the existing vector store
        embeddings = OpenAIEmbeddings()
        vectDB = Chroma(
            collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings
        )

    # main RAG framework
    llm_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        request_timeout=120,
        )

    retriever = vectDB.as_retriever(return_source_document=True)

    # prepare stuff prompt template
    template = """You are a helpful financial assistant. Your job is to provide the answer for the question based on the given context. 
    ## CONTEXT: {context}
    ## QUESTION: {question}
    ## ANSWER: """.strip()

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )

    rag_model = RetrievalQA.from_chain_type(
        llm=llm_model,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={"prompt":prompt}
    )
    print("Langchain RAG agent has been initialized.")
    return rag_model


def langchain_query_function(query: str) -> str:
    response = rag_model({"query": query})["result"]
    return response

lh = Lighthouz("LH_API_KEY") # To obtain a Lighthouz API key contact srijan@lighthouz.ai

benchmark_generator = Benchmark(lh)
benchmark_generator.generate_benchmark(file_path="ENTER_FILENAME_TO_GENERATE_RAG_BENCHMARK", benchmarks=["rag_benchmark", "out_of_context", "pii_leak", "prompt_injection"]) # You can also generate benchmarks from https://lighthouz.ai/studio

rag_model = langchain_llm_model()

evaluation = Evaluation(lh)
e_single = evaluation.evaluate_rag_model(
    response_function=langchain_query_function,
    benchmark_id="BENCHMARK_ID", # Enter the benchmark id from above or enter a previous benchmark id. Benchmark ids are on the dashboard https://lighthouz.ai/benchmarks/
    app_id="APP_ID", # App can be created on the dashboard at https://lighthouz.ai/dashboard/
)
print(e_single)

e_multiple = evaluation.evaluate_multiple_rag_models(
    response_functions=[langchain_query_function, hf_example_function],
    benchmark_id="BENCHMARK_ID",
    app_ids=["APP_ID1", "APP_ID2"],
)
print(e_multiple)
