from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from langchain.llms import HuggingFaceHub, HuggingFacePipeline 
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import textwrap
import os


def load_vector_store():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    print('Embeddings loaded!')
    load_vector_store = Chroma(persist_directory = 'vector stores/ncertdb', embedding_function = embeddings)
    print('Vector store loaded!')

    retriever = load_vector_store.as_retriever(
        search_kwargs = {"k" : 2},
    )
    return retriever

    
#model 
def load_model():  
    load_dotenv()  
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    repo_id = 'llmware/bling-sheared-llama-1.3b-0.1'
    llm = HuggingFaceHub(
        repo_id = repo_id,
        model_kwargs = {'max_new_tokens' : 100}
    )
    print(llm('HI!'))
    return llm


def qa_chain():
    retriever = load_vector_store()
    llm = load_model()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = 'stuff',
        retriever = retriever,
        return_source_documents = True,
        verbose = True 
    )
    return qa

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

qa = qa_chain()

response = qa('What are types of Embedded system?')
process_llm_response(response)