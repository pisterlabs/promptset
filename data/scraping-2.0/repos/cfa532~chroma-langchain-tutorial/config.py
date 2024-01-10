import chromadb
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import SentenceTransformerEmbeddings
# from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Init environment variables in .env file
from pprint import pprint
from dotenv import load_dotenv
load_dotenv()

VERBOSE = True

"""The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""

CHROMA_CLIENT = chromadb.HttpClient(host='192.168.0.5', port=8000)
LAW_COLLECTION_NAME = "law-docs"     # collection name for all public laws and regulations
cols = CHROMA_CLIENT.list_collections()
print(cols)
laws = CHROMA_CLIENT.get_or_create_collection(LAW_COLLECTION_NAME)
print(laws.count())
print(laws.peek(5))

# CHROMA_CLIENT.delete_collection(LAW_COLLECTION_NAME)
# CHROMA_CLIENT.delete_collection("OWsMe8-Gl3epd7ESBEq9C7LjYX2")
# cols = CHROMA_CLIENT.get_or_create_collection("OWsMe8-Gl3epd7ESBEq9C7LjYX2")
# CHROMA_CLIENT.reset()

LLM = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=-1, verbose=VERBOSE,)
CHAT_LLM = ChatOpenAI(temperature=0, model="gpt-4", max_tokens=1024, verbose=VERBOSE)     # ChatOpenAI cannot have max_token=-1

# EMBEDDING_FUNC = OpenAIEmbeddings()
# EMBEDDING_FUNC = DefaultEmbeddingFunction()
# EMBEDDING_FUNC = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# EMBEDDING_FUNC = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
EMBEDDING_FUNC = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

def print_object(obj):
    pprint(vars(obj))

def llm_chain(query:str, llm=CHAT_LLM):
    return LLMChain(llm=llm, prompt=PromptTemplate.from_template("{query}. Give your reply in Chinese."),
        # verbose=VERBOSE
    ).run(query)

class LegalCase:
    def __init__(self, lc):
        self.role = lc.role
        self.mid = lc.mid           # Memei id of the user obj, used as colleciton name in DB
        self.id = lc.id             # id of the case, used as doc_type metadata
        self.title = lc.title
        self.brief = lc.title
        self.plaintiff = lc.plaintiff
        self.defendant = lc.defendant
        self.attorney = lc.attorney
        self.judge = lc.judge