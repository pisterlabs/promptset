
import os
import typing
import openai
import logging

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown

logging.basicConfig(level=logging.DEBUG)

LOGGER = logging.getLogger(__file__)

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ["OPENAI_API_KEY"]

# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

# Using API provided by LangChain

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

response = index.query(query)

display(Markdown(response))


# Doing the process manually

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

docs = loader.load()
embed = embeddings.embed_query("Hi my name is Harrison")

db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"

# perform similarity search manually and use an LLm to do Q & A

docs = db.similarity_search(query)

retriever = db.as_retriever()

llm = ChatOpenAI(temperature = 0.0, model=llm_model)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

display(Markdown(response))

# using a QA Chain that does the above steps from line 77-85

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", # indicates to just stuff the document into the llm context, others are: map_reduce, refine, map_rerank, map_reduce is great for summarization
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

display(Markdown(response))

# You can also swap out the vector store type

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])