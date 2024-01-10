
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# load chroma from disk
db = Chroma(persist_directory="chroma_db", embedding_function=OpenAIEmbeddings())

# get query
query = str(input("Query: "))

# run a similarity search
docs = db.similarity_search(query)
docs_page_content = "\n".join([docs[i].page_content for i in range(len(docs))])

# Create the prompt template
template = """Address the following query based on the provided passages: {query}
Passages: {docs_page_content}."""
prompt = PromptTemplate(template=template, input_variables=['query', 'docs_page_content'])

inputs = {'query': query, 'docs_page_content': docs_page_content}

# Create the chain
llm = OpenAI()
llm_chain = LLMChain(prompt=prompt, llm=llm)
llm_chain.run(inputs)

