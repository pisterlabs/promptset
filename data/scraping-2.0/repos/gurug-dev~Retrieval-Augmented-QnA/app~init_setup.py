
import os
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

directory = "docs"
index_name = "langchain-demo"

model_name = "text-davinci-003"
# model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAPI_KEY")
os.environ["PINCONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
    environment=os.getenv("PINECONE_ENV")  # next to api key in console
)

embeddings = OpenAIEmbeddings()
llm = OpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")