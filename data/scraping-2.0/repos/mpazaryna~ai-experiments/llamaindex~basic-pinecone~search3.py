import os

import pinecone
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

# Load the stored environment variables
load_dotenv()

PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model_name = "text-embedding-ada-002"
text_field = "input"

index_name = PINECONE_INDEX  # Replace with your index name
namespace = "flow-docs"  # Replace with your namespace name

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)
vectorstore = Pinecone(
    pinecone.Index(index_name), embed.embed_query, text_field, namespace
)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo", temperature=0.0
)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Enter your query here.
query = "Tell me about Boston Globe and the Times"
answer = qa.run(query)

print("Question: ", query)
print("----")
print("Answer: ", answer)
