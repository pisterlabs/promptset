import os

import pinecone
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone

load_dotenv()

# initialize pinecone and openai
index_name = os.environ["PINECONE_INDEX"]
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],
    environment=os.environ["PINECONE_ENV"],
)
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# retrieve vector store
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# Query those docs to get the answer back
llm = OpenAI(temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])
chain = load_qa_chain(llm, chain_type="stuff")

query_qa = input("Enter a question: ")
# Search documents with semantic similarity
docs = docsearch.similarity_search(query_qa)

# Get answer from the embedded documents and the question
answer = chain.run(input_documents=docs, question=query_qa)
print(answer)
