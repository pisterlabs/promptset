import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm =  OpenAI(temperature=.1, verbose=True)
loader = PyPDFLoader('apple.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embedding=OpenAIEmbeddings(model='davinci'),
                                 collection_name='annualreport')

# convert the document vector store into something LangChain can read
vectorstore_info = VectorStoreInfo(
    name="apple",
    description="Apple quarterly consolidated financials",
    vectorstore=store,
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

prompt = input("Enter your search term> ")
response = agent_executor.run(prompt)
print(response)