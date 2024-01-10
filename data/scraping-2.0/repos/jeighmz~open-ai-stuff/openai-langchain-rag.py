
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(temperature = 0)

doc = 'https://arxiv.org/pdf/2310.11616.pdf'
loader = WebBaseLoader(doc)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()

db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

tool = create_retriever_tool(
    retriever, 
    "search_document",
    "Searches and returns documents regarding the query."
)
tools = [tool]

agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

result = agent_executor({"input": "hi, im bob"})

while True:
    user_input = input("User: ")
    if user_input == "exit":
        break
    result = agent_executor({"input": user_input})