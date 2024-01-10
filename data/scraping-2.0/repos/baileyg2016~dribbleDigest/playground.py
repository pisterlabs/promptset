from langchain import PromptTemplate, OpenAI, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
import json
from article import Article

from dotenv import load_dotenv

load_dotenv()

# Load the JSON data
with open('samples/sample-articles.json', 'r') as f:
  data = json.load(f)

# Create Article objects
articles = [Article(**article) for article in data]

prompt_template = "What are the top 3 most interesting articles for someone who likes the NBA?"

documents = [
  Document(
    page_content=str(article),
    metadata=article.to_dict(),
  )
  for article in articles
]

retriever = SVMRetriever.from_texts(
  [article.title for article in articles],
  OpenAIEmbeddings(),
)

embeddings = OpenAIEmbeddings()
# article_embeddings = [embeddings.(article.text) for article in articles]
vectorstore = Chroma.from_documents(documents, embeddings)

llm = OpenAI(temperature=0, model='gpt-4')
llm_chain = LLMChain(
  llm=llm,
  prompt=PromptTemplate.from_template(prompt_template),
  retriever=retriever,
)

llm_chain.run(prompt_template)





# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.agents.agent_toolkits import create_retriever_tool, create_conversational_retrieval_agent
# from langchain.chat_models import ChatOpenAI
# from langchain.agents import AgentExecutor

# # Set up the retriever
# loader = TextLoader('path/to/your/documents.txt')
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)
# embeddings = OpenAIEmbeddings()
# db = FAISS.from_documents(texts, embeddings)
# retriever = db.as_retriever()

# # Create a retriever tool
# tool = create_retriever_tool(
#     retriever,
#     "retriever_name",
#     "Description of the retriever"
# )
# tools = [tool]

# # Construct the agent
# llm = ChatOpenAI(temperature=0)
# agent_executor = create_conversational_retrieval_agent(llm, tools, verbose=True)

# # Test the agent
# result = agent_executor({"input": "your_input_here"})
