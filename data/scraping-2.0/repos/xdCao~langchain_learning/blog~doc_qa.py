import configparser
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

loader = PyPDFLoader(file_path="/Users/caohao/Downloads/1_个人简历.pdf")
content = loader.load()

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)

db = DocArrayInMemorySearch.from_documents(
    content,
    embeddings
)

retriever = db.as_retriever()

qa_stuff = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.0, openai_api_key=openai_api_key, openai_api_base=openai_api_base),
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)
query = "不苦有几段工作经历,详细介绍下分别做了什么"

# 通过索引进行查询
response = qa_stuff.run(query)

print(response)
