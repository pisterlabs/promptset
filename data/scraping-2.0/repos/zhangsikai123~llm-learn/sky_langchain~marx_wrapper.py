from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from sky_langchain.models.chatgpt import chatgpt_three_point_five_turbo
from sky_langchain.wrapper import ChatChainWrapper

llm = chatgpt_three_point_five_turbo

embedding = OpenAIEmbeddings()

url = "https://www.marxists.org/chinese/marx/capital/01.htm"
domain = "https://www.marxists.org/chinese/marx/capital/"


loader = WebBaseLoader(url)
# Split documents


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = loader.load()

splits = text_splitter.split_documents(documents)
# Embed and store splits


vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

wrapper = ChatChainWrapper(llm, retriever, chinese=True)
wrapper.run()
