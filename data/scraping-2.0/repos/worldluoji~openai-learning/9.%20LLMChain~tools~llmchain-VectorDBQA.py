
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import SpacyTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import TextLoader

llm = OpenAI(temperature=0)
# 通过一个 TextLoader 把文件加载进来，还通过 SpacyTextSplitter 给文本分段，确保每个分出来的 Document 都是一个完整的句子
loader = TextLoader('../data/ecommerce_faq.txt')
documents = loader.load()
text_splitter = SpacyTextSplitter(chunk_size=256, pipeline="zh_core_web_sm")
texts = text_splitter.split_documents(documents)

# 使用 OpenAIEmbeddings 来给文档创建 Embedding，通过 FAISS 把它存储成一个 VectorStore。最后，我们通过 VectorDBQA 的 from_chain_type 定义了一个 LLM。
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_documents(texts, embeddings)

faq_chain = VectorDBQA.from_chain_type(llm=llm, vectorstore=docsearch, verbose=True)

question = "请问你们的货，能送到三亚吗？大概需要几天？"
result = faq_chain.run(question)
print(result)