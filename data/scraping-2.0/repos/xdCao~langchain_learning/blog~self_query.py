import configparser
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config.get('api', 'openai_api_key')
openai_api_base = config.get('api', 'openai_api_base')

loaders = [
    # PyPDFLoader("docs/服务端开发与面试知识手册.pdf"),
    # PyPDFLoader("docs/服务端开发与面试知识手册.pdf"),
    # PyPDFLoader("docs/Go语言编程.pdf"),
    PyPDFLoader(file_path="/Users/caohao/Downloads/1_个人简历.pdf"),
    PyPDFLoader(file_path="docs/聊一聊商品发布.pdf")
]

docs = []
for loader in loaders:
    docs.extend(loader.load())

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30
)

trunks = r_splitter.split_documents(documents=docs)
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, openai_api_base=openai_api_base)

vector_db = Chroma.from_documents(
    documents=trunks,
    embedding=embedding,
    persist_directory="docs/chroma/"
)
vector_db.persist()
# 定义元数据的过滤条件
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="查找的内容必须来自 `/Users/caohao/Downloads/1_个人简历.pdf`",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="查找内容",
        type="integer",
    ),
]

# 创建SelfQueryRetriever
document_content_description = "个人简历"
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, openai_api_base=openai_api_base)
retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True
)

compressor = LLMChainExtractor.from_llm(llm)
compress_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 问题
question = "不苦在国际化中台做了什么？"

# 搜索相关文档
docs = compress_retriever.get_relevant_documents(question)

# 打印结果中的元数据信息
for d in docs:
    print(d)
