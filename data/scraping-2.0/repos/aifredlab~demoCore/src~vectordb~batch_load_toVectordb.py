from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from pymilvus import MilvusClient
#from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
import os

def loadPdf(file_path):
    # -------------------------------------
    # pdf 파일을 읽어서 chunk_size 단위로 배열로 만든다 
    # -------------------------------------
    loader = PyPDFLoader(file_path) # ex: "../doc/samsung_tooth_terms.pdf"
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs_list = text_splitter.split_documents(documents)

    # print (docs_list)
    # ex : Document(page_content='본 약관은 100% 재생펄프를 사용하여 제작한 친환경 인쇄물입니다. 장기상품개발팀 인쇄', metadata={'source': '../doc/samsung_tooth_terms.pdf', 'page': 153})

    # -------------------------------------
    # insert vector_db
    # -------------------------------------

    # - pymilvus를 사용해 vector를 저장하는 방법
    # client = MilvusClient(
    #     uri=os.environ.get('ZILLIZ_CLOUD_URI'),
    #     token=os.environ.get('ZILLIZ_CLOUD_API_KEY'), # for serverless clusters, or
    # )
    # 
    # client.insert(collection_name=COLLECTION_NAME, data=docs_list)


    # langchain api를 사용해 vector를 저장하는 방법:
    m = Milvus.from_documents(
        documents=docs_list,
        embedding=OpenAIEmbeddings(),
        connection_args={
            "uri": os.environ.get('ZILLIZ_CLOUD_URI'),
            "token": os.environ.get('ZILLIZ_CLOUD_API_KEY'), 
            "secure": True
        },
    )

    return

loadPdf("../doc/samsung_tooth_terms.pdf")
