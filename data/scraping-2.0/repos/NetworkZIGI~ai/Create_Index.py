import glob
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import re
import os
import sys
import envinfo
from tenacity import retry, wait_random_exponential, stop_after_attempt 
import json  
import openai  
from dotenv import load_dotenv  
from tenacity import retry, wait_random_exponential, stop_after_attempt  
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,   
    SearchField,  
    VectorSearch,  
    VectorSearchAlgorithmConfiguration,  
) 


##### 환경 변수 설정 ######
openai.api_type = "azure"  
openai.api_version = '2023-05-15'
openai.api_key = envinfo.openai_api_key
openai.api_base = envinfo.openai_api_base
service_endpoint = envinfo.cg_endpoint
key = envinfo.cg_key
credential = AzureKeyCredential(key)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def generate_embeddings(text):
    response = openai.Embedding.create(
        input=text, engine="text-embedding-ada-002")
    embeddings = response['data'][0]['embedding']
    return embeddings

def index_key_omit_letter(filename):
    return re.sub(r'[^a-zA-Z0-9_=-]', '', filename)

################# 문서를 Chunk로 나누기  #################
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=400)

#################  파일 불러오기  ##################
directory = '파일 디렉토리'
pdf_files = glob.glob(os.path.join(directory, '*.pdf'))
print(pdf_files)
contents_final = []
cnt=1
### TEXT 파일 Split ####
if(len(pdf_files)==0):
    print("처리 할 파일이 없습니다.")
    sys.exit()

print("Pdf 파일 Split 시작")
files_len = len(pdf_files)
for i, pdf_file in enumerate(pdf_files):
    print("{}파일 Split 처리 중{}/{}".format(pdf_file[len(directory)+1:],str(i+1),str(files_len)))
    print(str(cnt) +" : " + pdf_file)
    cnt=cnt+1
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    contents_final.extend(text_splitter.split_documents(documents))
    print("{}파일 Split 처리 완료{}/{}".format(pdf_file[len(directory)+1:],str(i+1),str(files_len)))    
print("pdf 파일 Chunking 완료")

#Congitive Serarch에 업로드하기 위한 문서 List 
input_data = []

# Generate embeddings for title and content fields
cnt = 1

print("Vector 데이터 생성 시작")
chunk_len = len(contents_final)
for i, chunk in enumerate(contents_final):
    print("{}파일 Embedding 처리 중{}/{}".format(chunk.metadata['source'][len(directory)+1:],str(i+1),str(chunk_len)))
    item = {}
    item["id"] = index_key_omit_letter(chunk.metadata['source']) + str(cnt)
    cnt = cnt+1
    
    item["source"] = chunk.metadata['source'].replace(directory+"/", '')
    item["Content"] = chunk.page_content
    item["content_vector"] = generate_embeddings(chunk.page_content)
    input_data.append(item)
    print("{}파일 Embedding 완료{}/{}".format(chunk.metadata['source'][len(directory)+1:],str(i+1),str(chunk_len)))

print("Vector 데이터 생성 완료")


# Output embeddings to docVectors.json file
embedding_file = "Embedding 파일명"
with open(embedding_file, "w",encoding='utf-8') as f:
    json.dump(input_data, f)
print("임베딩 파일 생성 완료")

######## Cognitive Search Index 생성 #######
load_dotenv()  
index_name = "zigi-index"

index_client = SearchIndexClient(
    endpoint=service_endpoint, credential=credential)
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="source", type=SearchFieldDataType.String,searchable=True, retrievable=True),
    SearchableField(name="Content", type=SearchFieldDataType.String,searchable=True, retrievable=True),
    SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, vector_search_dimensions=1536, vector_search_configuration="zigi-vector-config"),
]

vector_search = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(
            name="zigi-vector-config",
            kind="hnsw",
            hnsw_parameters={
                "m": 4,
                "efConstruction": 400,
                "efSearch": 1000,
                "metric": "cosine"
            }
        )
    ]
)

index = SearchIndex(name=index_name, 
                    fields=fields, 
                    vector_search=vector_search) 
result = index_client.create_or_update_index(index)

print(f' {result.name} created')

############### Document Uload  ############
print("Cognitive Search에 Document 업로드 시작")

with open(embedding_file, 'r',encoding='utf-8') as file:  
    documents = json.load(file)  
search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
result = search_client.upload_documents(documents)  
print(f"Uploaded {len(documents)+1} documents") 

print("Cognitive Search에 Document 업로드 완료")
