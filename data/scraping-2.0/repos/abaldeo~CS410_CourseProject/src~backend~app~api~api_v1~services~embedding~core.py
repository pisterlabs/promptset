import functools
import g4f
from g4f import Provider, models
from langchain.llms.base import LLM

from langchain_g4f import G4FLLM

from langchain.document_loaders import (UnstructuredPDFLoader,
                                        UnstructuredPowerPointLoader,
                                        TextLoader,
                                        DirectoryLoader)
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Zilliz
from langchain.prompts.prompt import PromptTemplate

# from dotenv import load_dotenv
import os
from app.core.config import settings, get_settings 
# load_dotenv()
from loguru import logger

ZILLIZ_CLOUD_COLLECTION_NAME = settings.ZILLIZ_CLOUD_COLLECTION_NAME
ZILLIZ_CLOUD_URI = settings.ZILLIZ_CLOUD_URI
ZILLIZ_CLOUD_API_KEY = settings.ZILLIZ_CLOUD_API_KEY

# print(ZILLIZ_CLOUD_COLLECTION_NAME)
# print(ZILLIZ_CLOUD_URI)
# print(ZILLIZ_CLOUD_API_KEY)

AWS_REGION_NAME = settings.AWS_REGION_NAME
S3_ENDPOINT_URL = settings.S3_ENDPOINT_URL
AWS_SECRET_ACCESS_KEY = settings.AWS_SECRET_ACCESS_KEY
AWS_ACCESS_KEY_ID = settings.AWS_ACCESS_KEY_ID


def create_vectorstore(embedding_model, recreate=False):
    vectorstore = Zilliz(
        embedding_model,
        collection_name=ZILLIZ_CLOUD_COLLECTION_NAME,
        drop_old=recreate,
        connection_args={
            "uri": ZILLIZ_CLOUD_URI,
            # "user": ZILLIZ_CLOUD_USERNAME,
            # "password": ZILLIZ_CLOUD_PASSWORD,
            "token": ZILLIZ_CLOUD_API_KEY,            
            "secure": False,
        },
        # On Zilliz Cloud
        # index_params = {
        #     # Always set this to AUTOINDEX or just omit it.
        #     "index_type": "AUTOINDEX",            
        #     # Default to IP. This is the only parameter you should think about.
        #     "metric_type": "COSINE",
        #     # No need to set `params` 
        #     # "params": {},
        # },
    )


    return vectorstore


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".pdf": (UnstructuredPDFLoader, {}),
	#".pdf": (PyMuPDFLoader, {}),
    # ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

# Loading pdf, docx, and txt files 
def load_document(file):
    import os 
    name, ext = os.path.splitext(file)

    if ext == ".pdf":
        from langchain.document_loaders import UnstructuredPDFLoader 
        loader = UnstructuredPDFLoader(file)
    elif ext == ".ppt" or ext == ".pptx":
        from langchain.document_loaders import UnstructuredPowerPointLoader
        loader = UnstructuredPowerPointLoader(file)
    elif ext == ".txt":
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file, encoding="utf-8")
    else:
        return None

    data = loader.load()
    return data


def split_s3_url(s3_url):
    from urllib.parse import urlparse
    parsed_url = urlparse(s3_url, allow_fragments=False)
    bucket_name = parsed_url.netloc
    s3_path = parsed_url.path.lstrip('/')
    return bucket_name, s3_path

def load_s3_file(file_name, bucket_name=settings.S3_BUCKET_NAME):
    from langchain.document_loaders import S3FileLoader
    if file_name.startswith('s3://') or not bucket_name:
        bucket_name,file_name = split_s3_url(file_name)
    # print(bucket_name, file_name)
    loader = S3FileLoader(bucket_name, file_name, 
                          region_name=AWS_REGION_NAME, endpoint_url=S3_ENDPOINT_URL,
                          aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return loader.load()

def create_s3_directory_loader(directory_path, bucket_name=settings.S3_BUCKET_NAME):
    from langchain.document_loaders import S3DirectoryLoader
    settings = get_settings()
    if directory_path.startswith('s3://'):
        bucket_name,file_path = split_s3_url(directory_path)
        directory_path  = os.path.dirname(file_path)
    # print(directory_path, bucket_name)
    return S3DirectoryLoader(
        prefix=directory_path,
        bucket=bucket_name,
        region_name=settings.AWS_REGION_NAME,
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,  
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
    )
    

def create_directory_loader(file_type, directory_path):
    if directory_path.startswith('s3://'):
        return create_s3_directory_loader(directory_path)
    loader_class, loader_kwargs = LOADER_MAPPING[file_type]    
    return DirectoryLoader(
        path=directory_path,
        glob=f"**/*{file_type}",
        loader_cls=loader_class,
        loader_kwargs=loader_kwargs,
        use_multithreading=True, 
        show_progress=True
    )

def get_loaders(dir):
  if dir.startswith('s3://'):
     return [create_s3_directory_loader(dir)]
  extensions=LOADER_MAPPING.keys()  
  return [create_directory_loader(extension,dir) for extension in extensions]


def clean_text(text):

    import string, re 
    # from nltk.corpus import stopwords
    # Lowercase the text
    text = text.lower().strip()
    text =  text.replace('[sound]','')
    text =  text.replace('[music]','')
    # text =  text.replace('[inaudible]','')
    
   # Remove extra spaces/lines
    text = re.sub(r'[ |\t]+', ' ', text)
    text = re.sub(r"\n+", "\n", text)

    # lines = (line.strip() for line in text.splitlines())
    # text = " ".join(iter(lines))
    text = "\n".join((line for line in text.splitlines() if line.strip()))


    punc_chars = string.punctuation.replace('.','')
    # remove punctuations
    text = ''.join(c for c in text if c not in punc_chars)
    
    # # Remove stop words
    # stop_words = set(stopwords.words('english'))
    # text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

def post_clean_text(text):
    import re 
    lines = (line.strip() for line in text.splitlines())
    text = " ".join(iter(lines))
    text = re.sub(r"\n+", " ", text)
    return text 

def get_text_splitter( chunk_size=1000, chunk_overlap=200, **kwargs):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, 
                                                   keep_separator=False, 
                                                   length_function = len, add_start_index = True, **kwargs)
    return text_splitter


def get_text_splitter2(model_name, chunk_size=500, chunk_overlap=20, **kwargs):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from app.api.api_v1.services.embedding.token_count import get_encoder_for_model
    encoding = get_encoder_for_model(model_name).name
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding, keep_separator=False,
                                                   chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                   add_start_index = True,  **kwargs)
    return text_splitter

def get_token_splitter(model_name, chunk_size=500, chunk_overlap=20, **kwargs): 
    from langchain.text_splitter import TokenTextSplitter 
    from app.api.api_v1.services.embedding.token_count import get_encoder_for_model
    encoder = get_encoder_for_model(model_name)
    encoding_name = encoder.name 
    token_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,encoding_name=encoding_name, **kwargs )
    return token_splitter

# Chunk data
def chunk_docs(docs,text_splitter, clean=True, post_clean=True):
    if not isinstance(docs, list): docs = [docs]  
    if clean:  clean_documents(docs)          
    chunks = text_splitter.split_documents(docs)
    add_chunk_metadata(chunks)
    if post_clean:  clean_documents(chunks, post_clean_text)    
    return chunks

def add_chunk_metadata(chunks):
    number = 1
    current_doc = ''
    for chunk in chunks:
        if current_doc != chunk.metadata.get('document_id',''):
            number = 1
        current_doc = chunk.metadata.get('document_id', '')
        chunk.metadata['chunk_id'] = get_hash_id(chunk.page_content)
        chunk.metadata['chunk_number'] = number
        number+=1
    

def chunk_texts(texts, text_splitter, clean=True,post_clean=True):
    if clean: texts = clean_text(texts)        
    if not isinstance(texts, list): texts = [texts]            
    chunks = text_splitter.create_documents(texts)
    if post_clean: clean_documents(chunks, post_clean_text)   
    return chunks

# def clean_texts(texts):
#     texts = [clean_text(text) for text in texts]
#     return texts

def clean_documents(documents, clean_func=clean_text):
    # Iterate over texts page_content category with this cleaning method.
    for doc in documents:
        doc.page_content = clean_func(doc.page_content)


@functools.lru_cache()
def get_embedding_model(model_name="BAAI/bge-base-en-v1.5" ):
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return hf

def create_doc_embeddings(doc_chunks, embedding_model):
    doc_text = [doc.page_content for doc in doc_chunks]
    return embedding_model.embed_documents(doc_text)
 
def create_text_embeddings(text_chunks, embedding_model):
    return [embedding_model.embed_query(chunk) for chunk in text_chunks]
 
def create_query_embeddings(query_text, embedding_model):
    return embedding_model.embed_query(query_text)

def load_all_docs(directory_path):
    loaders = get_loaders(directory_path)
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    return documents


def bulk_load_n_split_docs(directory_path="/workspace/CS410_CourseProject/src/backend/data/transcripts/",
                     file_ext=".txt",
                     gpt_model=settings.GPT_MODEL_NAME, 
                     split_seperators=None):
    if not gpt_model: gpt_model = 'gpt-3.5-turbo'
    if not split_seperators: split_seperators = ['.','\n','\n\n']
    if not file_ext:
        documents = load_all_docs(directory_path)
    else:
        loader = create_directory_loader(file_ext, directory_path)
        documents = loader.load()
    text_splitter = get_text_splitter2(gpt_model,separators=split_seperators)
    chunks = chunk_docs(documents, text_splitter)
    return chunks 

def ingest_documents(directory_path="/workspace/CS410_CourseProject/src/backend/data/transcripts/",
                     file_ext=".txt",
                     gpt_model='gpt-3.5-turbo', 
                     embedding_model="BAAI/bge-base-en-v1.5",
                     recreate_vectorstore=False,
                     split_seperators=None):
    from app.api.api_v1.services.embedding.utils import timer
    
    if not split_seperators: split_seperators = ['.','\n','\n\n']
    loader = create_directory_loader(file_ext, directory_path)
    documents = loader.load()
    text_splitter = get_text_splitter2(gpt_model,separators=split_seperators)
    chunks = chunk_docs(documents, text_splitter)
    model = get_embedding_model(embedding_model)
    vector_db = create_vectorstore(embedding_model=model, recreate=recreate_vectorstore)
    with timer() as t:
        vector_db.add_documents(chunks)    

def extract_metadata_from_filename(filename):
    import re 
     # Split the filename into parts using hyphen as the separator
    # parts = filename.split("-")

    # Extract the lecture ID from the second part
    try:
        match = re.search(r"(\d+-\d+)", filename)
        lecture_number  = match.group(0).replace('-','.').strip()
    except Exception:
        logger.warning(f"Could not extract lecture number from filename {filename}")
        lecture_number = 'n/a'

    # Extract the week number from the first part
    try:
        match = re.search(r"(\d+)_", filename)
        week_number = match.group(0).replace("_", "").strip()
        if lecture_number != 'n/a':
            lecture_week  =lecture_number.split(".")[0]
            if int(week_number) < int(lecture_week):
                week_number = lecture_week
    except Exception:
        logger.warning("Could not extract week number from filename {}", filename)
        week_number = "n/a"

    # Extract the lecture name from the remaining parts
    try:
        # lecture_title= " ".join(parts[4:]).split(".")[0]
        # lecture_title = lecture_title.title()
        match = re.search(r'-(\D+).+\.', filename)
        lecture_title = match.group(0).replace('-',' ').strip().title()
        lecture_title = lecture_title.split('.')[0]
    except Exception:
        logger.warning(f"Could not extract lecture name from filename {filename}")
        lecture_title = 'n/a'
    # Return a tuple of the week number, lecture ID, and lecture name
    return (week_number, lecture_number, lecture_title)


def get_hash_id(text: str):
    import hashlib 
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def enrich_document_metadata(documents):
    for doc in documents:
        metadata = doc.metadata
        metadata['document_id'] = metadata.get('id',  get_hash_id(doc.page_content) )
        doc.metadata["week_number"] = ''
        doc.metadata["lecture_number"] = ''
        doc.metadata["lecture_title"] = ''

        source = metadata.get('source')
        if source:
            filename = os.path.basename(source)
            ext = os.path.splitext(filename)[-1]
            if ext == ".txt":
                week_number, lecture_number, lecture_title = extract_metadata_from_filename(filename)
                doc.metadata['week_number'] = week_number
                doc.metadata['lecture_number'] = lecture_number
                doc.metadata['lecture_title'] = lecture_title


def main2():
    # import textwrap 
    # file='/workspace/CS410_CourseProject/src/backend/data/transcripts/01_10-1-text-clustering-motivation.en.txt'
    # doc = load_document(file)
    # print(doc)
        
    # directory_path = '/workspace/CS410_CourseProject/src/backend/data/transcripts/'
    # loader = DirectoryLoader(directory_path, glob="**/*txt", loader_cls=TextLoader, 
    #                                                             loader_kwargs={"encoding": "utf-8"}, use_multithreading=True,
    #                                                             show_progress=False)

    # documents = loader.load()
    # enrich_document_metadata(documents)
    # # for doc in documents:
    # #     print(doc.metadata)

  
    # # text_splitter = get_text_splitter(chunk_size=1000, chunk_overlap=200, separators=['.','\n','\n\n'])
    # text_splitter = get_text_splitter2("gpt4",separators=['.','\n','\n\n'])
    # # text_splitter = get_token_splitter("gpt4")
    # # chunks = chunk_texts(doc[0].page_content, text_splitter)
    # chunks = chunk_docs(documents, text_splitter)
    # # print(chunks)
    # # [print(textwrap.fill(chunk.page_content, 120) +  '\n' + '-'*20) for chunk in chunks]
    
    model = get_embedding_model()
    # print(model.model_name)
    
    # from app.api.api_v1.services.embedding.utils import timer
    # # with timer() as t:
    # #     vectors = create_doc_embeddings(chunks, model)
    # # print(len(vectors))
    
    vector_db = create_vectorstore(embedding_model=model, recreate=False)
    
    # with timer() as t:
    #     vector_db.add_documents(chunks)
    
    print(vector_db.similarity_search("text clustering", top_k=5))
        
    # g4f.debug.logging = False # enable logging
    # g4f.check_version = False # Disable automatic version checking
    # print(g4f.version) # check version
    # print(g4f.Provider.Ails.params)  # supported args

    # streamed completion
    # RESPONSE = g4f.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": "Hello"}],
    #     stream=True,
    # )

    # for message in RESPONSE:
    #     print(message, flush=True, end="")

    # llm: LLM = get_llm()

    # res = llm("hello")
    # print(res)  # Hello! How can I assist you today?
    # qa = create_qa_chain_with_sources(llm, vector_db)
    # answer=qa({"question":"What is text clustering"})
    # print(format_qa_response(answer))

def main():
    # file='s3://coursebuddy/cs410/slides/01_7-1-overview-text-mining-and-analytics-part-1_TM-3-overview.pdf'
    # file='cs410/transcripts/01_7-1-overview-text-mining-and-analytics-part-1.en.txt'
    # print(load_s3_file(file))
    # loader = create_directory_loader('','s3://coursebuddy/cs410/transcripts/')
    # loader = create_directory_loader('.pdf', '/workspaces/CS410_CourseProject/src/backend/data/slides/')
    docs = bulk_load_n_split_docs('s3://coursebuddy/cs410/')
    print(len(docs))

if __name__ == "__main__":
    main() 