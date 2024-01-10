import requests
from llama_index import SimpleDirectoryReader,download_loader
from llama_index.readers.web import SimpleWebPageReader
from llama_index import StorageContext, ServiceContext, VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.embeddings import HuggingFaceEmbedding
import os
from llama_index import set_global_service_context
from llama_index.extractors import (
    TitleExtractor,
    KeywordExtractor, EntityExtractor
)
from llama_index.ingestion import IngestionPipeline
from llama_index.text_splitter import TokenTextSplitter
from llama_index.llms import OpenAI

def save_all_files(input, output_dir):
    save_f = open(input, 'r')
    links = save_f.readlines()
    save_f.close()
    for i in links[4295:]:
        i = i[:-1]
        response = requests.get(i)
        print(i)
        if response.status_code == 200:
            name = i[34:].replace('/', '')
            print(output_dir+name+'.html')

            with open(output_dir+name+'.html', 'w', encoding='utf-8') as file:
                file.write(response.text)

def load_data(dir):
    UnstructuredReader = download_loader('UnstructuredReader')
    reader = SimpleDirectoryReader(
        input_dir='./data',
        file_extractor={".html": UnstructuredReader()}
    )
    return reader.load_data()

def build_faiss_db(documents):
    os.environ['OPENAI_API_KEY'] = os.environ.get('FLASK_APP')
    EMBED_MODEL = "jinaai/jina-embeddings-v2-small-en"
    d = 512
    #define LLM service 
    llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo", max_tokens=512)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(
    embed_model=embed_model, chunk_size=512, llm=llm)
    set_global_service_context(service_context)

    text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)

    #create metadata extractor
    title_extractor = TitleExtractor(nodes=1, llm=llm), #title is located on the first page, so pass 1 to nodes param
    keyword_extractor = KeywordExtractor(keywords=10, llm=llm) #let's extract 10 keywords for each node.
    entity_extractor = EntityExtractor(entities=10, llm=llm) #let's extract 10 entities for each node.

    pipeline = IngestionPipeline(
    transformations=[text_splitter, title_extractor, keyword_extractor, entity_extractor]
    )
    print('metadata extractor created')
    nodes = pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True,
        service_context=service_context,
    )
    vector_store.add(nodes, service_context=service_context)

    # if running for the first time, will download model weights first! 
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context,
        service_context=service_context
    )
    # save index to disk
    index.storage_context.persist(persist_dir="vectors/vector_index")
    print('index created')
        

if __name__ =="__main__":
    docs = load_data('build_knowledge/data/')
    print('data loaded')
    build_faiss_db(docs)
