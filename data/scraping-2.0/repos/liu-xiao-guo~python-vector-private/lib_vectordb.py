import os
from config import Config

## for vector store
from langchain.vectorstores import ElasticVectorSearch

def setup_vectordb(hf,index_name):
    # Elasticsearch URL setup
    print(">> Prep. Elasticsearch config setup")
    # endpoint = os.getenv('ES_SERVER', 'ERROR') 
    # username = os.getenv('ES_USERNAME', 'ERROR') 
    # password = os.getenv('ES_PASSWORD', 'ERROR')
    
    with open('simple.cfg') as f:
        cfg = Config(f)
    
    # print(cfg['ES_FINGERPRINT'])
    # print(cfg['ES_PASSWORD'])  
    endpoint = cfg['ES_SERVER']
    username = "elastic"
    password = cfg['ES_PASSWORD']
    
    ssl_verify = {
        "verify_certs": True,
        "basic_auth": (username, password),
        "ca_certs": "./python_es_client.pem",
    }

    url = f"https://{username}:{password}@{endpoint}:9200"

    return ElasticVectorSearch( embedding = hf, 
                                elasticsearch_url = url, 
                                index_name = index_name, 
                                ssl_verify = ssl_verify), url