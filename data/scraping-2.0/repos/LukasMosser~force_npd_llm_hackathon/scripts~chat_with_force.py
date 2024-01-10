from datasets import load_dataset, Value, Features, Sequence
from typing import Dict, List
from tqdm.auto import tqdm 
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv(".env.shared")
load_dotenv(".env.secret")

features = Features({'doc_id': Value('string'), 
                     'meta': Features({'_id': Value(dtype='string', id=None), 
                                               'corpus': Value(dtype='string', id=None),
                                                 'langdetect': Value(dtype='string', id=None), 
                                                 'possible_lanaguage': Value(dtype='string', id=None), 
                                                 'content_could_be_natural_language': Value(dtype='string', id=None)
                                                 }), 
                    'raw_content': Features({'Document Corpus': Value(dtype='string', id=None),
                                                 'Filename': Value(dtype='string', id=None),
                                                 'Content': Value('string')
                                                 })
})

force_llm_dataset_scrubbed = load_dataset("json", data_files="./data/force_llm_corpus_scrubbed_embedding_docs.jsonl", features=features).filter(lambda x: x['meta']["corpus"] == "NL_NOG_PR")

force_llm_dataset_scrubbed_embeddings = load_dataset("json", data_files="./data/force_llm_corpus_scrubbed_embeddings.jsonl", 
                                                     features=Features({'doc_id': Value('string'), 'embedding': Sequence(Value("float32")) }))
embedding_function = AzureOpenAIEmbeddings(api_version=os.environ["OPENAI_API_VERSION"], 
                                           base_url=os.environ["OPENAI_API_BASE"],
                                           azure_deployment=os.environ["ADA002_DEPLOYMENT"],
                                           api_key=os.environ["OPENAI_API_KEY"])

vectorstore = FAISS.from_texts(["Test"], embedding=embedding_function)
data = {}
for row in tqdm(force_llm_dataset_scrubbed["train"]):
    data[row["doc_id"]] = row

for row in tqdm(force_llm_dataset_scrubbed_embeddings["train"]):
    try:
        test = data[row["doc_id"]]
    except KeyError:
        print(row["doc_id"])
        continue
    text = data[row["doc_id"]]["raw_content"]["Content"]
    embedding = row["embedding"]
    vectorstore.add_embeddings([(text, embedding)])

print(vectorstore.similarity_search("I'm looking for data from Wintershall dea", k=10))