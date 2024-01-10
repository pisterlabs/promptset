import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
import hashlib
from IPython.display import Markdown, display
import json
from pathlib import Path
import os
import textwrap
from typing import List, Union
from llama_index.schema import Document
from llama_index import (
    ServiceContext,
    VectorStoreIndex
)
import re
import json
from llama_index.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from llama_index import LLMPredictor, ServiceContext
from llama_index.callbacks import CallbackManager, OpenInferenceCallbackHandler
from llama_index.callbacks.open_inference_callback import (
    as_dataframe,
    QueryData,
    NodeData,
)
from llama_index import (
    Prompt,
    get_response_synthesizer,
    load_index_from_storage,
    StorageContext
)
from llama_index.node_parser import SimpleNodeParser
import pandas as pd
from tqdm import tqdm
from llama_index import download_loader

class ParquetCallback:
    def __init__(
        self, data_path: Union[str, Path], max_buffer_length: int = 1000
    ):
        self._data_path = Path(data_path)
        self._data_path.mkdir(parents=True, exist_ok=False)
        self._max_buffer_length = max_buffer_length
        self._batch_index = 0

    def __call__(
        self,
        query_data_buffer: List[QueryData],
        node_data_buffer: List[NodeData],
    ) -> None:
        if len(query_data_buffer) > self._max_buffer_length:
            query_dataframe = as_dataframe(query_data_buffer)
            file_path = self._data_path / f"log-{self._batch_index}.parquet"
            query_dataframe.to_parquet(file_path)
            self._batch_index += 1
            query_data_buffer.clear()  # ⚠️ clear the buffer or it will keep growing forever!
            node_data_buffer.clear() 

response_template = """
## Question

{question}


## Answer
```
{response}
```

"""

template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
    "Find information from different urls. \n"
    "Find top 10 distinct related url which related to query. Rename field as results. \n"
    "Find a webp image url among related urls. Name field as image_url. \n"
    "Reduce each document to url and title and one small sentence. Rename snippet to sentence. \n"
    "Result should includes url, tittle, sentence."
    "Return the response in json format.\n"
)
            
def get_search_query_engine(path:str="demo-graz.parquet"):
    data_path = f"./prototype-search-application/resources/parquet/{path}"
    callback_handler = OpenInferenceCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.9, model_name="gpt-4"))
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        callback_manager=callback_manager
    )

    try:
        folder_name = re.sub('[^A-Za-z0-9]+', '_', path.strip())
        storage_context = StorageContext.from_defaults(persist_dir= f"./storage/index/{folder_name}")
        index = load_index_from_storage(storage_context)
        index_loaded = True
        print("Index loaded.")
    except Exception as e:
        print("Index not found.")
        index_loaded = False
        raise e
        
        
    if not index_loaded:
        print("Loading documents...")
        SimpleWebPageReader = download_loader("SimpleWebPageReader")
        loader = SimpleWebPageReader()
        
        df = pd.read_parquet(data_path)
        df['warc_date']=df['warc_date'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        docs = []
        for index, row in df.iterrows():
            metadata = dict(row)
            metadata['plain_text'] = ""
            docs.append(Document(doc_id= row["id"], text=row["plain_text"],metadata=metadata ))
        parser = SimpleNodeParser.from_defaults(chunk_size=1024)
        nodes = parser.get_nodes_from_documents(docs,show_progress=True)
        print("Building index...")
        index = VectorStoreIndex(
            nodes, service_context=service_context,show_progress=True
        )
    folder_name = re.sub('[^A-Za-z0-9]+', '_', path.strip())
    index.storage_context.persist(persist_dir= f"./storage/index/{folder_name}")
    qa_template = PromptTemplate(template)
    response_synthesizer = get_response_synthesizer(text_qa_template= qa_template)
    query_engine = index.as_query_engine(similarity_top_k = 3,response_synthesizer=response_synthesizer)
    return query_engine


def query(query: str, query_engine, is_api_results=True):
    response_md = query_engine.query(query)
    data= json.loads(str(response_md))["results"]
    image_url=""
    #image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/17-05-16-Graz_Rathaus-aDSC_1282.jpg/2560px-17-05-16-Graz_Rathaus-aDSC_1282.jpg"
    if is_api_results:
        data = data+ get_from_api(query)
    founded_webpages =[]
    final_results = []
    for r in data:
        if r["url"] not in founded_webpages:
            founded_webpages.append(r["url"])
            final_results.append(r)
            
    if "image_url" in data:
        image_url = data["image_url"]
            
    print(data)
    return final_results,image_url

def convert_to_mark_down_result(result,image_url):
    markdown_content = "Graz, city in Austria\n\n"
    markdown_content += f'<img src="{image_url}" alt="drawing" width="200"/>\n\n'
    markdown_content += "\n-------------\n"
    for item in result:
        markdown_content += f"[{item['title']}]({item['url']})\n- {item['sentence']}\n\n"
        markdown_content += "\n-------------\n"
    return markdown_content


def get_from_api(q:str):
    import requests
    from urllib.parse import quote
    q2 = quote(q)
    url = f"http://localhost:8000/search?q={q2}&index=demo-graz&lang=en&ranking=asc&limit=10"
    response = requests.get(url)
    data = []
    # Ensure the request was successful
    if response.status_code == 200:
        data = response.json()['results']
    else:
        print(f"Request failed with status code {response.status_code}")

    for item in data:
        item['sentence'] = item.pop('textSnippet')
        q_t = q.strip("\'")
        item['title'] = f"{q_t} - {item['url']}"
        
    return data