from typing import Optional, Any, Dict
from operator import add
import requests, json
from starlette.requests import Request
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import WikipediaLoader
from langchain import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from transformers import pipeline as hf_pipeline

import ray
from ray import serve

class LocalHuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_id):
        self.model = SentenceTransformer(model_id)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text)
        return list(map(float, embedding))
    
class StableLMPipeline(HuggingFacePipeline): 
    # Class is temporary, we are working with the authors of LangChain to make these unnecessary.
    
    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        response = self.pipeline(prompt, temperature=0.1, max_new_tokens=256, do_sample=True)
        print(f"Response is: {response}")
        text = response[0]["generated_text"][len(prompt) :]
        return text

    @classmethod
    def from_model_id(cls, model_id: str, task: str, device: Optional[str] = None, model_kwargs: Optional[dict] = None, **kwargs: Any,):
        pipeline = hf_pipeline(model=model_id, task=task, device=device, model_kwargs=model_kwargs, )
        return cls(pipeline=pipeline, model_id=model_id, model_kwargs=model_kwargs, **kwargs, )
    
template = """
<|SYSTEM|># StableLM Tuned (Alpha version)
- You are a helpful, polite, fact-based agent for answering questions. 
- Your answers include enough detail for someone to follow through on your suggestions. 
<|USER|>
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Please answer the following question using the context provided. 

CONTEXT: 
{context}
=========
QUESTION: {question} 
ANSWER: <|ASSISTANT|>"""

PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

@serve.deployment(autoscaling_config={"min_replicas": 1, "initial_replicas": 2, "max_replicas": 5})
class ParallelBuildVectorDBDeployment:
    FAISS_INDEX_PATH = "/home/ray/faiss_dist_built_index"

    def __init__(self):
        self.embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
        try:
            self.db = FAISS.load_local(self.FAISS_INDEX_PATH, self.embeddings)
        except:
            self.setup_db()
            
    def setup_db(self):
        topics = ['The Eras Tour', '2023 XFL season']
        loaders = [WikipediaLoader(query=topic, load_max_docs=20) for topic in topics]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20, length_function=len,)
        docs = add(*[loader.load() for loader in loaders])
        print([d.metadata['title'] for d in docs])
        chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
        db_shards = 8
        print(f"Loading chunks into vector store ... using {db_shards} shards")
        shards = np.array_split(chunks, db_shards)
        
        @ray.remote
        def process_shard(shard):
            embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
            result = FAISS.from_documents(shard, embeddings)
            return result
        
        futures = [process_shard.remote(shards[i]) for i in range(db_shards)]
        results = ray.get(futures)
        self.db = results[0]
        for i in range(1, db_shards):
            self.db.merge_from(results[i])
        self.db.save_local(self.FAISS_INDEX_PATH)
        
    def similarity_search(self, query):
        return self.db.similarity_search(query)

@serve.deployment(ray_actor_options={"num_gpus": 1.0}, num_replicas=2)
class QADeployment:
    def __init__(self, db):
        self.embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
        self.db = db
        self.llm = StableLMPipeline.from_model_id(
            model_id="stabilityai/stablelm-tuned-alpha-7b",
            task="text-generation",
            model_kwargs={"torch_dtype": torch.float16, "device_map": "auto", 'cache_dir':'/mnt/local_storage'}
        )
        self.chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=PROMPT)

    async def qa(self, query):
        search_results_ref = await self.db.similarity_search.remote(query)
        search_results = await search_results_ref
        result = self.chain({"input_documents": search_results, "question": query})
        return result["output_text"]
    
    async def __call__(self, request: Request) -> Dict:
        data = await request.json()
        data = json.loads(data)
        output = await self.qa(data['user_input'])
        return {"result": output }

vecdb_deployment = ParallelBuildVectorDBDeployment.bind()
entrypoint = QADeployment.bind(vecdb_deployment)