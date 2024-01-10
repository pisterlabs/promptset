from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from torch.mps import empty_cache
import torch
class GLM(LLM):
    max_token: int = 2048
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024
    def __init__(self):
        super().__init__()
    @property
    def _llm_type(self) -> str:
        return "GLM"
    def load_model(self, llm_device="gpu",model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True).half().cuda()
    def _call(self,prompt:str,history:List[str] = [],stop: Optional[List[str]] = None):
        response, _ = self.model.chat(
                    self.tokenizer,prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,temperature=self.temperature,
                    top_p=self.top_p)
        return response
    

import sys
modelpath = "./chatglm2-6b/"
sys.path.append(modelpath)
llm = GLM()
llm.load_model(model_name_or_path = modelpath)

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import BSHTMLLoader
text_loader_kwargs={'open_encoding': 'utf-8'}
loader = DirectoryLoader("./financial_text/html/", glob="**/*.html", show_progress=True,loader_cls=BSHTMLLoader, loader_kwargs=text_loader_kwargs)
docs = loader.load()


from langchain.text_splitter import CharacterTextSplitter

# Load the document, split it into chunks, embed each chunk and load it into the vector store.
# raw_documents = TextLoader('../../../state_of_the_union.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
documents = text_splitter.split_documents(docs)

from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# 中文Wikipedia数据导入示例：
embedding_model_name = 'WangZeJun/simbert-base-chinese'
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name,
                                                model_kwargs={'device': 'cuda'})

db = Chroma.from_documents(documents, embeddings)

retriever = db.as_retriever()