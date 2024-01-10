import torch
from utils.embedding import load_embedding
from utils.documents import load_documents
from utils.vendors import ChromaVectorStore
from utils.llm import load_llm

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA

from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

from utils.QuestionAnswerChain import QuestionAnswerChain

prompt_template = (
  "Below is an instruction that describes a task. "
  "Write a response in chinese that appropriately completes the request.\n\n"
  "### Instruction:\n{context}\n{question}\n\n### Response: ")
prompt_template = """
Answer the question in chinese based on the context below. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer.

### Context: {context}

### Question: {question}

### Answer: """

class QA:
  
  def __init__(self, embedding_source:str, embedding_model_path: str, model_path:str=None, lora_path: str=None, model_type:str = "alpaca", device:str = "cuda") -> None:
    self.device = device
    if device == "cuda" and not torch.cuda.is_available():
      self.device = "cpu"
    
      
    self.embedding_model_path = embedding_model_path
    self.model_path = model_path
    self.lora_path = lora_path
    self.model_type = model_type
    self.chain = None
    
    #self.embedding = HuggingFaceEmbeddings(model_name=self.embedding_model_path)
    self.embedding = load_embedding(embedding_source=embedding_source, embedding_model_path=embedding_model_path)
    #self.load_model()
    self.add_documents()
  
  def add_documents(self, path: str = "./data/", pattern: str = "**/*.txt"):
    # loader = DirectoryLoader(path, pattern)
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(        
    #   chunk_size = 2000,
    #   chunk_overlap  = 100,
    # )
    # documents = text_splitter.split_documents(documents)
    documents = load_documents(path, pattern)
    #self.docsearch = Chroma.from_documents(documents=documents, embedding=self.embedding)
    #self.store = ChromaVectorStore(documents, self.embedding)
    self.store = Chroma.from_documents(documents=documents, embedding=self.embedding)
    #self.load_chain()
  
  def load_chain(self):
    PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
    
    # self.qa = RetrievalQA.from_chain_type(
    #   llm = self.llm,
    #   chain_type="stuff",
    #   retriever=self.docsearch.as_retriever(search_kwargs={"k":1}),
    #   chain_type_kwargs={"prompt": PROMPT}
    # )
    #self.qa = load_qa_chain(self.llm, chain_type="stuff", prompt=PROMPT)
    self.llm = load_llm(self.model_type, self.device)
    #self.qa = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=PROMPT)
    self.qa = RetrievalQA.from_chain_type(self.llm, chain_type="stuff", retriever=self.store.as_retriever(search_kwargs={"k":4}, search_type="mmr"), chain_type_kwargs={"prompt": PROMPT})
    #self.qa = QuestionAnswerChain(self.model_type, self.device, prompt_template)
    
  def query(self, query:str) -> str:
    if len(query.strip()) == 0:
      return ""
    
    if self.chain is None:
      self.load_chain()
    
    #similar_docs = self.store.get_similiar_docs(query, k=1)
    
    return self.qa({"query": query})
  
  def check_similar(self, query:str, keyword:str):
    similar_docs = self.store.similarity_search(query, k=3)
    #print(similar_docs)
    return keyword in str(similar_docs)