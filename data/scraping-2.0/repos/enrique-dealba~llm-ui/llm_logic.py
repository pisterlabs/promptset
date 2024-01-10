import os
import config
import logging

from apikeys import open_ai_key
from prompts import db_template
from flask import jsonify, Response
from typing import Optional, Dict, Any, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms import OpenAI, HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from InstructorEmbedding import INSTRUCTOR


os.environ['OPENAI_API_KEY'] = open_ai_key

logging.basicConfig(level=logging.DEBUG)

class LLMServer:

  def __init__(self, use_local: bool = True,
               mode: str = 'vectordb-memory',
               template: str = "",
               init_model: bool = True):
    """
    Initialize the server with given mode and template.
    If init_model is True, initializes a local LLM otherwise it initializes 
    an OpenAI model (text-davinci-003 by default).
    """
    self.mode = mode
    self.tokenizer = None
    self.model = None
    self.llm = None
    self.embedding = None

    # Memory + Vector DB
    self.memory = None
    self.qa_chain = None

    self.template = template
    self.persist_directory = 'docs/chroma/'
    if init_model:
      self.initialize_model(use_local)
      self.initialize_mode()
      
  def initialize_mode(self):
    if self.mode == "non-vectordb":
      pass  # No special inits needed for this
    elif self.mode == "vectordb":
      self.init_embedding()
    elif self.mode == "vectordb-memory":
      self.init_embedding()
      self.init_qa()
    else:
      raise ValueError(f"Invalid mode: {self.mode}")

  def initialize_model(self, use_local: bool = True):
    if use_local:
      directory_path = config.MODEL_DIRECTORY_PATH
      self.tokenizer = AutoTokenizer.from_pretrained(directory_path)
      self.model = AutoModelForCausalLM.from_pretrained(directory_path)
    else:
        self.llm = OpenAI(temperature=0.2)
    
    if self.model:
      self.llm = self.initialize_local_model()
      logging.debug(f"LLM running on {self.model.device}")
        
  def initialize_local_model(self) -> HuggingFacePipeline:
    """
    Initializes a HuggingFacePipeline with the local model and its tokenizer.
    [Output] - HuggingFacePipeline: Returns a HuggingFacePipeline instance.
    """
    local_pipe = pipeline("text-generation",
                          model=self.model,
                          tokenizer=self.tokenizer,
                          max_length=500)
    return HuggingFacePipeline(pipeline=local_pipe)
    
  def init_embedding(self):
     self.embedding = HuggingFaceInstructEmbeddings()

  def init_qa(self):
    """
    Initializes a question-answering pipeline with ConversationBufferMemory,
    and Chroma (using the initialized embeddings).
    """
    if not self.embedding:
      self.init_embedding()

    self.memory = ConversationBufferMemory(memory_key="chat_history",
                                           return_messages=True)
    
    vector_db = Chroma(persist_directory=self.persist_directory,
                       embedding_function=self.embedding)
    
    self.qa_chain = ConversationalRetrievalChain.from_llm(llm=self.llm,
                                                          retriever=vector_db.as_retriever(),
                                                          memory=self.memory,
                                                          # return_source_documents=True
                                                          )

  def chat_with_model(self, prompt: str) -> Union[Tuple[Response, int], Response]:
    """
    Generates chat response from LLM.

    [Input] - prompt: The prompt (str) that the model responds to.

    [Output] - JSON: A JSON object containing either generated response
    or an error message.
    """
    if not prompt:
      return jsonify({'error': 'No prompt provided'}), 400
    
    llm_chain = LLMChain(llm=self.llm,
                         prompt=PromptTemplate(template=self.template,
                                               input_variables=["question"])
                                               )
    response = llm_chain.run(prompt)
    return jsonify({'response': response})
    
  def chat_with_db(self, prompt) -> Union[Tuple[Response, int], Response]:
    """
    Retrieves relevant context from a vector database before generating a response.
    """
    if not prompt:
      return jsonify({'error': 'No prompt provided'}), 400
    
    vector_db = Chroma(persist_directory=self.persist_directory,
                       embedding_function=self.embedding)
    
    vector_db_prompt = PromptTemplate(input_variables=['context', 'question'],
                                      template=db_template)
    
    vector_db_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                  retriever=vector_db.as_retriever(),
                                                  return_source_documents=True,
                                                  chain_type_kwargs={'prompt': vector_db_prompt}
                                                  )
    
    response = vector_db_chain({'query': prompt})['result']
    return jsonify({'response': response})
    
  def chat_with_memory(self, prompt) -> Union[Tuple[Response, int], Response]:
    """
    Uses past message history for current conversation to generate responses.
    """
    if not prompt:
      return jsonify({'error': 'No prompt provided'}), 400
        
    assert self.qa_chain is not None

    response = self.qa_chain({'question': prompt})['answer']
    return jsonify({'response': response})

  def error_handler(self, e) -> Tuple[Response, int]:
    """
    Logs errors and returns a JSON error message.
    """
    logging.error(f"Error occurred: {e}")
    return jsonify({'error': 'An error occurred while processing the request.'}), 500