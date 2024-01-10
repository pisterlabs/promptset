import os
import logging
import json
import openai
#from llama_index import SimpleDirectoryReader, VectorStoreIndex, LLMPredictor, ServiceContext, GPTVectorStoreIndex
from llama_index import SimpleDirectoryReader, GPTListIndex, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
logger = logging.getLogger("bot")
logger.setLevel("DEBUG")

class OpenAiHelper:
    def __init__(self, token):
        logging.info(f"Initializing OpenAI helper. Selected model: gpt3 di llamaindex")
        os.environ["OPENAI_API_KEY"] = token
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.max_input_size = 4096
        self.num_outputs = 512
        self.max_chunk_overlap_ratio = 0.1
        self.chunk_size_limit = 600
        
        #self.temperature = temperature
        #self.max_tokens = max_tokens
        #self.model = model
        #self.llm_predictor = LLMPredictor(llm=self)
        #self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor)

    def get_response(self, message_text):
        try:
            logging.info(f"Getting response from OpenAI. Message: {message_text}")
            
            #### CODICE CHE FUNZIONA #####
            
            prompt_helper = PromptHelper(self.max_input_size, self.num_outputs, self.max_chunk_overlap_ratio, chunk_size_limit= self.chunk_size_limit)
            llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-4", max_tokens=self.num_outputs))
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

            loader = SimpleDirectoryReader('villa_romana_nonni_arii', recursive=True, exclude_hidden=True)
            documents = loader.load_data()
            
            index = GPTVectorStoreIndex(documents, service_context=service_context, prompt_helper=prompt_helper)
            
            query_engine = index.as_query_engine(vector_store_query_mode="default")
            ai_response = query_engine.query(message_text)
            
            ##### FINE CODICE CHE FUNZIONA #######
            
            return str(ai_response)
            
        except Exception as e:
            logging.error(f"Failed to get response from OpenAI: {e}")
            raise