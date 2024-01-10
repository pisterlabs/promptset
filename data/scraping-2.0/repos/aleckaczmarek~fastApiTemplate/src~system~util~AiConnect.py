 
import os
from dotenv import load_dotenv
load_dotenv()
from fastapi import HTTPException 
from langchain.llms import Ollama

from system.transporters.Result import Result
from system.util.HttpUtils import  handleError

    # ollama = Ollama(base_url='http://localhost:11434',model="llama2")
    # print(ollama("why is the sky blue"))
class AiConnect(): 
    # ie 'http://127.0.0.1:2222'
     def __init__(self, llm_model_name):
         self.llm_model_name=llm_model_name
         llm_server_base_url=os.getenv('BASE_AI_URL') 
         self.llm_server_base_url=llm_server_base_url
         self.ollama = Ollama(base_url=llm_server_base_url,model=llm_model_name)
        #  self.connectToAi(llm_server_base_url,llm_model_name)
     
     async def connectToAi(self,llm_server_base_url,llm_model_name):
         try:
            self.ollama = Ollama(base_url=llm_server_base_url,model=llm_model_name)
         except  (Exception) as error: 
            return await handleError(error, "[ Ai Connect ] ") 
    
     async def getResponse(self,prompt):
         try:
            return  self.ollama(prompt)
         except  (Exception) as error: 
            return await handleError(error, "[ Ai Connect ] ") 
         
       
     