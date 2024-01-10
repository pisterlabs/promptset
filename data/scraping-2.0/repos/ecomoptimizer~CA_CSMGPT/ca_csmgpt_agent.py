# ca_csmgpt_agent.py

from ca_csmgpt.agents import ca_csmgpt 
from langchain.chat_models import ChatLiteLLM

def create_agent(config_path, verbose, max_num_turns):
   llm = ChatLiteLLM(temperature=0.2)  
   # existing agent creation logic
   ...
   return agent