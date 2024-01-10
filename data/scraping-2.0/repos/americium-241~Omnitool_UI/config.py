#Models, agent, key are in ui.settings_ui
import logging
from langchain.agents import AgentType 
import os 

#Define api_keys session_state if you want to hardcode them
os.environ["OPENAI_API_KEY"] ='' 
#os.environ["HUGGINGFACEHUB_API_TOKEN"] =''

# Logger level 
LOGGER_LEVEL=logging.INFO

# Add a keys in KEYS list and a text_input will be available in settings tab UI
KEYS=["OPENAI_API_KEY","HUGGINGFACEHUB_API_TOKEN"]
# Add a model name to the list
MODELS=['gpt-3.5-turbo', 'gpt-4', 'gpt-3.5-turbo-0613']
gguf_models={'llama':{'model':'TheBloke/Llama-2-7b-Chat-GGUF','file':'llama-2-7b-chat.Q2_K.gguf'},
            'mistral':{'model':'TheBloke/Mistral-7B-v0.1-GGUF','file':'mistral-7b-v0.1.Q2_K.gguf'}}
MODELS.extend(list(gguf_models.keys()))
#Monitoring langchain.agents and new_agents folder to add to agents list
#agents=[AgentType.OPENAI_MULTI_FUNCTIONS,AgentType.OPENAI_FUNCTIONS,AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, AgentType.ZERO_SHOT_REACT_DESCRIPTION,AgentType.CONVERSATIONAL_REACT_DESCRIPTION]
AGENTS=[eval('AgentType.'+a) for a in dir(AgentType) if a.isupper()] 

#LLM temperature 
TEMPERATURE=0

# Maximum thoughts iteration per query
MAX_ITERATIONS=20 

#Document embedding chunk size
CHUNK_SIZE=500
#Similarity document search 
SIMILARITY_MAX_DOC=5

#Audio recognition 
TIMEOUT_AUDIO=10
PHRASE_TIME_LIMIT=50