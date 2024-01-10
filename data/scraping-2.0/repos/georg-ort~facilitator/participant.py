# participant.py

# system imports
from abc import ABC, abstractmethod
from loguru import logger

# project imports
from config.config import Config
from facilitator import FacilitatorResponse
from history import History

# langchain imports
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import FileCallbackHandler


class Participant(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def respond(self, facilitator_response: FacilitatorResponse, history: History) -> str:
        pass


class HumanParticipant(Participant):
    
    def respond(self, facilitator_response: FacilitatorResponse, history: History) -> str:
        response = input(f"[\033[1;35;40m{self.name}\033[0;0m]: ")
        return response


class AIParticipant(Participant):
    def __init__(self, name, backstory, temperature=0.5):
        super().__init__(name)
        self.backstory = backstory
        
        self.llm = ChatOpenAI(temperature=temperature, model="gpt-3.5-turbo")  
        self.prompt = PromptTemplate.from_template(Config.AIPARTICIPANT_PROMPT_TEMPLATE)
        self.chain = LLMChain(prompt=self.prompt, 
                              llm=self.llm, 
                              verbose=True if Config.LOGGING_LEVEL > 1 else False )
        
    # Generate the response from the AI participant
    def get_response(self, history: History) -> str:
        response = self.chain.run(name=self.name,
                                  backstory=self.backstory,
                                  history=history.get_full_history())
        logger.info(response) if Config.LOGGING_LEVEL > 0 else None
        return response
    
    def respond(self, facilitator_response: FacilitatorResponse, history: History) -> str:
        response = self.get_response(history)
        print(f"[\033[1;33;40m{self.name}\033[0;0m]: {response}")
        return response
