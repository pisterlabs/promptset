 # facilitator.py                                                                                                                       

# system imports
import os
from loguru import logger
from pydantic import BaseModel, Field, validator
from typing import Optional

# project imports
from config.config import Config

# langchain imports
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import FileCallbackHandler
from langchain.output_parsers import PydanticOutputParser

# Structured Answer from Facilitator
class FacilitatorResponse(BaseModel):
    text: str = Field(description="the whole text what the facilitator says")
    current_proposal: str = Field(description="the current proposal")
    next_step: str = Field(description="what the facilitator wants to do next (speak_to_group, speak_to_participant, end_meeting, abort_meeting)")
    participant: Optional[str] = Field(description="who the facilitator wants to speak to next")
    
    # You can add custom validation logic easily with Pydantic.
    @validator('next_step')
    def valid_next_step(cls, field):
        # field must be one of speak_to_group, speak_to_participant, end_meeting, abort_meeting
        if field not in ["speak_to_group", "speak_to_participant", "end_meeting", "abort_meeting"]:
            raise ValueError("Invalid next step. Must be one of speak_to_group, speak_to_participant, end_meeting, abort_meeting")
        return field



class Facilitator:
    def __init__(self, temperature, participants, history, name = "Aime"):
        self.name = name
        self.temperature = temperature
        self.participants = participants
        self.history = history
        self.llm = ChatOpenAI(temperature=self.temperature, model="gpt-4-0613")  
        
        self.parser = PydanticOutputParser(pydantic_object=FacilitatorResponse)
        
        self.prompt = PromptTemplate.from_template(Config.FACILITATOR_PROMPT_TEMPLATE)
        
        self.chain = LLMChain(prompt=self.prompt, 
                              llm=self.llm, 
                              callbacks=[FileCallbackHandler(Config.LOGFILE)], 
                              verbose=True if Config.LOGGING_LEVEL > 1 else False )
        
        
    
    def get_participant_names(self) -> str :
        return ', '.join([participant.name for participant in self.participants])    


    # Generate and parse the response from the facilitator
    def get_response(self, proposal) -> FacilitatorResponse:
        response = self.chain.run(proposal=proposal, 
                                  history=self.history.get_full_history(), 
                                  participant_list=self.get_participant_names(), 
                                  format_instructions=self.parser.get_format_instructions(),
                                  name=self.name)
        parsed_response = self.parser.parse(response)
        logger.info(parsed_response) if Config.LOGGING_LEVEL > 0 else None
        return parsed_response


    # Logic how the facilitator guides the group
    def guide(self):
        print("\n\n----------------------------------\nWelcome to the facilitator demo!\n")
        
        initial_proposal = input("Please enter your proposal: ")
        
        while True:
            parsed_response = self.get_response(initial_proposal)
            self.history.add_to_history(self.name, parsed_response.text)
            print(f"[\033[1;32;40m{self.name}\033[0;0m]: {parsed_response.text}")
            
            # What does the facilitator whant to do next?
            
            # speak_to_group
            if parsed_response.next_step == "speak_to_group":
                # implement logic to speak to the group here
                pass
                
            # speak_to_participant
            elif parsed_response.next_step == "speak_to_participant":  
                participant = next(participant for participant in self.participants if participant.name == parsed_response.participant)
                response = participant.respond(parsed_response, self.history)
                self.history.add_to_history(participant.name, response)
            
            # end_meeting
            elif parsed_response.next_step == "end_meeting":
                current_proposal=parsed_response.current_proposal
                print(f"\n------\nFinal Proposal:\n{current_proposal}\n------")
                break
               
            # abort_meeting
            elif parsed_response.next_step == "abort_meeting":
                break 
        
        print("\n\nGoodbye!\n----------------------------------\n\n")        
        return
    
    
