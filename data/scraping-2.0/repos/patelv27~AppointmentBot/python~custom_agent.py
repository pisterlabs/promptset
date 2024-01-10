import logging
import re
import openai
from typing import Optional, Tuple, AsyncGenerator, Union
import typing
from vocode.streaming.agent import ChatGPTAgent, LLMAgent
from vocode.streaming.models.agent import AgentConfig, AgentType, ChatGPTAgentConfig, LLMAgentConfig
from vocode.streaming.agent.base_agent import BaseAgent, RespondAgent
from vocode.streaming.agent.factory import AgentFactory


#Used LLM agent rather than ChatGPT agent because of improved performance.
class CustomAgentConfig(LLMAgentConfig, type="agent_Custom"):
    pass


class CustomAgent(LLMAgent):
    def __init__(self, agent_config: CustomAgentConfig):
        super().__init__(agent_config=agent_config)

        #Initializing values for patient info. Can later be used to store actual info. 
        self.patient_info = {
            "name": None,
            "date of birth": None,
            "insurance payer ID": None,
            "insurancepayer name": None,
            "medical condition": None,
            "address": None,
            "phone number": None,
            "referral status": None,
        }

    async def generate_response(
        self,
        human_input,
        conversation_id: str,
        is_interrupt: bool = False,
    ) -> AsyncGenerator[str, None]:
        self.logger.debug("LLM generating response to human input")

        #searches through conversation memory. 
        def search_memory(phrase: str) -> bool:
            if phrase in self.patient_info and self.patient_info[phrase] is not None:
                return True

            for sentence in self.memory:
                full_phrase = "your {} is confirmed".format(phrase)
                if full_phrase in sentence:
                    self.patient_info[phrase]=True
                    return True

        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            self.memory.append(self.get_memory_entry(human_input, cut_off_response))
            yield cut_off_response
            return
        self.memory.append(self.get_memory_entry(human_input, ""))
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            sentences = self._agen_from_list([self.first_response])
        else:
            if not search_memory('name'):
                human_input += " Ask for my name, repeating it back to me as a question to see if it is correct. Prepare but don't yet ask for my date of birth "

            elif not search_memory('date of birth'):
                human_input += " Ask for my date of birth, repeating it back to me as a question to see if it is correct. Prepare but don't yet ask for my insurance payer name. "
            
            elif not search_memory('insurance payer name'):
                human_input += " Ask for my insurance payer name, repeating it back to me as a question to see if it is correct. Prepare but don't yet ask for my insurance payer ID "
            
            elif not search_memory('insurance payer ID'):
                human_input += " Ask for and confirm my insurance payer ID, repeating it back to me as a question to see if it is correct. Prepare but don't yet ask for my address "
            
            elif not search_memory('address'):
                human_input += " Ask for my address, repeating it back to me as a question to see if it is correct. Prepare but don't yet ask for my phone number "
            
            elif not search_memory('phone number'):
                human_input += " Ask for my phone number, repeating it back to me as a question to see if it is correct. Prepare but don't yet ask for my medical condition "
            
            elif not search_memory('medical condition'):
                human_input += " Ask for and confirm my medical condition, repeating it back to me as a question to see if it is correct. Prepare but don't yet ask whether I have a referral "
            
            elif not search_memory('referral status'):
                human_input += " Ask for and confirm whether I have a referral and to whom this referral is for, repeating it back to me as a question to see if it is correct. If my response is correct, say 'ok, your referral status is confirmed'. Prepare but don't yet ask whether I have an appointment selection. "
            
            else:
                human_input += " Ask me to choose an appointment preference between an appointment with Doctor House on August 1st at 2pm or with Doctor Strange on Augst 2nd at 3pm. If I have a referral, say the name of my referred Doctor then 'has an appointment on August 10th at 1pm'.' "

            self.logger.debug("Creating LLM prompt")
            prompt = self.create_prompt(human_input)
            self.logger.debug("Streaming LLM response")
            sentences = self._stream_sentences(prompt)
        response_buffer = ""
        async for sentence in sentences:
            sentence = sentence.replace(f"{self.sender}:", "")
            sentence = re.sub(r"^\s+(.*)", r" \1", sentence)
            response_buffer += sentence
            self.memory[-1] = self.get_memory_entry(human_input, response_buffer)

            yield sentence



class CustomAgentFactory(AgentFactory):
    def create_agent(
        self, agent_config: AgentConfig, logger: Optional[logging.Logger] = None
    ) -> BaseAgent:
        if agent_config.type == "agent_Custom":
            return CustomAgent(
                agent_config=typing.cast(CustomAgentConfig, agent_config),
            )
        raise Exception("Invalid agent config")