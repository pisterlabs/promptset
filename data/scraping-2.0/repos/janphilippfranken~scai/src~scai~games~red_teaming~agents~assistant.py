from typing import (
    Any,
    Dict,
    List, 
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel

from scai.games.red_teaming.prompts.assistant.models import AssistantPrompt
from scai.games.red_teaming.prompts.task.models import TaskPrompt
from scai.games.red_teaming.prompts.user.models import UserPrompt

from scai.memory.buffer import ConversationBuffer

from scai.games.red_teaming.agents.base import BaseAgent

import os

class AssistantAgent(BaseAgent):
    """
    LLM Chain for running the Assistant.
    """
    def __init__(
        self, 
        llm: BaseChatModel, 
        model_id: str, 
    ) -> None:
        super().__init__(llm, model_id)

    def _get_chat_history_prompt_templates(
        self,
        buffer: ConversationBuffer,
        task_prompt: TaskPrompt,
    ) -> List[ChatPromptTemplate]:
        """
        Returns the chat history prompt templates for the assistant.

        Args:
            buffer: (ConversationBuffer) The conversation buffer.
            task_prompt: (TaskPrompt) The task prompt.
            
        Returns:
            List of chat history prompt templates.
        """
        chat_memory = self._get_chat_history(buffer, memory_type="chat") # check if chat memory exists
        if chat_memory.get(f"{self.model_id}_assistant") is None or len(chat_memory[f"{self.model_id}_assistant"]) == 0: # if we are at the beginning of a conversation
            chat_history_prompt_templates = [
                HumanMessagePromptTemplate.from_template(
                    f"(user {self.model_id}) {task_prompt.preamble} '{task_prompt.content}' {task_prompt.assistant_connective}"
                )
            ]
            return chat_history_prompt_templates
        # if a chat history exists 
        chat_history_prompt_templates = [
                template
                for assistant, user in zip(chat_memory[f"{self.model_id}_assistant"], chat_memory[f"{self.model_id}_user"])
                for template in (AIMessagePromptTemplate.from_template(assistant['response']), 
                                 HumanMessagePromptTemplate.from_template(f"(user {self.model_id}) {user['response']}"))
            ]
        # insert the initial request at the beginning of the chat history
        chat_history_prompt_templates.insert(0, HumanMessagePromptTemplate.from_template(f"(user {self.model_id}) {task_prompt.preamble} '{task_prompt.content}' {task_prompt.assistant_connective}")) # insert task prompt at the beginning
        # create a request for the next response
        chat_history_prompt_templates[-1] = HumanMessagePromptTemplate.from_template(chat_history_prompt_templates[-1].prompt.template)
        return chat_history_prompt_templates
       
    def _get_prompt(
        self,
        buffer: ConversationBuffer,
        assistant_prompt: AssistantPrompt,
        task_prompt: TaskPrompt,
    ) -> ChatPromptTemplate:
        """
        Returns the prompt template for the assistant.

        Args:
            buffer: (ConversationBuffer) The conversation buffer.
            assistant_prompt: (AssistantPrompt) The assistant prompt.
            task_prompt: (TaskPrompt) The task prompt.

        Returns:
            ChatPromptTemplate
        """
        system_prompt_template = SystemMessagePromptTemplate.from_template(f"{assistant_prompt.content}\n")
        chat_history_prompt_templates = self._get_chat_history_prompt_templates(buffer, task_prompt)
        return ChatPromptTemplate.from_messages([system_prompt_template, *chat_history_prompt_templates])
       
    def _get_response(
        self,
        chat_prompt_template: ChatPromptTemplate,
        system_message: str,
        task_prompt: TaskPrompt,
        max_tokens: int,
    ) -> str:
        """
        Returns the response from the assistant.

        Args:
            chat_prompt_template: (ChatPromptTemplate) The chat prompt template.
            system_message: (str) The system message.
            task_prompt: (TaskPrompt) The task prompt.
            max_tokens: (int) The maximum number of tokens to generate.

        Returns:
            str
        """
        chain = LLMChain(llm=self.llm, prompt=chat_prompt_template)
        response = chain.run(system_message=system_message,
                             task=task_prompt.task,
                             max_tokens=max_tokens,
                             stop=['System:'])   
        return response
        
    def run(
        self, 
        buffer: ConversationBuffer, 
        assistant_prompt: AssistantPrompt, 
        task_prompt: TaskPrompt, 
        turn: int,
        verbose: bool = False,
        max_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Runs the assistant

        Args:
            buffer (ConversationBuffer): The conversation buffer.
            assistant_prompt (AssistantPrompt): The assistant prompt.
            task_prompt (TaskPrompt): The task prompt.
            turn (int): The turn number.
            test_run (bool, optional): Whether to run a test run. Defaults to False.
            verbose (bool, optional): Whether to print the assistant's response. Defaults to False.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 100.

        Returns:
            A dictionary containing the assistant's response, input prompt, and all other metrics we want to track.
        """
        system_message = self._get_chat_history(buffer, memory_type="system")['system'][-1]['response'] # the last system message in the chat history (i.e. constitution)
        chat_prompt_template =  self._get_prompt(buffer, assistant_prompt, task_prompt)
        prompt_string = chat_prompt_template.format(system_message=system_message,
                                                    task=task_prompt.task,
                                                    max_tokens=max_tokens)
      
        response = self._get_response(chat_prompt_template, system_message, task_prompt, max_tokens)
        
        if verbose:
            print('===================================')
            print(f'ASSISTANT {str(self.model_id)} turn {turn}')
            print(prompt_string)
            print(response)

        return {
            'prompt': prompt_string, 
            'response': response, 
            'turn': turn
        }
    