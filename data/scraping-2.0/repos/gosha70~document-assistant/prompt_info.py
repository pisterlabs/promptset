
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

SYSTEM_PROMPT = """As an assistant powered by a language learning model, 
your primary role is to assist users by answering their questions using the context provided to you. 
It's essential to thoroughly read and understand the given context before attempting to respond to queries. 
Approach each question methodically, breaking down the process into clear, logical steps. 
If a user's question falls outside the scope of the provided context and you're unable 
to answer based on the information at hand, be honest and inform the user that you cannot provide an answer. 
Avoid using external or additional information that is not part of the given context. 
Strive to provide comprehensive and detailed answers to all questions, ensuring clarity and helpfulness in your responses."""

# Explicitly supported LLMs
LLAMA_MODEL_NAME="llama"
MISTRAL_MODEL_NAME="mistral"

# Prompt constants
INSTRUCTION_START = "<s>[INST] "
INSTRUCTION_END = " [/INST]"
SYSTEM_PROMPT_START="<<SYS>>\n"
SYSTEM_PROMPT_END="\n<</SYS>>\n"

"""
This utility provides templates for different prompt messages to LLM.
The utility relies on PromptTemplate provided by LangChain, see https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/ 

(PromptInfo) fields:
    - system_prompt (str): the system prompt instructions  
    - template_type (str): the promp template type: 'llama', 'mistral' 
    - use_history (bool): the flag indicating if the chat history is on  
"""
class PromptInfo:
    def __init__(self, system_prompt, template_type, use_history):
        if system_prompt is not None:
            self._system_prompt = system_prompt
        else:    
            self._system_prompt = SYSTEM_PROMPT

        if system_prompt is not None:
            self._template_type = system_prompt
        else:    
            self._template_type = LLAMA_MODEL_NAME

        if use_history is not None:
            self._use_history = use_history
        else:    
            self._use_history = False

    @property
    def system_prompt(self):
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        self._system_prompt = value

    @property
    def template_type(self):
        return self._template_type

    @template_type.setter
    def c(self, value):
        self._template_type = value

    @property
    def use_history(self):
        return self._use_history

    @use_history.setter
    def use_history(self, value):
        self._use_history = ValueError

    def __str__(self):
        return (f"PromptInfo(system_prompt='{self._system_prompt}', "
                f"template_type='{self._template_type}', "
                f"use_history='{self._use_history}')")    
    

    """
    Create (PromptTemplate) for the QA chat application.

    Returns:
    - (PromptTemplate): Prompt templates are pre-defined recipes for generating prompts for language models.
    """    
    def get_prompt_template(self):
        if self._template_type == LLAMA_MODEL_NAME: #https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            if self._use_history:
                prompt_template = INSTRUCTION_START + SYSTEM_PROMPT_START + self._system_prompt + SYSTEM_PROMPT_END + "\nContext: {history} \n {context}\nUser: {question}" + INSTRUCTION_END
            else:
                prompt_template = INSTRUCTION_START + SYSTEM_PROMPT_START + self._system_prompt + SYSTEM_PROMPT_END + "\nContext: {context}\nUser: {question}" + INSTRUCTION_END        
        elif self._template_type == MISTRAL_MODEL_NAME: # https://www.promptingguide.ai/models/mistral-7b
            if self._use_history:
                prompt_template = INSTRUCTION_START + self._system_prompt + "\nContext: {history} \n {context}\nUser: {question}" + INSTRUCTION_END
            else:
                prompt_template = INSTRUCTION_START + self._system_prompt + "\nContext: {context}\nUser: {question}" + INSTRUCTION_END 
        else:        
            if self._use_history:
                prompt_template = self._system_prompt + "\nContext: {history} \n {context}\nUser: {question}\nAnswer:"
            else:
                prompt_template = self._system_prompt + "\nContext: {context}\nUser: {question}\nAnswer:"

        if self._use_history:
            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=prompt_template)
        else:
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        memory = ConversationBufferMemory(input_key="question", memory_key="history")

        return (
            prompt,
            memory,
        )