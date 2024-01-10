from langchain.chains import ConversationChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage

from utils.const import *

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def get_langchain_template (template_type, with_system=False) -> ChatPromptTemplate :
    
    assert template_type in TEMPLATES
    
    tem_path = TEMPLATES[template_type]
    
    if with_system :
        system_message_prompt = ChatPromptTemplate.from_template(template=read_prompt_template(TEMPLATES["SYSTEM"]))
        human_message_prompt = HumanMessagePromptTemplate.from_template(read_prompt_template(tem_path))

        prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    else :
        prompt = ChatPromptTemplate.from_template(
            template=read_prompt_template(tem_path)
        )
    return prompt
