from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
chat = ChatOpenAI(temperature=0)
def languageconvert(inputlang,outputlang,text):
    template = "you are bot that convert from text with {input_language} to {output_language}."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    humantemplate = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(humantemplate)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    chain.run(inputlang, outputlang, text)
    return ""