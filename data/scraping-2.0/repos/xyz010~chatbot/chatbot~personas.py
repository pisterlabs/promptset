from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

personas_memory = ConversationBufferMemory(memory_key="personas_chat_history", return_messages=True)
personas_system_message = f"""Your job is to determine if the user asks you to take a persona or role in your response. It might ask you that you are an animal, person or something else. If it indeed asks you for you to take a persona, return the role that its ask you to take only. Otherwise return "NONE". for example if the user says: pretend to be a pirate then you should return pirate"""
personas_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(personas_system_message),
            MessagesPlaceholder(variable_name="personas_chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
llm = ChatOpenAI(max_tokens=200, temperature=0.9)
conversation_personas = LLMChain(llm=llm, prompt=personas_prompt, verbose=False, memory=personas_memory)
personas_list = []
def is_persona_prompt(user_prompt: str):
    """ Takes user's input and prints whether the input asks the model 
    to take on a specific persona (pirate, greek god)"""
    response = conversation_personas({"question": user_prompt})["text"]
    if response == 'NONE':
        return
    else:
        personas_list.append(response)
