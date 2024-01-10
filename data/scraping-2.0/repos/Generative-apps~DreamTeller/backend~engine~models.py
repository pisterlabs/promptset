from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.schema import HumanMessage, AIMessage
from langchain import PromptTemplate
from backend.engine.prompt import prompt_template
from backend.engine.config import CONF
from json import dumps, loads

MEMORY_NAME = 'history'

dream_strory = """I was getting back home, only I lived in midtown tel aviv(in reality I live outside the city.
anyway, I was walking in the neighborhood when I realized all the stores and homes are locked and closed down.
it was night, but still it looks too closed - like it was preparing for a war.
I got to my house, which wasn't my house - instead it contains pepole I know I live with - it was like a militry unit.
We got up to the watch tower and fought cat-zombies that tried to get up and kill us!
the thing is, I was the only one that really fought ! everyone else was useless! tried to fight with no real effort.
I kept pushing them and yelling, demonstrating how to hit and kill the cat zombies - but with no luck. frustrated, I woke up.
"""

MESSAGE_TYPE = {'human': HumanMessage, 'ai': AIMessage}

# def get_raw_mem(mem_list):
#             raw_mem = []
#             for item in mem_list:
                 

class BaseModel:
    def send_dream_description(self, desc:str):
        return f"This is the description of the dream: {desc}"

    def answer_questions(self, answers:list[str]):
        msg = "here are my answers:\n"
        for i, ans in enumerate(answers):
            msg += f"{i+1}. {ans}\n"
        msg += "Now please provide the dream interpretation."
        return msg





class GPT35(BaseModel):
    conversation: ConversationChain
    memory: ConversationBufferMemory
    

    def __init__(self, memory=None) -> None:        
        openai_key = CONF.openai_key
        llm = OpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=openai_key)
        prompt = PromptTemplate(input_variables=["history", "input"], template=prompt_template)        
                

        # load memory
        if (memory):
            history = GPT35.load_memory(memory)
            self.memory = ConversationBufferMemory(chat_memory=history, memory_key=MEMORY_NAME, human_prefix="HUMAN")
        else:
            self.memory = ConversationBufferMemory( memory_key=MEMORY_NAME, human_prefix="HUMAN")
        
        self.conversation = ConversationChain(
            llm=llm, 
            verbose=True, 
            memory=self.memory,
            prompt=prompt


        )
        super().__init__()

    def send_dream_description(self, desc: str):
        msg = super().send_dream_description(desc)
        return self.conversation.predict(input=msg)
    
    def answer_questions(self, answers:list[str]):
        msg = super().answer_questions(answers)
        return self.conversation.predict(input=msg)

    def get_raw_memory(self) -> str:
        # self.memory.dict()
        return [{'content': m.content, 'type': m.type} for m in  self.conversation.memory.chat_memory.messages ]

        
    
    @staticmethod
    def load_memory(mem:str):  
        messages = [MESSAGE_TYPE[m['type']](content=m['content'], type=m['type']) for m in mem]         
        return ChatMessageHistory(messages=messages)        
    

    
    
    
# model = GPT35()
# model.send_dream_description(dream_strory)
# mem = model.get_raw_memory()

# del model
# model = GPT35(memory=mem)

# # model = GPT35(memory=mem)
# model.answer_questions(["frustratuing, unsecure, a failure",
#                         "i dont know, try to understand from the dream",
#                          "absoulutly not",
#                          "i think it has something to do with me trying to proove myself to others, mainlt prossinaly "])
# mem = model.get_raw_memory()






