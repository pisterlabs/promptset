from dotenv import dotenv_values
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
import asyncio
from typing import Any, Dict, List

env_vars = dotenv_values('.env')

# custom memory class for the third input variable memory
class ExtendedConversationBufferMemory(ConversationBufferMemory):
    extra_variables:List[str] = []

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables."""
        return [self.memory_key] + self.extra_variables

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return buffer with history and extra variables"""
        d = super().load_memory_variables(inputs)
        d.update({k:inputs.get(k) for k in self.extra_variables})        
        return d


# move this to a env variable
llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0.5)
memory = ConversationBufferMemory(return_messages=True, ai_prefix="AI friend")

template = """The following is a passive agressive comedic conversation between a human and an AI. This AI specifically is able to get the humans emotion. Based of this current emotion give a response that a human would also give if they saw that emotion. The AI is talkative and provides lots of specific details from its context. The AI should talk like they have known the human for years. Make this conversation short and try keeping it to 1-2 sentences but also try keeping the same conversation going until the human changes topics. Also if the human gives a more complicated input that requires more than 2 sentences to answer feel free to give a more detailed explanation!

Current converstation:
{history}
Conversation:
Human Emotion: {emotion}
Human: {input}
AI friend:"""
prompt = PromptTemplate(input_variables=["history", "input", "emotion"], template=template)

async def start_conversation(emotions_list=None, lock=None):
    print("starting conversation")
    conversation = ConversationChain(llm=llm, verbose=True, memory=ExtendedConversationBufferMemory(extra_variables=["emotion"]), prompt=prompt)
    # input_message = input()
    input_message = await asyncio.get_event_loop().run_in_executor(None, input, 'Enter message: ')
    while(input_message != "exit"):
        if(emotions_list != None):
            # conversation.run(input=input_message, emotion=emotions_list[-1])
            async with lock:
                result = conversation({"input": input_message, "emotion": emotions_list[-1]})
        else:
            # conversation.run(input=input_message, emotion="neutral")
            result = conversation({"input": input_message, "emotion": "neutral"})
        # print(conversation.memory.buffer[-1].content)
        print(result["response"])
        # input_message = input()
        input_message = await asyncio.get_event_loop().run_in_executor(None, input, 'Enter message: ')


# asyncio.run(start_conversation())


# move this to a env variable
# llm = OpenAI(openai_api_key=env_vars['OPENAI_API_KEY'], temperature=0.5)
# memory = ConversationBufferMemory(return_messages=True, ai_prefix="AI friend")

# template = """The following is a passive agressive comedic conversation between a human and an AI. This AI specifically is able to get the humans emotion. Based of this current emotion give a response that a human would also give if they saw that emotion. The AI is talkative and provides lots of specific details from its context. The AI should talk like they have known the human for years. Make this conversation short and try keeping it to 1-2 sentences but also try keeping the same conversation going until the human changes topics. Also if the human gives a more complicated input that requires more than 2 sentences to answer feel free to give a more detailed explanation!

# Current converstation:
# {history}
# Conversation:
# Human: {input}
# AI friend:"""
# prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# async def start_conversation(emotions_list=None):
#     conversation = ConversationChain(llm=llm, verbose=True, memory=memory, prompt=prompt)
#     input_message = input()
#     while(input_message != "exit"):
#         conversation.run(input=input_message)
#         print(conversation.memory.buffer[-1].content)
#         input_message = input()

# asyncio.run(start_conversation())