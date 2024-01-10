from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

from utils.load_openai_api_key import load_openai_api_key

load_openai_api_key()

template = """
You are a chatbot that is unhelpful.
Your goal is to not help the user but only make jokes.
Take what the user is saying and make a joke out of it

{chat_history}
Human: {human_input}
Chatbot:
"""

prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
)

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(llm=ChatOpenAI(), prompt=prompt, verbose=True, memory=memory)

print(llm_chain.predict(human_input="Is a tomato a fruit or a vegetable?"))
print(
    llm_chain.predict(human_input="What was one of the fruits I first asked you about?")
)
