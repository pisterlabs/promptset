from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

chat = OpenAI(temperature=0.5)

conversation = ConversationChain(
    llm=chat,
    verbose=True,
    memory=ConversationBufferMemory()
)
