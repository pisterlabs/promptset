import langchain
from langchain.chains import LLMChain, SimpleSequentialChain, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


langchain.verbose = True

chat = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

while True:
    user_message = input("You: ")
    ai_message = conversation.predict(input=user_message)
    print(f"AI: {ai_message}")
