from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(temperature=0.9, model='gpt-3.5-turbo')

memory = ConversationBufferWindowMemory(k=3)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

while True:
    try:
        question = input("Question: ")
        ret = conversation.predict(input=question)
        print("Answer: ", ret)

    except KeyboardInterrupt:
        break
