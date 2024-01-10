from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# initialize the LLM ~ large language model
llm = ChatOpenAI()

conversation_buf = ConversationChain(llm=llm, memory=ConversationBufferMemory())

res = conversation_buf("Good morning AI!")

print("\n", res)


res = conversation_buf("What is the tallest tree?")
print("\n", res)

