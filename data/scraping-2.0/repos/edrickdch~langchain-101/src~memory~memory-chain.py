from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(temperature=0.9)
conversation = ConversationChain(llm=chat, memory=ConversationBufferMemory())

out1 = conversation.run(
    "Answer briefly. \
                        What are the first 3 colors of a rainbow?"
)
out2 = conversation.run("And the next 4?")


print(out1)
print(out2)
