from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ChatMessageHistory
from prompts import *


llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106")
history = [SystemMessage(content=assistant_message)]
user_message = ""

def converse():
    global user_message
    ai_message = llm.invoke(input=history)
    print("Bot: ", ai_message.content)
    history.append(AIMessage(content = ai_message.content))
    user_message = input("\nUser: ")
    history.append(HumanMessage(content= user_message))


while user_message != "quit":
    converse()

print(llm.get_num_tokens_from_messages(history))
