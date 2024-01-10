from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)

def humanconversationanswer (message):
    k=chat([HumanMessage(content=message)])
    return chat.AIMessage.content


