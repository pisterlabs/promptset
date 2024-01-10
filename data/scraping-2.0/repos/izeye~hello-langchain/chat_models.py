from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(temperature=0)
predicted_messages = chat.predict_messages(
    [HumanMessage(content="Translate this sentence from English to French. I love programming.")])
print(predicted_messages)

predicted = chat.predict("Translate this sentence from English to French. I love programming.")
print(predicted)
