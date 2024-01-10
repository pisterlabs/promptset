from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

chat = ChatOpenAI(model="gpt-3.5-turbo")
result = chat(
    [
        HumanMessage(content="こんちには！"),
        AIMessage(content="こんにちは！"),
        HumanMessage(content="元気ですか?"),
    ]
)

print(result.content)
