from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

chat = ChatOpenAI(model="gpt-3.5-turbo")
result = chat(
    [
        SystemMessage(content="あなたは親しい友人として振舞ってください."),
        HumanMessage(content="こんちには！"),
    ]
)

print(result.content)
