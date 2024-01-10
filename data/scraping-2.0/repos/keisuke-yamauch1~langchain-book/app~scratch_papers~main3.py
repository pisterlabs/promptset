from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

if __name__ == "__main__":
    chat = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0
    )

    messages = [
        SystemMessage(content="You are a helpful assinstant."),
        HumanMessage(content="こんにちは！私はジョンと言います！"),
        AIMessage(content="こんにちは、ジョンさん！どのようにお手伝いできますか？"),
        HumanMessage(content="私の名前がわかりますか？")
    ]

    result = chat(messages)
    print(result.content)
