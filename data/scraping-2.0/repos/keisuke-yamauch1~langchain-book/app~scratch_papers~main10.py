import langchain
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

langchain.verbose = True
langchain.debug = True

if __name__ == "__main__":
    chat = ChatOpenAI(model_name="gpt-4", temperature=0)
    conversation = ConversationChain(
        llm=chat,
        memory=ConversationBufferMemory()
    )

    while True:
        user_message = input("You: ")
        ai_message = conversation.run(input=user_message)
        print(f"AI: {ai_message}")
