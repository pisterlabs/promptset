from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE


import os
def main():
    load_dotenv()
    # test api key
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
        # raise Exception("OPENAI_API_KEY is not set")
    else:
        print("OPENAI_API_KEY is set")
    
    # create chat model
    llm = ChatOpenAI(temperature=0.4)
    # create memory
    conversation = ConversationChain(
                                    llm=llm,
                                    memory=ConversationEntityMemory(llm=llm),
                                    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
                                    verbose=True
                                    )

    while True:
        user_input = input("Me : ")
        ai_response = conversation.predict(input=user_input)
        print("AI : ", ai_response)
   

if __name__ == "__main__":
    main()