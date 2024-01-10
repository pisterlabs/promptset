# https://python.langchain.com/en/latest/modules/memory/types/buffer_window.html

from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import os

def main():
    load_dotenv()

    # Load the OpenAI API key from the environment variable
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set")
        exit(1)
    else:
        print("OPENAI_API_KEY is set")

    system_message = "You are a helpful assistant."
    user_input = input(f"System Message (default=`{system_message}`): ")
    if len(user_input) > 0:
        system_message = user_input
        print(f"System Message: {system_message}")
       
    k = 2
    try:
        k = int(input(f"keep the last `k` interactions in memory (default k={k}): "))
        print(f"k={k}")
    except ValueError:
        pass
        
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])
    llm = ChatOpenAI(temperature=0)
    # memory = ConversationBufferMemory(return_messages=True)
    memory = ConversationBufferWindowMemory( k=k, return_messages=True)
    memory.save_context({"Human": "hi"}, {"AI": "whats up"})
    conversation = ConversationChain(
      memory=memory, prompt=prompt, llm=llm, verbose=True
    )

    while True:
        user_input = input("> ")

        response = conversation.predict(input=user_input)
        # messages.append(HumanMessage(content=user_input))
        # assistant_response = chat(messages)
        # messages.append(AIMessage(content=assistant_response.content))

        print(f"Assistant: {response}\n")


if __name__ == '__main__':
    main()
