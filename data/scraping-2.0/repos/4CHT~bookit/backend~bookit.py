import datetime
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)


def main():

    # Load environment variables from .env file
    load_dotenv()

    # Now you can use the environment variable
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    current_date = datetime.date.today()

    llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                temperature = 0,
                max_tokens = 256,
                openai_api_key=openai_api_key)


    # Prompt
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                Today is the {current_date}
                You are a chatbot that is responsible to handle a restaurant's booking reservations, you sound as human as possible, answering in short sentences only.
                Your goal is to gather the number of people and date of reservation, make sure there is a place available.

                If there is a place available, you ask for the name of the person and book the table.
                If there is no place available, you can propose an alternative date.

                To end the chat, you confirm the details (number of persons, date and name) with the client
                """.format(current_date=current_date)
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

    # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory

    conversation = LLMChain(llm=llm, prompt=prompt, memory = memory)


    # conversation flow
    while True:
        query = input("Human: ")
        print(conversation({"question": query}))



if __name__ == "__main__": 
    main()