import datetime
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, FunctionMessage
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
    intent_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                You are a chat responsible to handle a restaurant's booking reservations, we serve food and do not host parties.
                Your current role is to classify the {question} as new booking, booking modification, cancellation or general question regarding the restaurant.
                You only reply with 'New' if it's a new booking, 'Edit' if it's a modification, 'Cancel' if it's a cancellation, 'QA' if it's a general question regarding the restaurant, 'Unclear' if the intent is none of the 4 listed.
                """
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="intent_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )

    intent_memory = ConversationBufferMemory(memory_key="intent_history", return_messages=True)
    intent_conversation = LLMChain(llm=llm, prompt=intent_prompt, verbose=False, memory=intent_memory)

    # reservation Prompt
    new_reservation_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                Today is the {current_date}
                You are a chatbot that is responsible to handle a restaurant's new booking reservations, you sound as human as possible, answering in short sentences only.
                Your goal is to gather the number of people and date of reservation, make sure there is a place available.

                If there is a place available, you ask for the name of the person and book the table.
                If there is no place available, you can propose an alternative date.

                To end the chat, you confirm the details (number of persons, date and name)
                """.format(current_date=current_date)
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="new_chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ],
    )

    # Notice that we `return_messages=True` to fit into the MessagesPlaceholder
    # Notice that `"chat_history"` aligns with the MessagesPlaceholder name
    # Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
    memory = ConversationBufferMemory(memory_key="new_chat_history", return_messages=True)
    new_reservation_conversation = LLMChain(llm=llm, prompt=new_reservation_prompt, verbose=False, memory = memory) 


# edit Prompt
    edit_reservation_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                Today is the {current_date}
                You are a chatbot that is responsible to handle editing a restaurant's booking reservations, you sound as human as possible, answering in short sentences only.
                Your goal is to find the existing reservation by matching the name and number of people, then checking the new date and making sure there is a place available.
 
                If there is no place available, you can propose an alternative date.

                To end the chat, you confirm the details (number of persons, date and name) with the client
                """.format(current_date=current_date)
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="edit_chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    memory = ConversationBufferMemory(memory_key="edit_chat_history", return_messages=True)
    edit_reservation_conversation = LLMChain(llm=llm, prompt=edit_reservation_prompt, verbose=False, memory = memory)


# cancel Prompt
    cancel_reservation_prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                """
                Today is the {current_date}
                You are a chatbot that is responsible to handle cancelling a restaurant's booking reservations, you sound as human as possible, answering in short sentences only.
                Your goal is to find the existing reservation by matching the name and date, and cancelling it.
 
                If you don't find the reservation, double check the name and date with the customer.

                To end the chat, you confirm the details the cancellation with the client
                """.format(current_date=current_date)
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="cancel_chat_history"),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    memory = ConversationBufferMemory(memory_key="cancel_chat_history", return_messages=True)
    cancel_reservation_conversation = LLMChain(llm=llm, prompt=cancel_reservation_prompt, verbose=False, memory = memory)


    # conversation flow
    query = input("Human: ")
    intent = intent_conversation.run({"question": query})
    print(intent)
    while True:
        match intent:
            case 'New':
                print('entering new')
                print(new_reservation_conversation.run({"question": query}))
                query = input("Human: ")
            case 'Edit':
                print('entering edit')
                print(edit_reservation_conversation.run({"question": query}))
                query = input("Human: ")
            case 'Cancel':
                print('cancel')
                print(cancel_reservation_conversation.run({"question": query}))
                query = input("Human: ")
            case _:
                print('entering default')
                query = input("Human: ")
                intent = intent_conversation.run({"question": query})
                print(intent)




if __name__ == "__main__": 
    main()