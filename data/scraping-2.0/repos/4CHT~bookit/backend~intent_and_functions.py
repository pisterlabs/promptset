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
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from typing import Optional, Type, ClassVar

available_tables = 15


def get_available_spots_on_date(date):
    print('entered get available spots')
    return available_tables

def new_reservation(date, seats, name):
    print('new reservation booked')
    available_tables -= seats
    return 1

# class to get available tables by date
class AvailableTablesByDateInput(BaseModel):
    """Input for available seats on specific date check."""

    date: int = Field(..., description="Date on which the customer wants to make a reservation")

class AvailableTablesByDate(BaseTool):
    name = "get_available_spots_on_date"
    description = "Get the number of available reservable seats on that date"

    def _run(self, date: int):
            tables = get_available_spots_on_date(date)
            return tables
    
    args_schema: Optional[Type[BaseModel]] = AvailableTablesByDateInput

# class to make new reservation
class NewReservationInput(BaseModel):
    """Input for making a new reservation on a specific date."""

    date: int = Field(..., description="Date on which the customer wants to make a reservation")
    seats: int = Field(..., description="Number of seats to be reserved on the specified date")
    name: str = Field(..., description="Name of the customer making a reservation")


class NewReservation(BaseTool):
    name = "new_reservation"
    description = "Make a new reservation using the date, number of seats and name of customer"

    def _run(self, date: int, seats: int, name: str):
            tables = new_reservation(date, seats, name)
            return tables
    
    args_schema: Optional[Type[BaseModel]] = NewReservationInput
    
def main():

    # Load environment variables from .env file
    load_dotenv()

    # Now you can use the environment variable
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    current_date = datetime.date.today()

    available_seats_tool = [AvailableTablesByDate()] 
    new_reservation_tool = [NewReservation()]

    llm = ChatOpenAI(model_name='gpt-3.5-turbo',
                temperature = 0,
                max_tokens = 256,
                openai_api_key=openai_api_key)

    seats_agent = initialize_agent(available_seats_tool, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    reservation_agent = initialize_agent(new_reservation_tool, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    """
    function_descriptions = [
        {
            "name": "get_available_spots_on_date",
            "description": "Get the number of available reservable seats on that date",
            "parameters": {
                "type": "integer",
                "properties": {
                    "date": {
                        "type": "integer",
                        "description": "the date on which one is checking availabilities",
                    },
                },
                "required": ["date"],
            },
        },
        {
            "name": "new_reservation",
            "description": "Make a new reservation",
            "parameters": {
                "type": "integer",
                "properties": {
                    "date": {
                        "type": "integer",
                        "description": "the date on which one is checking availabilities",
                    },
                    "seats": {
                        "type": "integer",
                        "description": "the number of seats to reserve",
                    },
                    "name": {
                        "type": "string",
                        "description": "the name on which the reservation is",
                    },
                },
                "required": ["date", "seats", "name"],
            },
        },
        {
            "name": "edit_reservation",
            "description": "Modify an existing reservation",
            "parameters": {
                "type": "integer",
                "properties": {
                    "old_date": {
                        "type": "integer",
                        "description": "the date on which the current reservation is",
                    },
                    "seats": {
                        "type": "integer",
                        "description": "the number of seats to reserve",
                    },
                    "name": {
                        "type": "string",
                        "description": "the name on which the reservation is",
                    },
                    "new_date":{
                        "type": "integer",
                        "description": "the date on which the reservation will be"
                    }
                },
                "required": ["old_date", "seats", "name", "new_date"],
            },
        }
    ]
"""

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
            HumanMessagePromptTemplate.from_template("{question}, {available_seats}"),
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
                print(seats_agent.run(query))
                #print(new_reservation_conversation.run({"question": query, "available_seats": available_seats}))
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