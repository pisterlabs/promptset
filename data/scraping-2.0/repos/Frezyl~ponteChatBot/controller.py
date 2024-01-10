from typing import Annotated

import openai
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPBasicCredentials

import backend
import database
import mocked_parts
from backend import authenticate, format_new_person_message, message_data_base, mock_db

app = FastAPI()


@app.get("/mock_messages")
async def get_messages(number_of_messages: int = 10):
    """
    Returns the last n messages
    :param number_of_messages: the number of messages to return
    :return: the last n messages
    """
    return mock_db.get_messages(number_of_messages)


@app.post("/mock_messages", dependencies=[Depends(authenticate), Depends(backend.rate_limit_database.check_rate_limit)])
async def send_message(
        request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(database.security)]
):
    """
    Sends a message to the mock chatbot
    :param request: The request object containing the message
    :param credentials: The credentials of the user
    :return: The response of the chatbot
    """
    body: dict = await request.json()
    message_from_body = body.get('message', False)
    message = await process_message_and_check_ratelimit(credentials, message_from_body)
    messages = mock_db.get_messages()
    messages.append(message)
    response = await mocked_parts.call_external_service(messages)
    mock_db.add_message(message)
    mock_db.add_message(response['choices'][0]['message'])
    return response['choices'][0]['message']['content']


async def process_message_and_check_ratelimit(credentials, message_from_body):
    """
    Processes a message and checks the rate limit
    :param credentials: User credentials
    :param message_from_body: Message from the request body
    :return: The processed message
    """
    if not message_from_body:
        raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Missing message in request body"
        )
    # user_exists = rate_limit_database.check_user(credentials.username)
    # if user_exists is False:
    #     rate_limit_database.add_event(credentials.username)
    # else:
    #     if rate_limit_database.check_rate_limit(
    #             credentials.username, 3
    #     ) is False:
    #         raise HTTPException(
    #                 status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests"
    #         )
    #     else:
    #        rate_limit_database.add_event(credentials.username)
    message = format_new_person_message(message_from_body)
    return message


@app.post(
        "/GPTmessages", dependencies=[Depends(authenticate), Depends(backend.rate_limit_database.check_rate_limit)]
)
async def send_GPT_message(
        request: Request, credentials: Annotated[HTTPBasicCredentials, Depends(database.security)]
):
    """
    Sends a message to the real chatbot (ChatGPT 3.5)
    :param request: The request object containing the message
    :param credentials: The credentials of the user (username and password)
    :return: The response of the chatbot
    """

    body = await request.json()
    message_from_body = body['message']
    message = await process_message_and_check_ratelimit(credentials, message_from_body)
    messages = message_data_base.query_user_history(credentials.username)
    messages = messages.append(message)
    response = await openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages
    )
    message_data_base.add_message_to_user(message, credentials.username)
    message_data_base.add_message_to_user(
            response['choices'][0]['message'], credentials.username
    )
    return response['choices'][0]['message']['content']
