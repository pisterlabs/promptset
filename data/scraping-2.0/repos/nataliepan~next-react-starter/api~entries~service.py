from entries.models import Entry
from server import db
from utils.common import generate_response
from utils.http_code import HTTP_200_OK, HTTP_201_CREATED, HTTP_400_BAD_REQUEST
from auth_middleware import token_required
from context import SYSTEM_PROMPT
from langchain.llms import Clarifai
from langchain.chains import LLMChain
from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
import os
import json

pat_openai=os.environ.get("NEXT_PUBLIC_OPENAI_API_KEY")

def create_entry(user, input_data):
    """
    It creates a new entry
    :param request: The request object
    :param input_data: This is the data that is passed to the function
    :return: A response object
    """
    # create_validation_schema = CreateSignupInputSchema()
    # errors = create_validation_schema.validate(input_data)
    # if errors:
    #     return generate_response(message=errors)
    # check_username_exist = User.query.filter_by(
    #     username=input_data.get("username")
    # ).first()
    # check_email_exist = User.query.filter_by(email=input_data.get("email")).first()
    # if check_username_exist:
    #     return generate_response(
    #         message="Username already exist", status=HTTP_400_BAD_REQUEST
    #     )
    # elif check_email_exist:
    #     return generate_response(
    #         message="Email  already taken", status=HTTP_400_BAD_REQUEST
    #     )

    new_entry = Entry(user_id=user.id, **input_data)  # Create an instance of the User class

    db.session.add(new_entry)  # Adds new User record to database
    db.session.commit()  # Comment

    return generate_response(
        data=new_entry.as_dict(), message="Entry Created", status=HTTP_201_CREATED
    )

def get_entries(user, request):
    """
    It takes in a request and input data, validates the input data,
    checks if the user exists,
    checks if the password is correct, and returns a response
    :param request: The request object
    :param input_data: The data that is passed to the function
    :return: A dictionary with the keys: data, message, status
    """
    # create_validation_schema = CreateSignupInputSchema()
    # errors = create_validation_schema.validate(input_data)
    # if errors:
    #     return generate_response(message=errors)
    # check_username_exist = User.query.filter_by(
    #     username=input_data.get("username")
    # ).first()
    # check_email_exist = User.query.filter_by(email=input_data.get("email")).first()
    # if check_username_exist:
    #     return generate_response(
    #         message="Username already exist", status=HTTP_400_BAD_REQUEST
    #     )
    # elif check_email_exist:
    #     return generate_response(
    #         message="Email  already taken", status=HTTP_400_BAD_REQUEST
    #     )


    entries = Entry.query.filter_by(user_id = user.id)

    return generate_response(
        data=[entry.as_dict() for entry in entries], message="Found entries", status=HTTP_201_CREATED
    )

def sentiment_analysis(token, message):

    llmName="GPT-4"
    llmAuthor="openai"
    llmApp="chat-completion"

    llm = ChatOpenAI(openai_api_key=pat_openai)
    # llm=Clarifai(pat=token, user_id=llmAuthor, app_id=llmApp, model_id=llmName)

    prompt = SystemMessage(content=SYSTEM_PROMPT)
    new_prompt = (
    prompt
    + "{input}"
    )

    new_prompt.format_messages(input=message)

    conversation = LLMChain(
    prompt=new_prompt,
    llm=llm
    )

    data = conversation.run(message)

<<<<<<< Updated upstream
    print(data)
=======
    # Convert the dictionary to a JSON string
    # response_json = json.dumps(data)
    # print(response_json)
>>>>>>> Stashed changes
    return generate_response(data=data, message="Sentiment Analysis", status=HTTP_201_CREATED)