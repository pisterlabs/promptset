"""Example usage:
poetry run python queried_results_to_app_response.py --user_role architect \
    --building_type residential \
    --initial_user_message "What are the structural requirements?" \
    --pinecone_response_list "[{\"topic\": \"text about topic\", \"index\": \"802.1c\", \"section_text\": \"text in section\"}, {\"topic\": \"text about second topic\", \"index\": \"52.1b\", \"section_text\": \"more text\"}]"


a more realistic example:
poetry run python queried_results_to_app_response.py --user_role "building inspector" \
    --building_type hospitals \
    --initial_user_message "What are some checklist items for surgical clinics compliance?" \
    --pinecone_response_list "[{\"topic\": \"clinics compliance requirements\", \"content\": [{\"index\": \"802.1c\", \"section_text\": \"all hospitals must have 1 sprinkler per room and hallway\"}]}]"
"""

import argparse
import sys
import openai
import json
import os

import requests

# Initialize GPT-4 with your private key
openai.organization = os.getenv("OPENAI_ORGANIZATION")
openai.api_key = os.getenv("OPENAI_API_KEY")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
REQUEST_HEADER = {
    "Content-Type": "application/json",
    "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY"),
}


class AppResponseMachine:
    SYSTEM_MESSAGE = (
        "output: a human readable summary of the query results relevant to "
        "user question, reference relevant sections of the code (bold format: "
        "*section code*)). Section codes, title and text are pulled from our building "
        "code database (user unaware). Give an apology if no relevant results are possible. "
        "Speak as if data came from yourself. "
        "You don't have to use every piece of data, just the most relevant ones. "
        "Correct spelling and grammar, even in codes."
    )
    def __init__(
        self, user_role, building_type
    ):
        self.messages_history = []
        self.user_role = user_role
        self.building_type = building_type
        self.system_message = AppResponseMachine.SYSTEM_MESSAGE
        self.all_responses = []

        self.add_system_message_to_history()

    def add_message_to_history(self, message):
        self.messages_history.append({"role": "user", "content": message})
        return self.messages_history

    def add_system_message_to_history(self):
        self.messages_history.append({"role": "system", "content": self.system_message})
        return self.messages_history

    def add_assistant_message_to_history(self, message):
        self.messages_history.append({"role": "assistant", "content": message})
        return self.messages_history

    def add_user_message_to_history(self, message):
        self.messages_history.append({"role": "user", "content": str(message)})
        return self.messages_history

    def make_request(self):
        request_body = {
            "model": "gpt-4-0613",
            "messages": self.messages_history,
        }
        response = requests.post(OPENAI_URL, headers=REQUEST_HEADER, json=request_body)
        self.all_responses.append(response.json())

        # parse the response, add to history
        self.add_assistant_message_to_history(response.json()["choices"][0]["message"]["content"])

        return response.json()


def prep_input(pinecone_response_list: str):
    # first turn str into list of tuples
    pinecone_response_list = json.loads(pinecone_response_list.replace('\\"', '"'))
    docs = []
    for item in pinecone_response_list:
        # item is a dict
        query = item["topic"]
        content_ret = str(item["content"])
        docs.append(f"Query: {query}\nData: {content_ret}\n")
    return "\n".join(docs)



def get_gpt_prompt(user_role, building_type, initial_user_message, pinecone_response_list):
    gpt_prompt = (
        f"I am a {user_role} looking for information about {building_type} building codes. "
        f"Here is my question: \n\n'{initial_user_message}'\n\n"
        f"Relevant Documents:\n\n{prep_input(pinecone_response_list)}"
    )
    return gpt_prompt


# if main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT-4 assistant for building codes')
    parser.add_argument(
        '--user_role',
        type=str,
        help='User role (e.g. architect, engineer, etc.)'
    )
    parser.add_argument(
        '--building_type',
        type=str,
        help='Building type (e.g. residential, commercial, etc.)'
    )
    parser.add_argument(
        '--initial_user_message',
        type=str,
        help='Initial user message (e.g. What are the structural requirements?)'
    )

    # Handling list type argument
    parser.add_argument(
        '--pinecone_response_list',
        type=str,
        help='List of tuples (query, (index, section_text))',
    )

    args = parser.parse_args()

    gpt_prompt = get_gpt_prompt(args.user_role, args.building_type, args.initial_user_message, args.pinecone_response_list)

    arm = AppResponseMachine(args.user_role, args.building_type)
    arm.add_user_message_to_history(gpt_prompt)
    response = arm.make_request()

    # Printing GPT-4's response
    print(response["choices"][0]["message"]["content"])
    sys.exit(0)
