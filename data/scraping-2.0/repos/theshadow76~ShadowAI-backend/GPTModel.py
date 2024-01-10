import openai
import json
import requests  # Required for making API calls

from env import OpenAIKeys
from image_generation.generate import generate_image
from help.helper_functions import get_data_from_txt
from Firebase import get_firebase_database
from Model import Text2TextModel
from tools.tools import GetLinkData, SearchStable, UsePythonCodeInterpreter

class OpenAIModel(Text2TextModel):
    def __init__(self, model: str | None = None):
        openai.api_key = OpenAIKeys.OPENAI_API_KEY
        self.model = model if model else "gpt-3.5-turbo-0613"
        self.processing = False
        self.db = get_firebase_database()
        self.transforms = {
            "\n": " ",
            r"\{([^}]+)\}": "{}"
        }
        self.available_functions = {
            "generate_image": generate_image,
            "fetch_all_messages": self.fetch_all_messages,
            "GetLinkData" : GetLinkData,
            "SearchStable" : SearchStable,
            "UsePythonCodeInterpreter" : UsePythonCodeInterpreter
        }
        self.logger = self.setup_logger()

    def _get_functions(self, **kwargs):
        return [
            {
                "name": "generate_image",   
                "description": get_data_from_txt(OpenAIKeys.GENERATE_IMAGE_PATH),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string", "description" : "The image prompt"},
                    },
                    "required": ["description"],
                },
            },
            {
                "name" : "fetch_all_messages",
                "description" : get_data_from_txt(OpenAIKeys.FETCH_ALL_MESSAGES_PATH, self.transforms).format(kwargs['user_id']),
                "parameters" : {
                    "type" : "object",
                    "properties" : {
                        "chat_id" : {"type" : "string", "description" : f"The id of the chat, please use this id: {kwargs['chat_id']}"}
                    },
                    "required" : ["chat_id"]
                }
            },
            {
                "name" : "GetLinkData",
                "description" : get_data_from_txt(OpenAIKeys.GET_LINK_DATA_PATH),
                "parameters" : {
                    "type" : "object",
                    "properties" : {
                        "link" : {"type" : "string", "description" : "The link of the provided url"}
                    },
                }
            },
            {
                "name" : "UsePythonCodeInterpreter",
                "description" : get_data_from_txt(OpenAIKeys.CODE_INTERPRETER_PATH),
                "parameters" : {
                    "type" : "object",
                    "properties" : {
                        "code" : {"type" : "string", "description" : "The code you want to execute"}
                    },
                }
            }
        ]

    def fetch_all_messages(self, description, chat_id):
        """Get the previous messages of the user in the same chat"""
        self.processing = False

        chats_collection = self.db.collection('chat').document(chat_id).collection('messages')

        try:
            query_snapshot = chats_collection.get()
            chat_docs = list(query_snapshot)

            for doc in chat_docs:
                chat_data = doc.to_dict()
                description.append(
                    {
                        'content': chat_data['content'],
                        'isUserMessage': chat_data['sender'] == 'bot',
                        'dateTime': chat_data['timestamp']
                    }
                )

            # title_abbar = description[-1]['content'] if description else None
            self.processing = True
            # is_loading = False

            # Log the messages to console for debugging
            self.logger.debug(f"Fetched {len(description)} messages from chat {chat_id}")
            return description

        except Exception as e:
            self.logger.error(f"An error occurred: {e}")
            return {"error": str(e)}  # Returning error message for better debugging


    def run_conversation(self, prompt, chat_id, user_id):
        functions = self._get_functions(user_id=user_id, chat_id=chat_id)
        prevmessages = self.fetch_all_messages(chat_id=chat_id, description=[])
        print(prevmessages)
        # prevmessages_clean = prevmessages[0]['content']
        messages = [{"role": "system", "content": get_data_from_txt(OpenAIKeys.SYSTEM_PATH)}, {"role" : "user", "content" : f"{prompt}"}]
        
        for i in prevmessages:
            if i['isUserMessage'] == False:
                messages.append({"role" : "user", "content" : f"{i['content']}"})
            elif i['isUserMessage']:
                messages.append({"role" : "assistant", "content" : f"{i['content']}"})
            else:
                self.logger.debug("Messages not added, skipping...")

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=functions,
            function_call="auto",
        )
        response_message = response["choices"][0]["message"]

        if response_message.get("function_call"):
            function_name = response_message["function_call"]["name"]
            function_to_call = self.available_functions.get(function_name, None)
            if function_to_call is None:
                return {"Response" : response_message}
            
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(**function_args)  # Changed line

            messages.append(response_message)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )
            second_response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
            )
            return second_response
        else:
            return {"Response" : response_message}

    