from google.auth import credentials
from google.oauth2 import service_account
import google.cloud.aiplatform as aiplatform
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
import vertexai
import json  # add this line
from langchain.chat_models import ChatVertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage


class LanguageModel:
    credentials = None
    project_id = None
    chatVertexAI = ChatVertexAI()

    def __init__(self, model_name: str):
        self.model_name = model_name

        # Load the service account json file
        # Update the values in the json file with your own
        with open(
            "service_account.json"
        ) as f:  # replace 'serviceAccount.json' with the path to your file if necessary
            service_account_info = json.load(f)
            self.project_id = service_account_info["project_id"]
    

        self.credentials = service_account.Credentials.from_service_account_info(
            service_account_info
        )

        # Initialize Google AI Platform with project details and credentials
        aiplatform.init(
            credentials=self.credentials,
        )

    async def start_chat(self, messages= None, **kwargs):
        if messages is None:
            messages = [
                SystemMessage(content="Hello."),
                HumanMessage(content="World."),
            ]

        self.chatVertexAI(messages)


    async def get_response(self, human_msg) -> str:

        # Initialize Vertex AI with project and location
        vertexai.init(project=self.project_id, location="us-central1")

        chat_model = ChatModel.from_pretrained("chat-bison@001")
        parameters = {
            "temperature": 0.8,
            "max_output_tokens": 1024,
            "top_p": 0.8,
            "top_k": 40,
        }
        chat = chat_model.start_chat(  # Initialize the chat with model
            # chat context and examples go here
        )
        # Send the human message to the model and get a response
        response = chat.send_message(human_msg, **parameters)
        # Return the model's response
        return response.text
