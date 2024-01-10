import os
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

load_dotenv()
api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_endpoint = os.environ.get("AZURE_OPENAI_API_ENDPOINT")

# Initialize AzureChatOpenAI
azure_chat = AzureChatOpenAI(
    azure_deployment="carecoach-gpt35-16k",  # Replace with your deployment name
    api_key=api_key,
    azure_endpoint=azure_endpoint,
)

message = HumanMessage(
    content="Translate this sentence from English to French. I love programming."
)

# Send a chat message
response = azure_chat([message])

# Print the response
print(response)
