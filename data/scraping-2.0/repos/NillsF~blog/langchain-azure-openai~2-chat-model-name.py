import os
import dotenv

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage

# Load environment variables from .env file
dotenv.load_dotenv(dotenv_path='./chat.env')

# Create an instance of the AzureChatOpenAI class using Azure OpenAI
llm = AzureChatOpenAI(
    deployment_name=os.getenv("OPENAI_DEPLOYMENT_NAME"),
    temperature=0.7,
    openai_api_version="2023-05-15")

# Testing chat llm  
res = llm([HumanMessage(content="Tell me a joke about a penguin sitting on a fridge.")])
print(res)
