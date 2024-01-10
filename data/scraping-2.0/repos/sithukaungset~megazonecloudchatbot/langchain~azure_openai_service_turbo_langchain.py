# Make sure you have the latest openai and langchain version running:
# pip install openai --upgrade
# pip install langchain --upgrade

import os
import openai
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Load environment variables (set OPENAI_API_KEY and OPENAI_API_BASE in .env)
load_dotenv()

# Configure Azure OpenAI Service API
openai.api_type = "azure"
openai.api_version = "2023-03-15-preview"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Init LLM and embeddings model
llm = AzureChatOpenAI(deployment_name="gpt354k",
                      temperature=0.7, openai_api_version="2023-03-15-preview")

system_message = "You are an AI assistant that tells jokes."

system_message_prompt = SystemMessagePromptTemplate.from_template(
    system_message)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt])

chain = LLMChain(llm=llm, prompt=chat_prompt)
result = chain.run(f"Tell me a dad joke")
print(result)
