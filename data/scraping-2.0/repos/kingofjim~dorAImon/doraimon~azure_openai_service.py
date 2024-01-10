import os
from dotenv import load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.memory import ChatMessageHistory
from langchain.chat_models import AzureChatOpenAI


class AzureOpenAIService:
    def __init__(self):
        load_dotenv()

        self.chatbot = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_version="2023-07-01-preview",
            deployment_name="gpt-35-turbo-16k_canadaeast",
            openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
            openai_api_type="azure",
            temperature=0.7,
        )

        system_template = """Your are a assistant to help HR to find the best candidate for the job."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(
            system_template)

        init_message = system_message_prompt.format()
        self.history = ChatMessageHistory()
        self.history.add_message(init_message)

    def chat(self, message):
        messages = self.generate_prompt(message)
        response = self.chatbot(messages)
        self.history.add_ai_message(response.content)

        return response

    def chatbot(self, prompt):
        response = self.chatbot(prompt)
        self.history.add_ai_message(response.content)

        return response

    def generate_prompt(self, message):
        # system_template = ""
        human_template = """{user_message}"""

        # system_message_prompt = SystemMessagePromptTemplate.from_template(
        #     system_template)

        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template)

        chat_prompt = ChatPromptTemplate.from_messages([
            # system_message_prompt,
            human_message_prompt
        ])

        messages = chat_prompt.format_prompt(
            user_message=message
        ).to_messages()

        for message in messages:
            self.history.add_message(message)

        return self.history.messages
