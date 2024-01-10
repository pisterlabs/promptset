import configparser
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import revolt

# You'll need to install the Revolt.py library, which is a Python wrapper for the Revolt API.
# pip install revolt.py

# The main function.
class AIAssistant:
    def __init__(self, config_file_path: str):
        self.config_file_path = config_file_path
        self.openai_api_key = None

    def load_config(self):
        config_parser = configparser.ConfigParser()
        config_parser.read(self.config_file_path)

        openai_api_key = config_parser.get("openai", "api_key")
        revolt_api_url = config_parser.get("revolt", "api_url")
        revolt_bot_token = config_parser.get("revolt", "bot_token")


        return openai_api_key, revolt_api_url, revolt_bot_token

    def generate_response(self, question: str) -> str:
        """Generates a response to the given prompt, taking into account past interactions and the config variables."""

        template = """Question: {question}
        Answer: Let's think step by step."""

        prompt = PromptTemplate(template=template, input_variables=["question"])

        # llm = OpenAI(openai_api_key=self.openai_api_key)
        # llm_chain = LLMChain(prompt=prompt, llm=llm)
        # response = llm_chain.run(question)

        response = "I hear yar, I hear you."
        # Return the response.
        return response

    async def on_message(self, message: revolt.Message):
        """Event listener for new messages."""

        # Don't reply to own messages
        if message.author.id == self.client.user.id:
            return

        # If the message is "quit", disconnect the bot.
        if message.content == "quit":
            await self.client.logout()

        # Generate a response to the message's content.
        response = self.generate_response(question=message.content)

        # Send the response to the channel the message was received from.
        await message.channel.send(response)

    def run(self):
        """Starts the Revolt chat bot interaction."""
        self.openai_api_key, revolt_api_url, revolt_bot_token = self.load_config()

        self.client = Client(api_url=revolt_api_url)

        self.client.run(revolt_bot_token)

        # Set up the event listener for new messages.
        self.client.event(self.on_message)

if __name__ == "__main__":
    # Get the path to the config file from the user.
    config_file_path = input("Enter the path to the config file: ")

    # Create an AI assistant with the optional API URL.
    ai_assistant = AIAssistant(config_file_path=config_file_path)

    # Run the Revolt chat bot.
    ai_assistant.run()
