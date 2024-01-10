import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_file = "env_vars.env"
if not load_dotenv(env_file):
    raise ValueError(f"Failed to load the environment variables from {env_file}")

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
else:
    # Set OpenAI API key
    openai.api_key = openai_api_key

class GPT_Stock:
    def __init__(self, stock_symbols):
        """Initialize the object with the stock symbols"""
        self.stock_symbols = stock_symbols
        # Begin the conversation with the GPT-3 model with a system message
        self.messages = [{"role": "system", "content": f"You are a stock analysis. Provide a summary comparison of the following companies: {', '.join(stock_symbols)}. Make sure that you provide a paragraph for each company and a summary of your comparison, that is all you need."}]

    # Method to generate a comparison analysis using OpenAI's GPT-3
    def CustomChatGPT(self, user_input):
        """Append user message to the conversation, make an API call to get a reply from the model, and append the model's reply to the conversation"""
        self.messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        ChatGPT_reply = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": ChatGPT_reply})
        return ChatGPT_reply

    # Method to get the comparison analysis
    def get_comparison_analysis(self):
        """Get the comparison analysis by providing the appropriate user input"""
        user_input = "Please provide a detailed comparison analysis."
        comparison_analysis = self.CustomChatGPT(user_input)
        return comparison_analysis