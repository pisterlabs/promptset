import openai
import json
import os
from dotenv import load_dotenv
# Initialize OpenAI and GitHub API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
class ChatGPTAgent:
    def __init__(self, prompt):
        self.prompt = prompt

    # Initialize a session with OpenAI's chat models
    @staticmethod
    def chat_with_gpt3(prompt, systemprompt):
        messages = [{"role": "user", "content": prompt}]
        if systemprompt:
            messages.insert(0, {"role": "system", "content": systemprompt})
        else:
            messages.insert(0,{"role": "system", "content": "You are a helpful AI assistant"})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            temperature=0.9
        )
        return response['choices'][0]['message']['content']
            # Initialize a session with OpenAI's chat models
    @staticmethod
    def function_with_gpt3(prompt, systemprompt=None):
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=messages,
            functions=[
                {
                    "name": "get_current_price",
                    "description": "Get the current price of a stock",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock": {
                                "type": "string",
                                "description": "The stock to get the price of"
                            },
                    "unit": {
                        "type": "string",
                        "enum": ["USD", "EUR", "GBP"]
                    }
                        },
                "required": ["stock"]
                    }
                }
            ],
            function_call="auto",
        )
        message = response['choices'][0]['message']
        if message.get("function_call"):
            function_name = message['function_call']['name']

            function_response = get_current_price(
                stock=message.get("stock"),
                unit=message.get("unit"),
            )

            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "user", "content": "What is the stock price of AAPL?"},
                    message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": function_response,
                    },
                ],
            )
            return second_response

    def generate(self, input_data):
        return ChatGPTAgent.chat_with_gpt3(self.prompt, input_data)

def get_current_price(stock, unit="USD"):
    """Get the current price of a given stock"""
    print("replace with real API call")
    price = 100
    stock_info = {
        "stock": stock,
        "price": price,
        "unit": unit,
    }
    return json.dumps(stock_info)

def get_price(stock):
    prompt = f"whats the price of {stock} right now?"
    print(prompt)
    result = ChatGPTAgent.function_with_gpt3(prompt)
    message = result['choices'][0]['message']['content']
    return message