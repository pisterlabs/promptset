import os
import json
from openai import OpenAI
from gepetto.response import ChatResponse, FunctionResponse
class MistralModel():
    name = "Minxie"
    def get_token_price(self, token_count, direction="output", model_engine="mistralai/Mistral-7B-Instruct-v0.1"):
        return (0.50 / 1000000) * token_count

    async def chat(self, messages, temperature=1.0, model="mistralai/Mistral-7B-Instruct-v0.1"):
        """Chat with the model.

        Args:
            messages (list): The messages to send to the model.
            temperature (float): The temperature to use for the model.

        Returns:
            str: The response from the model.
            tokens: The number of tokens used.
            cost: The estimated cost of the request.
        """
        api_key = os.getenv("ANYSCALE_API_KEY")
        api_base = os.getenv("ANYSCALE_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        # print(str(response.choices[0].message))
        tokens = response.usage.total_tokens
        cost = (0.50 / 1000000) * tokens
        message = str(response.choices[0].message.content)
        return ChatResponse(message, tokens, cost)

    async def function_call(self, messages = [], tools = [], temperature=0.7, model="mistralai/Mistral-7B-Instruct-v0.1"):
        api_key = os.getenv("ANYSCALE_API_KEY")
        api_base = os.getenv("ANYSCALE_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=api_base)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": tools[0]["function"]["name"]}},
        )
        # print(str(response.choices[0].message))
        tokens = response.usage.total_tokens
        cost = (0.50 / 1000000) * tokens
        message = response.choices[0].message
        parameters = json.loads(message.tool_calls[0].function.arguments)
        return FunctionResponse(parameters, tokens, cost)
