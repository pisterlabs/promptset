"""
Demonstrates how to use the openai_helper package to create a chatbot that can generate
random numbers.
"""
from random import randint
from openai_helper import FunctionCallManager, ChatSession

functions = FunctionCallManager()


@functions.register
def random_number(min_number: int, max_number: int) -> int:
    """
    Generate a random number from min_number to max_number.

    Args:
        min_number (int): The minimum bound for the random number.
        max_number (int): The maximum bound for the random number.

    Returns:
        int: A random number between min_number and max_number.
    """
    return randint(min_number, max_number)


if __name__ == "__main__":
    message = {
        "role": "user",
        "content": "Generate a random number between 1 and 10000",
    }
    chat = ChatSession(functions, model="gpt-4", verbose=True)

    response = chat.send_messages([message], temperature=0, max_tokens=500)

    print(response.choices[0].message.content)

    if response.choices[0].finish_reason == "function_call":
        function_call = response.choices[0].message.function_call
        function_output = chat.handle_function(function_call, verbose=True)

    response = chat.send_messages(
        [message, function_output], temperature=0, max_tokens=500
    )
    print(response.choices[0].message.content)
