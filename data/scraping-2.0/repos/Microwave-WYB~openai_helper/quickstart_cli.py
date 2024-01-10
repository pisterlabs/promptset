"""
Demonstrates how to use the openai_helper package to create a chatbot that can generate
random numbers in a CLI interface.
"""
from random import randint
from openai_helper import FunctionCallManager, ChatSession, HistoryManager

functions = FunctionCallManager()


@functions.register
def random_number(min_number: int, max_number: int) -> int:
    """
    Generate a random number from min to max.

    Args:
        min_number (int): The minimum bound for the random number.
        max_number (int): The maximum bound for the random number.

    Returns:
        int: A random number between min and max.
    """
    return randint(min_number, max_number)


if __name__ == "__main__":
    SYSTEM_PROMPT = """
    You are a random number generator assistant. You will help your user to generate random numbers.
    If the user did not provide a range, you need to ask the user for a range.
    """
    message = {"role": "system", "content": SYSTEM_PROMPT}
    history_manager = HistoryManager(
        token_threshold=3000,
        max_tokens=8000,
        compacting_method="fifo",
        keep_top=1,
        keep_bottom=5,
        messages=[message],
        verbose=True,
    )
    chat = ChatSession(functions, model="gpt-4", verbose=True)
    chat.start(no_confirm=True, history_manager=history_manager)
