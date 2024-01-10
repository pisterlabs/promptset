import asyncio
import openai
import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

MAX_TOKENS = 8192
GPT_MODEL = "gpt-4"
# Initialize conversation history
conversation_history = [
    {
        "role": "system",
        "content": "You are a helpful assistant, do not say you are an AI or explain what you are.",
    },
]


async def generate_text(prompt, conversation_history):
    # Add user message to the conversation history
    conversation_history.append({"role": "user", "content": prompt})

    completion = await asyncio.to_thread(
        openai.ChatCompletion.create,
        model=GPT_MODEL,
        messages=conversation_history,
    )

    # Extract assistant response and add it to the conversation history
    assistant_response = completion.choices[0]["message"]["content"]
    conversation_history.append({"role": "assistant", "content": assistant_response})

    return assistant_response


def count_tokens(string: str, encoding_model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_model_name)
    # encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def truncate_conversation_history(conversation_history, max_tokens=MAX_TOKENS):
    tokens = 0
    truncated_history = []
    system_message = conversation_history[0]

    for message in reversed(conversation_history[1:]):  # Skip the system message
        message_tokens = count_tokens(message["content"], GPT_MODEL)
        tokens += message_tokens

        if tokens > max_tokens:
            break

        truncated_history.insert(0, message)

    # Keep the system message at the beginning
    truncated_history.insert(0, system_message)
    return truncated_history


async def main():
    global conversation_history
    while True:
        try:
            # Truncate the conversation history if it exceeds the token limit
            conversation_history = truncate_conversation_history(conversation_history)

            # Call generate_text with the truncated conversation history
            user_input = input("Ask something (type 'quit' to exit): ").strip()

            if user_input.lower() == "quit":
                break

            response = await generate_text(user_input, conversation_history)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
