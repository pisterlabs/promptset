import os
import openai
from typing import List
# import markdown2

openai.api_key = "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"

conversation_history: List[dict] = []


def send_message_to_chatgpt(message: str) -> str:
    conversation_history.append({'role': 'user', 'content': message})

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                     {'role': 'system',
                      'content': 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.'}
                 ] + conversation_history,
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({'role': 'assistant', 'content': assistant_message})
    return assistant_message


def save_conversation_as_md(filename: str):
    with open(filename, 'w') as f:
        for message in conversation_history:
            f.write(f"## {message['role']}:\n")
            f.write(f"{message['content']}\n")


def main():
    base_dir = "generated_code"
    chat_history_dir = "chat_history/test_cases_gen"
    os.makedirs(chat_history_dir, exist_ok=True)

    for i in range(6, 50):

        subfolder_name = f"code_{str(i).zfill(2)}"
        filename = f"code_{str(i).zfill(2)}.py"
        path = os.path.join(base_dir, subfolder_name, filename)
        print(path)
        with open(path) as f:
            code = f.read()

            # prompt 1
            message = f"{code}\nfor this code, your task is to make it runnable without bugs  give me the full code where the main function calling this function"
            send_message_to_chatgpt(message)

            # prompt 2
            message = "then write a class to test every line of the code you just generated"
            send_message_to_chatgpt(message)

            # save the conversation
            filename = f"conversation_{str(i).zfill(2)}.md"
            save_conversation_as_md(os.path.join(chat_history_dir, filename))

            # clear the conversation history for the next round
            conversation_history.clear()


if __name__ == "__main__":
    main()
