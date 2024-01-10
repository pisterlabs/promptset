import json
from openai import OpenAI
import keys

# OpenAI and chat history file details
openai_client = OpenAI(api_key=keys.openai_api_key)
chat_history_file = 'chat_history.json'

def read_chat_history_from_file(filename=chat_history_file):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []  # Return an empty list if the file doesn't exist

def write_chat_history_to_file(chathistory, filename=chat_history_file):
    with open(filename, 'w') as file:
        json.dump(chathistory, file, indent=4)

def summarize_chat_history(chat_history):
    # Combine the chat history into a single string
    chat_text = " ".join([message["content"] for message in chat_history])

    # Generate summary using OpenAI
    prompt = f"Please summarize the following conversation into key points. Be sure to be as detailed as possible but use as few of words as possible. Make sure you retain the important parts of the conversation: {chat_text}"
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "system", "content": prompt}]
    )

    return response.choices[0].message.content

def main():
    chathistory = read_chat_history_from_file()
    if chathistory:
        summary = summarize_chat_history(chathistory)
        # Overwrite the chat history with the summary
        summarized_history = [{"role": "system", "content": summary}]
        write_chat_history_to_file(summarized_history)
        print("Chat history summarized and saved.")
    else:
        print("No chat history to summarize.")

main()
