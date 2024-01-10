from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import datetime
from dotenv import load_dotenv
import os


def save_conversation_to_file(conversation, timestamp):
    filename = f"chat_{timestamp}.txt"
    with open(filename, "w") as file:
        for message in conversation:
            file.write(f"{message['role'].title()}: {message['content']}\n")
    print(f"Conversation saved to {filename}")


def main():
    load_dotenv()  # Load environment variables from .env file
      # Get API key from environment variable

    print("ChatGPT-4 Command Line Interface")
    print("Type 'quit' to exit and save the conversation.\n")

    conversation_history = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        conversation_history.append(user_input)

        try:
            response = client.completions.create(engine="gpt-4-1106-preview",  # You might need to change this depending on the available models
            prompt="\n".join(conversation_history),
            max_tokens=150,  # Adjust as needed
            stop=None,  # You can set stopping criteria here
            temperature=0.7  # Adjust as needed)

            chatgpt_response = response.choices[0].text.strip()
            print("ChatGPT:", chatgpt_response)
            conversation_history.append("ChatGPT: " + chatgpt_response)
        except Exception as e:
            print("An error occurred:", e)

    save_conversation_to_file(conversation_history, timestamp)


if __name__ == "__main__":
    main()