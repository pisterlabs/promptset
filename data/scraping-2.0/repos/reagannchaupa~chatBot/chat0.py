# Demonstrates a simple interaction with a chatbot

import openai


def main():
    # Setup the API and start chat
    openai.api_key = "OPENAI_API_KEY"
    chat()


# Define how to get a response from chatbot
def ask_chatbot(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150,
    )
    return response["choices"][0]["text"].strip()


# Use that function to chat with the chatbot
def chat():
    # Continuously try to get and print a response from chatbot until user stops.
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Bot: Goodbye!")
                break
            bot_response = ask_chatbot("You: " + user_input + "\nBot:")
            print(f"Bot: {bot_response}")
    except:
        print("Something went wrong!")


if __name__ == "__main__":
    main()
