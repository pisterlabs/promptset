import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_KEY")

def chat_with_openai(lists,init_text):
    """
    Send a series of messages to the OpenAI API and get a response.

    Parameters:
    - lists: A list of lists where each sub-list has two elements:
      1. A 0 or 1, where 0 indicates "assistant" and 1 indicates "user".
      2. The message content.

    Returns:
    - A string containing the assistant's response.
    """
    print("Chatting with OpenAI...")
    
    # Convert the list of lists to the desired format for the API
    messages = []

    messages.append({"role": "system","content":"You are a legal chatbot that is designed to help a user understand "
                                                "the nuances of a legal document, read the document attached in this "
                                                "message and answer all the queries of the user clearly, remember to "
                                                "always act like the legal chatbot that you are, always answer the question of "
                                                "the user to the point and do not include information they didn't ask"
                                                "unless they explicitly asked to explain and start with a very small and brief"
                                                "welcome message to the user by summarizing only the very key nuances of the "
                                                "document  Document:"+init_text})

    for item in lists:
        role = "assistant" if item[0] == 0 else "user"
        content = item[1]
        messages.append({"role": role, "content": content})

    # Send the formatted messages to the OpenAI API
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # Extract the assistant's response from the API completion and return it
    return completion["choices"][0]["message"]["content"]


# Driver Code
# if __name__ == "__main__":
#     # Initial message list
#     messages_list = []

#     with open('sample.txt', 'r') as file:
#         legal_document = file.read()
#         while True:
#             # Get assistant's response based on the current message list
#             response = chat_with_openai(messages_list,legal_document)

#             # Display the assistant's response
#             print(f"Assistant: {response}")

#             # Add assistant's response to the list for context in future interactions
#             messages_list.append([0, response])

#             # Get user input
#             user_message = input("You: ")

#             # Add user message to the list
#             messages_list.append([1, user_message])

#             # Check if user wants to continue
#             continue_chat = input("Continue chatting? (yes/no): ").strip().lower()
#             if continue_chat != "yes":
#                 print("Goodbye!")
#                 break
