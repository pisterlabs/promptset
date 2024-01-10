import openai  # Import the OpenAI library
import time    # Import the time module for adding delays

openai.api_key = '*****************************************'


# Function to send a prompt to the chatbot and receive the response
def chatbot_pipeline(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    generated_text = response['choices'][0]['message']['content']
    return generated_text


# Function to facilitate a delay before sending the prompt to the chatbot
def slow_chatbot_pipeline(prompt, delay_seconds=10):
    time.sleep(delay_seconds)
    return chatbot_pipeline(prompt)


# Call the chat_with_chatbot function to start the conversation
def chat_with_chatbot():
    user_prompt = ""

    while user_prompt.lower() != "exit" or user_prompt.lower() == "thank you":
        user_prompt = input("User: ")
        if user_prompt.lower() == "exit" or user_prompt.lower() == "thank you":
            print("Chat ended. Thank you!")
            break

        chatbot_response = slow_chatbot_pipeline(user_prompt)
        print("Chatbot: ", chatbot_response)


# Call the chat_with_chatbot function
chat_with_chatbot()
